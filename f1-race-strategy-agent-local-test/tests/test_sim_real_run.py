"""One command for the real runs (SPEC 0021, src/sim/real_run.py): the smoke test on the
committed Spain cache in a tmp directory, resumption, every stage's skip, the Sol scripts
(emitted, never submitted) and the reference rebuild. Offline: no data is fetched, nothing is
written under reports/calibration or data/references."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.sim import calibrate, cross_race, event_rates, fuel_prior, jobs, real_run
from src.sim.compute import load_profile
from src.sim.real_run import CURRENT, PENDING, RAN, SKIPPED, STAGES, RealRun

ROOT = real_run.ROOT
ENTRY_KEYS = {"stage", "status", "reason", "inputs", "outputs", "code_version", "wall_time_s", "fingerprint"}
COMMITTED = (real_run.REPORTS_DIR, ROOT / "data" / "references")
N_ARMS = len(cross_race.ARMS)   # one h_fold task per committed arm on a one-race run


def small_profile(name="local", scheduler="local"):
    p = load_profile("local")
    return p.model_copy(update={"name": name, "scheduler": scheduler,
                                "g": p.g.model_copy(update={"laps_every": 20, "n_boot": 20}),
                                "sim": p.sim.model_copy(update={"n_samples": 5}),
                                "parallel": p.parallel.model_copy(update={"n_jobs": 1}),
                                "wandb": p.wandb.model_copy(update={"mode": "disabled"})})


def snapshot(dirs=COMMITTED):
    """(path, size, mtime) of every file under the committed report and reference directories."""
    return sorted((str(f), f.stat().st_size, f.stat().st_mtime_ns) for d in dirs for f in Path(d).rglob("*")
                  if f.is_file())


def make_run(tmp, profile, races, log, **kw):
    kw = {"periods": tmp / "no_periods.csv", "race_features": tmp / "no_features.csv",
          "reference_dir": tmp / "references", "tau_multipliers": (1.0,), "n_boot": 50, "log": log.append, **kw}
    (tmp / "references").mkdir(exist_ok=True)
    return RealRun(races, profile, tmp / "out", **kw)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real_run")
    before = snapshot()
    # The smoke run stays on the 2021 Spain cache so its cost and expectations do not grow
    # with every newly cached race.
    _, notes = real_run.cached_races()
    log = []
    manifest = real_run.execute(make_run(tmp, small_profile(), ["2021_04_spain"], log, race_notes=notes,
                                         race_source="auto: every cached race"))
    return tmp, manifest, log, before


def test_cached_races_are_the_committed_spain_cache():
    # Every cached season has its event schedule in the sqlite (dataset 3 added a seeded
    # 2018-25 sample); the sprint weekends have no FP2/FP3 practice prior, so discovery
    # leaves them out with a note until the simulator supports them.
    assert real_run.cached_races() == ([
        "2019_06_monaco", "2019_10_united_kingdom", "2020_16_sakhir", "2021_04_spain",
        "2022_02_saudi_arabia", "2023_01_bahrain", "2023_07_spain", "2024_07_emilia_romagna",
        "2024_08_monaco", "2024_12_united_kingdom", "2025_11_austria",
    ], [
        "2024/2024-06-30_Austrian_Grand_Prix: sprint weekend (no FP2/FP3 practice prior), left out",
        "2024/2024-11-03_S\u00e3o_Paulo_Grand_Prix: sprint weekend (no FP2/FP3 practice prior), left out",
        "2025/2025-11-30_Qatar_Grand_Prix: sprint weekend (no FP2/FP3 practice prior), left out",
    ])


def test_smoke_real_run_on_the_spain_cache_finishes_with_a_complete_manifest(smoke):
    tmp, manifest, log, before = smoke
    assert manifest["complete"] and [e["stage"] for e in manifest["stages"]] == list(STAGES)
    assert manifest["races"] == ["2021_04_spain"] and manifest["mode"] == "in-process, n_jobs 1"
    by = {e["stage"]: e for e in manifest["stages"]}
    for e in manifest["stages"]:
        assert ENTRY_KEYS <= set(e), e["stage"]
        assert e["status"] in (RAN, SKIPPED), e
        assert all("present" in v for v in e["inputs"].values()), e["stage"]
        assert e["code_version"]["sim_code_version"] == real_run.calibration.SIM_CODE_VERSION
        assert e["wall_time_s"] >= 0
        if e["status"] == SKIPPED:   # skipped with its reason, printed
            assert e["reason"] and any(e["reason"] in line for line in log), e["stage"]
        for p in e["outputs"]:
            assert Path(p).exists(), p
    assert {s: by[s]["status"] for s in STAGES} == {
        "eligibility": RAN, "cross_race": RAN, "event_rates": SKIPPED, "fuel_prior": SKIPPED, "references": RAN,
        "report": RAN}
    assert "neutralised periods table" in by["event_rates"]["reason"]
    assert not by["event_rates"]["inputs"]["periods"]["present"]
    assert "1 eligible race(s), fewer than 8" in by["fuel_prior"]["reason"]
    assert by["references"]["reason"] == "no stale reference"
    out = tmp / "out"
    assert json.loads((out / "real_run" / "eligibility.json").read_text())[0]["eligible"]
    assert (out / "cross_race" / "run.json").is_file() and (out / "cross_race" / "summary.md").is_file()
    assert any(line.startswith("Decision for to_snapshot") for line in by["report"]["detail"]["decisions"])
    assert snapshot() == before   # nothing under reports/calibration or data/references changed


def test_a_rerun_resumes_every_stage_that_ran(smoke):
    tmp, first, _, _ = smoke
    log = []
    manifest = real_run.execute(make_run(tmp, small_profile(), ["2021_04_spain"], log))
    status = {e["stage"]: e["status"] for e in manifest["stages"]}
    assert status == {s: CURRENT if e["status"] == RAN else SKIPPED
                      for s, e in ((e["stage"], e) for e in first["stages"])}
    ran = {e["stage"]: e for e in first["stages"]}
    for e in manifest["stages"]:
        if e["status"] == CURRENT:
            assert e["ran_at"] == ran[e["stage"]]["ran_at"] and e["outputs"] == ran[e["stage"]]["outputs"]
            assert e["ran_wall_time_s"] == ran[e["stage"]]["wall_time_s"]


def test_with_no_race_every_stage_skips_with_its_reason(tmp_path):
    log = []
    manifest = real_run.execute(make_run(tmp_path, small_profile(), [], log, reference_dir=tmp_path / "none"))
    by = {e["stage"]: e for e in manifest["stages"]}
    assert manifest["complete"] and all(e["status"] == SKIPPED for e in manifest["stages"])
    assert "no race" in by["eligibility"]["reason"]
    assert "no eligibility table" in by["cross_race"]["reason"]
    assert "0 eligible race(s), fewer than 8" in by["fuel_prior"]["reason"]
    assert "no reference directory" in by["references"]["reason"]
    assert "the cross-race pass is skipped" in by["report"]["reason"]
    assert all(any(e["reason"] in line for line in log) for e in manifest["stages"])


def test_event_rates_run_from_the_periods_table_when_it_is_there(tmp_path):
    periods, feats = event_rates.synthetic_periods(range(2020, 2024))
    periods.to_csv(tmp_path / "periods.csv", index=False)
    feats.to_csv(tmp_path / "features.csv", index=False)
    run = make_run(tmp_path, small_profile(), [], [], periods=tmp_path / "periods.csv",
                   race_features=tmp_path / "features.csv")
    e = real_run.stage_event_rates(run)
    assert e["status"] == RAN and e["inputs"]["periods"]["present"] and e["inputs"]["periods"]["sha256"]
    assert all(Path(p).is_file() for p in e["outputs"])
    assert (tmp_path / "out" / "event_rates" / "summary.md").is_file()


def _factors(n):
    ids = [f"2021_{i:02d}_race" for i in range(1, n + 1)]
    return pd.DataFrame({"race_id": ids, "total_laps": 60, "sc_count": [i % 2 for i in range(n)],
                         "sc_laps": [3 * (i % 2) for i in range(n)], "vsc_count": 0, "vsc_laps": 0,
                         "sc_start_laps": ["10" if i % 2 else "" for i in range(n)], "vsc_start_laps": ""})


@pytest.mark.parametrize("n", [7, 8])
def test_without_a_periods_table_the_event_rates_use_8_cached_races(tmp_path, n):
    run = make_run(tmp_path, small_profile(), [], [])
    cross = tmp_path / "out" / "cross_race"
    cross.mkdir(parents=True)
    _factors(n).to_csv(cross / "factors.csv", index=False)
    assert real_run.stage_event_rates(run)["status"] == SKIPPED   # the pass did not run here
    run.entries["cross_race"] = {"status": RAN}
    e = real_run.stage_event_rates(run)
    if n < 8:
        assert e["status"] == SKIPPED and "holds 7 eligible race(s), fewer than 8" in e["reason"]
    else:
        assert e["status"] == RAN and e["detail"]["source"] == "cached races only (8)"
        assert "cached races only (8)" in (tmp_path / "out" / "event_rates" / "summary.md").read_text()


def test_the_scheduler_field_not_the_name_picks_slurm(tmp_path):
    assert load_profile("sol").scheduler == "slurm" and load_profile("local").scheduler == "local"
    assert not make_run(tmp_path, small_profile("sol", "local"), [], []).slurm
    assert make_run(tmp_path, small_profile("local", "slurm"), [], []).slurm
    p = load_profile("sol")
    assert p.model_copy(update={"scheduler": "local"}).hash == p.hash   # where units run is not an output


def test_the_fuel_prior_runs_from_8_eligible_races(tmp_path, monkeypatch):
    calls = []

    def fake_run(races, out, profile, features_path=None, log=print, **kw):
        calls.append(list(races))
        Path(out).mkdir(parents=True, exist_ok=True)
        for f in real_run.FUEL_OUTPUTS:
            (Path(out) / f).write_text("x")
    monkeypatch.setattr(fuel_prior, "run", fake_run)
    run = make_run(tmp_path, small_profile(), [], [])
    ids = [f"2022_{i:02d}_race" for i in range(1, 9)]
    run.elig = [{"race_id": r, "eligible": True} for r in ids[:7]] + [{"race_id": ids[7], "eligible": False}]
    assert real_run.stage_fuel_prior(run)["status"] == SKIPPED and not calls
    run.elig[-1]["eligible"] = True
    assert real_run.stage_fuel_prior(run)["status"] == RAN and calls == [ids]


def _stale_sidecar(directory, **change):
    meta = json.loads(sorted((ROOT / "data" / "references" / "v1").glob("*.json"))[0].read_text())
    meta = {**meta, "sim_code_version": "000000stale0", **change}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{meta['key']}{len(list(directory.glob('*.json')))}.json").write_text(json.dumps(meta))
    return meta


def test_a_stale_reference_is_rebuilt_with_its_own_inputs_once(tmp_path):
    refs = tmp_path / "references"
    meta = _stale_sidecar(refs / "v1")
    _stale_sidecar(refs / "v1", world_config_hash="ffffffffffff")
    built = []

    def fake_build(scenario_set, sim_for, root, **kw):
        built.append((scenario_set.version, kw))
        from src.eval.reference import reference_key, reference_path
        path = reference_path(root, scenario_set.version, reference_key(
            meta["set_digest"], meta["world_config_hash"], real_run.calibration.SIM_CODE_VERSION, **kw))
        path.write_text("x")
        return [(path, [])]
    run = make_run(tmp_path, small_profile(), [], [], reference_dir=refs)
    e = real_run.stage_references(run, build=fake_build)
    assert e["status"] == RAN and e["reason"] == "2 stale, 1 not rebuilt"
    assert built == [("v1", {"seed": 0, "n_samples": 50, "max_stops": 2, "hindsight": False})]
    results = {r["status"]: r for r in e["detail"]["stale"]}
    assert results["rebuilt"]["world_config"] == "default (SimConfig())"
    assert "no world config with hash ffffffffffff" in results["not rebuilt"]["reason"]
    assert real_run.rebuild_reference(meta, run.profile, refs)["status"] == "already rebuilt"
    other = {**meta, "scenario_set": "v0"}
    assert "no scenario set" in real_run.rebuild_reference(other, run.profile, refs)["reason"]


def _no_sbatch(monkeypatch):
    """Fail on any process started with sbatch in its arguments."""
    real = subprocess.Popen

    class Guard(real):
        def __init__(self, args, *a, **kw):
            assert "sbatch" not in " ".join(map(str, args if isinstance(args, (list, tuple)) else [args]))
            super().__init__(args, *a, **kw)
    monkeypatch.setattr(subprocess, "Popen", Guard)


def _task_body(script: Path) -> str:
    """The array task as Slurm runs it, without the template's module and venv set-up."""
    lines = script.read_text().splitlines()
    keep = [ln for ln in lines if ln.startswith(("OUT=", "MANIFEST="))]
    cut = next(i for i, ln in enumerate(lines) if ln.startswith("mkdir -p logs"))
    return "\n".join(["set -euo pipefail", *keep, *lines[cut + 1:]]).replace("python -m ", f"{sys.executable} -m ")


def test_sol_emits_the_arrays_from_the_templates_and_submits_nothing(tmp_path, monkeypatch):
    _no_sbatch(monkeypatch)
    profile = small_profile("sol", "slurm")
    path = tmp_path / "sol.yaml"
    path.write_text(yaml.safe_dump(profile.model_dump(mode="json")))
    log = []
    run = make_run(tmp_path, load_profile(str(path)), ["2021_04_spain"], log, profile_ref=str(path))
    manifest = real_run.execute(run)
    by = {e["stage"]: e for e in manifest["stages"]}
    assert manifest["complete"] and manifest["mode"].startswith("slurm")
    assert by["cross_race"]["status"] == PENDING and by["cross_race"]["detail"]["pending"] == 1 + N_ARMS
    assert by["report"]["status"] == SKIPPED and "pending" in by["report"]["reason"]
    assert not (tmp_path / "out" / "cross_race" / "run.json").exists()
    assert not list((tmp_path / "out" / "cross_race").rglob("*.parquet"))   # nothing ran here
    slurm = tmp_path / "out" / "real_run" / "slurm"
    commands = by["cross_race"]["detail"]["sbatch"]
    assert commands == [f"sbatch --array=0-0%50 {slurm / 'cross_race.sbatch'}",
                        f"sbatch --array=0-{N_ARMS - 1}%50 {slurm / 'h_fold_arms.sbatch'}"]
    assert all(any(c in line for line in log) for c in commands)
    for name, template in (("cross_race", "g_race.sbatch"), ("h_fold_arms", "h_fold.sbatch")):
        text = (slurm / f"{name}.sbatch").read_text()
        header = [ln for ln in (ROOT / "slurm" / template).read_text().splitlines()
                  if ln.startswith("#SBATCH") and "--job-name" not in ln]
        assert all(ln in text.splitlines() for ln in header) and "source \"${SIM_VENV" in text
        assert f"OUT={tmp_path / 'out' / 'cross_race'}" in text and "--profile " + str(path) in text
    assert len((slurm / "h_fold_arms_tasks.txt").read_text().splitlines()) == N_ARMS

    # one task of each array, as Slurm would run it, lands where the next real-run looks
    for name in ("cross_race", "h_fold_arms"):
        r = subprocess.run(["bash", "-c", _task_body(slurm / f"{name}.sbatch")], cwd=ROOT, capture_output=True,
                           text=True, env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"}, timeout=300)
        assert r.returncode == 0, r.stderr
    again = real_run.execute(make_run(tmp_path, load_profile(str(path)), ["2021_04_spain"], [],
                                      profile_ref=str(path)))
    cross = {e["stage"]: e for e in again["stages"]}["cross_race"]
    assert cross["status"] == PENDING and cross["detail"]["pending"] == N_ARMS - 1
    assert cross["detail"]["sbatch"] == [f"sbatch --array=0-{N_ARMS - 2}%50 {slurm / 'h_fold_arms.sbatch'}"]
    assert len((slurm / "h_fold_arms_tasks.txt").read_text().splitlines()) == N_ARMS - 1


def test_a_long_phase_is_split_into_arrays_with_an_offset(tmp_path):
    from argparse import Namespace
    emitter = real_run.SlurmEmitter(tmp_path, "sol")
    races = [f"2022_{i:02d}_race" for i in range(1, 3)]
    items = [("h_fold", r, None, str(tmp_path / "out"), Namespace(arm=f"arm{a}", pool=races))
             for a in range(1100) for r in races[:1]]
    emitter.emit(items, "tau_grid")
    assert emitter.commands == [
        f"sbatch --array=0-999%50 {tmp_path / 'h_fold_tau_grid.sbatch'}",
        f"sbatch --array=0-99%50 --export=ALL,SIM_TASK_OFFSET=1000 {tmp_path / 'h_fold_tau_grid.sbatch'}"]


def test_the_cli_takes_auto_or_ids_not_both():
    with pytest.raises(SystemExit):
        calibrate.main(["real-run", "--races", "auto", "2021_04_spain"])
