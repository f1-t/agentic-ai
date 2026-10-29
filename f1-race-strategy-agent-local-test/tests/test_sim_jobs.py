"""Compute profiles and resumable batch jobs. Offline; synthetic races except one Spain run."""
import json
import math
import os
import sys
import time

import pandas as pd
import pytest

from src.sim import calibration, calibrators, compute, jobs
from src.sim.config import EventEffects, SimConfig
from src.sim.data import CACHE_DIR
from test_sim import RACE, synthetic_practice
from test_sim_position import race_frame

A_PLAN = [("SOFT", 20), ("MEDIUM", 30)]
B_PLAN = [("SOFT", 10), ("MEDIUM", 25), ("SOFT", 15)]


@pytest.fixture(scope="module")
def synthetic():
    frame = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN))
    cfg = SimConfig(effects=EventEffects(lap1_extra_s=0.0), pre_race_base_delta_s=0.0)
    return jobs.RaceData(RACE.race_id, frame, RACE, synthetic_practice(), None, cfg)


@pytest.fixture
def fake_load(monkeypatch, synthetic):
    calls = []

    def load(race_id, profile):
        calls.append(race_id)
        return synthetic
    monkeypatch.setattr(jobs, "load_race", load)
    return calls


def small(profile, **g):
    """The profile with a small lap set, for the synthetic races."""
    return profile.model_copy(update={"g": profile.g.model_copy(update={"laps_every": 20, "n_boot": 10, **g})})


# --- profiles -------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["local", "sol"])
def test_profile_files_validate_and_load(name):
    p = compute.load_profile(name)
    assert p.name == name
    assert p.wandb.tags == [name]
    assert p.hash == compute.load_profile(name).hash and len(p.hash) == 12


def test_profile_numbers_follow_the_spec():
    loc, sol = compute.load_profile("local"), compute.load_profile("sol")
    assert (loc.sim.n_samples, sol.sim.n_samples) == (50, 500)
    assert (loc.parallel.n_jobs, sol.parallel.n_jobs) == (8, 1)
    assert (loc.parallel.eval_chunk_size, sol.parallel.eval_chunk_size) == (50, 20)
    assert (loc.g.laps_every, sol.g.laps_every) == (5, 2)
    assert (loc.g.n_boot, sol.g.n_boot) == (200, 2000)
    assert (loc.g.cadences, sol.g.cadences) == ([1, 5, None], [1, 2, 5, None])
    assert (loc.h.loro_folds, sol.h.loro_folds, loc.h.loso, sol.h.loso) == (20, None, False, True)
    assert (loc.h.mlp.ensemble, sol.h.mlp.ensemble, loc.h.mlp.epochs, sol.h.mlp.epochs) == (3, 10, 200, 2000)
    assert (loc.wandb.mode, sol.wandb.mode, loc.build.workers) == ("online", "offline", 4)
    assert loc.hash != sol.hash


def test_profile_selection_flag_then_env_then_local(monkeypatch):
    monkeypatch.delenv("SIM_PROFILE", raising=False)
    assert compute.load_profile().name == "local"
    monkeypatch.setenv("SIM_PROFILE", "sol")
    assert compute.load_profile().name == "sol"
    assert compute.load_profile("local").name == "local"
    with pytest.raises(FileNotFoundError, match="available"):
        compute.load_profile("nope")


def test_profile_rejects_unknown_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(open(compute.profile_path("local")).read() + "extra: 1\n")
    with pytest.raises(Exception):
        compute.load_profile(str(bad))


def test_hash_ignores_wandb_and_build_only():
    p = compute.load_profile("local")
    same = p.model_copy(update={"wandb": p.wandb.model_copy(update={"mode": "offline"})})
    other = p.model_copy(update={"sim": p.sim.model_copy(update={"n_samples": 51})})
    assert same.hash == p.hash and other.hash != p.hash


def test_wandb_is_optional(monkeypatch):
    p = compute.load_profile("local")
    monkeypatch.setitem(sys.modules, "wandb", None)   # import raises ImportError
    with compute.wandb_run(p, "g_race") as run:
        assert run is None
    off = p.model_copy(update={"wandb": p.wandb.model_copy(update={"mode": "disabled"})})
    with compute.wandb_run(off, "g_race") as run:
        assert run is None


def test_device_resolves_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert compute.load_profile("local").device() == "cpu"
    assert compute.load_profile("sol").device() == "cpu"


# --- g_race ---------------------------------------------------------------------------
def test_laps_k_density_and_cadences(synthetic, fake_load):
    p = compute.load_profile("local")
    assert jobs.laps_k_for(synthetic.frame, 50, p) == list(range(0, 41, 5))
    assert jobs.laps_k_for(synthetic.frame, 50, compute.load_profile("sol")) == list(range(0, 41, 2))
    rows = jobs.g_rows(synthetic, synthetic.config, small(p), laps_k=[0, 5, 10])
    by = {c: sorted(g["k"].unique()) for c, g in rows.groupby("k_subsample")}
    assert by == {"1": [0, 5, 10], "5": [0, 5, 10], "none": [0]}
    assert {"profile", "config_hash", "sim_code_version", "pit_loss_s"} <= set(rows.columns)
    assert "cadence" not in rows   # reserved for SimConfig.calibration_every


def test_job_twice_gives_one_output_and_skips(tmp_path, fake_load):
    p = small(compute.load_profile("local"))
    job = jobs.JOBS["g_race"]
    assert jobs.run_task(job, "syn", p, tmp_path) == "done"
    assert jobs.run_task(job, "syn", p, tmp_path) == "skipped"
    assert fake_load == ["syn"]
    assert sorted(f.name for f in (tmp_path / "g").iterdir()) == ["syn.parquet", "syn.parquet.meta.json"]
    meta = json.loads((tmp_path / "g" / "syn.parquet.meta.json").read_text())
    assert meta["sim_code_version"] == calibration.SIM_CODE_VERSION and meta["profile_hash"] == p.hash


def test_changed_profile_or_missing_meta_reruns(tmp_path, fake_load):
    p = small(compute.load_profile("local"))
    job = jobs.JOBS["g_race"]
    jobs.run_task(job, "syn", p, tmp_path)
    q = p.model_copy(update={"sim": p.sim.model_copy(update={"n_samples": 7})})
    assert jobs.run_task(job, "syn", q, tmp_path) == "done"
    (tmp_path / "g" / "syn.parquet.meta.json").unlink()   # an interrupted write leaves no meta
    assert jobs.run_task(job, "syn", q, tmp_path) == "done"


def test_aggregate_over_a_partial_set(tmp_path, fake_load):
    p = small(compute.load_profile("local"))
    job = jobs.JOBS["g_race"]
    jobs.run_task(job, "syn", p, tmp_path)
    status = jobs.aggregate(job, ["syn", "other"], p, tmp_path)
    assert (status["expected"], status["done"], status["missing"]) == (2, 1, ["other"])
    summary = pd.read_csv(tmp_path / "g" / "summary.csv")
    assert "k_subsample" in summary and "cadence" not in summary
    assert json.loads((tmp_path / "g" / "status.json").read_text())["missing"] == ["other"]
    assert jobs.aggregate(job, ["syn", "other"], p, tmp_path)["done"] == 1   # safe to re-run


def test_aggregate_with_nothing_done(tmp_path):
    status = jobs.aggregate(jobs.JOBS["g_race"], ["a"], compute.load_profile("local"), tmp_path)
    assert (status["done"], status["files"]) == (0, [])


# --- manifests and cli -----------------------------------------------------------------
def test_manifest_forms(tmp_path):
    (tmp_path / "a.txt").write_text("x\n\ny\n")
    (tmp_path / "b.json").write_text('["x", "y"]')
    (tmp_path / "c.json").write_text('{"units": ["x", "y"]}')
    assert all(jobs.read_manifest(tmp_path / f) == ["x", "y"] for f in ("a.txt", "b.json", "c.json"))


def test_cli_task_id_runs_one_manifest_entry(tmp_path, fake_load, capsys, monkeypatch):
    monkeypatch.setattr(jobs, "load_profile", lambda name=None: small(compute.load_profile(name)))
    m = tmp_path / "m.json"
    argv = ["g_race", "--profile", "local", "--manifest", str(m), "--task-id", "1", "--out", str(tmp_path / "o")]
    jobs.main(["manifest", "g_race", "--manifest", str(m), "--out", str(tmp_path), "--races", "r0", "r1"])
    assert jobs.main(argv) == 0 and jobs.main(argv) == 0
    out = capsys.readouterr().out
    assert "r1 [local]: done" in out and "r1 [local]: skipped" in out
    assert (tmp_path / "o" / "g" / "r1.parquet").is_file() and not (tmp_path / "o" / "g" / "r0.parquet").exists()


def test_eval_manifest_is_chunks_of_the_profile_size():
    args = jobs.build_parser().parse_args(["manifest", "eval_chunk", "--out", ".", "--n-items", "101", "--profile", "local"])
    assert jobs.JOBS["eval_chunk"].units(args) == ["chunk_0000", "chunk_0001", "chunk_0002"]
    args.profile = "sol"
    assert len(jobs.JOBS["eval_chunk"].units(args)) == 6


# --- h_fold, eval_chunk, build_weekend ---------------------------------------------------
R0, R1 = "2021_01_aaa", "2021_02_bbb"   # synthetic races in the canonical id form


def h_args(tmp_path, calibrator, *races):
    args = jobs.build_parser().parse_args(["h_fold", "--out", str(tmp_path), "--task-id", "0",
                                           "--calibrator", calibrator, "--races", *races])
    args.pool = args.races
    return args


def test_h_fold_runs_one_loro_fold(tmp_path, fake_load, monkeypatch):
    seen = []
    real = calibrators.loro

    def spy(dataset, name, fold, **kw):
        seen.append((sorted(dataset.races), name, fold, kw["cadences"], kw["windows"]))
        return real(dataset, name, fold, **kw)
    monkeypatch.setattr(calibrators, "loro", spy)
    p = small(compute.load_profile("local"))
    args = h_args(tmp_path, "shrinkage", R0, R1)
    job = jobs.JOBS["h_fold"]
    assert jobs.run_task(job, R0, p, tmp_path, args) == "done"
    assert jobs.run_task(job, R0, p, tmp_path, args) == "skipped"
    assert seen == [([R0, R1], "shrinkage", R0, p.g.cadences, ["pre_race", "to_snapshot"])]
    rows = pd.read_parquet(tmp_path / "h" / "shrinkage" / f"{R0}.parquet")
    assert set(rows["window"]) == {"pre_race", "to_snapshot"} and set(rows["calibrator"]) == {"shrinkage"}
    assert set(rows["race_id"]) == {R0} and set(rows["fold"]) == {R0}
    assert set(rows["calibration_every"]) == {"1", "5", "none"} and "cadence" not in rows
    rmse = json.loads((tmp_path / "h" / "shrinkage" / f"{R0}.rmse.json").read_text())
    assert (rmse["held_out"], rmse["train"], rmse["seed"], rmse["protocol"]) == ([R0], [R1], p.h.seed, "race")
    fuel = rmse["targets"]["to_snapshot/fuel_s_per_lap/N=1"]
    assert fuel["n"] > 0 and fuel["rmse"] == pytest.approx(math.sqrt(fuel["sse"] / fuel["n"]))
    status = jobs.aggregate(job, [R0, R1], p, tmp_path, args)
    assert status["missing"] == [R1]
    total = json.loads((tmp_path / "h" / "shrinkage" / "rmse.json").read_text())
    h_summary = pd.read_csv(tmp_path / "h" / "shrinkage" / "summary.csv")
    assert set(h_summary["calibration_every"].astype(str)) == {"1", "5", "none"} and "cadence" not in h_summary
    assert total["to_snapshot/fuel_s_per_lap/N=1"]["rmse"] == pytest.approx(fuel["rmse"])
    assert 0.0 <= total["to_snapshot/fuel_s_per_lap/N=1"]["coverage_68"] <= 1.0


def test_h_fold_logs_the_fold_assignment_to_wandb(tmp_path, fake_load, monkeypatch):
    logged = []

    class Config:
        def update(self, d, allow_val_change=False):
            logged.append(d)

    monkeypatch.setitem(sys.modules, "wandb", type(sys)("wandb"))
    sys.modules["wandb"].run = type("Run", (), {"config": Config()})()
    args = h_args(tmp_path, "none", R0, R1)
    args.wandb = True
    jobs.run_task(jobs.JOBS["h_fold"], "season:2021", small(compute.load_profile("local")), tmp_path, args)
    a = logged[0]["fold_assignment"]
    assert (a["fold"], a["protocol"], a["held_out"], a["train"], a["calibrator"]) == (
        "season:2021", "season", [R0, R1], [], "none")
    on_disk = json.loads((tmp_path / "h" / "none" / "season_2021.rmse.json").read_text())
    assert {k: on_disk[k] for k in a} == a


def test_h_fold_with_a_learned_stub_says_it_needs_data(tmp_path, fake_load):
    p = small(compute.load_profile("local"))
    with pytest.raises(NotImplementedError, match="needs the multi-race dataset"):
        jobs.run_task(jobs.JOBS["h_fold"], R0, p, tmp_path, h_args(tmp_path, "ridge", R0, R1))


def test_h_fold_folds_subsample_and_loso():
    loc = jobs.build_parser().parse_args(["manifest", "h_fold", "--out", ".", "--profile", "local",
                                          "--races", *[f"2021_R{i}" for i in range(30)]])
    folds = jobs.JOBS["h_fold"].units(loc)
    assert len(folds) == 20 and folds == jobs.JOBS["h_fold"].units(loc) and not any(f.startswith("season") for f in folds)
    sol = jobs.build_parser().parse_args(["manifest", "h_fold", "--out", ".", "--profile", "sol",
                                          "--races", "2021_A", "2022_B"])
    assert jobs.JOBS["h_fold"].units(sol) == ["2021_A", "2022_B", "season:2021", "season:2022"]


def test_unknown_calibrator_names_the_registered_ones(tmp_path, fake_load):
    with pytest.raises(KeyError, match="unknown calibrator 'not_there'.*shrinkage"):
        jobs.run_task(jobs.JOBS["h_fold"], R0, small(compute.load_profile("local")), tmp_path,
                      h_args(tmp_path, "not_there", R0, R1))


def test_race_ids_default_to_spain_and_accept_the_alias():
    assert jobs.DEFAULT_RACES == ["2021_04_spain"]
    args = jobs.build_parser().parse_args(["manifest", "g_race", "--out", ".", "--races", "2021_Spain"])
    assert jobs.JOBS["g_race"].units(args) == ["2021_04_spain"]
    args.races = None
    assert jobs.JOBS["g_race"].units(args) == ["2021_04_spain"]
    sol = jobs.build_parser().parse_args(["manifest", "h_fold", "--out", ".", "--profile", "sol",
                                          "--races", "2021_Spain", "2022_01_bahrain"])
    assert jobs.JOBS["h_fold"].units(sol) == ["2021_04_spain", "2022_01_bahrain", "season:2021", "season:2022"]
    assert jobs.split_race_id("2021_Spain") == jobs.split_race_id("2021_04_spain") == (2021, 4)
    assert jobs.JOBS["g_race"].output("o", "2021_Spain") == jobs.JOBS["g_race"].output("o", "2021_04_spain")


def test_eval_chunk_uses_the_installed_runner(tmp_path):
    p = compute.load_profile("local")
    job = jobs.JOBS["eval_chunk"]
    with pytest.raises(NotImplementedError, match="eval runner"):
        jobs.run_task(job, "chunk_0000", p, tmp_path)
    jobs.set_eval_runner(lambda chunk, prof: pd.DataFrame({"chunk": [chunk], "n_samples": [prof.sim.n_samples]}))
    try:
        assert jobs.run_task(job, "chunk_0000", p, tmp_path) == "done"
        assert jobs.run_task(job, "chunk_0000", p, tmp_path) == "skipped"
        status = jobs.aggregate(job, ["chunk_0000", "chunk_0001"], p, tmp_path)
        assert status["missing"] == ["chunk_0001"]
        assert pd.read_parquet(tmp_path / "eval" / "all.parquet")["n_samples"].tolist() == [50]
    finally:
        jobs.set_eval_runner(None)


def test_build_weekend_is_a_stub(tmp_path):
    with pytest.raises(NotImplementedError, match="owned by the data pipeline"):
        jobs.run_task(jobs.JOBS["build_weekend"], "2021_04_spain", compute.load_profile("local"), tmp_path)


# --- Spain, end to end ------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isdir(os.path.join(CACHE_DIR, "2021")), reason="no cache")
def test_local_profile_runs_g_on_spain_in_under_a_minute(tmp_path):
    p = compute.load_profile("local")
    start = time.perf_counter()
    assert jobs.run_task(jobs.JOBS["g_race"], "2021_Spain", p, tmp_path) == "done"   # the alias
    elapsed = time.perf_counter() - start
    print(f"local profile, G on Spain: {elapsed:.1f} s")
    assert elapsed < 60
    rows = pd.read_parquet(tmp_path / "g" / "2021_04_spain.parquet")
    assert set(rows["window"]) == set(p.g.windows) and set(rows["k_subsample"]) == {"1", "5", "none"}
    assert "cadence" not in rows
    assert set(rows["race_id"]) == {"2021_04_spain"}
    assert jobs.run_task(jobs.JOBS["g_race"], "2021_04_spain", p, tmp_path) == "skipped"
