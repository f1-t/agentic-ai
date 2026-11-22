"""src/eval/rerun.py, the corrected-protocol rerun: set definitions, the --final guard, the tables
and the provenance file. Stubs only: no race is loaded (test9's held-out races least of all)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.eval import rerun
from src.eval.scenarios import load_eval_set


def args(tmp_path, **kw):
    base = dict(set="dev20", sets=str(rerun.SETS_FILE), out_dir=str(tmp_path), world_config="default",
                belief_config="default", rule="as_of", final=False, jobs=1, profile=None, harness=False)
    return SimpleNamespace(**{**base, **kw})


def test_the_sets_are_the_calibration_races_and_the_registered_held_out_races():
    dev = rerun.set_spec("dev20")
    assert len(dev["races"]) == 19 and dev["development"] and dev["per_race"] == 25 and dev["seed"] == 0
    assert "2025_23_qatar" not in dev["races"] and "2022_11_austria" in dev["races"]
    test = rerun.set_spec("test9")
    held, _ = load_eval_set()
    assert sorted(test["races"]) == sorted(held) and len(test["races"]) == 9
    assert not test["development"] and test["eras"] and test["per_race"] == 25
    assert not set(dev["races"]) & set(held)
    with pytest.raises(KeyError, match="unknown set"):
        rerun.set_spec("nope")


class _Built:
    kind, items, skipped, digest = "report set", [], [], "0" * 64

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")


def test_a_report_set_is_built_only_with_final(tmp_path, monkeypatch):
    calls = []

    def build(races, version, **kw):
        calls.append((races, version, kw))
        return _Built()
    monkeypatch.setattr("src.sim.compute.load_profile", lambda name=None: None)
    with pytest.raises(SystemExit, match="--final"):
        rerun.stage_scenarios(args(tmp_path, set="test9"), build=build)
    assert calls == []
    rerun.stage_scenarios(args(tmp_path / "t", set="test9", final=True), build=build)
    races, version, kw = calls[0]
    assert version == "test9" and kw["development"] is False and kw["rule"] == "as_of" and kw["max_per_race"] == 25
    with pytest.raises(SystemExit, match="exists"):   # a set is built once
        rerun.stage_scenarios(args(tmp_path / "t", set="test9", final=True), build=build)
    rerun.stage_scenarios(args(tmp_path / "s", rule="stratified"), build=build)
    assert "max_per_race" not in calls[-1][2] and calls[-1][2]["development"] is True


def test_the_help_says_a_report_set_needs_final():
    assert "--final" in rerun.parser().format_help() and "held-out" in rerun.parser().format_help()


def rows():
    rng = np.random.default_rng(0)
    out = []
    for race in ("2019_06_x", "2021_04_y", "2024_11_z"):
        for d in ("AAA", "BBB"):
            for lap in (5, 10):
                sid = f"{race}:{d}:{lap:02d}"
                for pol, scale in (("agent", 5.0), ("fixed_one_stop", 12.0), ("run_to_flag", 25.0)):
                    out.append({"scenario_id": sid, "race_id": race, "driver": d, "lap": lap, "policy": pol,
                                "status": "ok", "regret_s": float(abs(rng.normal(0, scale))), "regret_pct": 0.1,
                                "stop_agree": True, "first_pit_error": 1, "within_mc": False, "reason": None})
    out.append({**out[0], "policy": "fixed_one_stop", "status": "invalid", "regret_s": np.nan,
                "reason": "lap_sum: 50 laps planned"})
    return pd.DataFrame(out)


def test_tail_and_status_tables_and_eras():
    r = rows()
    t = rerun.tail_table(r).set_index("policy")
    g = r[(r["policy"] == "run_to_flag") & (r["status"] == "ok")]["regret_s"]
    assert t.loc["run_to_flag", "worst_s"] == g.max() and t.loc["run_to_flag", "share_over_10s"] == (g > 10).mean()
    assert list(rerun.status_table(r)["reason"]) == ["lap_sum: 50 laps planned"]
    assert [rerun.era_of(x) for x in ("2019_06_x", "2020_16_s", "2021_04_y", "2022_01_b", "2025_11_a")] == \
        ["2019-20", "2019-20", "2021", "2022", "2024-25"]


def test_provenance_records_the_model_never_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "voyager")
    monkeypatch.setenv("VOYAGER_MODEL", "some-model")
    monkeypatch.setenv("VOYAGER_API_KEY", "sk-test-secret")
    monkeypatch.setenv("VOYAGER_BASE_URL", "https://example.invalid")
    prov = json.loads(rerun.write_provenance(args(tmp_path)).read_text())
    assert prov["model"]["LLM_BACKEND"] == "voyager" and prov["model"]["VOYAGER_MODEL"] == "some-model"
    text = (tmp_path / "provenance.json").read_text()
    assert "sk-test-secret" not in text and "example.invalid" not in text


def test_tables_write_every_output_and_a_provenance_file(tmp_path):
    spec = tmp_path / "sets.yaml"
    spec.write_text("t:\n  races: [2019_06_x, 2021_04_y, 2024_11_z]\n  per_race: 4\n  seed: 0\n  development: false\n"
                    "  eras: true\n")
    rows().to_csv(tmp_path / "baselines.csv", index=False)
    a = args(tmp_path, set="t", sets=str(spec))
    rerun._log_command(tmp_path, ["tables", "--set", "t"])
    out = rerun.stage_tables(a)
    names = {p.name for p in out.iterdir()}
    assert {"combined.csv", "summary.csv", "paired_agent_minus_x.csv", "chart_regret_s.csv", "tail.csv",
            "tail_by_era.csv", "chart_regret_s_by_era.csv", "status.csv"} <= names
    paired = pd.read_csv(out / "paired_agent_minus_x.csv").set_index("policy")
    assert set(paired.index) == {"fixed_one_stop", "run_to_flag"} and paired.loc["run_to_flag", "clusters"] == 6
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["commands"] == ["python -m src.eval.rerun tables --set t"] and prov["sim_code_version"]
    assert "pit_loss" in prov["future_blindness"]["fixed_leaks"]
    assert prov["as_of_tables"]["compound_steps"]["kind"] == "as_of"
    assert prov["as_of_tables"]["compound_steps"]["files"]["configs/sim/compound_steps.json"]
