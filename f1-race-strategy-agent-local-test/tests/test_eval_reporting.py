"""The pre-registered reporting (configs/eval/reporting.yaml, src/eval/reporting.py) on a SYNTHETIC
nine-race set: race ids shaped like the held-out ones and made-up rows. Nothing is built or loaded
(test9 least of all)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.eval import reporting, rerun

# shaped like the nine held-out race ids (season_round_event); rounds and slugs invented
FRESH = ["2021_07_france", "2021_18_mexico_city", "2022_05_miami", "2022_03_australia", "2022_14_belgium"]
EARLIER = ["2019_06_monaco", "2020_17_sakhir", "2024_11_austria", "2025_11_austrian"]
SPEC = reporting.load()


def synthetic_rows(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for race in FRESH + EARLIER:
        for d in ("AAA", "BBB", "CCC"):
            for lap in (5, 20, 40):
                sid = f"{race}:{d}:{lap:02d}"
                for pol, scale in (("agent", 3.0), ("run_to_flag", 12.0), ("one_stop_total", 11.0),
                                   ("actual_strategy", 6.0), ("belief_optimizer", 3.0), ("fixed_one_stop", 15.0),
                                   ("fixed_two_stop", 25.0), ("greedy_crossover", 80.0)):
                    out.append({"scenario_id": sid, "race_id": race, "driver": d, "lap": lap, "policy": pol,
                                "status": "ok", "regret_s": float(abs(rng.normal(0, scale))), "reason": None,
                                "regret_pct": 0.1, "stop_agree": True, "first_pit_error": 1, "within_mc": False})
    rows = pd.DataFrame(out)
    # unavailable and invalid rows are counted, never dropped
    rows.loc[(rows.policy == "greedy_crossover") & (rows.lap != 5), "status"] = "unavailable"
    rows.loc[(rows.policy == "agent") & (rows.scenario_id == "2019_06_monaco:AAA:05"), "status"] = "invalid"
    return rows


def test_the_groups_are_the_fresh_five_the_earlier_four_and_all_nine_in_order():
    gs = reporting.groups(FRESH + EARLIER, SPEC)
    assert [g[0] for g in gs] == ["fresh five", "earlier four", "all nine"]
    assert sorted(gs[0][2]) == sorted(FRESH) and sorted(gs[1][2]) == sorted(EARLIER) and len(gs[2][2]) == 9
    assert reporting.groups(["2021_04_spain", "2022_06_spain"], SPEC)[0][0] == "development (all)"
    with pytest.raises(ValueError, match="no held-out group"):
        reporting.groups(FRESH + EARLIER + ["2023_07_spain"], SPEC)


def test_the_tables_split_three_ways_with_both_intervals_and_the_verdicts():
    out = reporting.tables(synthetic_rows(), SPEC, rerun.era_of)
    p = out["reporting_paired"]
    assert list(dict.fromkeys(p["group"])) == ["fresh five", "earlier four", "all nine"]
    first = p[p.group == "fresh five"]
    assert list(first["policy"]) == ["run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer",
                                     "fixed_one_stop", "fixed_two_stop", "greedy_crossover"]
    assert list(first["role"][:2]) == ["lead", "lead"] and first["role"].iloc[-1] == "footnote"
    allnine = p[p.group == "all nine"].set_index("policy")
    assert allnine.loc["run_to_flag", "races"] == 9 and allnine.loc["run_to_flag", "race_driver_clusters"] == 27
    # agent regret is far below run_to_flag's: both intervals below zero, "ahead"
    assert allnine.loc["run_to_flag", "ci_high_race_driver"] < 0 and allnine.loc["run_to_flag", "ci_high_race"] < 0
    assert allnine.loc["run_to_flag", "verdict"] == "ahead"
    assert allnine.loc["belief_optimizer", "verdict"] in ("level", "ahead", "behind")
    # the invalid agent row is out of every comparison: 80 of 81 scenarios paired
    assert allnine.loc["run_to_flag", "paired"] == 80
    t = out["reporting_tail"]
    assert "share_over_20s (stated)" in t and {"share_over_10s", "share_over_30s", "median_s", "worst_s"} <= set(t)
    c = out["reporting_counts"].query("group == 'all nine'").set_index("policy")
    assert c.loc["greedy_crossover", "unavailable"] == 54 and c.loc["agent", "invalid"] == 1
    assert set(out["reporting_paired_by_era"]["era"]) == {"2019-20", "2021", "2022", "2024-25"}


def test_verdict_words():
    assert reporting.verdict(-3.0, -0.5) == "ahead" and reporting.verdict(0.2, 4.0) == "behind"
    assert reporting.verdict(-1.0, 1.0) == "level" and reporting.verdict(np.nan, 1.0) is None


def _report_set(tmp_path, sampling):
    sets = tmp_path / "sets.yaml"
    sets.write_text("h:\n  races: [" + ", ".join(FRESH + EARLIER) + "]\n  per_race: 9\n  seed: 0\n"
                    "  development: false\n  eras: true\n")
    (tmp_path / "scenarios.json").write_text(json.dumps({"version": "h", "development": False, "items": [],
                                                         "sampling": sampling}))
    synthetic_rows().to_csv(tmp_path / "baselines.csv", index=False)
    return SimpleNamespace(set="h", sets=str(sets), out_dir=str(tmp_path), world_config="default",
                           belief_config="default", rule="as_of", final=False, jobs=1, profile=None, harness=False)


def test_tables_refuse_a_report_set_built_under_other_reporting_choices(tmp_path):
    a = _report_set(tmp_path, {"rule": "as_of", reporting.SAMPLING_KEY: "0" * 64})
    with pytest.raises(SystemExit, match="changed after the set was built"):
        rerun.stage_tables(a)
    a = _report_set(tmp_path, {"rule": "as_of"})   # none recorded
    with pytest.raises(SystemExit, match="none recorded"):
        rerun.stage_tables(a)


def test_tables_run_and_stamp_provenance_when_the_hash_matches(tmp_path):
    a = _report_set(tmp_path, reporting.record({"rule": "as_of"}))
    t = rerun.stage_tables(a)
    assert {"reporting_paired.csv", "reporting_tail.csv", "reporting_counts.csv", "reporting_header.txt"} <= \
        {p.name for p in t.iterdir()}
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["reporting"]["sha256"] == reporting.sha256()
    assert prov["reporting"]["text"] == reporting.REPORTING_FILE.read_text()   # verbatim


def test_the_scenarios_stage_records_the_reporting_hash(tmp_path, monkeypatch):
    class Built:
        kind, items, skipped, digest, sampling = "report set", [], [], "0" * 64, {"rule": "as_of"}

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"sampling": self.sampling}))
    built = Built()
    monkeypatch.setattr("src.sim.compute.load_profile", lambda name=None: None)
    a = SimpleNamespace(set="dev20", sets=str(rerun.SETS_FILE), out_dir=str(tmp_path), rule="as_of", final=False,
                        profile=None)
    rerun.stage_scenarios(a, build=lambda *x, **k: built)
    assert json.loads((tmp_path / "scenarios.json").read_text())["sampling"][reporting.SAMPLING_KEY] == reporting.sha256()
