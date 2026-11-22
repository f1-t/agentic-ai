"""src/eval/smoke.py's lint, on synthetic trajectories, rows and provenance: one case per finding
type, and a clean episode with none. No race is loaded and no model is called."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.eval import rerun, smoke

PIT_LOSS = {"pit_loss_s": {"low": 18.0, "median": 22.0, "high": 26.0}, "sd_s": 2.4, "track_status": "green"}
TYRE = {"compound": "SOFT", "modelled": True, "deg_per_lap_s": {"low": 0.02, "median": 0.05, "high": 0.1},
        "tyre_life": {"longest_run_seen_laps": 20.0, "prior_life_laps": 30.0, "prior_sd_laps": 8.0}}
EVAL = {"remaining": {"mean_s": 5100.0}, "gap_reading": "STAY_OUT 1 lap is faster than PIT MEDIUM in 70% of draws",
        "probability_improves_outcome": 0.3, "alternative": {"action": {"action": "pit"}, "valid": True, "errors": []}}
SUBMIT = {"tool": "submit_recommendation", "call_index": 9, "output": {"accepted": True}}
RULE_OK = ("Below the 3-lap minimum stint: stopping at the end of lap 1 makes the current SOFT stint 1 lap. "
           "The earliest legal stop is at the end of lap 3, the latest at the end of lap 49.")


def call(tool, output, i=1, error=None):
    return {"tool": tool, "call_index": i, "output": output, "error": error}


def episode(*events, submit=True):
    return [*events, SUBMIT] if submit else list(events)


def findings(events, identity=None):
    return smoke.lint_trajectory("s", events, identity)


def test_a_clean_episode_has_no_finding():
    ev = episode(call("estimate_pit_loss", PIT_LOSS), call("estimate_tyre_degradation", TYRE, 2),
                 call("evaluate_strategy", EVAL, 3), call("get_race_snapshot", {"track_status": "SC"}, 4),
                 call("evaluate_strategy", {"error": RULE_OK, "recoverable": True}, 5),
                 call("compare_plans", {"valid": [True, True], "summaries": [{"mean_s": 1.0}, {"mean_s": 2.0}],
                                        "gap_readings": ["a is faster than b in 60% of draws"],
                                        "errors": [[], []]}, 6))
    assert findings(ev) == []


@pytest.mark.parametrize("events, expect", [
    (episode(call("estimate_pit_loss", PIT_LOSS), submit=False), "did not submit"),
    (episode(call("estimate_pit_loss", {"error": "KeyError: 'pit_loss_s'"})), "exception text"),
    (episode(call("compare_plans", {"note": "Traceback (most recent call last): ..."})), "exception text"),
    (episode(call("estimate_pit_loss", {**PIT_LOSS, "sd_s": float("nan")})), "non-finite"),
    (episode(call("estimate_tyre_degradation", {**TYRE, "deg_per_lap_s": {"low": 0.1, "median": float("inf"),
                                                                           "high": 0.2}})), "non-finite"),
    (episode(call("estimate_pit_loss", {**PIT_LOSS, "pit_loss_s": {"low": 1.0, "median": None, "high": 3.0}})),
     "pit_loss_s.median is not a number"),
    (episode(call("estimate_pit_loss", {k: v for k, v in PIT_LOSS.items() if k != "track_status"})),
     "track_status missing"),
    (episode(call("estimate_tyre_degradation", {k: v for k, v in TYRE.items() if k != "tyre_life"})),
     "tyre_life missing"),
    (episode(call("get_model_report", {"compounds": {"SOFT": {**TYRE, "tyre_life": {"prior_life_laps": 30.0,
                                                                                      "prior_sd_laps": None}}}})),
     "without its sd"),
    (episode(call("evaluate_strategy", {**EVAL, "gap_reading": None})), "gap_reading missing"),
    (episode(call("compare_plans", {"valid": [True, True], "summaries": [{"mean_s": 1.0}, {"mean_s": 2.0}],
                                    "gap_readings": [None]})), "gap_readings missing"),
    (episode(call("compare_plans", {"valid": [True, False], "summaries": [None, None], "gap_readings": []})),
     "summaries[0].mean_s"),
    (episode(call("evaluate_strategy", {"error": "Stint of 1 laps is below the 3-lap minimum."})),
     "rule reply without a legal alternative"),
    (episode(call("evaluate_strategy", {**EVAL, "alternative": {"valid": False, "errors": [
        "Single compound: this plan runs only SOFT over the race."]}})), "rule reply without a legal alternative"),
])
def test_each_finding_type(events, expect):
    found = findings(events)
    assert any(expect in f for f in found), found


def test_a_real_identity_in_a_tool_output_is_a_finding():
    def identity(value):   # the harness's identity_hits, stubbed: the real season and a driver code
        return [(p, tok) for p, text in smoke._strings(value) for tok in ("2021", "HAM") if tok in text]
    ev = episode(call("get_race_snapshot", {"track_status": "green", "event": "2021 Spanish Grand Prix",
                                            "nearby_cars": [{"driver": "HAM"}]}))
    found = findings(ev, identity)
    assert any("'2021'" in f for f in found) and any("'HAM'" in f for f in found)


def rows(**over):
    base = {"scenario_id": "r:A:10", "race_id": "r", "policy": "agent", "status": "ok", "regret_s": 1.0,
            "decision": "PIT", "reason": None}
    return pd.DataFrame([{**base, **over}])


def test_row_findings():
    assert smoke.lint_rows(rows()) == []
    assert any("agent error row" in f for f in smoke.lint_rows(rows(status="error", reason="APIError: boom")))
    assert any("invalid agent row" in f for f in smoke.lint_rows(rows(status="invalid", reason="no plan")))
    assert any("negative regret" in f for f in smoke.lint_rows(rows(regret_s=-0.5)))
    assert any("no decision" in f for f in smoke.lint_rows(rows(decision=None)))
    combined = pd.concat([rows(), rows(policy="fixed_one_stop", regret_s=-2.0)], ignore_index=True)
    assert any("fixed_one_stop" in f and "negative" in f for f in smoke.lint_rows(rows(), combined))


def test_provenance_findings():
    good = {"model": {"LLM_BACKEND": "voyager", "VOYAGER_MODEL": "qwen3-30b-a3b"}, "commit": "abc",
            "world": {"config_hash": "w"}, "belief": {"config_hash": "b"},
            "loop": {"max_steps": 10, **{k: None for k in smoke.LOOP_KEYS}}}
    assert smoke.lint_provenance(good) == []
    assert any("model id" in f for f in smoke.lint_provenance({**good, "model": {"LLM_BACKEND": "voyager"}}))
    assert any("commit" in f for f in smoke.lint_provenance({**good, "commit": None}))
    assert any("belief config" in f for f in smoke.lint_provenance({**good, "belief": {}}))
    assert any("AGENT_TURN_REPEAT" in f for f in smoke.lint_provenance(
        {**good, "loop": {k: v for k, v in good["loop"].items() if k != "AGENT_TURN_REPEAT"}}))


def sc(sid, race, lap, total=60, status="green"):
    return SimpleNamespace(scenario_id=sid, race_id=race, lap=lap,
                           snapshot=SimpleNamespace(track_status=status, race=SimpleNamespace(total_laps=total)))


def test_the_pick_covers_lap_0_sc_late_and_mid_over_races():
    items = [sc(f"r{r}:D:{lap}", f"r{r}", lap, status="sc" if (r, lap) == (3, 33) else "green")
             for r in range(6) for lap in (0, 12, 30, 33, 55)]
    chosen = smoke.pick(items, 6, seed=0)
    assert len(chosen) == 6 and len({c.race_id for c in chosen}) == 6
    assert any(c.lap == 0 for c in chosen) and any(c.snapshot.track_status == "sc" for c in chosen)
    assert any(60 - c.lap <= smoke.LATE_LAPS for c in chosen) and any(c.lap in (30, 33) and c.snapshot.track_status == "green" for c in chosen)
    assert [c.scenario_id for c in chosen] == [c.scenario_id for c in smoke.pick(items, 6, seed=0)]


def test_the_smoke_set_is_development_only_and_final_is_refused(tmp_path):
    spec = rerun.set_spec("smoke")
    smoke.check_development(spec, spec["races"])     # passes: development races only
    test9 = rerun.set_spec("test9")
    with pytest.raises(SystemExit, match="held-out"):
        smoke.check_development(spec, spec["races"] + test9["races"][:1])
    with pytest.raises(SystemExit, match="held-out"):
        smoke.check_development({**spec, "development": False}, spec["races"])
    a = rerun.parser().parse_args(["smoke", "--final", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="--final is refused"):
        rerun.stage_smoke(a)
