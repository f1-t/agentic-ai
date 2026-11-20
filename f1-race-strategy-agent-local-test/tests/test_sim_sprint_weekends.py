"""Sprint weekends (docs/eval.md "Sprint weekends"): the practice prior reads the FP sessions
a weekend has plus the Sprint; a conventional weekend reads what it always did. Synthetic
weekends and 2025 Qatar (a calibration race) only: no held-out race is loaded here."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval import scenarios as scen_mod
from src.sim import calibrate, calibration, cross_race, data, pace
from src.sim.config import SimConfig
from test_sim_pace import practice_laps

CONVENTIONAL = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
SPRINT = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
SPRINT_2021_22 = ["Practice 1", "Qualifying", "Practice 2", "Sprint", "Race"]   # the cached 2021-22 layout
HAS_QATAR = (Path(data.CACHE_DIR) / "2025" / "2025-11-30_Qatar_Grand_Prix").is_dir()


@pytest.mark.parametrize("names, base, want", [
    (CONVENTIONAL, calibrate.PRACTICE_SESSIONS, ("FP1", "FP2", "FP3")),
    (CONVENTIONAL, pace.PRACTICE_SESSIONS, ("FP2", "FP3")),
    (SPRINT, calibrate.PRACTICE_SESSIONS, ("FP1", "Sprint")),
    (SPRINT, pace.PRACTICE_SESSIONS, ("FP1", "Sprint")),
    (SPRINT_2021_22, calibrate.PRACTICE_SESSIONS, ("FP1", "FP2", "Sprint")),
    (SPRINT_2021_22, pace.PRACTICE_SESSIONS, ("FP1", "FP2", "Sprint")),
    (None, calibrate.PRACTICE_SESSIONS, ("FP1", "FP2", "FP3")),   # schedule not cached: unchanged
])
def test_a_sprint_weekend_reads_fp1_and_the_sprint_and_a_conventional_one_is_unchanged(monkeypatch, names,
                                                                                        base, want):
    monkeypatch.setattr(calibrate, "weekend_sessions", lambda *a, **k: names)
    assert calibrate.practice_sessions(2099, 1, base) == want


def test_the_prior_and_the_pace_read_a_synthetic_sprint_weekend_without_fp2_or_fp3(monkeypatch):
    monkeypatch.setattr(calibrate, "weekend_sessions", lambda *a, **k: SPRINT)
    asked = []

    def load(season, gp, kind="R", **kw):
        assert kind not in ("FP2", "FP3"), f"a sprint weekend has no {kind}"
        asked.append(kind)
        return SimpleNamespace(laps=practice_laps(seed=len(asked)))

    monkeypatch.setattr(calibrate, "load_session", load)
    f = calibrate._practice(2099, 1)
    assert asked == ["FP1", "Sprint"] and set(f["session"]) == {"FP1", "Sprint"}
    assert set(f.loc[f["run_class"] == "long_run", "session"]) == {"FP1", "Sprint"}

    # the pace offsets: cross_race's pace gate and the calibrate commands pick the same sessions
    seen = {}
    monkeypatch.setattr(pace, "load_practice", lambda season, gp, sessions, **kw: seen.setdefault("s", sessions))
    cross_race.pace_practice(SimpleNamespace(race_id="2099_01_sprintland", config=SimConfig()))
    assert tuple(seen["s"]) == ("FP1", "Sprint")
    assert calibrate._pace_sessions(Namespace(pace_sessions=None, season=2099, gp=1)) == ["FP1", "Sprint"]
    assert calibrate._pace_sessions(Namespace(pace_sessions="FP2", season=2099, gp=1)) == ["FP2"]


def test_sprint_support_lives_outside_the_physics_modules():
    # a conventional weekend's inputs and SIM_CODE_VERSION do not move with it
    assert "calibrate" not in calibration.PHYSICS_MODULES and "cross_race" not in calibration.PHYSICS_MODULES


@pytest.mark.skipif(not HAS_QATAR, reason="2025 Qatar not in the committed cache")
def test_2025_qatar_reads_fp1_and_the_sprint_from_the_cache():
    assert calibrate.practice_sessions(2025, 23, calibrate.PRACTICE_SESSIONS) == ("FP1", "Sprint")
    assert calibrate.practice_sessions(2021, 4, calibrate.PRACTICE_SESSIONS) == calibrate.PRACTICE_SESSIONS
    assert calibrate.practice_sessions(2023, 1, pace.PRACTICE_SESSIONS) == pace.PRACTICE_SESSIONS
    f = calibrate._practice(2025, 23, SimConfig().long_run)
    long_run = f[f["run_class"] == "long_run"]
    assert set(f["session"]) == {"FP1", "Sprint"} and long_run["driver"].nunique() == 20
    assert set(long_run["session"]) == {"FP1", "Sprint"}


@pytest.mark.parametrize("development", [False, True])
def test_2025_qatar_is_never_sampled_into_a_scenario_set(tmp_path, monkeypatch, development):
    from src.sim import jobs

    def no_load(*a, **k):
        raise AssertionError("a calibration-only race was loaded for sampling")
    monkeypatch.setattr(jobs, "load_race", no_load)
    with pytest.raises(scen_mod.CalibrationOnly, match="2025_23_qatar.*25 laps per tyre set"):
        scen_mod.build_set(["2021_04_spain", "2025_23_qatar"], "x", eval_set=tmp_path / "missing.json",
                           development=development)
