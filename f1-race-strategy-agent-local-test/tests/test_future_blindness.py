"""Future-blindness: nothing a policy sees at a snapshot may change when the race after the
snapshot changes. The harness, the alterations, the observed surfaces and the known leaks are
described in tests/future_blindness.py.

  test_harness_builds_what_load_race_builds   the alteration enters where jobs.load_race reads
                                              the race session
  test_alteration_*                           each alteration keeps the past and changes the
                                              future (and the world that scores plans)
  test_future_blind                           every alteration, every surface, every snapshot,
                                              with the known leaks neutralised: guards against
                                              new leaks
  test_leak_<name>                            one known leak, the others neutralised: expected
                                              to fail (xfail strict) until its fix lands; then
                                              remove the xfail and the KNOWN_LEAKS entry

The synthetic race and one real snapshot run in the default suite; the real-race sweep is
`slow`. A fix branch runs just its own checks by name, e.g.
    pytest tests/test_future_blindness.py -m "slow or not slow" -k "stint_limit or future_blind"
"""
import functools

import numpy as np
import pandas as pd
import pytest

import future_blindness as fb
from src.sim import SimConfig, Simulator, jobs

SYNTHETIC = "2099_01_synthetic"
# (driver, lap) snapshots per race: early, mid, late, and under a neutralisation where the race has one
POINTS = {
    SYNTHETIC: [("EEE", 5), ("AAA", 32)],                                              # SC laps 31-33
    "2021_04_spain": [("HAM", 5), ("NOR", 8), ("HAM", 30), ("NOR", 30), ("VER", 55)],  # SC laps 7-10
    "2022_02_saudi_arabia": [("VER", 5), ("LEC", 17), ("ALB", 25), ("VER", 38), ("LEC", 44)],  # SC 15-20, VSC 36-41
    "2024_07_emilia_romagna": [("LEC", 5), ("NOR", 30), ("ALO", 50), ("VER", 55)],     # no SC or VSC
}
REAL = [r for r in POINTS if r != SYNTHETIC]
# the default suite's one real-race case: a single snapshot under the SC, scrambled
SMALL_REAL = ("2022_02_saudi_arabia", [("LEC", 17)])


def _race(race_id, *marks):
    return pytest.param(race_id, marks=[*marks, *([] if race_id == SYNTHETIC else [pytest.mark.slow])], id=race_id)


@functools.lru_cache(maxsize=None)
def race(race_id):
    """(RaceInput, recorded RaceData, cache of recorded observations by point)."""
    inp = fb.synthetic_input(SYNTHETIC) if race_id == SYNTHETIC else fb.real_input(race_id)
    return inp, fb.build(inp), {}


def differences(race_id, alteration, neutralise, points=None):
    inp, orig, recorded = race(race_id)
    return fb.check(inp, orig, POINTS[race_id] if points is None else points, alteration, neutralise, recorded)


def report(diffs) -> str:
    return "\n".join(map(str, diffs))


# --- the harness itself -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("race_id", REAL)
def test_harness_builds_what_load_race_builds(race_id):
    _, built, _ = race(race_id)
    loaded = jobs.load_race(race_id, fb.PROFILE)
    pd.testing.assert_frame_equal(built.frame, loaded.frame)
    assert built.race == loaded.race
    pd.testing.assert_frame_equal(built.results, loaded.results)
    for got, want in ((built.practice, loaded.practice), (built.qualifying, loaded.qualifying)):
        assert (got is None) == (want is None)
        if got is not None:
            pd.testing.assert_frame_equal(got, want)


def test_diff_reports_the_first_differing_path_and_both_values():
    a = {"x": [1.0, {"y": "a"}], "z": 3}
    assert fb.diff(a, {"x": [1.0 + 1e-12, {"y": "a"}], "z": 3}) == []
    assert fb.diff(a, {"x": [1.5, {"y": "b"}], "z": 3}) == [("/x[0]", 1.0, 1.5), ("/x[1]/y", "a", "b")]
    assert fb.diff({"x": [1]}, {"x": [1, 2]}) == [("/x[len]", 1, 2)]


@pytest.mark.parametrize("alteration", sorted(fb.FRAME_ALTERATIONS))
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_alteration_keeps_the_past_and_changes_the_future(race_id, alteration):
    """Rows up to the snapshot lap are the recorded ones, in order; the laps after it are not;
    the world (the full-window fit that scores plans) sees the change."""
    inp, orig, _ = race(race_id)
    world = Simulator(orig.frame, SimConfig())
    pit_losses = set()
    for lap in sorted({k for _, k in POINTS[race_id]}):
        alt = fb.FRAME_ALTERATIONS[alteration](inp, lap)
        pd.testing.assert_frame_equal(fb.past_rows(alt.laps, lap), fb.past_rows(inp.laps, lap))
        later = alt.laps[alt.laps["LapNumber"] > lap]
        if alteration == "truncate":
            assert later["LapNumber"].max() == lap + 1 and later["LapTime"].isna().all()
            continue
        was = inp.laps[inp.laps["LapNumber"] > lap]
        for col in ("LapTime", "TrackStatus", "Compound", "TyreLife", "PitInTime"):
            assert not later[col].reset_index(drop=True).equals(was[col].reset_index(drop=True)), col
        assert not later.groupby("Driver")["LapNumber"].max().equals(was.groupby("Driver")["LapNumber"].max())
        data = fb.build(alt, altered=True)
        pit_losses.add(data.race.pit_loss_s)
        moved = Simulator(data.frame, SimConfig())
        assert moved.degradation.coefs != world.degradation.coefs
    # a late snapshot can leave the observed pit loss (a median over every stop) where it was
    assert alteration == "truncate" or pit_losses - {orig.race.pit_loss_s}


def test_aggregates_shift_every_whole_race_constant():
    _, orig, _ = race(SYNTHETIC)
    alt = fb.shift_aggregates(orig)
    assert alt.race.pit_loss_s != orig.race.pit_loss_s and alt.race.compounds != orig.race.compounds
    assert not alt.results["classified"].equals(orig.results["classified"])
    pd.testing.assert_frame_equal(alt.frame, orig.frame)


def test_every_known_leak_has_its_own_test():
    assert set(fb.KNOWN_LEAKS) == {name[len("test_leak_"):] for name in globals() if name.startswith("test_leak_")}


# --- the guard: no leak beyond the known ones ---------------------------------------------------
@pytest.mark.parametrize("alteration", fb.ALTERATIONS)
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_future_blind(race_id, alteration):
    """Every surface at every snapshot is the same in the recorded and the altered race once
    the known leaks (fb.KNOWN_LEAKS) are restored to their recorded values. A failure is a new
    leak: the message names each snapshot and surface, the first differing field and both values."""
    diffs = differences(race_id, alteration, list(fb.KNOWN_LEAKS))
    assert not diffs, report(diffs)


def test_future_blind_small_real_race():
    """test_future_blind's guard on one real snapshot, in the default suite."""
    race_id, points = SMALL_REAL
    diffs = differences(race_id, "scramble", list(fb.KNOWN_LEAKS), points)
    assert not diffs, report(diffs)


# --- the known leaks, one test each: expected failures until fixed ------------------------------
def _leak_params(name, leaking):
    """Races for test_leak_<name>, xfail (strict) on those where the leak shows today."""
    lk = fb.KNOWN_LEAKS[name]
    reason = f"known leak {name}: {lk.what} ({lk.where})"
    return [_race(r, *([pytest.mark.xfail(strict=True, reason=reason)] if r in leaking else [])) for r in POINTS]


def _leak(name, race_id):
    """The leak's alteration with every other known leak neutralised: no difference once it is fixed."""
    diffs = differences(race_id, fb.KNOWN_LEAKS[name].alteration, [n for n in fb.KNOWN_LEAKS if n != name])
    assert not diffs, report(diffs)


@pytest.mark.parametrize("race_id", _leak_params("race_length", POINTS))
def test_leak_race_length(race_id):
    _leak("race_length", race_id)


# Imola 2024: every dry compound was raced from lap 1, so the race-wide list equals the as-of
# list at every snapshot there (the aggregates alteration still shows the tools read it)
@pytest.mark.parametrize("race_id", _leak_params("compound_list", [r for r in POINTS if r != "2024_07_emilia_romagna"]))
def test_leak_compound_list(race_id):
    _leak("compound_list", race_id)


@pytest.mark.parametrize("race_id", _leak_params("pit_loss", POINTS))
def test_leak_pit_loss(race_id):
    _leak("pit_loss", race_id)


@pytest.mark.parametrize("race_id", _leak_params("stint_limit", POINTS))
def test_leak_stint_limit(race_id):
    _leak("stint_limit", race_id)


@pytest.mark.parametrize("race_id", _leak_params("frame_filter", POINTS))
def test_leak_frame_filter(race_id):
    _leak("frame_filter", race_id)
