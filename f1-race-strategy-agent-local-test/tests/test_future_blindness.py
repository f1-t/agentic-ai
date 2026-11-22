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
import dataclasses
import functools

import numpy as np
import pandas as pd
import pytest

import future_blindness as fb
from src.sim import SimConfig, Simulator, jobs

SYNTHETIC = "2099_01_synthetic"
# (driver, lap) snapshots per race: the start (lap 0), early, mid, late, under a neutralisation
# where the race has one, and a backmarker lapped by the end at a late lap (last entry)
POINTS = {
    SYNTHETIC: [("EEE", 20), ("GGG", 40)],                                             # GGG 49 laps; more below
    "2021_04_spain": [("PER", 0), ("HAM", 5), ("NOR", 8), ("HAM", 30), ("NOR", 30), ("VER", 55),
                      ("MAZ", 55)],                                                    # SC laps 7-10; MAZ 64 of 66
    "2022_02_saudi_arabia": [("OCO", 0), ("VER", 5), ("LEC", 17), ("ALB", 25), ("VER", 38), ("LEC", 44),
                             ("STR", 44)],                                             # SC 15-20, VSC 36-41; STR 49 of 50
    "2024_07_emilia_romagna": [("PER", 0), ("LEC", 5), ("NOR", 30), ("ALO", 50), ("VER", 55),
                               ("SAR", 55)],                                           # no SC or VSC; SAR 62 of 63
}
# the scenario sampler is compared up to this point's moment
SELECTION_POINT = {SYNTHETIC: ("AAA", 32), "2021_04_spain": ("HAM", 30), "2022_02_saudi_arabia": ("LEC", 17),
                   "2024_07_emilia_romagna": ("NOR", 30)}
# the synthetic start and SC snapshots (SC laps 31-33), kept out of the default suite's budget
SYNTHETIC_SLOW = [("CCC", 0), ("AAA", 32)]
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
    return list(_differences(race_id, alteration, tuple(neutralise), None if points is None else tuple(points)))


# One check per (race, alteration, neutralised leaks, points), shared by every test that asks for
# it: with no known leak left, test_future_blind[r-scramble], test_leak_stint_limit and each fixed
# leak's test run the same check, and the check is deterministic. Each test keeps its own name
# and assertion, so a leak that comes back still fails under every test that guards it.
@functools.lru_cache(maxsize=None)
def _differences(race_id, alteration, neutralise, points):
    inp, orig, recorded = race(race_id)
    return tuple(fb.check(inp, orig, POINTS[race_id] if points is None else list(points), alteration,
                          list(neutralise), recorded))


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
    """Rows that ended by the snapshot moment are the recorded ones, in order; nothing after it
    is (and no lap in progress ends by it); the world (the full-window fit that scores plans)
    sees the change."""
    inp, orig, _ = race(race_id)
    world = Simulator(orig.frame, SimConfig())
    pit_losses = set()
    for driver, lap in POINTS[race_id]:
        alt = fb.FRAME_ALTERATIONS[alteration](inp, driver, lap)
        t = fb.moment(inp.laps, driver, lap)
        assert fb.moment(alt.laps, driver, lap) == t
        pd.testing.assert_frame_equal(fb.past_rows(alt.laps, driver, lap), fb.past_rows(inp.laps, driver, lap))
        past = fb.is_past(alt.laps, driver, lap)
        later = alt.laps[~past]
        assert not (later["Time"] <= t).any()   # a row in the future ends after the moment
        if alteration == "truncate":
            assert later["LapTime"].isna().all() and later["Driver"].is_unique
            continue
        was = inp.laps[~fb.is_past(inp.laps, driver, lap)]
        for col in ("LapTime", "TrackStatus", "Compound", "TyreLife", "PitInTime"):
            assert not later[col].reset_index(drop=True).equals(was[col].reset_index(drop=True)), col
        assert not later.groupby("Driver")["LapNumber"].max().equals(was.groupby("Driver")["LapNumber"].max())
        data = fb.build(alt, altered=True)
        pit_losses.add(data.race.pit_loss_s)
        moved = Simulator(data.frame, SimConfig())
        assert moved.degradation.coefs != world.degradation.coefs
    # a late snapshot can leave the observed pit loss (a median over every stop) where it was
    assert alteration == "truncate" or pit_losses - {orig.race.pit_loss_s}


@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_the_past_ends_at_the_snapshot_moment_not_at_a_lap_number(race_id):
    """By lap number and by time the past differs: at a mid-race point some car's lap L ends
    after the moment (it is future) or some car's lap L + 1 ends before it (it is past)."""
    inp, _, _ = race(race_id)
    driver, lap = SELECTION_POINT[race_id]
    past = fb.is_past(inp.laps, driver, lap)
    by_lap = inp.laps["LapNumber"] <= lap
    assert (past != by_lap).any()
    assert past[inp.laps["Driver"] == driver].equals(by_lap[inp.laps["Driver"] == driver])


def test_aggregates_shift_every_whole_race_constant():
    _, orig, _ = race(SYNTHETIC)
    alt = fb.shift_aggregates(orig)
    assert alt.race.pit_loss_s != orig.race.pit_loss_s   # compounds: the nominated three, not a race aggregate
    assert not alt.results["classified"].equals(orig.results["classified"])
    pd.testing.assert_frame_equal(alt.frame, orig.frame)


def test_every_known_leak_has_its_own_test():
    assert set(fb.KNOWN_LEAKS) | set(fb.FIXED_LEAKS) == \
        {name[len("test_leak_"):] for name in globals() if name.startswith("test_leak_")}
    assert not set(fb.KNOWN_LEAKS) & set(fb.FIXED_LEAKS)


# --- the guard: no leak beyond the known ones ---------------------------------------------------
@pytest.mark.parametrize("alteration", fb.ALTERATIONS)
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_future_blind(race_id, alteration):
    """Every surface at every snapshot is the same in the recorded and the altered race once
    the known leaks (fb.KNOWN_LEAKS) are restored to their recorded values. A failure is a new
    leak: the message names each snapshot and surface, the first differing field and both values."""
    diffs = differences(race_id, alteration, list(fb.KNOWN_LEAKS))
    assert not diffs, report(diffs)


@pytest.mark.slow
@pytest.mark.parametrize("alteration", fb.ALTERATIONS)
def test_future_blind_synthetic_start_and_sc(alteration):
    diffs = differences(SYNTHETIC, alteration, list(fb.KNOWN_LEAKS), SYNTHETIC_SLOW)
    assert not diffs, report(diffs)


def test_future_blind_small_real_race():
    """test_future_blind's guard on one real snapshot, in the default suite."""
    race_id, points = SMALL_REAL
    diffs = differences(race_id, "scramble", list(fb.KNOWN_LEAKS), points)
    assert not diffs, report(diffs)


# --- the known leaks, one test each: expected failures until fixed ------------------------------
def _leak_params(name, leaking):
    """Races for test_leak_<name>, xfail (strict) on those where the leak shows today."""
    if name in fb.FIXED_LEAKS:   # fixed: a plain regression check
        return [_race(r) for r in POINTS]
    lk = fb.KNOWN_LEAKS[name]
    reason = f"known leak {name}: {lk.what} ({lk.where})"
    return [_race(r, *([pytest.mark.xfail(strict=True, reason=reason)] if r in leaking else [])) for r in POINTS]


def _leak(name, race_id):
    """The leak's alteration with every other known leak neutralised: no difference once it is fixed."""
    lk = fb.KNOWN_LEAKS.get(name) or fb.FIXED_LEAKS[name]
    diffs = differences(race_id, lk.alteration, [n for n in fb.KNOWN_LEAKS if n != name])
    assert not diffs, report(diffs)


@pytest.mark.parametrize("race_id", _leak_params("race_length", POINTS))
def test_leak_race_length(race_id):
    _leak("race_length", race_id)


# Imola 2024: every dry compound was raced from lap 1, so the race-wide list equals the as-of
# list at every snapshot there (the aggregates alteration still shows the tools read it)
@pytest.mark.parametrize("race_id", _leak_params("compound_list", [r for r in POINTS if r != "2024_07_emilia_romagna"]))
def test_leak_compound_list(race_id):
    _leak("compound_list", race_id)


# fixed (fb.FIXED_LEAKS): the frame scrambled after the snapshot, and every whole-race constant
# shifted (the pit loss by 6.5 s), with the remaining known leaks neutralised
@pytest.mark.parametrize("alteration", ["scramble", "aggregates"])
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_leak_pit_loss(race_id, alteration):
    diffs = differences(race_id, alteration, list(fb.KNOWN_LEAKS))
    assert not diffs, report(diffs)


# fixed (fb.FIXED_LEAKS): the scramble that moved R8's limit, with the remaining known leaks
# neutralised, moves nothing a policy sees
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_leak_stint_limit(race_id):
    """R8's whole-race stint limit is retired (docs/rules.md R8): the scramble that moved it moves
    nothing a policy sees, the belief's wear cliff included."""
    diffs = differences(race_id, "scramble", list(fb.KNOWN_LEAKS))
    assert not diffs, report(diffs)


@pytest.mark.parametrize("race_id", _leak_params("frame_filter", POINTS))
def test_leak_frame_filter(race_id):
    _leak("frame_filter", race_id)


@pytest.mark.parametrize("race_id", _leak_params("lap_window", POINTS))
def test_leak_lap_window(race_id):
    _leak("lap_window", race_id)


@pytest.mark.parametrize("race_id", _leak_params("gap_behind", POINTS))
def test_leak_gap_behind(race_id):
    _leak("gap_behind", race_id)


# --- checks the alterations cannot make ---------------------------------------------------------
SNAPSHOT_NEXT_ROW = ("calibration.snapshot_at needs rows from after the snapshot moment: it locates the current "
                     "stint with race_stints over the whole frame and needs a stint covering lap k + 1 "
                     "(src/sim/calibration.py:577 _current_stint, raising 'no stint covers lap k + 1'), reads the "
                     "gaps off the lap k + 1 row (:585 _gaps) and the car behind off lap k + 1 rows (:602 _nearby)")


# fixed: fb.BUILDER (calibration.snapshot_as_of) reads rows completed by the moment only
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_snapshot_builds_from_the_past_alone(race_id):
    """The snapshot built from a race cut at the snapshot moment (every row that ends after it
    gone, the lap in progress too) is the recorded snapshot, the known leaks neutralised. Lap 0
    is left out: its state (the starting tyres) lives in the lap-1 row."""
    inp, orig, _ = race(race_id)
    failures = []
    for driver, lap in [p for p in POINTS[race_id] if p[1] > 0]:
        want = fb.plain(fb.BUILDER(orig.frame, orig.race, driver, lap))
        try:
            data, fix, _ = fb.altered(inp, orig, "cut", driver, lap, list(fb.KNOWN_LEAKS))
            snap = fb.BUILDER(data.frame, data.race, driver, lap)
            fixer = fix(fb.BUILDER(orig.frame, orig.race, driver, lap))
            got = fb.plain(fixer(snap) if fixer else snap)
        except Exception as exc:   # noqa: BLE001  a failure to build is the finding
            failures.append(f"{driver}:{lap}: {type(exc).__name__}: {exc}")
            continue
        d = fb.diff(want, got)
        if d:
            failures.append(f"{driver}:{lap}: {len(d)} field(s) differ; first {d[0]}")
    assert not failures, "\n".join(failures)


SELECTION = ("the scenario sampler picks with the whole race: eligible drivers are those who finish within "
             "two laps (src/eval/scenarios.py:127 eligible_drivers), decision laps include the laps beside every "
             "SC and VSC period of the race (src/sim/calibration.py:643 default_laps_k) and stop min_stint_laps "
             "before the driver's last lap (src/eval/scenarios.py:137 decision_laps)")


# fixed: sample_race's default rule "as_of" (src/eval/scenarios.py decision_laps_as_of)
@pytest.mark.parametrize("race_id", [_race(r) for r in POINTS])
def test_scenario_selection_up_to_the_moment(race_id):
    """The scenarios the sampler picks with a decision lap <= L are the same in the recorded
    and the scrambled race."""
    from src.eval.scenarios import sample_race
    inp, orig, _ = race(race_id)
    driver, lap = SELECTION_POINT[race_id]
    data = fb.build(fb.scramble(inp, driver, lap), altered=True)

    def picked(d):
        return sorted((s.driver, s.lap) for s in sample_race(d.frame, d.race)[0] if s.lap <= lap)
    want, got = picked(orig), picked(data)
    assert want == got, f"recorded only: {sorted(set(want) - set(got))}; altered only: {sorted(set(got) - set(want))}"


IDENTITY = ("no switch hides the real race: the agent's view carries season, event and driver code, and the "
            "race id reaches the tool outputs (src/snapshot.py:71 from_contract; src/sim/agent_tools.py race_state)")


# fixed: the model sees pseudonyms (src/snapshot.py anonymised, src/tools.py present), AGENT_SHOW_IDENTITY off
@pytest.mark.parametrize("race_id", [_race(r) for r in REAL])
def test_no_real_identity_reaches_the_agent(race_id, monkeypatch):
    """The view shown to the agent and every tool output and error string (the recorded
    observation at the race's first point) name no season, event, circuit or real driver code."""
    monkeypatch.delenv("AGENT_SHOW_IDENTITY", raising=False)
    inp, orig, recorded = race(race_id)
    driver, lap = POINTS[race_id][0]
    if (driver, lap) not in recorded:
        recorded[driver, lap] = fb.observe(orig, driver, lap)
    obs = recorded[driver, lap]
    shown = {k: v for k, v in obs["tools"].items() if not k.startswith(fb.MODEL_UNSEEN_TOOLS)}
    hits = fb.identity_hits({"snapshot": obs["snapshot"]["view"], "tools": shown}, race_id, orig.frame)
    assert not hits, "\n".join(f"{p}: {t!r}" for p, t in hits[:20])


# --- cross-race artifacts: provenance -------------------------------------------------------------
# fixed: every policy-side artifact is in src/sim/provenance.py with the races it records
@pytest.mark.parametrize("name", list(fb.POLICY_SIDE_ARTIFACTS))
def test_policy_side_artifact_records_its_races(name):
    races = fb.provenance(name)
    assert races is not None
    kind = fb.POLICY_SIDE_ARTIFACTS[name].kind
    assert kind in ("as_of", "a_priori", "not_as_of") and (kind != "a_priori" or races == [])
    assert not set(races) & fb.tyre_history_held_out(), "a held-out race feeds a policy-side table"


# fixed: each table is read as of the race (proxy.as_of_config, race_history, model_report.calibration_error)
@pytest.mark.parametrize("race_id", [*REAL, "2021_01_bahrain", "2019_10_united_kingdom", "1990_01_synthetic"])
def test_policy_side_artifacts_fitted_only_on_earlier_races(race_id):
    """Every table or constant the policy side reads records its races, all before this race."""
    later = fb.artifacts_not_before(race_id)
    assert not later, "\n".join(f"{n}: {why}" for n, why in later.items())


PRIORS = ("a race dated before every cross-race table still prices from them: the cross-race step and life "
          "ratio (configs/sim/compound_steps.json via src/sim/proxy.py) and calibration_error_s "
          "(reports/calibration/midrace_summary.csv via src/sim/model_report.py:29)")


# fixed: with no earlier race the tier-5 step is unavailable, calibration_error_s is NaN and every
# as-of factor is its documented a priori constant (race_history.A_PRIORI, EVENT_PRIOR)
def test_race_before_every_table_uses_documented_priors():
    """A synthetic race dated 1990 runs (no failure) and nothing it shows the agent comes from a
    table fitted on later races."""
    inp = dataclasses.replace(fb.synthetic_input(SYNTHETIC), race_id="1990_01_synthetic")
    data = fb.build(inp)
    obs = fb.observe(data, "AAA", 32)
    hits = fb.later_table_hits(obs)
    assert not hits, "\n".join(f"{p}: {v!r}" for p, v in hits[:20])
