"""docs/eval.md "Information contract": at a snapshot (race R, driver D, after lap L), with t
the session time D completed lap L, everything a policy can read is a function only of R's
timing records completed by t, this weekend's earlier sessions, the regulations and the tyre
allocation, and artifacts built from earlier races.

The future-blindness harness builds one synthetic race three ways, each the way lap_frame
builds a race frame (the whole-race derived columns recomputed on the rows it is given):

    full       the race as run
    truncated  only the rows completed by t
    scrambled  every row completed after t altered: lap times, compounds (rain later), tyre
               ages, stops, track status, positions and clock, a car retired, the snapshot
               driver lapped (fewer laps) and the race made longer

and requires the policy-visible outputs to be identical: the snapshot and the agent's view,
the belief's fit, every agent tool's output and error strings, the belief optimizer's plan,
the continuation behind the action compares and the baseline and rule-baseline plans, and
the scenario sampler's decision points up to t. The legacy switches (snapshot_at, the "raced"
compounds, policy_view lap/race, the legacy sampling rule) reproduce the old behaviour and
are shown to fail the same check, so the harness has teeth.

Items 1 (R8 stint limit) and 2 (pit loss) of the leakage audit are other branches': the
harness runs the belief with R8 off and a fixed RaceParams pit loss.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src import tools
from src.agent import run_agent
from src.baselines.domain import PolicyUnavailable
from src.contracts import RaceParams
from src.eval import decisions, policies
from src.eval.scenarios import Scenario, decision_laps_as_of, eligible_drivers, sample_race
from src.eval.score import score_scenario
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.sim import SimConfig, Simulator, agent_tools, calibrate, calibration
from src.sim.compute import load_profile
from src.sim.config import PolicyView
from src.sim.data import CACHE_DIR, derive_columns, visible_rows
from src.snapshot import RACE_LABEL, RaceSnapshot

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
TOTAL = 40
L = 12
D = "DDD"
PACE = {"SOFT": (80.0, 0.10), "MEDIUM": (80.8, 0.05), "HARD": (81.5, 0.03)}
CARS = [("AAA", 0.0, [("SOFT", 14), ("MEDIUM", 26)]), ("BBB", 0.3, [("MEDIUM", 18), ("HARD", 22)]),
        ("CCC", 0.5, [("SOFT", 12), ("HARD", 28)]), (D, 0.8, [("SOFT", 16), ("MEDIUM", 24)]),
        ("EEE", 1.1, [("MEDIUM", 20), ("SOFT", 20)]), ("FFF", 1.6, [("SOFT", 10), ("MEDIUM", 15), ("SOFT", 15)])]
SC_LAPS = (30, 31, 32)   # after the snapshot
LOCAL = load_profile("local")
PROFILE = LOCAL.model_copy(update={"sim": LOCAL.sim.model_copy(update={"n_samples": 12, "compare_n_samples": 12})})
# the belief as the eval runs it, R8 off: the stint limit is item 1, fix/tyre-life-no-hindsight's
BELIEF = agent_tools.belief_config().model_copy(update={"stint_limit_margin_laps": None})
LEGACY_VIEW = PolicyView(window_cut="lap", clean="race")


def raw_race(seed=0):
    """Rows of lap_frame's _columns for a 40-lap race, every car on the lead lap, with noise,
    traffic outliers for the robust filter to cut, pit loss on the in-laps and an SC after L."""
    rng = np.random.default_rng(seed)
    rows = []
    for code, offset, plan in CARS:
        lap, t = 0, 0.0
        for stint, (comp, n) in enumerate(plan, start=1):
            base, slope = PACE[comp]
            for age in range(1, n + 1):
                lap += 1
                pit_in = age == n and stint < len(plan)
                sc = lap in SC_LAPS
                lt = base + slope * age + offset - 0.05 * lap + rng.normal(0.0, 0.25)
                lt += 5.0 * (lap == 1) + 20.0 * pit_in + 4.0 * (age == 1 and stint > 1)
                lt += rng.choice([0.0, 0.7, 2.5], p=[0.84, 0.09, 0.07]) + 30.0 * sc   # borderline, clear
                rows.append({"driver": code, "lap": lap, "stint": float(stint), "compound": comp,
                             "tyre_age": float(age), "lap_time_s": lt, "track_status": "4" if sc else "1",
                             "pit_in": pit_in, "pit_out": age == 1 and stint > 1, "start_s": t,
                             "end_s": t + lt, "team": f"T{code}", "position": np.nan})
                t += lt
    df = pd.DataFrame(rows)
    df["position"] = df.groupby("lap")["end_s"].rank(method="first")
    return df


def t_cut(raw):
    return float(raw.loc[(raw["driver"] == D) & (raw["lap"] == L), "end_s"].iloc[0])


def scramble_after(raw, t, seed=1):
    """`raw` with everything completed after t altered (the module docstring's list)."""
    rng = np.random.default_rng(seed)
    df = raw.copy()
    late = df["end_s"].to_numpy() > t
    n = int(late.sum())
    df.loc[late, "lap_time_s"] += rng.uniform(-3.0, 6.0, n)
    df.loc[late, "compound"] = df.loc[late, "compound"].map({"SOFT": "HARD", "MEDIUM": "INTERMEDIATE",
                                                             "HARD": "SOFT"})
    df.loc[late, "tyre_age"] += 7.0
    df.loc[late, "stint"] += 1.0
    df.loc[late, "pit_in"] = rng.random(n) < 0.3
    df.loc[late, "pit_out"] = rng.random(n) < 0.3
    df.loc[late, "track_status"] = rng.choice(["1", "4", "671"], n)
    df.loc[late, "position"] = rng.permutation(n).astype(float)
    df.loc[late, ["start_s", "end_s"]] += 3.0
    retired = late & (df["driver"] == "BBB").to_numpy()           # BBB retires after t
    lapped = late & (df["driver"] == D).to_numpy() & (df["lap"] > TOTAL - 2).to_numpy()   # D lapped
    df = df[~(retired | lapped)]
    last = df[df["driver"] == "AAA"].iloc[-1]
    extra = [{**last.to_dict(), "lap": TOTAL + i, "start_s": last["end_s"] + 90.0 * (i - 1),
              "end_s": last["end_s"] + 90.0 * i} for i in (1, 2, 3)]           # the race runs longer
    return pd.concat([df, pd.DataFrame(extra, index=range(len(raw), len(raw) + len(extra)))])


def frames():
    """{"full", "truncated", "scrambled"}: each race frame as lap_frame would build it."""
    raw = raw_race()
    t = t_cut(raw)
    return {"full": derive_columns(raw.copy()),
            "truncated": derive_columns(raw[raw["end_s"] <= t].copy()),
            "scrambled": derive_columns(scramble_after(raw, t))}


def practice():
    from test_sim import synthetic_practice
    return synthetic_practice(coefs=PACE)


def race_for(frame, compounds="nominated"):
    return RaceParams(race_id="syn_contract", total_laps=TOTAL, pit_loss_s=20.0, min_stint_laps=5,
                      compounds=calibrate.race_compounds(frame, compounds))


# each switch back to the old behaviour, by the leakage audit's item
LEGACY = {"clean": {"clean": "race"},                       # item 3
          "window_cut": {"window_cut": "lap"},              # item 6
          "builder": {"builder": calibration.snapshot_at},  # item 4 (and the old gaps)
          "compounds": {"compounds": "raced"}}              # item 5


def policy_outputs(frame, *, legacy=()):
    """Every policy-visible output at (D, L) as JSON text, under the leak-free defaults, with
    each switch named in `legacy` (LEGACY) set as it was before."""
    old = {k: v for name in legacy for k, v in LEGACY[name].items()}
    view = PolicyView(**{k: old[k] for k in ("clean", "window_cut") if k in old})
    cfg = BELIEF.model_copy(update={"policy_view": view})
    race = race_for(frame, old.get("compounds", "nominated"))
    snap = old.get("builder", calibration.snapshot_as_of)(frame, race, D, L)
    sim = Simulator(frame, cfg, practice())
    kw = {"sim": sim, "profile": PROFILE}
    rest = snap.remaining_laps
    fit = sim.fitted_for(snap)
    scenario = Scenario(scenario_id=f"syn_contract:{D}:{L:02d}", race_id="syn_contract", driver=D, lap=L,
                        snapshot=snap, actual_stints=[])
    out = {
        "snapshot": snap.model_dump(mode="json"),
        "view": RaceSnapshot.from_contract(snap).anonymised().to_dict(),
        "fit": {"coefs": {c: list(v) for c, v in fit.degradation.coefs.items()},
                "fit_report": fit.degradation.fit_report, "ref_lap_s": fit.ref_lap_s,
                "noise_sd": fit._lap_noise_sd()},
        "race_state": agent_tools.race_state(snap, config=cfg),
        "model_report": agent_tools.model_report(snap, ages={"MEDIUM": 30}, **kw),
        "pit_loss": agent_tools.pit_loss(snap, **kw),
        "simulate_plan": agent_tools.simulate_plan(snap, [["SOFT", 6], ["MEDIUM", rest - 6]], **kw),
        "simulate_error": agent_tools.simulate_plan(snap, [["SOFT", 2], ["WET", rest - 2]], **kw),
        "compare_plans": agent_tools.compare_plans(
            snap, [[["SOFT", 6], ["MEDIUM", rest - 6]], {"action": "stay_out", "laps": 2},
                   {"action": "pit", "compound": "MEDIUM"}, {"action": "pit", "compound": "HARD"}], **kw),
        "evaluate_pit": agent_tools.evaluate_action(snap, {"action": "pit", "compound": "MEDIUM"}, **kw),
        "evaluate_stay": agent_tools.evaluate_action(snap, {"action": "stay", "laps": 3}, **kw),
        "evaluate_error": agent_tools.evaluate_action(snap, {"action": "pit", "compound": "WET"}, **kw),
        "belief_optimum": decisions.belief_optimum(snap, sim, 2, n_samples=12).as_pairs(),
        "to_plan": decisions.to_plan(snap, {"decision": "PIT", "compound": "MEDIUM"}, sim, 2,
                                     n_samples=12).as_pairs(),
    }
    for name, policy in [*((n, policies.BASELINE_POLICIES[n]) for n in ("fixed_one_stop", "fixed_two_stop",
                                                                         "greedy_crossover")),
                         *policies.RULE_POLICIES.items()]:
        try:
            out[name] = policy(scenario, sim, 2).plan.as_pairs()
        except PolicyUnavailable as exc:   # an answer too: it must not depend on the future either
            out[name] = f"unavailable: {exc}"
    return {name: json.dumps(agent_tools._jsonable(v), sort_keys=True, default=str) for name, v in out.items()}


@pytest.fixture(scope="module")
def race_frames():
    return frames()


@pytest.fixture(scope="module")
def as_of(race_frames):
    return {name: policy_outputs(f) for name, f in race_frames.items()}


@pytest.mark.parametrize("variant", ["truncated", "scrambled"])
def test_every_policy_visible_output_ignores_what_happened_after_t(as_of, variant):
    a, b = as_of["full"], as_of[variant]
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []


def test_the_harness_runs_the_policies_it_checks(as_of):
    got = {k: json.loads(v) for k, v in as_of["full"].items()}
    assert got["simulate_plan"]["valid"] and not got["simulate_error"]["valid"]
    assert got["simulate_error"]["errors"] and got["evaluate_error"]["error"]
    assert got["compare_plans"]["valid"][:3] == [True, True, True]
    assert all(got[e]["gap_mean_s"] is not None for e in ("evaluate_pit", "evaluate_stay"))
    assert got["snapshot"]["race"]["total_laps"] == TOTAL
    planned = [n for n in ("fixed_one_stop", "fixed_two_stop", "greedy_crossover", "run_to_flag",
                           "one_stop_total", "belief_optimum", "to_plan") if isinstance(got[n], list)]
    assert len(planned) >= 5
    for name in planned:
        assert sum(n for _, n in got[name]) == TOTAL - L, name


@pytest.mark.parametrize("switch, expect", [
    ("clean", {"fit", "model_report"}),                  # the race-wide robust filter
    ("window_cut", {"fit", "model_report"}),             # rows that ended after t
    ("builder", {"snapshot", "view", "race_state"}),     # the driver's own lap count
    ("compounds", {"snapshot", "view", "race_state"}),   # rain to come
])
def test_each_legacy_switch_fails_the_same_check(race_frames, switch, expect):
    """Each switch back to the old behaviour reads the future on its own: the harness has teeth."""
    a = policy_outputs(race_frames["full"], legacy=[switch])
    b = policy_outputs(race_frames["scrambled"], legacy=[switch])
    assert expect <= {k for k in a if a[k] != b[k]}


def test_the_world_reads_the_whole_race(race_frames):
    """The full window (the world, the hindsight reference) keeps every lap: it is the scorer."""
    snap = calibration.snapshot_as_of(race_frames["full"], race_for(race_frames["full"]), D, L)
    world = BELIEF.model_copy(update={"fit_window": "full"})
    fits = [Simulator(race_frames[v], world, practice()).fitted_for(snap).degradation.coefs
            for v in ("full", "scrambled")]
    assert json.dumps(fits[0], sort_keys=True) != json.dumps(fits[1], sort_keys=True)


def test_every_legacy_switch_together_moves_the_plans_too(race_frames):
    a, b = (policy_outputs(race_frames[v], legacy=list(LEGACY)) for v in ("full", "scrambled"))
    assert {"snapshot", "fit", "model_report", "compare_plans", "race_state", "simulate_plan"} <= {
        k for k in a if a[k] != b[k]}


# --- item 3: the clean flag and the other derived columns, as of t --------------------------
def test_window_rows_are_the_visible_rows_with_their_own_filter(race_frames):
    full = race_frames["full"]
    raw = raw_race()
    t = t_cut(raw)
    sim = Simulator(full, BELIEF, practice())
    snap = calibration.snapshot_as_of(full, race_for(full), D, L)
    rows = sim.window_rows(snap, L)
    expect = derive_columns(raw[raw["end_s"] <= t].copy())
    cols = ["driver", "lap", "clean", "pace_resid_s", "driver_offset_s", "gap_ahead_s", "gap_behind_s", "free_air"]
    pd.testing.assert_frame_equal(rows[cols].reset_index(drop=True), expect[cols].reset_index(drop=True))
    # the race-wide filter saw every lap: its flags or residuals on these rows differ
    whole = full.loc[rows.index]
    assert not np.allclose(whole["pace_resid_s"].fillna(0), rows["pace_resid_s"].fillna(0))


def test_legacy_policy_view_reads_the_frame_as_before(race_frames):
    full = race_frames["full"]
    legacy = BELIEF.model_copy(update={"policy_view": LEGACY_VIEW})
    sim = Simulator(full, legacy, practice())
    snap = calibration.snapshot_as_of(full, race_for(full), D, L)
    rows = sim.window_rows(snap, L)
    pd.testing.assert_frame_equal(rows, full[full["lap"] <= L])
    assert calibration.config_hash(legacy) != calibration.config_hash(BELIEF)
    assert calibration.config_hash(SimConfig()) == calibration.config_hash(SimConfig(policy_view=PolicyView()))


# --- item 6: the window is cut by session time ----------------------------------------------------
def test_window_is_cut_at_the_snapshot_drivers_time(race_frames):
    full = race_frames["full"]
    t = t_cut(full)
    sim = Simulator(full, BELIEF, practice())
    snap = calibration.snapshot_as_of(full, race_for(full), D, L)
    rows = sim.window_rows(snap, L)
    assert rows["end_s"].max() <= t
    slower_lap_l = full[(full["lap"] == L) & (full["end_s"] > t)]
    assert len(slower_lap_l) and not rows.index.isin(slower_lap_l.index).any()   # lap L, finished later
    assert sim.cut_time(D, L) == t and sim.cut_time("nobody", L) == full.loc[full["lap"] == L, "end_s"].min()
    assert sim.cut_time(D, 0) is None


def test_visible_rows_leave_out_rows_without_an_end_time():
    f = pd.DataFrame({"end_s": [1.0, np.nan, 3.0, 2.0]})
    assert visible_rows(f, 2.0).index.tolist() == [0, 3]


# --- items 4 and 5: scheduled distance and nominated compounds ---------------------------------
def test_snapshot_carries_the_scheduled_distance_and_the_nominated_compounds(race_frames):
    scr = race_frames["scrambled"]
    assert int(scr[scr["driver"] == D]["lap"].max()) == TOTAL - 2          # lapped in this version
    assert "INTERMEDIATE" in set(scr["compound"])                          # it rained, after t
    snap = calibration.snapshot_as_of(scr, race_for(scr), D, L)
    assert snap.race.total_laps == TOTAL and snap.remaining_laps == TOTAL - L
    assert snap.race.compounds == ["HARD", "MEDIUM", "SOFT"]
    # the legacy builder and compound source tell the policy both
    old = calibration.snapshot_at(scr, race_for(scr, "raced"), D, L)
    assert old.race.total_laps == TOTAL - 2 and "INTERMEDIATE" in old.race.compounds


def test_a_compound_raced_by_t_joins_the_nominated_three():
    raw = raw_race()
    raw.loc[(raw["driver"] == "AAA") & (raw["lap"] <= 3), "compound"] = "INTERMEDIATE"   # a wet start
    f = derive_columns(raw)
    snap = calibration.snapshot_as_of(f, race_for(f), D, L)
    assert snap.race.compounds == ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT"]


def test_race_compounds_sources():
    f = derive_columns(raw_race())
    assert calibrate.race_compounds(f) == ["HARD", "MEDIUM", "SOFT"]
    assert calibrate.race_compounds(f.assign(compound="SOFT"), "raced") == ["SOFT"]
    with pytest.raises(ValueError, match="compound source"):
        calibrate.race_compounds(f, "practice")


def test_snapshot_as_of_on_an_in_lap_and_for_a_car_not_running():
    f = derive_columns(raw_race())
    snap = calibration.snapshot_as_of(f, race_for(f), "AAA", 14)   # AAA's in-lap; MEDIUM goes on
    # no record by t holds the set going on: the car shows on the SOFT set it came in on, the
    # same as from the race cut at t (snapshot_at read MEDIUM off the lap 15 row)
    assert (snap.compound, snap.tyre_age, snap.stints) == ("SOFT", 14, [])
    assert calibration.snapshot_at(f, race_for(f), "AAA", 14).compound == "MEDIUM"
    assert calibration.snapshot_as_of(f[f["end_s"] <= f.loc[(f["driver"] == "AAA") & (f["lap"] == 14), "end_s"].iloc[0]],
                                      race_for(f), "AAA", 14) == snap
    with pytest.raises(ValueError, match="not running"):
        calibration.snapshot_as_of(f[f["lap"] <= 20], race_for(f), "AAA", 25)


def test_snapshot_as_of_gaps_are_known_at_t(race_frames):
    full = race_frames["full"]
    t = t_cut(full)
    snap = calibration.snapshot_as_of(full, race_for(full), D, L)
    lap_l = full[full["lap"] == L].sort_values("end_s")
    ahead = lap_l[lap_l["end_s"] < t].iloc[-1]
    assert snap.car_ahead == ahead["driver"] and snap.gap_ahead_s == pytest.approx(t - ahead["end_s"])
    behind = [c for c in snap.field.nearby if c.gap_s > 0]
    if behind:   # the car behind as of the previous crossing, which had not caught up by t
        prev = full[(full["lap"] == L - 1)].set_index("driver")["end_s"]
        assert prev[behind[0].driver] <= t and behind[0].gap_s == pytest.approx(prev[behind[0].driver] - prev[D])
    # the legacy builder reads the car behind's crossing at the end of lap L, after t
    old = calibration.snapshot_at(full, race_for(full), D, L)
    assert old.gap_ahead_s == pytest.approx(snap.gap_ahead_s)


def test_world_scores_a_lapped_driver_over_the_scheduled_distance(race_frames):
    """Decision (docs/eval.md): every plan, the reference's included, covers the scheduled
    distance and the world scores all of it, so regret compares like with like even when the
    car was lapped or later retired; actual_strategy runs its last set on to the flag."""
    scr = race_frames["scrambled"]
    snap = calibration.snapshot_as_of(scr, race_for(scr), D, L)
    actual = [(c, n) for c, n, _, _ in calibration.race_stints(scr, D)]
    scenario = Scenario(scenario_id="syn:DDD:12", race_id="syn", driver=D, lap=L, snapshot=snap,
                        actual_stints=actual)
    world = Simulator(scr, SimConfig(stint_limit_margin_laps=None))
    pols = {"actual_strategy": policies.BASELINE_POLICIES["actual_strategy"],
            "fixed_one_stop": policies.BASELINE_POLICIES["fixed_one_stop"]}
    rows = score_scenario(scenario, world, pols, belief=Simulator(scr, BELIEF, practice()), n_samples=4)
    assert sum(n for _, n in actual) == TOTAL - 2 and rows[0]["status"] != "no_reference"
    assert any(row["status"] == "ok" for row in rows)
    for row in rows:
        assert row["remaining_laps"] == TOTAL - L
        if row["status"] == "ok":
            assert sum(n for _, n in json.loads(row["plan"])) == TOTAL - L
    assert sum(n for _, n in json.loads(rows[0]["opt_plan"])) == TOTAL - L


def test_actual_strategy_runs_its_last_set_on_to_the_scheduled_distance():
    got = policies._actual_to_distance([("SOFT", 12), ("MEDIUM", 26)], TOTAL)
    assert [(s.compound, s.laps) for s in got] == [("SOFT", 12), ("MEDIUM", 28)]
    same = policies._actual_to_distance([("SOFT", 12), ("MEDIUM", 28)], TOTAL)   # a legacy set: no change
    assert [(s.compound, s.laps) for s in same] == [("SOFT", 12), ("MEDIUM", 28)]


# --- S: scenario selection reads nothing after the sampled lap -----------------------------------
def test_as_of_decision_laps_read_nothing_after_each_lap(race_frames):
    full, scr = race_frames["full"], race_frames["scrambled"]
    race = race_for(full)
    t = t_cut(full)

    def known(frame):
        out = set()
        for code, *_ in CARS:
            ends = frame[frame["driver"] == code].set_index("lap")["end_s"]
            out |= {(code, k) for k in decision_laps_as_of(frame, race, code) if k == 0 or ends.get(k, np.inf) <= t}
        return out
    assert known(full) == known(scr)
    assert {k for c, k in known(full) if c == D} == {0, 5, 10}
    # with `neutralised`: SC laps and the first green lap after them, from the driver's own record
    assert {30, 31, 32, 33} <= set(decision_laps_as_of(full, race, D, neutralised=True))
    # the legacy rule drops the car that retires and the lap before the SC
    assert "BBB" not in eligible_drivers(scr, TOTAL) and "BBB" in eligible_drivers(full, TOTAL)
    assert 29 in calibration.default_laps_k(full, TOTAL) and 29 not in decision_laps_as_of(full, race, D, neutralised=True)


def test_sample_race_as_of_keeps_a_car_that_later_retires(race_frames):
    scr = race_frames["scrambled"]
    items, skipped = sample_race(scr, race_for(scr), drivers=["BBB"])
    laps = [s.lap for s in items]
    assert laps and max(laps) <= L + 2 and all(s.snapshot.race.total_laps == TOTAL for s in items)
    legacy, _ = sample_race(scr, race_for(scr), drivers=["BBB"], rule="legacy")
    assert legacy == []
    with pytest.raises(ValueError, match="sampling rule"):
        sample_race(scr, race_for(scr), rule="hindsight")


# --- item 7: the model sees no real race or driver name --------------------------------------------
class Scripted(ModelBackend):
    name = "test"
    model_id = "scripted"

    def __init__(self, turns):
        self.turns, self.seen = iter(turns), []

    def run_turn(self, **kwargs):
        self.seen.append(json.dumps(kwargs["messages"], default=str))
        return next(self.turns)


def every_tool_then_submit():
    calls = [("get_race_snapshot", {}), ("estimate_tyre_degradation", {"compound": "SOFT"}),
             ("estimate_tyre_degradation", {"compound": "WET"}), ("estimate_pit_loss", {}),
             ("get_model_report", {"ages": {"MEDIUM": 30}}),
             ("compare_plans", {"candidates": [{"action": "stay_out", "laps": 2}, {"action": "pit", "compound": "HARD"},
                                               [["MEDIUM", 1]]]}),
             ("evaluate_strategy", {"action": {"action": "pit", "compound": "MEDIUM"}}),
             ("evaluate_strategy", {"action": {"action": "pit", "compound": "WET"}})]
    turns = [ModelTurn(tool_calls=[ToolCall(str(i), n, a)]) for i, (n, a) in enumerate(calls)]
    return turns + [ModelTurn(tool_calls=[ToolCall("s", "submit_recommendation", {
        "decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6, "rationale": "r", "key_risks": []})])]


@pytest.fixture(scope="module")
def spain_snapshot():
    data = agent_tools._race_data("2021_04_spain", load_profile())
    return data, calibration.snapshot_as_of(data.frame, data.race, "HAM", 20)


def _shown(backend, traj):
    return " ".join(backend.seen) + json.dumps([e["output"] for e in traj], default=str)


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_the_model_sees_pseudonyms_in_the_prompt_and_every_tool_output(spain_snapshot, monkeypatch):
    monkeypatch.delenv("AGENT_SHOW_IDENTITY", raising=False)
    data, snap = spain_snapshot
    backend = Scripted(every_tool_then_submit())
    stats = {}
    rec, traj = run_agent(snap, backend=backend, verbose=False, max_steps=12, stats=stats)
    assert rec is not None and len(traj) == 9
    shown = _shown(backend, traj)
    assert "Car A" in shown and RACE_LABEL in shown
    leaks = [w for w in ["spain", "Spain", "2021_04_spain"] if w in shown]
    leaks += re.findall(r"(?<![\w.])2021(?![\w.]?\d)", shown)
    leaks += [c for c in set(data.frame["driver"]) if f'"{c}"' in shown or f" {c} " in shown or f"'{c}'" in shown]
    assert leaks == []
    assert any(e["error"] for e in traj)   # the error strings went through the same mapping
    # the harness keeps the real names
    assert stats["identity"]["race_id"] == "2021_04_spain" and stats["identity"]["labels"]["HAM"] == "Car A"
    view = tools.get_race_snapshot()
    assert "season" not in view and view["driver"] == "Car A" and view["event"] == RACE_LABEL


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_show_identity_is_the_ablation_switch(spain_snapshot, monkeypatch):
    monkeypatch.setenv("AGENT_SHOW_IDENTITY", "1")
    _, snap = spain_snapshot
    backend = Scripted(every_tool_then_submit())
    rec, traj = run_agent(snap, backend=backend, verbose=False, max_steps=12)
    shown = _shown(backend, traj)
    assert '"HAM"' in shown and '"season": 2021' in shown and '"event": "spain"' in shown


def test_present_maps_codes_race_and_year_in_strings_and_keys():
    from src.contracts import FieldState, NearbyCar
    f = derive_columns(raw_race())
    snap = calibration.snapshot_as_of(f, race_for(f), D, L).model_copy(
        update={"field": FieldState(nearby=[NearbyCar(driver="CCC", position=3, gap_s=-1.0)])})
    shown = tools.set_snapshot(snap, identity=False)
    assert shown.driver == "Car A" and shown.nearby_cars[0]["driver"] == "Car B"
    out = tools.present({"error": "No simulator for race 'syn_contract' (DDD behind CCC, 2019)",
                         "DDD": ["CCC", "SOFT"]})
    assert out == {"error": f"No simulator for race '{RACE_LABEL}' (Car A behind Car B, <year>)",
                   "Car A": ["Car B", "SOFT"]}
    tools.set_snapshot(snap, identity=True)
    assert tools.present("DDD 2019") == "DDD 2019"
