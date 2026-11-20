"""The compound proxy (src/sim/proxy.py): the source blend and its weights, anchoring, the
ordering constraint, uncertainty, R8 for an unraced compound and the off switch, on synthetic
weekends; the validation's leave-one-race-out weights; then the 2022 Saudi Arabia case (SOFT run
in practice, never raced) on the calibration race."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim import SimConfig, Simulator, proxy
from src.sim.config import CompoundProxy, load_compound_steps
from test_sim import RACE, TRUE, at_lap, synthetic_frame, synthetic_practice
from test_sim_position import race_frame

FUEL = -0.04
HARD_PRACTICE = (83.5, 0.01)     # practice HARD, a step of +1.5 s and -0.02 s/lap from practice MEDIUM
STEPS = {"SOFT-MEDIUM": {"pace_frac": {"value": -0.02, "sd": 0.004, "n": 4},
                         "wear_ratio": {"value": 2.0, "sd": 0.5, "n": 4},
                         "life_ratio": {"value": 0.5, "sd": 0.1, "n": 4}},
         "MEDIUM-HARD": {"pace_frac": {"value": -0.01, "sd": 0.003, "n": 4},
                         "wear_ratio": {"value": 1.5, "sd": 0.6, "n": 4},
                         "life_ratio": {"value": 0.8, "sd": 0.2, "n": 4}}}
# 1 / mse: practice weighs 1, qualifying 2, the step 1 for pace; practice 1, the step 3 for wear
WEIGHTS = {"pace": {"practice": 1.0, "qualifying": 2.0, "cross_race_step": 1.0},
           "wear": {"practice": 1.0, "cross_race_step": 3.0}}
SD_FLOOR = {"pace": 0.6, "wear": 0.04}


def config(enabled=True, steps=STEPS, sources=None, weights=WEIGHTS, **kw):
    px = dict(enabled=enabled, steps=steps, weights=weights, sd_floor=SD_FLOOR)
    if sources is not None:
        px["sources"] = sources
    return SimConfig(compound_proxy=CompoundProxy(**px), pre_race_base_delta_s=0.0, **kw)


def practice(hard=True):
    coefs = {**TRUE, **({"HARD": HARD_PRACTICE} if hard else {})}
    return synthetic_practice(coefs=coefs)


def quali(gap_hard_medium=None, gap_soft_medium=-0.8):
    """Each of four drivers: Q1 MEDIUM and SOFT, Q2 SOFT (a segment 0.4 s faster), driver
    offsets 0, 0.3, 0.6, 0.9 s; HARD in Q1 too when `gap_hard_medium` is given. One slow
    (aborted) lap per driver that the trim must drop."""
    rows = []
    for k, drv in enumerate("ABCD"):
        off = 0.3 * k
        laps = [("Q1", "MEDIUM", 90.0), ("Q1", "SOFT", 90.0 + gap_soft_medium), ("Q2", "SOFT", 89.6 + gap_soft_medium)]
        if gap_hard_medium is not None:
            laps.append(("Q1", "HARD", 90.0 + gap_hard_medium))
        for seg, comp, t in laps:
            rows.append({"driver": drv, "compound": comp, "lap_time_s": t + off, "segment": seg, "end_s": 0.0,
                         "timed": True})
        rows.append({"driver": drv, "compound": "SOFT", "lap_time_s": 99.0, "segment": "Q3", "end_s": 0.0,
                     "timed": True})
    return pd.DataFrame(rows)


def world(enabled=True, steps=STEPS, prac=None, q=None, frame=None, sources=None, weights=WEIGHTS, **kw):
    return Simulator(synthetic_frame(fuel=FUEL) if frame is None else frame,
                     config(enabled, steps, sources, weights, **kw), prac, qualifying=q)


def tiers(sim, c):
    e = sim.degradation.fit_report["proxy"][c]
    return e["pace_tier"], e["wear_tier"]


def used(sim, c, kind="pace"):
    return {p["source"]: p["weight"] for p in sim.degradation.fit_report["proxy"][c][f"{kind}_sources"]}


# --- source selection and the blend ---------------------------------------------------------------
def test_raced_compounds_are_tier_one_and_unchanged():
    on, off = world(prac=practice()), world(enabled=False, prac=practice())
    for c in ("SOFT", "MEDIUM"):
        assert tiers(on, c) == (1, 1)
        assert on.degradation.coefs[c] == off.degradation.coefs[c]
    assert on.degradation.fuel_slope == off.degradation.fuel_slope
    assert "HARD" not in off.modelled and "HARD" in on.modelled


def test_every_source_that_exists_enters_the_blend():
    full = world(prac=practice(), q=quali(gap_hard_medium=0.9))
    assert used(full, "HARD") == pytest.approx({"practice": 0.25, "qualifying": 0.5, "cross_race_step": 0.25})
    assert used(full, "HARD", "wear") == pytest.approx({"practice": 0.25, "cross_race_step": 0.75})
    # no HARD long run: qualifying and the step for pace, the step alone for wear
    nq = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9))
    assert used(nq, "HARD") == pytest.approx({"qualifying": 2 / 3, "cross_race_step": 1 / 3})
    assert used(nq, "HARD", "wear") == {"cross_race_step": 1.0}
    # neither: the step, always there
    assert used(world(), "HARD") == {"cross_race_step": 1.0}
    assert "HARD" not in world(steps={}).modelled
    assert full.degradation.fit_report["proxy"]["HARD"]["pace_source"] == "practice+qualifying+cross_race_step"


def test_the_blend_is_the_weighted_mean_of_the_sources():
    full = world(prac=practice(), q=quali(gap_hard_medium=0.9)).degradation
    base_m, wear_m = full.coefs["MEDIUM"]
    e = full.fit_report["proxy"]["HARD"]
    parts = {p["source"]: p["value"] for p in e["pace_sources"]}
    assert parts["qualifying"] == pytest.approx(base_m + 0.9)
    assert parts["cross_race_step"] == pytest.approx(base_m / 0.99)
    assert full.coefs["HARD"][0] == pytest.approx(0.25 * parts["practice"] + 0.5 * parts["qualifying"]
                                                  + 0.25 * parts["cross_race_step"])
    w = {p["source"]: p["value"] for p in e["wear_sources"]}
    assert full.coefs["HARD"][1] == pytest.approx(max(0.25 * w["practice"] + 0.75 * w["cross_race_step"], 0.0))
    assert proxy.describe("HARD", e).startswith(
        "HARD: pace from practice long runs 25%, qualifying 50%, the cross-race step 25% (anchor MEDIUM)")


def test_sources_can_be_restricted():
    one = world(prac=practice(), q=quali(gap_hard_medium=0.9), sources=["cross_race_step"])
    assert used(one, "HARD") == {"cross_race_step": 1.0}
    nopq = world(prac=practice(), q=quali(gap_hard_medium=0.9), sources=["practice", "cross_race_step"])
    assert set(used(nopq, "HARD")) == {"practice", "cross_race_step"}


# --- anchoring -----------------------------------------------------------------------------------
def test_practice_is_the_practice_step_from_the_raced_anchor():
    sim = world(prac=practice(), sources=["practice"])
    base_m, wear_m = sim.degradation.coefs["MEDIUM"]
    pp = sim.physics_prior().coefs
    base_h, wear_h = sim.degradation.coefs["HARD"]
    assert base_h == pytest.approx(base_m + pp["HARD"][0] - pp["MEDIUM"][0])
    assert wear_h == pytest.approx(wear_m + pp["HARD"][1] - pp["MEDIUM"][1])


def test_the_race_sets_the_level_of_a_borrowed_compound():
    frame = synthetic_frame(fuel=FUEL)
    slow = frame.assign(lap_time_s=frame["lap_time_s"] + 2.0)
    kw = {"prac": practice(), "q": quali(gap_hard_medium=0.9)}
    a = world(frame=frame, **kw).degradation.coefs["HARD"][0]
    b = world(frame=slow, **kw).degradation.coefs["HARD"][0]
    # practice and qualifying add a step in seconds; the step scales with the anchor (a share of the lap)
    assert b - a == pytest.approx(2.0 * (0.25 + 0.5 + 0.25 / 0.99), abs=1e-6)


def test_qualifying_alone_is_the_qualifying_gap_from_the_anchor():
    sim = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9),
                weights={"pace": {"qualifying": 1.0, "cross_race_step": 0.0}, "wear": {}})
    base_m = sim.degradation.coefs["MEDIUM"][0]
    assert sim.degradation.coefs["HARD"][0] == pytest.approx(base_m + 0.9, abs=1e-9)


def test_qualifying_gaps_take_out_driver_and_segment_and_trim_slow_laps():
    gaps = proxy.qualifying_gaps(quali(gap_hard_medium=0.9))
    assert gaps["gap"]["SOFT"] - gaps["gap"]["MEDIUM"] == pytest.approx(-0.8, abs=1e-9)
    assert gaps["gap"]["HARD"] - gaps["gap"]["MEDIUM"] == pytest.approx(0.9, abs=1e-9)
    assert gaps["n_drivers"]["HARD"] == 4
    # one driver linking a compound is not enough
    one = quali(gap_hard_medium=0.9)
    one = one[(one["compound"] != "HARD") | (one["driver"] == "A")]
    assert "HARD" not in proxy.qualifying_gaps(one)["gap"]


# --- ordering ------------------------------------------------------------------------------------
def test_ordering_clamps_a_proxied_value_and_never_a_raced_one():
    # a step table that would make HARD faster than MEDIUM and wear more
    bad = {"MEDIUM-HARD": {"pace_frac": {"value": 0.02, "sd": 0.0, "n": 1},
                           "wear_ratio": {"value": 0.5, "sd": 0.0, "n": 1},
                           "life_ratio": {"value": 0.8, "sd": 0.0, "n": 1}}}
    on, off = world(steps=bad), world(enabled=False)
    base_m, wear_m = on.degradation.coefs["MEDIUM"]
    assert on.degradation.coefs["HARD"] == pytest.approx((base_m, wear_m))
    assert on.degradation.fit_report["proxy"]["HARD"]["ordering"] == ["base raised", "wear lowered"]
    assert on.degradation.coefs["MEDIUM"] == off.degradation.coefs["MEDIUM"]


def test_ordering_puts_a_proxied_soft_between_its_neighbours():
    # SOFT never raced: the race has MEDIUM and HARD; practice SOFT slower than MEDIUM must be lifted
    frame = synthetic_frame(fuel=FUEL)
    frame = frame.assign(compound=frame["compound"].map({"SOFT": "HARD", "MEDIUM": "MEDIUM"}),
                         lap_time_s=np.where(frame["compound"] == "SOFT", frame["lap_time_s"] + 3.0, frame["lap_time_s"]))
    prac = synthetic_practice(coefs={"SOFT": (85.0, 0.01), "MEDIUM": TRUE["MEDIUM"], "HARD": (83.0, 0.10)})
    sim = world(frame=frame, prac=prac)
    (bs, ws), (bm, wm) = sim.degradation.coefs["SOFT"], sim.degradation.coefs["MEDIUM"]
    assert bs <= bm and ws >= wm
    assert tiers(sim, "SOFT") == (2, 2)


def test_a_proxied_wear_is_floored_at_zero():
    prac = synthetic_practice(coefs={**TRUE, "HARD": (83.0, -0.2)})
    assert world(prac=prac).degradation.coefs["HARD"][1] == 0.0


# --- uncertainty ---------------------------------------------------------------------------------
def test_proxied_sd_is_at_least_the_blend_error_and_the_step_spread_and_moves_with_the_anchor():
    sim = world(prac=practice(), q=quali(gap_hard_medium=0.9))
    e = sim.degradation.fit_report["proxy"]["HARD"]
    base_m = sim.degradation.coefs["MEDIUM"][0]
    spread = base_m / 0.99 * 0.003 / 0.99    # the step's spread times its value
    assert e["pace_sd_s"] == pytest.approx(max(SD_FLOOR["pace"], spread))
    assert sim.degradation.fit_report["sd"]["base_s"]["HARD"] >= SD_FLOOR["pace"]
    wide = world(steps={**STEPS, "MEDIUM-HARD": {**STEPS["MEDIUM-HARD"],
                                                 "pace_frac": {"value": -0.01, "sd": 0.05, "n": 4}}})
    assert wide.degradation.fit_report["proxy"]["HARD"]["pace_sd_s"] > 3.0   # the spread rules
    assert sim.degradation.fit_report["shrinkage_weight"]["base_s"]["HARD"] == 0.0
    cov = sim.degradation.fit_report["cov"]
    at = {n: i for i, n in enumerate(cov["names"])}
    m = np.array(cov["matrix"])
    assert np.all(np.linalg.eigvalsh(m) > -1e-12)
    # base[HARD] = g_eff * base[MEDIUM] + a step: their covariance is g_eff times MEDIUM's variance
    g = 0.25 + 0.5 + 0.25 / 0.99
    assert m[at["base_s[HARD]"], at["base_s[MEDIUM]"]] == pytest.approx(g * m[at["base_s[MEDIUM]"], at["base_s[MEDIUM]"]])


def test_posterior_draws_cover_the_proxied_compound():
    cfg = config(degradation={"name": "linear_fuel", "params": {"draw": "posterior", "wear_lower_bound": 0.0}})
    sim = Simulator(synthetic_frame(fuel=FUEL), cfg, practice())
    plan = [["SOFT", 37], ["HARD", 13]]   # R8: HARD's practice age 10 + 3
    b = sim.simulate_batch(Snapshot.at_start(RACE, "X", "SOFT"), [plan], seed=0, n_samples=200)
    assert b.valid[0], b.errors[0]
    # the HARD stint's spread: 13 laps of a base with sd >= 0.6 s
    assert b.stint_seconds[0][1].std() > 13 * 0.45


def test_model_report_and_agent_view_name_the_sources_and_the_qualifying_gap():
    sim = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9))
    rep = sim.model_report(Snapshot.at_start(RACE, "X", "SOFT"), calibration_summary=None)
    assert "pace_from_qualifying+cross_race_step:HARD" in rep.flags and "wear_from_cross_race_step:HARD" in rep.flags
    assert rep.compounds["HARD"].modelled and rep.compounds["HARD"].n_race_rows == 0
    from src.sim import agent_tools
    out = agent_tools.model_report(Snapshot.at_start(RACE, "X", "SOFT"), sim=sim, n_samples=5)
    entry = out["compounds"]["HARD"]
    assert (entry["pace_source"], entry["wear_source"]) == ("qualifying+cross_race_step", "cross_race_step")
    assert [p["source"] for p in entry["pace_sources"]] == ["qualifying", "cross_race_step"]
    assert entry["pace_sources"][0]["weight"] == pytest.approx(2 / 3)
    assert entry["price_basis"].startswith("HARD: pace from qualifying 67%, the cross-race step 33% (anchor MEDIUM)")
    # the measured qualifying gap on fresh tyres, for a proxied and for a raced compound
    assert entry["qualifying_gap_s"]["anchor"] == "MEDIUM"
    assert entry["qualifying_gap_s"]["gap_s"] == pytest.approx(0.9, abs=1e-9)
    assert "qualifying gap to MEDIUM +0.90 s" in entry["price_basis"]
    soft = out["compounds"]["SOFT"]
    assert soft["pace_source"] == "race" and soft["qualifying_gap_s"]["gap_s"] == pytest.approx(-0.8, abs=1e-9)


# --- belief: no hindsight ------------------------------------------------------------------------
def test_belief_tier_one_uses_only_laps_up_to_the_snapshot():
    belief = Simulator(synthetic_frame(fuel=FUEL), config(fit_window="to_snapshot"), practice())
    early = belief.fitted_for(at_lap(0)).degradation.fit_report["proxy"]
    assert all(e["pace_tier"] == 2 for e in early.values())          # no race laps yet: practice
    later = belief.fitted_for(at_lap(20)).degradation.fit_report["proxy"]
    assert later["SOFT"]["pace_tier"] == 1 and later["HARD"]["pace_tier"] == 2
    # the frame after the snapshot does not move the belief's proxied compound
    frame = synthetic_frame(fuel=FUEL)
    cut = frame[frame["lap"] <= 20]
    a = Simulator(frame, config(fit_window="to_snapshot"), practice()).fitted_for(at_lap(20)).degradation
    b = Simulator(cut, config(fit_window="to_snapshot"), practice()).fitted_for(at_lap(20)).degradation
    assert a.coefs["HARD"] == pytest.approx(b.coefs["HARD"], abs=1e-12)


# --- R8 ------------------------------------------------------------------------------------------
def limit_frame():
    """synthetic_frame without its age-100 rows, so the race's own limits are 20 laps."""
    f = synthetic_frame(fuel=FUEL)
    return f[f["tyre_age"] < 100]


def test_r8_caps_an_unraced_compound_at_its_practice_age():
    prac = practice()
    sim = world(frame=limit_frame(), prac=prac)
    assert sim.stint_limits["HARD"] == prac.loc[prac["compound"] == "HARD", "tyre_age"].max() == 10
    assert sim.stint_limit_sources == {"HARD": "practice"}
    snap = Snapshot.at_start(RACE, "X", "SOFT")
    ok = sim.validate(snap, Plan_of([["SOFT", 18], ["MEDIUM", 19], ["HARD", 13]]))
    assert ok == []
    errs = sim.validate(snap, Plan_of([["SOFT", 18], ["MEDIUM", 18], ["HARD", 14]]))
    assert any("no HARD raced: the oldest HARD tyre run in practice, 10, + 3" in e for e in errs)


def test_r8_without_practice_scales_the_nearest_raced_limit_by_the_life_ratio():
    sim = world(frame=limit_frame())
    assert sim.stint_limits["HARD"] == np.floor(20 / 0.8)
    assert sim.stint_limit_sources == {"HARD": "life_ratio:MEDIUM"}


def Plan_of(pairs):
    from src.contracts import Plan
    return Plan.of(pairs)


# --- off switch ----------------------------------------------------------------------------------
def test_off_switch_restores_the_unproxied_simulator():
    off = world(enabled=False, prac=practice(), q=quali(gap_hard_medium=0.9))
    plain = Simulator(synthetic_frame(fuel=FUEL), SimConfig(pre_race_base_delta_s=0.0,
                                                           compound_proxy={"enabled": False}), practice())
    assert off.modelled == {"SOFT", "MEDIUM"}
    assert "proxy" not in off.degradation.fit_report
    assert off.stint_limit_sources == {} and "HARD" not in off.stint_limits
    errs = off.validate(Snapshot.at_start(RACE, "X", "SOFT"), Plan_of([["SOFT", 20], ["HARD", 30]]))
    assert errs and errs[0].startswith("Unknown compound 'HARD'")
    assert off.degradation.coefs == plain.degradation.coefs


def test_the_committed_step_table_loads_with_both_pairs():
    steps = load_compound_steps()
    assert set(steps) == {"SOFT-MEDIUM", "MEDIUM-HARD"}
    for pair in steps.values():
        assert {"pace_frac", "wear_ratio", "life_ratio"} <= set(pair)
        assert all(e["n"] >= 2 for e in pair.values())
    px = SimConfig().compound_proxy
    assert px.steps == steps
    assert set(px.weights["pace"]) == {"practice", "qualifying", "cross_race_step"}
    assert set(px.weights["wear"]) == {"practice", "cross_race_step"}
    assert all(w > 0 for k in px.weights.values() for w in k.values())
    assert px.sd_floor["pace"] > 0 and px.sd_floor["wear"] > 0


def test_hindsight_targets_leave_a_proxied_compound_out():
    from src.sim.calibrators import RaceInput, hindsight_targets
    frame = race_frame(("AAA", "T", [("SOFT", 20), ("MEDIUM", 30)]), ("BBB", "T", [("SOFT", 10), ("MEDIUM", 40)]))
    targets = hindsight_targets(RaceInput("2021_01_syn", frame, RACE, practice(), None, config()))
    assert "wear_s_per_lap[MEDIUM]" in targets
    assert not any("HARD" in k for k in targets)


def test_eval_rows_can_count_plans_on_a_proxied_compound():
    from src.eval.score import _on_proxy, proxied_compounds
    snap = Snapshot.at_start(RACE, "X", "SOFT")
    on = world(prac=practice(hard=False))
    assert proxied_compounds(on, snap) == {"HARD": {"pace": "cross_race_step", "wear": "cross_race_step"}}
    assert _on_proxy(Plan_of([["SOFT", 37], ["HARD", 13]]), proxied_compounds(on, snap)) == "HARD"
    assert _on_proxy(Plan_of([["SOFT", 25], ["MEDIUM", 25]]), proxied_compounds(on, snap)) == ""
    assert proxied_compounds(world(enabled=False), snap) is None   # no columns with the proxy off


# --- 2022 Saudi Arabia: SOFT run in practice, never raced (a calibration race) --------------------
SAUDI = "2022_02_saudi_arabia"


@pytest.fixture(scope="module")
def saudi():
    from src.sim import jobs
    from src.sim.compute import load_profile
    return jobs.load_race(SAUDI, load_profile())


def test_saudi_world_prices_soft(saudi):
    world_on = Simulator(saudi.frame, saudi.config, saudi.practice, qualifying=saudi.qualifying)
    assert "SOFT" not in Simulator(saudi.frame, saudi.config.model_copy(update={"compound_proxy": CompoundProxy(
        enabled=False)}), saudi.practice).modelled
    assert "SOFT" in world_on.modelled
    e = world_on.degradation.fit_report["proxy"]["SOFT"]
    assert e["pace_tier"] in (2, 3, 5) and e["wear_tier"] in (2, 5)
    assert world_on.stint_limit_sources.get("SOFT") == "practice"
    assert world_on.stint_limits["SOFT"] == saudi.practice.loc[saudi.practice["compound"] == "SOFT", "tyre_age"].max()


def test_saudi_belief_continuation_is_accepted_by_the_world(saudi):
    from src.eval import decisions
    from src.sim import calibration
    from src.sim.engine import Simulator as S
    belief = S(saudi.frame, SimConfig.from_file("configs/sim/belief.yaml"), saudi.practice,
               qualifying=saudi.qualifying)
    world_on = S(saudi.frame, saudi.config, saudi.practice, qualifying=saudi.qualifying)
    drivers = calibration._finisher_ids(saudi.frame, 0.9)[:3]
    for d in drivers:
        for k in (10, 20):
            snap = calibration.snapshot_at(saudi.frame, saudi.race, d, k)
            for sub in ({"decision": "PIT", "compound": "SOFT"}, {"decision": "STAY_OUT", "reassess_after_laps": 3}):
                plan = decisions.to_plan(snap, sub, belief, 2, seed=0, n_samples=5)
                b = world_on.simulate_batch(snap, [plan], seed=0, n_samples=5)
                assert b.valid[0], b.errors[0]
                assert not any("Unknown compound" in e for e in b.errors[0])


def test_weights_are_one_over_the_mse_and_leave_a_race_out():
    from src.sim.compound_steps import weights_from
    rows = pd.DataFrame([
        {"race_id": "r1", "case": "practice", "available": True, "pace_err_s": 1.0, "wear_err_s_per_lap": 0.1},
        {"race_id": "r2", "case": "practice", "available": True, "pace_err_s": 3.0, "wear_err_s_per_lap": 0.1},
        {"race_id": "r1", "case": "qualifying", "available": True, "pace_err_s": 0.5, "wear_err_s_per_lap": np.nan},
        {"race_id": "r1", "case": "cross_race_step", "available": True, "pace_err_s": 2.0, "wear_err_s_per_lap": 0.2},
        {"race_id": "r2", "case": "cross_race_step", "available": False, "pace_err_s": 9.0, "wear_err_s_per_lap": 9.0}])
    w, mse = weights_from(rows)
    assert w["pace"] == pytest.approx({"practice": 1 / 5.0, "qualifying": 1 / 0.25, "cross_race_step": 1 / 4.0})
    assert w["wear"] == pytest.approx({"practice": 1 / 0.01, "cross_race_step": 1 / 0.04})
    assert mse["pace"]["practice"]["n"] == 2
    w1, _ = weights_from(rows, exclude=["r2"])
    assert w1["pace"]["practice"] == pytest.approx(1.0)
