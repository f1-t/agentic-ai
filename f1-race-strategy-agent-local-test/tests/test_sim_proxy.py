"""The compound proxy (src/sim/proxy.py): tier selection, anchoring, the ordering constraint,
uncertainty, R8 for an unraced compound and the off switch, on synthetic weekends; then the
2022 Saudi Arabia case (SOFT run in practice, never raced) on the calibration race."""
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


def config(enabled=True, steps=STEPS, **kw):
    return SimConfig(compound_proxy=CompoundProxy(enabled=enabled, steps=steps), pre_race_base_delta_s=0.0, **kw)


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


def world(enabled=True, steps=STEPS, prac=None, q=None, frame=None, **kw):
    return Simulator(synthetic_frame(fuel=FUEL) if frame is None else frame, config(enabled, steps, **kw), prac,
                     qualifying=q)


def tiers(sim, c):
    e = sim.degradation.fit_report["proxy"][c]
    return e["pace_tier"], e["wear_tier"]


# --- tier selection ------------------------------------------------------------------------------
def test_raced_compounds_are_tier_one_and_unchanged():
    on, off = world(prac=practice()), world(enabled=False, prac=practice())
    for c in ("SOFT", "MEDIUM"):
        assert tiers(on, c) == (1, 1)
        assert on.degradation.coefs[c] == off.degradation.coefs[c]
    assert on.degradation.fuel_slope == off.degradation.fuel_slope
    assert "HARD" not in off.modelled and "HARD" in on.modelled


def test_tier_selection_practice_then_qualifying_then_step():
    assert tiers(world(prac=practice()), "HARD") == (2, 2)
    # no HARD long run, HARD in qualifying: pace from qualifying, wear from the step
    assert tiers(world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9)), "HARD") == (3, 5)
    # neither: the step for both
    assert tiers(world(prac=practice(hard=False), q=quali()), "HARD") == (5, 5)
    assert tiers(world(), "HARD") == (5, 5)
    # nothing to price it with: HARD stays unknown
    assert "HARD" not in world(steps={}).modelled


def test_tiers_can_mix_and_be_restricted():
    sim = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9))
    e = sim.degradation.fit_report["proxy"]["HARD"]
    assert (e["pace_source"], e["wear_source"]) == ("qualifying", "cross_race_step")
    assert proxy.describe("HARD", e) == ("HARD: pace from qualifying (anchor MEDIUM), "
                                         "wear from the cross-race step (anchor MEDIUM)")


# --- anchoring -----------------------------------------------------------------------------------
def test_tier_two_is_the_practice_step_from_the_raced_anchor():
    sim = world(prac=practice())
    base_m, wear_m = sim.degradation.coefs["MEDIUM"]
    pp = sim.physics_prior().coefs
    base_h, wear_h = sim.degradation.coefs["HARD"]
    assert base_h == pytest.approx(base_m + pp["HARD"][0] - pp["MEDIUM"][0])
    assert wear_h == pytest.approx(wear_m + pp["HARD"][1] - pp["MEDIUM"][1])


def test_the_race_sets_the_level_of_a_borrowed_compound():
    frame = synthetic_frame(fuel=FUEL)
    slow = frame.assign(lap_time_s=frame["lap_time_s"] + 2.0)
    for kw in ({"prac": practice()}, {"q": quali(gap_hard_medium=0.9)}, {}):
        a = world(frame=frame, **kw).degradation.coefs["HARD"][0]
        b = world(frame=slow, **kw).degradation.coefs["HARD"][0]
        # tier 5 scales with the anchor (a share of the lap); 2 and 3 add a step in seconds
        expect = 2.0 * (1 / 0.99) if not kw else 2.0
        assert b - a == pytest.approx(expect, abs=1e-6)


def test_tier_three_is_the_qualifying_gap_from_the_anchor():
    sim = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9))
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
def test_proxied_sd_is_wider_down_the_tiers_and_moves_with_the_anchor():
    t2, t5 = world(prac=practice()), world()
    sd2 = t2.degradation.fit_report["sd"]["base_s"]["HARD"]
    sd5 = t5.degradation.fit_report["sd"]["base_s"]["HARD"]
    assert sd2 >= 0.5 and sd5 >= 1.0 and sd5 > sd2
    assert t5.degradation.fit_report["shrinkage_weight"]["base_s"]["HARD"] == 0.0
    cov = t2.degradation.fit_report["cov"]
    at = {n: i for i, n in enumerate(cov["names"])}
    m = np.array(cov["matrix"])
    assert np.all(np.linalg.eigvalsh(m) > -1e-12)
    # base[HARD] = base[MEDIUM] + a step: their covariance is MEDIUM's variance
    assert m[at["base_s[HARD]"], at["base_s[MEDIUM]"]] == pytest.approx(m[at["base_s[MEDIUM]"], at["base_s[MEDIUM]"]])


def test_posterior_draws_cover_the_proxied_compound():
    cfg = config(degradation={"name": "linear_fuel", "params": {"draw": "posterior", "wear_lower_bound": 0.0}})
    sim = Simulator(synthetic_frame(fuel=FUEL), cfg, practice())
    plan = [["SOFT", 37], ["HARD", 13]]   # R8: HARD's practice age 10 + 3
    b = sim.simulate_batch(Snapshot.at_start(RACE, "X", "SOFT"), [plan], seed=0, n_samples=200)
    assert b.valid[0], b.errors[0]
    # the HARD stint's spread: 13 laps of a base with sd >= 0.5 s
    assert b.stint_seconds[0][1].std() > 13 * 0.4


def test_model_report_and_agent_view_name_the_tiers():
    sim = world(prac=practice(hard=False), q=quali(gap_hard_medium=0.9))
    rep = sim.model_report(Snapshot.at_start(RACE, "X", "SOFT"), calibration_summary=None)
    assert "pace_from_qualifying:HARD" in rep.flags and "wear_from_cross_race_step:HARD" in rep.flags
    assert rep.compounds["HARD"].modelled and rep.compounds["HARD"].n_race_rows == 0
    from src.sim import agent_tools
    out = agent_tools.model_report(Snapshot.at_start(RACE, "X", "SOFT"), sim=sim, n_samples=5)
    entry = out["compounds"]["HARD"]
    assert (entry["pace_source"], entry["wear_source"]) == ("qualifying", "cross_race_step")
    assert entry["price_basis"].startswith("HARD: pace from qualifying")
    assert out["compounds"]["SOFT"]["pace_source"] == "race"


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
    assert SimConfig().compound_proxy.steps == steps


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
