"""SimConfig.belief_cliff_edge (docs/eval.md "Belief priors"): the belief's wear-cliff edge drawn
per sample ("draw", the default) or at its point, max(seen so far, prior) ("point")."""
import numpy as np

from src.contracts import Plan
from src.sim import Simulator, agent_tools, calibration
from src.sim.calibration import config_hash
from src.sim.config import SimConfig
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_sim_agent_tools import syn_frame


def test_the_switch_keeps_the_hash_at_its_default():
    base = SimConfig()
    assert config_hash(base) == config_hash(base.model_copy(update={"belief_cliff_edge": "draw"}))
    assert config_hash(base) != config_hash(base.model_copy(update={"belief_cliff_edge": "point"}))


def _long_stint_times(edge):
    df = syn_frame()
    cfg = agent_tools.belief_config().model_copy(update={"belief_cliff_edge": edge})
    sim = Simulator(df, cfg, synthetic_practice())
    snap = calibration.snapshot_at(df, SYN_RACE, "CCC", 5)
    cl = sim.cliffs_for(snap)
    c = next(c for c, x in sorted(cl.items()) if x.prior is not None and x.prior_sd > 0 and c != snap.compound
             and c in sim.fitted_for(snap).modelled)
    plan = Plan.of([[snap.compound, 1, snap.tyre_age], [c, snap.remaining_laps - 1, 0]])
    return sim, snap, plan


def test_point_edges_take_no_draw_and_never_cost_more_on_a_long_stint():
    sim_d, snap, plan = _long_stint_times("draw")
    sim_p, _, _ = _long_stint_times("point")
    td = sim_d.simulate_batch(snap, [plan], seed=1, n_samples=40, rules=["lap_sum", "known_compounds"]).times[0]
    tp = sim_p.simulate_batch(snap, [plan], seed=1, n_samples=40, rules=["lap_sum", "known_compounds"]).times[0]
    assert np.isfinite(td).all() and np.isfinite(tp).all()
    # the cost past the edge is convex in the edge: averaged over drawn edges it is at least the
    # cost at the point edge (same seeds otherwise)
    assert tp.mean() <= td.mean() + 1e-9
