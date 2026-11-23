"""The scorecard-chosen configuration is the default everywhere the evaluation reads one
(docs/eval.md "Simulator fidelity", "the current configuration"): the judge (score.race_simulators'
default world), the belief (configs/sim/belief.yaml, the agent's tools, the rerun pipeline, the
fidelity scorecard's base), and each earlier choice stays available as a named file or ablation
factor. Config-hash pins: a change to a default changes these on purpose. No race cache."""
import numpy as np
import pytest

from src.contracts import Snapshot
from src.eval import rerun, score
from src.sim import SimConfig, agent_tools, calibration, fidelity
from src.sim.config import (WORLD_DEGRADATION, WORLD_EDGE, WORLD_SC_QUEUE, WORLD_TYRE_INVENTORY, WORLD_WEAR_GUARD,
                             ScQueue, WearGuard, world_default)
from src.sim.engine import Simulator

ROOT = fidelity.ROOT
DEFAULT_WORLD_HASH = "9b98afd55974"       # driver effects, edge as_of, hybrid SC/VSC pooled history, wear guard, SC queue, tyre sets
PRE_ASOF_EDGE_WORLD_HASH = "c821efe0bc98"   # the default world from the tyre sets (#143) to the as-of edge
DEFAULT_BELIEF_HASH = "aaf95a09bc50"      # belief.yaml with driver effects, SC/VSC history lengths, cliff edge point, tyre sets
PRE_TYRE_INVENTORY_WORLD_HASH = "7c974c32c67c"   # the default world from the SC queue (#142) to the tyre sets
PRE_TYRE_INVENTORY_BELIEF_HASH = "9d691bbbcf24"  # the default belief from v3 (#138) to the tyre sets
PRE_SC_QUEUE_WORLD_HASH = "15e3f31d98a7"  # the default world from v3 (#138) to the SC queue's adoption
PRE_V3_WORLD_HASH = "5069228b35e7"        # the default world from the SC alignment (#125) to v3
PRE_V3_BELIEF_HASH = "351f4124ff22"
PRE_SC_ALIGNMENT_WORLD_HASH = "abc455a9bbb6"   # the default world from the adoption (#114) to the SC alignment
PRE_SC_ALIGNMENT_BELIEF_HASH = "f7b872d14c0c"
PRE_SCORECARD_WORLD_HASH = "d230d3a56b0b"   # the default world from PR #108 until the adoption
PRE_SCORECARD_BELIEF_HASH = "d479652b0fe7"


def default_world():
    return score.default_world_config(world_default(SimConfig()))


def belief():
    return SimConfig.from_file(score.BELIEF_CONFIG)


def test_the_default_world_is_the_scorecard_choice():
    w = default_world()
    assert score.WORLD_EVENTS_DEFAULT is True and WORLD_EDGE == "as_of"
    assert (w.events.name, w.events.params) == ("hybrid", {"lap_times": "recorded", "duration": "history", "pooled": True})
    assert w.wear_cliff.enabled and w.wear_cliff.world_edge == "as_of"
    assert WORLD_WEAR_GUARD == {"enabled": True, "scale_sd_by_drivers": True}
    assert w.wear_guard == WearGuard(enabled=True, scale_sd_by_drivers=True)
    assert (w.wear_guard.min_drivers, w.wear_guard.min_rows) == (3, 40)
    assert not SimConfig().wear_guard.enabled   # the SimConfig default stays off; the eval world sets it
    assert WORLD_SC_QUEUE == {"enabled": True, "mode": "single", "held_behind_laps": 1.25, "placement": "record"}
    assert w.sc_queue == ScQueue(enabled=True, mode="single", held_behind_laps=1.25, placement="record")
    assert w.sc_queue.spacing_s == 0.5
    assert not SimConfig().sc_queue.enabled     # the SimConfig default stays off; the eval world sets it
    assert WORLD_TYRE_INVENTORY is True and w.tyre_inventory
    assert not SimConfig().tyre_inventory       # the SimConfig default stays off; the eval world sets it
    assert w.degradation.model_dump() == {"name": "linear_fuel", "params": {"driver_effects": "race"}} \
        == {"name": WORLD_DEGRADATION["name"], "params": WORLD_DEGRADATION["params"]}
    assert w.traffic.name == "none" and w.stint_limit_margin_laps is None and w.plan_on == "draws"
    assert calibration.config_hash(w) == DEFAULT_WORLD_HASH
    # the default world as a file, and the switch that turns the events off
    assert SimConfig.from_file(ROOT / "configs/sim/world_hybrid.yaml") == w
    assert score.default_world_config(world_default(SimConfig()), on=False).events.name == "none"


def test_the_default_belief_is_the_scorecard_choice_and_the_tools_read_it():
    b = belief()
    p = b.degradation.params
    assert b.fit_window == "to_snapshot" and b.degradation.name == "linear_fuel"   # no hinge
    assert p["driver_effects"] == "race" and p["wear_lower_bound"] == 0.0 and "sd_scale" not in p
    assert b.lap_noise == "persistent" and b.plan_on == "draws" and b.traffic.name == "none"
    assert (b.fuel_anchor.prior, b.fuel_anchor.practice) == ("anchored", "anchored")
    assert b.calibrator.name == "as_of_history" and b.wear_cliff.enabled
    assert (b.events.name, b.events.params) == ("rate", {"as_of": True, "duration": "history"})
    assert b.belief_cliff_edge == "point" and SimConfig().belief_cliff_edge == "draw"
    assert b.tyre_inventory and agent_tools.belief_config().tyre_inventory
    assert agent_tools.belief_config().belief_cliff_edge == "point"
    assert calibration.config_hash(b) == DEFAULT_BELIEF_HASH
    assert agent_tools.BELIEF_CONFIG == score.BELIEF_CONFIG == fidelity.BELIEF_FILE
    assert rerun.belief_path("default") == str(score.BELIEF_CONFIG)


def test_the_belief_mirrors_are_the_base_plus_one_block():
    b = belief().model_dump()
    for name, block in (("belief_traffic", "traffic"), ("belief_learned", "calibrator")):
        other = SimConfig.from_file(ROOT / f"configs/sim/{name}.yaml").model_dump()
        assert {k for k in b if b[k] != other[k]} == {block}, name


def test_every_earlier_choice_stays_available_as_a_file():
    old_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_scorecard.yaml")
    old_b = SimConfig.from_file(ROOT / "configs/sim/legacy/belief_pre_scorecard.yaml")
    assert calibration.config_hash(old_w) == PRE_SCORECARD_WORLD_HASH
    assert calibration.config_hash(old_b) == PRE_SCORECARD_BELIEF_HASH
    assert (old_w.events.name, old_w.wear_cliff.world_edge) == ("none", "history")
    # the earlier belief is the one before the SC alignment without its driver effects, nothing else
    b = belief()
    pre = SimConfig.from_file(ROOT / "configs/sim/legacy/belief_pre_sc_alignment.yaml")
    assert "driver_effects" not in old_b.degradation.params
    assert old_b.degradation.params == {k: v for k, v in b.degradation.params.items() if k != "driver_effects"}
    assert {k for k, v in pre.model_dump().items() if v != old_b.model_dump()[k]} == {"degradation"}
    # before the SC alignment: the same configs with geometric lengths and the provisional rates
    pre_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_sc_alignment.yaml")
    assert calibration.config_hash(pre_w) == PRE_SC_ALIGNMENT_WORLD_HASH
    assert calibration.config_hash(pre) == PRE_SC_ALIGNMENT_BELIEF_HASH
    assert pre_w.events.params == {"lap_times": "recorded"} and pre.events.params == {"as_of": True}
    # before v3: the same configs with the guard off (world) and the cliff edge drawn (belief)
    v3_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_v3.yaml")
    v3_b = SimConfig.from_file(ROOT / "configs/sim/legacy/belief_pre_v3.yaml")
    assert calibration.config_hash(v3_w) == PRE_V3_WORLD_HASH
    assert calibration.config_hash(v3_b) == PRE_V3_BELIEF_HASH
    assert not v3_w.wear_guard.enabled and v3_b.belief_cliff_edge == "draw"
    # before the SC queue: the default world without it (the belief unchanged)
    sq_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_sc_queue.yaml")
    assert calibration.config_hash(sq_w) == PRE_SC_QUEUE_WORLD_HASH and not sq_w.sc_queue.enabled
    ti_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_tyre_inventory.yaml")
    ti_b = SimConfig.from_file(ROOT / "configs/sim/legacy/belief_pre_tyre_inventory.yaml")
    assert calibration.config_hash(ti_w) == PRE_TYRE_INVENTORY_WORLD_HASH and not ti_w.tyre_inventory
    assert calibration.config_hash(ti_b) == PRE_TYRE_INVENTORY_BELIEF_HASH and not ti_b.tyre_inventory
    # before the as-of edge: the default world with the prior-only edge (the belief unchanged)
    ae_w = SimConfig.from_file(ROOT / "configs/sim/legacy/world_pre_asof_edge.yaml")
    assert calibration.config_hash(ae_w) == PRE_ASOF_EDGE_WORLD_HASH and ae_w.wear_cliff.world_edge == "history_only"
    assert ae_w.model_copy(update={"wear_cliff": default_world().wear_cliff}) == default_world()
    assert ti_w.model_copy(update={"tyre_inventory": True}) == ae_w
    assert ti_b.model_copy(update={"tyre_inventory": True}) == b
    assert sq_w.model_copy(update={"sc_queue": ti_w.sc_queue}) == ti_w
    assert v3_w.model_copy(update={"wear_guard": sq_w.wear_guard}) == sq_w
    assert v3_b.model_copy(update={"belief_cliff_edge": "point"}) == ti_b
    assert pre_w.model_copy(update={"events": v3_w.events}) == v3_w
    assert pre.model_copy(update={"events": v3_b.events}) == v3_b
    # the traffic world is the default world plus traffic only (test_traffic_eval_params too)
    assert SimConfig.from_file(score.WORLD_TRAFFIC_CONFIG).model_copy(
        update={"traffic": default_world().traffic}) == default_world()


def test_the_ablation_toggles_each_adopted_choice_back_off():
    ab = fidelity.load_ablation()
    skipped = []
    specs = {s.label: s for s in fidelity.ablation_specs(fidelity.make_spec(), ab, skipped=skipped)}
    assert not skipped
    base_w, base_b = specs["base"].world_config(), specs["base"].belief_config()
    assert calibration.config_hash(base_w) == DEFAULT_WORLD_HASH
    assert calibration.config_hash(base_b) == DEFAULT_BELIEF_HASH
    w = {n: specs[n].world_config() for n in specs}
    assert w["world_events_off"].events.name == "none"
    assert w["cliff_edge_history"].wear_cliff.world_edge == "history"
    assert w["world_linear"].degradation.params["driver_effects"] == "off"
    assert w["world_hinge"].degradation.name == "linear_fuel_hinge"
    assert w["world_hinge"].degradation.params["driver_effects"] == "race"
    assert specs["belief_no_driver_effects"].belief_config().degradation.params["driver_effects"] == "off"
    off = specs["sc_alignment_off"]
    assert off.world_config().events.params["duration"] == "geometric" and not off.world_config().events.params["pooled"]
    assert off.belief_config().events.params["duration"] == "geometric"
    assert not w["wear_guard_off"].wear_guard.enabled
    assert w["wear_guard_off"].model_copy(update={"wear_guard": base_w.wear_guard}) == base_w
    assert w["wear_guard_unscaled"].wear_guard == WearGuard(enabled=True)
    assert specs["belief_cliff_edge_draw"].belief_config().belief_cliff_edge == "draw"
    assert specs["belief_cliff_edge_draw"].belief_config().model_copy(update={"belief_cliff_edge": "point"}) == base_b
    assert w["sc_queue_off"].sc_queue == ScQueue()
    prior_only = w["cliff_edge_history_only"].wear_cliff
    assert calibration.config_hash(w["sc_queue_off"].model_copy(
        update={"tyre_inventory": False, "wear_cliff": prior_only})) == PRE_SC_QUEUE_WORLD_HASH
    assert w["sc_queue_off"].model_copy(update={"sc_queue": base_w.sc_queue}) == base_w
    assert w["sc_queue_clock"].sc_queue.placement == "clock" and w["sc_queue_clock"].sc_queue.held_behind_laps == 1.25
    assert w["sc_queue_old"].sc_queue.held_behind_laps is None and w["sc_queue_old"].sc_queue.enabled
    assert w["sc_queue_field"].sc_queue.mode == "field"
    assert not base_b.sc_queue.enabled and specs["sc_queue_both"].belief_config().sc_queue.enabled
    assert specs["sc_queue_both"].belief_config().sc_queue.placement == "clock"
    ti = specs["tyre_inventory_off"]
    assert calibration.config_hash(ti.world_config().model_copy(update={"wear_cliff": prior_only})) == \
        PRE_TYRE_INVENTORY_WORLD_HASH
    # the as-of edge switched back to the prior alone is the world before it
    assert prior_only.world_edge == "history_only"
    assert calibration.config_hash(w["cliff_edge_history_only"]) == PRE_ASOF_EDGE_WORLD_HASH
    assert calibration.config_hash(ti.belief_config()) == PRE_TYRE_INVENTORY_BELIEF_HASH
    tb = specs["traffic_both"]
    assert tb.world_config().traffic.name == "ghost" and tb.belief_config().traffic.params["source"] == "projected"
    # traffic aside, traffic_both keeps every adopted choice
    assert tb.world_config().model_copy(update={"traffic": base_w.traffic}) == base_w


def test_provenance_names_each_choice():
    w = rerun.config_choices(rerun.default_world(), "world")
    assert calibration.config_hash(rerun.default_world()) == DEFAULT_WORLD_HASH
    assert (w["events"], w["tyre_life_edge"], w["driver_effects"], w["hinge"], w["traffic"]) == \
        ("hybrid", "as_of", "race", False, "none")
    assert "up to the scored snapshot" in w["tyre_life_edge_basis"]
    assert w["wear_guard"] == {"enabled": True, "min_drivers": 3, "min_rows": 40, "scale_sd_by_drivers": True}
    assert w["sc_queue"] == {"enabled": True, "mode": "single", "spacing_s": 0.5, "held_behind_laps": 1.25,
                             "placement": "record"}
    b = rerun.config_choices(belief(), "belief")
    assert (b["driver_effects"], b["lap_noise"], b["traffic"], b["calibrator"], b["sd_scale"], b["plan_on"]) == \
        ("race", "persistent", "none", "as_of_history", None, "draws")
    assert b["belief_cliff_edge"] == "point" and "wear_guard" not in b and "sc_queue" not in b
    assert b["fuel_anchor"]["prior"] == "anchored" and b["tyre_life_edge"].startswith("as_of_snapshot")
    assert w["tyre_inventory"] is True and b["tyre_inventory"] is True


def test_belief_config_as_runs_the_tools_on_the_given_belief_and_restores():
    before = agent_tools.belief_config()
    other = before.model_copy(update={"lap_noise": "residual"})
    with agent_tools.belief_config_as(other):
        assert agent_tools.belief_config() is other
    assert agent_tools.belief_config() is before


def test_the_hybrid_world_draws_the_same_neutralisations_for_every_plan():
    """Common random numbers: a plan's sample k sees the same drawn SC/VSC whether it is scored
    alone (a policy's plan) or among other plans (the reference's optimizer batch)."""
    from test_sim_neutralised_world import sc_frame, snap_at
    df = sc_frame().assign(compound="SOFT", tyre_age=lambda f: f["lap"], stint=1, clean=True, team="T",
                           position=1.0)
    cfg = SimConfig(events={"name": "hybrid", "params": {"p_sc": 0.4, "d_sc": 2.0, "p_vsc": 0.3, "d_vsc": 1.0}},
                    rules=["lap_sum"], compound_proxy={"enabled": False}, min_samples=3)
    world = Simulator(df, cfg, pit_loss_s=20.0)
    snap: Snapshot = snap_at(3).model_copy(update={"tyre_age": 3})
    a, b = [["SOFT", 6], ["SOFT", 11]], [["SOFT", 10], ["SOFT", 7]]
    alone = world.simulate_batch(snap, [a], seed=7, n_samples=40)
    among = world.simulate_batch(snap, [b, a], seed=7, n_samples=40)
    assert np.array_equal(alone.times[0], among.times[1])
    assert np.ptp(alone.times[0]) > 0   # the draws differ between samples: events are drawn
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(alone.times[0], world.simulate_batch(snap, [a], seed=8, n_samples=40).times[0])


def test_the_belief_has_no_safety_car_queue():
    """The judge's queue (config.WORLD_SC_QUEUE) is placed by the record: hindsight. No belief config
    the evaluation, the agent's tools or the scorecard read picks it up."""
    import glob
    for path in sorted(glob.glob(str(ROOT / "configs/sim/belief*.yaml"))):
        assert not SimConfig.from_file(path).sc_queue.enabled, path
    assert not agent_tools.belief_config().sc_queue.enabled
    assert not fidelity.make_spec().belief_config().sc_queue.enabled
    assert not SimConfig.from_file(score.BELIEF_CONFIG).sc_queue.enabled
    # the judge's worlds carry it
    for name in ("world_hybrid", "world", "world_difficulty"):
        assert SimConfig.from_file(ROOT / f"configs/sim/{name}.yaml").sc_queue == default_world().sc_queue, name
