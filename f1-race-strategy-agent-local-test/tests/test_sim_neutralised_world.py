"""Safety cars and VSC in the scoring world (events "recorded" and "hybrid") and the stratified
scenario sampler (docs/eval.md "Pit loss and track status", "Decision-critical sets")."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim.config import SimConfig
from src.sim.engine import Simulator
from src.sim.events import HybridEvents, RateEvents, RecordedEvents, Scenario
from test_sim import RACE


def sc_frame():
    """Three cars, 20 laps, an SC on laps 8-10; BBB pits under it (in lap 9)."""
    rows = []
    for d, off in (("AAA", 0.0), ("BBB", 0.3), ("CCC", 0.6)):
        t = 0.0
        for lap in range(1, 21):
            status = "4" if lap in (8, 9, 10) else "1"
            lt = 120.0 + off if status == "4" else 90.0 + off
            rows.append({"driver": d, "lap": lap, "lap_time_s": lt, "track_status": status,
                         "pit_in": d == "BBB" and lap == 9, "pit_out": d == "BBB" and lap == 10,
                         "start_s": t, "end_s": t + lt})
            t += lt
    return pd.DataFrame(rows)


def snap_at(k, status="green"):
    return Snapshot.at_start(RACE.model_copy(update={"total_laps": 20}), "AAA", "SOFT").model_copy(
        update={"laps_completed": k, "track_status": status})


def test_recorded_events_draw_the_drivers_own_laps_and_the_field_lap_times():
    ev = RecordedEvents().bind(sc_frame())
    out = ev.draw(snap_at(8, "sc"), None)
    assert out.sc_laps == [9, 10] and out.vsc_laps == []
    assert out.lap_time_s == {9: pytest.approx(120.3), 10: pytest.approx(120.3)}   # BBB's pit laps left out
    assert RecordedEvents(lap_times="factor").bind(sc_frame()).draw(snap_at(8, "sc"), None).lap_time_s == {}


def test_hybrid_ends_the_period_in_progress_as_recorded_and_draws_the_rest():
    params = {"p_sc": 0.3, "d_sc": 2.0, "p_vsc": 0.0}
    ev = HybridEvents(**params).bind(sc_frame())
    snap = snap_at(8, "sc")
    for seed in range(20):
        out = ev.draw(snap, np.random.default_rng(seed))
        assert out.sc_laps[:2] == [9, 10] and out.lap_time_s == {9: pytest.approx(120.3), 10: pytest.approx(120.3)}
        assert 11 not in out.sc_laps or 12 in out.sc_laps or out.sc_laps[2] == 11   # drawn from lap 11 on
    later = [tuple(ev.draw(snap, np.random.default_rng(s)).sc_laps[2:]) for s in range(20)]
    assert len(set(later)) > 1                       # later periods are drawn, not the record
    green = snap_at(3)                               # a green snapshot draws exactly as "rate"
    for seed in range(5):
        a = ev.draw(green, np.random.default_rng(seed))
        b = RateEvents(**params).draw(green, np.random.default_rng(seed))
        assert (a.sc_laps, a.vsc_laps, a.lap_time_s) == (b.sc_laps, b.vsc_laps, {})
    assert ev.in_progress(snap_at(10, "sc")) == []   # the SC ended on lap 10


def test_engine_uses_the_recorded_sc_lap_time():
    from test_sim import synthetic_frame
    sim = Simulator(synthetic_frame())
    snap = Snapshot.at_start(RACE, "AAA", "SOFT").model_copy(update={"laps_completed": 10})
    plan = [["SOFT", 10], ["MEDIUM", 30]]
    base = sim.simulate(snap, plan, scenario=Scenario(sc_laps=[12]), return_laps=True)
    rec = sim.simulate(snap, plan, scenario=Scenario(sc_laps=[12], lap_time_s={12: 150.0}), return_laps=True)
    assert rec.lap_times_s[1] == 150.0 and base.lap_times_s[1] != 150.0
    assert rec.lap_times_s[0] == base.lap_times_s[0] and rec.lap_times_s[2:] == base.lap_times_s[2:]


def test_world_prices_a_stop_under_the_recorded_sc_at_the_sc_factor():
    """A stop at the end of the next lap, inside the SC in progress, pays sc_pit_loss_factor."""
    df = sc_frame().assign(compound="SOFT", tyre_age=lambda f: f["lap"], stint=1, clean=True, team="T",
                           position=1.0)
    cfg = SimConfig(events={"name": "hybrid", "params": {"p_sc": 0.0, "p_vsc": 0.0}}, rules=["lap_sum"],
                    compound_proxy={"enabled": False}, min_samples=3)
    world = Simulator(df, cfg, pit_loss_s=20.0)
    snap = snap_at(8, "sc").model_copy(update={"tyre_age": 8})
    res = world.simulate(snap, [["SOFT", 1], ["SOFT", 11]], rules=["lap_sum"])
    assert res.pit_loss_total_s == pytest.approx(20.0 * cfg.effects.sc_pit_loss_factor)
    green = world.simulate(snap_at(12).model_copy(update={"tyre_age": 12}), [["SOFT", 1], ["SOFT", 7]],
                           rules=["lap_sum"])
    assert green.pit_loss_total_s == pytest.approx(20.0)


# --- the stratified sampler -----------------------------------------------------------------------
def test_record_strata_and_pool_read_only_laps_up_to_the_sampled_one():
    from src.eval.scenarios import record_stratum, stratified_pool
    race = RACE.model_copy(update={"total_laps": 20, "min_stint_laps": 3})
    df = sc_frame()
    assert record_stratum(df, "AAA", 8) == "neutral" and record_stratum(df, "AAA", 12) == "neutral"
    assert record_stratum(df, "CCC", 13) is None and record_stratum(df, "AAA", 7) is None
    assert record_stratum(df.assign(track_status="1"), "AAA", 10) == "window"   # BBB pitted on lap 9
    pool = stratified_pool(df, race)
    assert pool[:3] == [("AAA", 0), ("BBB", 0), ("CCC", 0)] and max(k for _, k in pool) == 16
    for k in range(0, 18):
        cut = df[df["lap"] <= max(k, 1)]
        for d in ("AAA", "BBB", "CCC"):
            assert record_stratum(cut, d, k) == record_stratum(df, d, k), (d, k)
        assert [p for p in stratified_pool(cut, race) if p[1] <= k] == [p for p in stratified_pool(df, race) if p[1] <= k]


@pytest.fixture(scope="module")
def stratified():
    from src.eval.scenarios import sample_stratified
    from test_sim import synthetic_frame
    from test_sim_position import race_frame
    df = race_frame(("AAA", "T1", [("SOFT", 15), ("MEDIUM", 35)]), ("BBB", "T2", [("SOFT", 20), ("MEDIUM", 30)]),
                    ("CCC", "T3", [("SOFT", 25), ("MEDIUM", 25)]))
    world = Simulator(synthetic_frame(), SimConfig(), pit_loss_s=20.0)
    quotas = {"pit_opt": 2, "close": 2, "neutral": 1, "window": 3, "rest": 3}
    run = lambda seed: sample_stratified(df, RACE, world, quotas=quotas, seed=seed, n_samples=1, every=4)
    return df, run


def test_stratified_sample_is_seeded_weighted_and_never_shown_to_a_policy(stratified):
    from src.eval.scenarios import STRATA
    from src.eval.score import policy_scenario
    df, run = stratified
    items, _ = run(0)
    again, _ = run(0)
    assert [x.scenario_id for x in items] == [x.scenario_id for x in again]
    assert {x.stratum for x in items} <= set(STRATA) and all(x.weight >= 1 for x in items)
    # the weights add up to the pool, stratum by stratum
    from src.eval.scenarios import stratified_pool
    pool = stratified_pool(df, RACE.model_copy(update={"total_laps": 50}), 4)
    assert sum(x.weight for x in items) == pytest.approx(len(pool))
    seen = policy_scenario(items[0], object())
    assert seen.stratum is None and seen.weight is None
    assert "stratum" not in items[0].snapshot.model_dump_json() and "weight" not in items[0].snapshot.model_dump_json()


def test_scenario_digest_ignores_the_unset_stratum():
    from src.eval.scenarios import ScenarioSet
    s = Snapshot.at_start(RACE, "AAA", "SOFT")
    from src.eval.scenarios import Scenario as Item
    item = Item(scenario_id="x", race_id="syn", driver="AAA", lap=0, snapshot=s, actual_stints=[])
    a = ScenarioSet(version="t", items=[item]).compute_digest()
    raw = item.model_dump(mode="json")
    del raw["stratum"], raw["weight"]
    import hashlib
    import json
    from src.eval.scenarios import _digest_view
    assert a == hashlib.sha256(json.dumps([_digest_view(raw)], sort_keys=True).encode()).hexdigest()
