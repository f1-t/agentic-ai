"""Neutralised-period lengths from history (events.RateEvents duration "history")."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams, Snapshot
from src.sim import race_history
from src.sim.events import HybridEvents, RateEvents

RACE = RaceParams(race_id="2022_12_france", total_laps=53, pit_loss_s=20.0)


def _snap(status="green", k=20, sc=(), vsc=()):
    s = Snapshot.at_start(RACE, "AAA", "MEDIUM")
    return s.model_copy(update={"laps_completed": k, "track_status": status, "sc_laps": list(sc), "vsc_laps": list(vsc),
                                "standing_start": False})


def test_period_lengths_as_of_reads_earlier_races_only():
    table = race_history.load_periods()
    later = table[table["race_id"].map(lambda r: r > "2022_12_france")]
    asof = race_history.period_lengths("2022_12_france", "sc", "as_of")
    assert not set(asof["races"]) & set(later["race_id"])
    assert "2022_21_brazil" in race_history.period_lengths("2022_12_france", "sc", "pooled")["races"]
    assert asof["pmf"].sum() == pytest.approx(1.0)


def test_period_lengths_prior_with_no_history():
    empty = pd.DataFrame(columns=["race_id", "kind", "start_lap", "end_lap", "total_laps", "wet"])
    r = race_history.period_lengths("2019_01_australia", "sc", "as_of", periods=empty)
    assert r["n"] == 0
    assert np.allclose(r["pmf"], race_history._prior_pmf(race_history.EVENT_PRIOR.d_sc))


def test_elapsed_counts_the_run_ending_at_the_snapshot():
    assert RateEvents.elapsed(_snap("sc", 20, sc=[5, 18, 19, 20])) == 3
    assert RateEvents.elapsed(_snap("vsc", 20, vsc=[20])) == 1


def test_history_duration_shortens_with_laps_elapsed():
    ev = RateEvents(as_of=True, duration="history")
    def mean_left(e):
        s = _snap("sc", 20, sc=list(range(21 - e, 21)))
        out = []
        for i in range(400):
            sc = ev.draw(s, np.random.default_rng(i)).sc_laps
            run = 0
            while 21 + run in sc:
                run += 1
            out.append(run)
        return np.mean(out)
    assert mean_left(1) > mean_left(4) > mean_left(7) - 1e-9
    geo = RateEvents(as_of=True)
    s = _snap("sc", 20, sc=list(range(14, 21)))   # 7 laps in: the geometric still expects ~4 more
    left = [len([x for x in geo.draw(s, np.random.default_rng(i)).sc_laps if x <= 30]) for i in range(200)]
    assert np.mean(left) > mean_left(7) + 1.0


def test_geometric_default_unchanged():
    a = RateEvents(as_of=True).draw(_snap("sc", 20, sc=[19, 20]), np.random.default_rng(3))
    b = RateEvents(as_of=True, duration="geometric").draw(_snap("sc", 20, sc=[19, 20]), np.random.default_rng(3))
    assert a == b
    with pytest.raises(ValueError):
        RateEvents(duration="poisson")


def test_hybrid_passes_the_duration_to_later_periods():
    h = HybridEvents(lap_times="recorded", duration="history", pooled=True)
    assert h.rate.duration == "history" and h.rate.pooled
    assert h.rate.rates(RACE)[0] == pytest.approx(race_history.pooled_event_rates()["p_sc"])
