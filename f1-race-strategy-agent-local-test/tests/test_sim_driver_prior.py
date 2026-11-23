"""The qualifying-pace driver-effect prior (degradation.params driver_prior "qualifying"), a switch, off:
measured on the fidelity scorecard on 2026-11-22 and not adopted. Synthetic tables only."""
import numpy as np
import pandas as pd
import pytest


def _quali():
    rows = []
    for d, off in (("AAA", -0.5), ("BBB", 0.0), ("CCC", 0.5)):
        for seg, base in (("Q1", 90.0), ("Q2", 89.5)):
            rows.append({"driver": d, "segment": seg, "compound": "SOFT", "lap_time_s": base + off, "timed": True})
            rows.append({"driver": d, "segment": seg, "compound": "SOFT", "lap_time_s": base + off + 3, "timed": True})
    rows.append({"driver": "DDD", "segment": "Q1", "compound": "SOFT", "lap_time_s": 140.0, "timed": True})  # no push lap
    return pd.DataFrame(rows)


def test_qualifying_driver_offsets_are_centred_and_skip_non_push_laps():
    from src.sim import proxy
    got = proxy.qualifying_driver_offsets(_quali())
    assert set(got) == {"AAA", "BBB", "CCC"}
    assert got["AAA"] == pytest.approx(-0.5) and got["CCC"] == pytest.approx(0.5) and sum(got.values()) == pytest.approx(0)
    assert proxy.qualifying_driver_offsets(None) is None


def test_the_driver_prior_moves_the_driver_effects_prior_mean():
    from src.sim import degradation
    rng = np.random.default_rng(0)
    rows = []
    for d, off in (("AAA", -1.0), ("BBB", 0.0), ("CCC", 1.0)):
        for lap in range(1, 4):   # three laps each: the data barely identify the offsets
            rows.append({"driver": d, "lap": float(lap), "compound": "MEDIUM", "tyre_age": float(lap),
                         "lap_time_s": 90.0 + off + 0.05 * lap + rng.normal(0, 0.3), "clean": True})
    data = pd.DataFrame(rows)
    prior = degradation.LinearFuelDegradation.fit(data.assign(lap=np.nan), fuel_prior=-0.05)
    kw = dict(min_samples=3, prior=prior, shrinkage={"base_s": 0.5, "wear_s_per_lap": 0.03, "fuel_s_per_lap": 0.01,
                                                     "driver_s": 0.1}, fuel_prior=-0.05, driver_effects="race")
    off = degradation.LinearFuelDegradation.fit(data, **kw).fit_report["driver_effects"]["value"]
    on = degradation.LinearFuelDegradation.fit(data, driver_prior={"AAA": -1.0, "BBB": 0.0, "CCC": 1.0},
                                               **kw).fit_report["driver_effects"]["value"]
    assert abs(on["CCC"] - on["AAA"]) > abs(off["CCC"] - off["AAA"]) + 0.5
    # "qualifying" unresolved (no qualifying): no prior, the fit as before
    same = degradation.LinearFuelDegradation.fit(data, driver_prior="qualifying", **kw).fit_report["driver_effects"]["value"]
    assert same == off
