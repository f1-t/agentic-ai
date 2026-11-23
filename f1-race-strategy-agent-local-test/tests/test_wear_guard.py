"""The judge's thin-fit wear guard (src/sim/wear_guard.py, SimConfig.wear_guard). Synthetic models and
the committed history table; no cache."""
import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, calibration, wear_guard
from src.sim.config import WearGuard
from src.sim.race_ids import parse_race_id


class _Model:
    def __init__(self, coefs, sd):
        self.coefs = coefs
        self.fit_report = {"sd": {"wear_s_per_lap": sd}}


def _frame(spec):
    """spec: {compound: (n_drivers, rows per driver, start age)}"""
    rows = []
    for c, (nd, nr, a0) in spec.items():
        for d in range(nd):
            for i in range(nr):
                rows.append({"driver": f"{c[:1]}{d}", "compound": c, "lap": i + 1, "tyre_age": a0 + i, "clean": True})
    return pd.DataFrame(rows)


def test_off_by_default_and_out_of_the_config_hash():
    assert SimConfig().wear_guard == WearGuard() and not WearGuard().enabled
    assert calibration.config_hash(SimConfig()) == calibration.config_hash(
        SimConfig().model_copy(update={"wear_guard": WearGuard()}))
    assert calibration.config_hash(SimConfig()) != calibration.config_hash(
        SimConfig().model_copy(update={"wear_guard": WearGuard(enabled=True)}))


def test_a_thin_compound_is_shrunk_and_keeps_its_mean_lap(monkeypatch):
    monkeypatch.setattr(wear_guard, "wear_prior", lambda rid, c: (0.05, 0.05, "test"))
    m = _Model({"HARD": (90.0, 0.16), "MEDIUM": (89.0, 0.07)}, {"HARD": 0.05, "MEDIUM": 0.004})
    f = _frame({"HARD": (1, 10, 2), "MEDIUM": (10, 30, 1)})
    out, ch = wear_guard.guarded(m, f, "2022_21_brazil", WearGuard(enabled=True))
    assert set(ch) == {"HARD"} and out.coefs["MEDIUM"] == m.coefs["MEDIUM"]
    b, w = out.coefs["HARD"]
    assert w == pytest.approx(0.105)   # equal precisions: the midpoint
    ages = f.loc[f["compound"] == "HARD", "tyre_age"]
    assert (b + w * ages).mean() == pytest.approx((90.0 + 0.16 * ages).mean())
    scaled, _ = wear_guard.guarded(m, f, "2022_21_brazil", WearGuard(enabled=True, scale_sd_by_drivers=True))
    assert scaled.coefs["HARD"][1] < w   # one car: a wider fit sd, pulled further toward the history


def test_a_compound_with_no_race_rows_is_left_to_the_proxy(monkeypatch):
    monkeypatch.setattr(wear_guard, "wear_prior", lambda rid, c: (0.05, 0.05, "test"))
    m = _Model({"SOFT": (88.0, 0.2)}, {"SOFT": 0.08})
    out, ch = wear_guard.guarded(m, _frame({"MEDIUM": (10, 30, 1)}), "2022_21_brazil", WearGuard(enabled=True))
    assert ch == {} and out is m


def test_the_history_is_as_of_and_never_holds_a_held_out_race():
    t = wear_guard.load_table()
    assert not set(t["race_id"]) & wear_guard.held_out()
    for rid in ("2021_04_spain", "2022_21_brazil", "2024_07_emilia_romagna"):
        key = parse_race_id(rid)[:2]
        assert all(parse_race_id(r)[:2] < key for r in wear_guard.races_for(rid))
    w, sd, src = wear_guard.wear_prior("2019_10_united_kingdom", "HARD")
    assert src == "a_priori" and sd == wear_guard.ROLE_PRIOR_SD   # nothing earlier in the table
    w, sd, src = wear_guard.wear_prior("2024_07_emilia_romagna", "SOFT")
    assert src.startswith("history") and np.isfinite(w) and sd >= wear_guard.SD_FLOOR
