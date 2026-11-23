"""The belief's thin-fit guard (src/sim/belief_guard.py, SimConfig.belief_guard). Synthetic models; no cache."""
import pandas as pd
import pytest

from src.sim import SimConfig, belief_guard, calibration
from src.sim.config import BeliefGuard, CompoundProxy


class _Model:
    def __init__(self, coefs, sd_b, sd_w):
        self.coefs = coefs
        names = [f"{p}[{c}]" for c in coefs for p in ("base_s", "wear_s_per_lap")]
        m = [[0.0] * len(names) for _ in names]
        for c in coefs:
            i, j = names.index(f"base_s[{c}]"), names.index(f"wear_s_per_lap[{c}]")
            m[i][i], m[j][j] = sd_b[c] ** 2, sd_w[c] ** 2
        self.fit_report = {"sd": {"base_s": sd_b, "wear_s_per_lap": sd_w}, "cov": {"names": names, "matrix": m},
                           "n_race_rows_per_compound": {}}


def _frame(spec):
    """spec: {compound: (n_drivers, rows per driver, start age)}"""
    rows = []
    for c, (nd, nr, a0) in spec.items():
        for d in range(nd):
            for i in range(nr):
                rows.append({"driver": f"{c[:1]}{d}", "compound": c, "lap": i + 1, "tyre_age": a0 + i, "clean": True})
    return pd.DataFrame(rows)


def _cfg():
    # one source with a fixed step so the target is known: the cross-race step, MEDIUM = HARD x (1 - 0.01)
    # and MEDIUM wear = HARD wear x 2
    steps = {"MEDIUM-HARD": {"pace_frac": {"value": -0.01, "sd": 0.0, "n": 5},
                             "wear_ratio": {"value": 2.0, "sd": 0.0, "n": 5},
                             "life_ratio": {"value": 1.5, "sd": 0.0, "n": 5}}}
    return CompoundProxy(sources=["cross_race_step"], steps=steps, weights={}, sd_floor={"pace": 0.5, "wear": 0.05})


def test_off_by_default_and_out_of_the_config_hash():
    assert SimConfig().belief_guard == BeliefGuard() and not BeliefGuard().enabled
    assert calibration.config_hash(SimConfig()) == calibration.config_hash(
        SimConfig().model_copy(update={"belief_guard": BeliefGuard()}))
    assert calibration.config_hash(SimConfig()) != calibration.config_hash(
        SimConfig().model_copy(update={"belief_guard": BeliefGuard(enabled=True)}))


def test_a_few_car_compound_moves_toward_the_well_covered_one(monkeypatch):
    monkeypatch.setattr(belief_guard, "wear_prior", lambda rid, c: (0.05, 0.05, "test"))
    m = _Model({"HARD": (92.0, 0.05), "MEDIUM": (89.0, 0.06)}, {"HARD": 0.5, "MEDIUM": 0.05},
               {"HARD": 0.05, "MEDIUM": 0.005})
    f = _frame({"HARD": (2, 5, 1), "MEDIUM": (12, 8, 1)})
    out, ch = belief_guard.guarded(m, f, "2019_10_united_kingdom", BeliefGuard(enabled=True, min_rows=1),
                                   proxy_cfg=_cfg())
    assert set(ch) == {"HARD"} and out.coefs["MEDIUM"] == m.coefs["MEDIUM"]
    a = ch["HARD"]["mean_age"]
    target = ch["HARD"]["level_target_s"]
    assert target == pytest.approx(89.0 / 0.99 + ch["HARD"]["wear_target"] * a)
    lvl0 = 92.0 + 0.05 * a
    lvl = out.coefs["HARD"][0] + out.coefs["HARD"][1] * a
    assert min(lvl0, target) < lvl < max(lvl0, target)   # between the fit and the target
    assert m.coefs["HARD"] == (92.0, 0.05)                # the input is not changed
    # mode "wear" keeps the level at the mean visible age
    out_w, _ = belief_guard.guarded(m, f, "2019_10_united_kingdom", BeliefGuard(enabled=True, min_rows=1, mode="wear"),
                                    proxy_cfg=_cfg())
    assert out_w.coefs["HARD"][0] + out_w.coefs["HARD"][1] * a == pytest.approx(lvl0)


def test_nothing_qualifies_then_the_model_is_returned(monkeypatch):
    monkeypatch.setattr(belief_guard, "wear_prior", lambda rid, c: (0.05, 0.05, "test"))
    m = _Model({"MEDIUM": (89.0, 0.06)}, {"MEDIUM": 0.05}, {"MEDIUM": 0.005})
    out, ch = belief_guard.guarded(m, _frame({"MEDIUM": (12, 8, 1)}), "2022_21_brazil",
                                   BeliefGuard(enabled=True), proxy_cfg=_cfg())
    assert ch == {} and out is m
