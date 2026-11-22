"""docs/eval.md "Belief priors": the switches on the belief's practice prior and its tyre-life
edge, each off by default. Synthetic tables only; the committed ones are as of the race
(tests/test_future_blindness.py)."""
import numpy as np
import pandas as pd
import pytest

from src.sim import calibration, race_history
from src.sim.config import SimConfig

HISTORY = pd.DataFrame({"race_id": ["2020_03_a", "2021_02_b", "2022_05_c"],
                        "base_delta_s": [0.0, 1.0, 9.0],
                        "base_delta_s[HARD]": [-2.0, -1.0, 50.0],
                        "base_delta_s[MEDIUM]": [1.0, 2.0, 0.0],
                        "base_delta_s[SOFT]": [1.0, np.nan, 0.0]})


def test_a_compound_delta_is_the_common_one_plus_its_shrunk_as_of_offset():
    common = race_history.as_of_factor("2022_01_q", "base_delta_s", HISTORY)
    hard = race_history.as_of_compound_delta("2022_01_q", "HARD", HISTORY)
    # offsets -2 and -2 (the 2022 race is later: never read), median -2, shrunk by 2 / (2 + 2)
    k = race_history.COMPOUND_DELTA_PRIOR_RACES
    assert hard["offset"] == pytest.approx(-2.0 * 2 / (2 + k))
    assert hard["mean"] == pytest.approx(common["mean"] + hard["offset"]) and hard["sd"] == common["sd"]
    assert hard["races"] == ["2020_03_a", "2021_02_b"]
    soft = race_history.as_of_compound_delta("2022_01_q", "SOFT", HISTORY)   # the NaN race does not count
    assert soft["n"] == 1 and soft["offset"] == pytest.approx(1.0 / (1 + k))


def test_no_earlier_race_gives_the_common_delta():
    got = race_history.as_of_compound_delta("2019_01_q", "HARD", HISTORY)
    assert got["offset"] == 0.0 and got["n"] == 0
    assert got["mean"] == race_history.as_of_factor("2019_01_q", "base_delta_s", HISTORY)["mean"]


class _Practice:
    def __init__(self, coefs, n_rows):
        self.coefs = coefs
        names = [f"{p}[{c}]" for c in coefs for p in ("base_s", "wear_s_per_lap")]
        self.fit_report = {"n_rows": n_rows, "cov": {"names": names, "matrix": np.eye(len(names)).tolist()}}


def test_practice_anchoring_moves_every_compound_but_the_best_covered_one():
    cfg = SimConfig()
    pm = _Practice({"HARD": (92.0, 0.02), "MEDIUM": (90.0, 0.05), "SOFT": (90.5, 0.1)},
                   {"HARD": 10, "MEDIUM": 80, "SOFT": 30})
    quali = {"gap": {"MEDIUM": 0.0, "SOFT": -0.6}, "cov": {("MEDIUM", "MEDIUM"): 0.0, ("SOFT", "SOFT"): 0.01,
                                                            ("SOFT", "MEDIUM"): 0.0}, "n_drivers": {"SOFT": 8, "MEDIUM": 8}}
    got = race_history.anchored_gap_offsets(pm, cfg, "2019_01_first", quali, ("pace", "wear"))
    assert set(got) == {"HARD", "SOFT"} and all(e["ref"] == "MEDIUM" for e in got.values())
    # no earlier race: no cross-race step, equal weights; SOFT is the mean of practice (+0.5) and qualifying (-0.6)
    assert got["SOFT"]["pace"] == pytest.approx((0.5 - 0.6) / 2 - 0.5)
    assert got["HARD"]["pace"] == 0.0   # practice alone: unchanged
    assert race_history.anchored_gap_offsets(_Practice({"MEDIUM": (90.0, 0.05)}, {"MEDIUM": 9}), cfg, "x") == {}
    assert race_history.anchored_gap_offsets(None, cfg, "x") == {}


@pytest.mark.parametrize("params", [{"base_delta": "nope"}, {"practice_anchor": "nope"}])
def test_unknown_modes_are_refused(params):
    with pytest.raises(ValueError):
        race_history.AsOfHistoryCalibrator(**params)
    race_history.AsOfHistoryCalibrator()   # the defaults


def test_the_new_config_switch_keeps_the_hash_at_its_default():
    base = SimConfig()
    assert calibration.config_hash(base) == calibration.config_hash(base.model_copy(update={"belief_cliff_edge": "draw"}))
    assert calibration.config_hash(base) != calibration.config_hash(base.model_copy(update={"belief_cliff_edge": "point"}))


MAP = pd.DataFrame({"race_id": ["2020_03_a", "2021_02_b", "2022_05_c"],
                    "practice_base[MEDIUM]": [90.0, 90.0, 90.0], "practice_wear[MEDIUM]": [0.15, 0.20, 0.0],
                    "judge_base[MEDIUM]": [91.0, 91.0, 99.0], "judge_wear[MEDIUM]": [0.05, 0.10, 9.0],
                    "practice_base[HARD]": [91.0, 91.0, 91.0], "practice_wear[HARD]": [0.0, 0.0, 0.0],
                    "judge_base[HARD]": [90.0, 89.0, 0.0], "judge_wear[HARD]": [0.04, 0.02, 9.0]})


def test_the_practice_map_moves_wear_and_relative_base_by_earlier_races_only():
    k = race_history.COMPOUND_DELTA_PRIOR_RACES
    med = race_history.as_of_practice_map("2022_01_q", "MEDIUM", MAP)
    assert med["races"] == ["2020_03_a", "2021_02_b"] and med["n"] == 2
    assert med["wear"] == pytest.approx(-0.10 * 2 / (2 + k))
    # base changes: MEDIUM +1, +1; HARD -1, -2; race means 0, -0.5; MEDIUM relative +1, +1.5 (median 1.25)
    assert med["base"] == pytest.approx(1.25 * 2 / (2 + k))
    hard = race_history.as_of_practice_map("2022_01_q", "HARD", MAP)
    assert hard["wear"] == pytest.approx(0.03 * 2 / (2 + k)) and hard["base"] == pytest.approx(-1.25 * 2 / (2 + k))
    none = race_history.as_of_practice_map("2019_01_q", "HARD", MAP)
    assert (none["wear"], none["base"], none["n"]) == (0.0, 0.0, 0)
    assert race_history.as_of_practice_map("2022_01_q", "SOFT", MAP)["n"] == 0   # no SOFT columns


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
