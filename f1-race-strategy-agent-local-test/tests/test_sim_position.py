"""Team-mate and finishing position checks on synthetic races. Offline, no model."""
import pandas as pd
import pytest

from src.sim import Simulator, calibration
from test_sim import RACE, TRUE, hand_time, synthetic_frame


def driver_rows(driver, team, plan, offset=0.0, positions=(1, 1)):
    """Laps that follow the true model plus a per lap pace offset, pit loss on the in lap."""
    rows, lap, t = [], 0, 0.0
    for i, (comp, n) in enumerate(plan):
        base, slope = TRUE[comp]
        for age in range(1, n + 1):
            lap += 1
            pit_in = age == n and i < len(plan) - 1
            lt = base + slope * age + offset + (RACE.pit_loss_s if pit_in else 0.0)
            rows.append({"driver": driver, "team": team, "lap": lap, "stint": i + 1, "compound": comp,
                         "tyre_age": age, "lap_time_s": lt, "track_status": "1", "pit_in": pit_in,
                         "pit_out": age == 1 and i > 0, "start_s": t, "end_s": t + lt,
                         "clean": not (pit_in or (age == 1 and i > 0)), "position": float(positions[0])})
            t += lt
    rows[-1]["position"] = float(positions[1])
    return rows


def race_frame(*drivers):
    return pd.DataFrame([r for d in drivers for r in driver_rows(*d)])


@pytest.fixture(scope="module")
def sim():
    return Simulator(synthetic_frame())


A_PLAN = [("SOFT", 20), ("MEDIUM", 30)]
B_PLAN = [("SOFT", 10), ("MEDIUM", 40)]
C_PLAN = [("SOFT", 15), ("MEDIUM", 34)]  # a lap down


def test_pairwise_gap_matches_hand_computation(sim):
    # AAA is slower on paper by 15.15 s, but BBB runs 0.4 s a lap quicker than the model
    frame = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN, -0.4))
    pairs, summary = calibration.pairwise(sim, frame, RACE)
    row = pairs.loc["T"]
    assert row["pred_gap_s"] == pytest.approx(hand_time(A_PLAN) - hand_time(B_PLAN))
    assert row["real_gap_s"] == pytest.approx(hand_time(A_PLAN) - (hand_time(B_PLAN) - 0.4 * 50))
    assert row["abs_err_s"] == pytest.approx(20.0)
    assert not row["tie"] and not row["sign_agree"]
    assert summary == {"n_pairs": 1, "n_skipped_unequal_laps": 0, "sign_agree_rate": 0.0,
                       "mae_gap_s": 20.0, "bias_gap_s": -20.0}


def test_pairwise_skips_and_ties(sim):
    frame = race_frame(
        ("AAA", "T1", A_PLAN), ("BBB", "T1", A_PLAN),            # identical, a tie
        ("CCC", "T2", A_PLAN), ("DDD", "T2", C_PLAN),            # unequal laps
        ("EEE", "T3", A_PLAN),                                   # team-mate retired
        ("FFF", "T4", A_PLAN), ("GGG", "T4", A_PLAN, 0.5), ("HHH", "T4", A_PLAN, 1.0))
    with pytest.warns(UserWarning, match="T4"):
        pairs, summary = calibration.pairwise(sim, frame, RACE)
    assert list(pairs.index) == ["T1"] and pairs.loc["T1", "tie"]
    assert summary["n_pairs"] == 1 and summary["n_skipped_unequal_laps"] == 1
    assert pd.isna(summary["sign_agree_rate"])


def three_driver_frame():
    # real order B, A, C. The model has A quicker than B, so A and B both come out second.
    return race_frame(("AAA", "T1", A_PLAN, 0.0, (1, 2)), ("BBB", "T2", B_PLAN, -0.4, (2, 1)),
                      ("CCC", "T3", C_PLAN, 0.0, (3, 3)))


def test_positions_from_lap_proxies(sim):
    table, summary = calibration.positions(sim, three_driver_frame(), RACE)
    assert table["pred_pos"].tolist() == [2, 2, 3]  # C is a lap down, so both others are ahead
    assert table["real_pos"].tolist() == [2, 1, 3] and table["start_pos"].tolist() == [1, 2, 3]
    assert table["abs_err"].tolist() == [0, 1, 0]
    assert summary == {"n_drivers": 3, "within_2_rate": 1.0, "mae_pos": 0.33, "exact_rate": 0.667,
                       "baseline_within_2_rate": 1.0, "baseline_mae": 0.67, "source": "lap_proxies"}


def test_positions_prefer_results_when_they_cover_every_finisher(sim):
    results = pd.DataFrame({"grid": [3.0, 1.0, 2.0], "classified": [3.0, 1.0, 2.0]},
                           index=pd.Index(["AAA", "BBB", "CCC"], name="driver"))
    table, summary = calibration.positions(sim, three_driver_frame(), RACE, results)
    assert table["real_pos"].tolist() == [3, 1, 2] and table["start_pos"].tolist() == [3, 1, 2]
    assert summary["source"] == "grid_classified"
    assert summary["mae_pos"] == 1.0 and summary["baseline_mae"] == 0.0

    partial = results.drop("CCC")
    assert calibration.positions(sim, three_driver_frame(), RACE, partial)[1]["source"] == "lap_proxies"


def test_observed_sc_factor_is_mean_with_median_alongside(sim):
    frame = race_frame(("AAA", "T", A_PLAN))
    for lap, f in [(30, 1.2), (31, 1.2), (32, 1.8)]:
        m = frame["lap"] == lap
        frame.loc[m, ["track_status", "clean", "lap_time_s"]] = ["4", False, f * sim.ref_lap_s]
    frame["end_s"] = frame["lap_time_s"].cumsum()
    frame["start_s"] = frame["end_s"] - frame["lap_time_s"]
    _, summary, _ = calibration.run(sim, frame, RACE)
    assert summary["observed_sc_factor"] == pytest.approx(1.4, abs=1e-3)
    assert summary["observed_sc_factor_median"] == pytest.approx(1.2, abs=1e-3)
    assert pd.isna(summary["observed_vsc_factor"])
    assert "observed_pit_loss_s" in summary
    assert "observed_lap1_extra_s" in summary
