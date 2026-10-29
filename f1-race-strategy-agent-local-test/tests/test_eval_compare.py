"""Paired comparison tables and cluster-bootstrap intervals (src/eval/compare.py). Synthetic rows."""
import numpy as np
import pandas as pd
import pytest

from src.eval.compare import cluster_bootstrap, main, paired_table, regret_table


def rows(spec):
    """spec: {policy: {(driver, lap): regret or None}}; None is an unavailable row."""
    out = []
    for policy, cells in spec.items():
        for (driver, lap), regret in cells.items():
            out.append({"scenario_id": f"r:{driver}:{lap:02d}", "race_id": "r", "driver": driver, "lap": lap,
                        "policy": policy, "status": "unavailable" if regret is None else "ok",
                        "regret_s": np.nan if regret is None else float(regret)})
    return pd.DataFrame(out)


GRID = [(d, lap) for d in ("AAA", "BBB", "CCC", "DDD") for lap in (0, 5, 10)]


# --- cluster bootstrap -------------------------------------------------------------
def test_constant_values_give_a_point_interval():
    assert cluster_bootstrap([2.0] * 6, ["a", "a", "b", "b", "c", "c"]) == pytest.approx((2.0, 2.0))


def test_one_cluster_has_no_interval():
    low, high = cluster_bootstrap([1.0, 2.0, 3.0], ["a"] * 3)
    assert np.isnan(low) and np.isnan(high)


def test_interval_brackets_the_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    values = rng.normal(5.0, 2.0, 40)
    clusters = [i // 4 for i in range(40)]
    low, high = cluster_bootstrap(values, clusters, seed=3)
    assert low < values.mean() < high
    assert cluster_bootstrap(values, clusters, seed=3) == (low, high)
    assert cluster_bootstrap(values, clusters, seed=4) != (low, high)


def test_clusters_widen_the_interval_over_independent_draws():
    # every cluster is one value repeated: resampling clusters sees 5 values, not 50
    values = np.repeat(np.arange(5.0), 10)
    clustered = cluster_bootstrap(values, np.repeat(np.arange(5), 10))
    independent = cluster_bootstrap(values, np.arange(50))
    assert clustered[1] - clustered[0] > 2 * (independent[1] - independent[0])


# --- paired table ------------------------------------------------------------------
def test_paired_difference_wins_and_sign():
    ref = {k: 1.0 for k in GRID}
    worse = {k: 4.0 for k in GRID}
    (row,) = paired_table(rows({"agent": ref, "fixed": worse}), "agent").to_dict("records")
    assert (row["paired"], row["clusters"]) == (12, 4)
    assert row["mean_diff_s"] == -3.0 and (row["ci_low_s"], row["ci_high_s"]) == pytest.approx((-3.0, -3.0))
    assert (row["reference_wins"], row["ties"], row["policy_wins"]) == (12, 0, 0)
    assert (row["reference_regret_s"], row["policy_regret_s"]) == (1.0, 4.0)


def test_only_scenarios_both_scored_are_paired():
    ref = {k: 2.0 for k in GRID}
    other = {k: (None if k[0] == "AAA" else 2.5) for k in GRID}   # AAA unavailable
    other[("BBB", 0)] = 2.0                                          # one tie
    (row,) = paired_table(rows({"agent": ref, "greedy": other}), "agent").to_dict("records")
    assert (row["paired"], row["clusters"]) == (9, 3)
    assert (row["reference_wins"], row["ties"], row["policy_wins"]) == (8, 1, 0)
    assert row["mean_diff_s"] == pytest.approx(-0.5 * 8 / 9)


def test_tie_tolerance_and_explicit_others():
    data = rows({"agent": {k: 1.0 for k in GRID}, "a": {k: 1.05 for k in GRID}, "b": {k: 0.0 for k in GRID}})
    table = paired_table(data, "agent", ["a"], tie_s=0.1)
    assert list(table["policy"]) == ["a"] and table["ties"].iloc[0] == 12
    every = paired_table(data, "agent")
    assert list(every["policy"]) == ["a", "b"] and every.set_index("policy").loc["b", "policy_wins"] == 12


def test_a_policy_with_nothing_in_common_gives_an_empty_row():
    data = rows({"agent": {("AAA", 0): 1.0}, "other": {("BBB", 0): 2.0}})
    (row,) = paired_table(data, "agent").to_dict("records")
    assert row["paired"] == 0 and np.isnan(row["mean_diff_s"]) and np.isnan(row["ci_low_s"])


def test_unknown_reference_raises():
    with pytest.raises(KeyError):
        paired_table(rows({"a": {("AAA", 0): 1.0}}), "agent")


# --- regret table -------------------------------------------------------------------
def test_common_subset_puts_every_policy_on_the_same_scenarios():
    data = rows({"good": {k: 1.0 for k in GRID},
                 "patchy": {k: (3.0 if k[0] in ("AAA", "BBB") else None) for k in GRID}})
    own = regret_table(data).set_index("policy")
    common = regret_table(data, common=True).set_index("policy")
    assert own.loc["good", "scenarios"] == 12 and own.loc["patchy", "scenarios"] == 6
    assert (common["scenarios"] == 6).all() and (common["clusters"] == 2).all()
    assert common.loc["patchy", "mean_regret_s"] == 3.0


def test_cli_prints_and_writes_both_tables(tmp_path, capsys):
    path = tmp_path / "rows.csv"
    rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 2.0 for k in GRID}}).to_csv(path, index=False)
    assert main([str(path), "--reference", "agent", "--out", str(tmp_path / "v1")]) == 0
    printed = capsys.readouterr().out
    assert "agent against each policy" in printed and "| agent | fixed |" in printed
    paired = pd.read_csv(tmp_path / "v1_paired.csv")
    ci = pd.read_csv(tmp_path / "v1_regret_ci.csv")
    assert paired["mean_diff_s"].iloc[0] == -1.0
    assert set(ci["subset"]) == {"own", "common"} and set(ci["policy"]) == {"agent", "fixed"}
