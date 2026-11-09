"""Paired comparison tables and cluster-bootstrap intervals (src/eval/compare.py). Synthetic rows."""
import numpy as np
import pandas as pd
import pytest

from src.eval.compare import cluster_bootstrap, main, metric_table, paired_table


def rows(spec, race="r", **extra):
    """spec: {policy: {(driver, lap): regret or None}}; None is an unavailable row. `extra`
    columns are {name: {(driver, lap): value}}, the same for every policy."""
    out = []
    for policy, cells in spec.items():
        for (driver, lap), regret in cells.items():
            row = {"scenario_id": f"{race}:{driver}:{lap:02d}", "race_id": race, "driver": driver, "lap": lap,
                   "policy": policy, "status": "unavailable" if regret is None else "ok",
                   "regret_s": np.nan if regret is None else float(regret)}
            row.update({name: values.get((driver, lap), np.nan) for name, values in extra.items()})
            out.append(row)
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
    assert (row["metric"], row["paired"], row["clusters"]) == ("regret_s", 12, 4)
    assert row["mean_diff"] == -3.0 and (row["ci_low"], row["ci_high"]) == pytest.approx((-3.0, -3.0))
    assert (row["reference_wins"], row["ties"], row["policy_wins"]) == (12, 0, 0)
    assert (row["reference_mean"], row["policy_mean"]) == (1.0, 4.0)


def test_only_scenarios_both_scored_are_paired():
    ref = {k: 2.0 for k in GRID}
    other = {k: (None if k[0] == "AAA" else 2.5) for k in GRID}   # AAA unavailable
    other[("BBB", 0)] = 2.0                                          # one tie
    (row,) = paired_table(rows({"agent": ref, "greedy": other}), "agent").to_dict("records")
    assert (row["paired"], row["clusters"]) == (9, 3)
    assert (row["reference_wins"], row["ties"], row["policy_wins"]) == (8, 1, 0)
    assert row["mean_diff"] == pytest.approx(-0.5 * 8 / 9)


def test_tie_tolerance_and_explicit_others():
    data = rows({"agent": {k: 1.0 for k in GRID}, "a": {k: 1.05 for k in GRID}, "b": {k: 0.0 for k in GRID}})
    table = paired_table(data, "agent", ["a"], tie=0.1)
    assert list(table["policy"]) == ["a"] and table["ties"].iloc[0] == 12
    every = paired_table(data, "agent")
    assert list(every["policy"]) == ["a", "b"] and every.set_index("policy").loc["b", "policy_wins"] == 12


def test_a_policy_with_nothing_in_common_gives_an_empty_row():
    data = rows({"agent": {("AAA", 0): 1.0}, "other": {("BBB", 0): 2.0}})
    (row,) = paired_table(data, "agent").to_dict("records")
    assert row["paired"] == 0 and np.isnan(row["mean_diff"]) and np.isnan(row["ci_low"])


def test_unknown_reference_metric_or_cluster_raises():
    data = rows({"a": {("AAA", 0): 1.0}})
    with pytest.raises(KeyError):
        paired_table(data, "agent")
    with pytest.raises(KeyError):
        paired_table(data, "a", metric="lap_time")
    with pytest.raises(KeyError):
        metric_table(data, cluster="team")


# --- metrics -----------------------------------------------------------------------
def test_position_regret_skips_rows_without_a_position():
    pos = {k: (1.0 if k[1] == 0 else np.nan) for k in GRID}        # traffic gave a position at lap 0 only
    data = rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 2.0 for k in GRID}}, position_regret=pos)
    (row,) = paired_table(data, "agent", metric="position_regret").to_dict("records")
    assert (row["metric"], row["paired"], row["mean_diff"]) == ("position_regret", 4, 0.0)
    assert metric_table(data, metric="position_regret")["scenarios"].tolist() == [4, 4]


def test_brier_is_the_squared_error_of_p_pit_and_covers_agent_rows_only():
    data = rows({"agent": {k: 1.0 for k in GRID[:3]}, "fixed": {k: 2.0 for k in GRID[:3]}},
                pit_now_opt={GRID[0]: True, GRID[1]: False, GRID[2]: True})
    data.loc[data["policy"] == "agent", "p_pit"] = [0.9, 0.2, 0.5]   # baselines state no p_pit
    table = metric_table(data, metric="brier").set_index("policy")
    assert table.loc["agent", "mean"] == pytest.approx((0.01 + 0.04 + 0.25) / 3)
    assert "fixed" not in table.index
    assert metric_table(rows({"a": {("AAA", 0): 1.0}}), metric="brier").empty


# --- clusters ----------------------------------------------------------------------
def test_race_clusters_resample_whole_races():
    one = pd.concat([rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 1.0 + i for i, k in enumerate(GRID)}},
                          race=r) for r in ("r1", "r2", "r3")], ignore_index=True)
    by_driver = paired_table(one, "agent").iloc[0]
    by_race = paired_table(one, "agent", cluster="race").iloc[0]
    assert (by_driver["clusters"], by_race["clusters"]) == (12, 3)
    assert by_race["mean_diff"] == by_driver["mean_diff"]
    single = paired_table(rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 2.0 for k in GRID}}),
                          "agent", cluster="race").iloc[0]
    assert single["clusters"] == 1 and np.isnan(single["ci_low"])   # one race: nothing to resample


# --- metric table -------------------------------------------------------------------
def test_common_subset_puts_every_policy_on_the_same_scenarios():
    data = rows({"good": {k: 1.0 for k in GRID},
                 "patchy": {k: (3.0 if k[0] in ("AAA", "BBB") else None) for k in GRID}})
    own = metric_table(data).set_index("policy")
    common = metric_table(data, common=True).set_index("policy")
    assert own.loc["good", "scenarios"] == 12 and own.loc["patchy", "scenarios"] == 6
    assert (common["scenarios"] == 6).all() and (common["clusters"] == 2).all()
    assert common.loc["patchy", "mean"] == 3.0


def test_cli_prints_and_writes_both_tables(tmp_path, capsys):
    path = tmp_path / "rows.csv"
    rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 2.0 for k in GRID}}).to_csv(path, index=False)
    assert main([str(path), "--reference", "agent", "--out", str(tmp_path / "v1")]) == 0
    printed = capsys.readouterr().out
    assert "agent against each policy: regret_s" in printed and "| regret_s | agent | fixed |" in printed
    paired = pd.read_csv(tmp_path / "v1_regret_s_paired.csv")
    ci = pd.read_csv(tmp_path / "v1_regret_s_ci.csv")
    assert paired["mean_diff"].iloc[0] == -1.0
    assert set(ci["subset"]) == {"own", "common"} and set(ci["policy"]) == {"agent", "fixed"}


def test_cli_metric_and_cluster_flags(tmp_path, capsys):
    path = tmp_path / "rows.csv"
    rows({"agent": {k: 1.0 for k in GRID}, "fixed": {k: 2.0 for k in GRID}},
         position_regret={k: 0.5 for k in GRID}).to_csv(path, index=False)
    assert main([str(path), "--reference", "agent", "--metric", "position_regret", "--cluster", "race",
                 "--out", str(tmp_path / "v1")]) == 0
    assert "position_regret, paired, 95% CI, clusters by race" in capsys.readouterr().out
    assert (tmp_path / "v1_position_regret_paired.csv").is_file()
