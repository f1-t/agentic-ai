"""Paired comparison tables with confidence intervals, from the rows `score` writes.

    python -m src.eval.compare reports/eval/v1_baselines_split.csv --reference fixed_one_stop
    python -m src.eval.compare rows.csv --reference agent --metric position_regret --out reports/eval/v1_agent

Two tables:

- `metric_table`: each policy's mean of a metric with a confidence interval. With
  `common=True` only the scenarios every listed policy has a value for count, so the means
  are over the same scenarios. The plain summary (score.summarize) averages each policy over
  whatever it scored, which differs between policies.
- `paired_table`: the reference policy against each other policy on the scenarios both have
  a value for. diff is reference minus other per scenario, so a negative mean means the
  reference does better. Wins, ties and losses count scenarios where the reference is
  better, within `tie`, or worse.

Metrics (METRICS), all lower is better: regret_s (default), regret_pct, position_regret
(needs world traffic), first_pit_error, and brier, the squared error of an agent row's
stated p_pit against pit_now_opt. Rows without a value for the metric are left out, so
brier only covers agent rows and position_regret only runs with traffic on.

Intervals are a cluster bootstrap. cluster="driver" (default) resamples (race_id, driver)
groups whole: the same car at laps 0, 5, 10 is strongly correlated, and resampling single
scenarios would make the intervals too narrow. cluster="race" resamples races whole, the
conservative choice once a set spans many races, since every scenario of a race shares the
world model's fit for it. With one cluster there is nothing to resample and the interval
is NaN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .score import REGRET_TOL_S, to_markdown

METRICS = ("regret_s", "regret_pct", "position_regret", "first_pit_error", "brier")
CLUSTERS = {"driver": ("race_id", "driver"), "race": ("race_id",)}


def _values(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The ok rows with a value for `metric`, in a `value` column."""
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; choose from {', '.join(METRICS)}")
    ok = rows[rows["status"] == "ok"]
    if metric == "brier":
        if not {"p_pit", "pit_now_opt"} <= set(ok.columns):
            return ok.iloc[0:0].assign(value=pd.Series(dtype=float))
        truth = ok["pit_now_opt"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        value = (pd.to_numeric(ok["p_pit"], errors="coerce") - truth) ** 2
    else:
        value = pd.to_numeric(ok[metric], errors="coerce") if metric in ok else pd.Series(np.nan, index=ok.index)
    return ok.assign(value=value).dropna(subset=["value"])


def _cluster_ids(df: pd.DataFrame, cluster: str) -> list[tuple]:
    if cluster not in CLUSTERS:
        raise KeyError(f"unknown cluster {cluster!r}; choose from {', '.join(CLUSTERS)}")
    return list(df[list(CLUSTERS[cluster])].itertuples(index=False, name=None))


def cluster_bootstrap(values, clusters, *, n_boot: int = 2000, level: float = 0.95,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile interval for the mean of `values`, resampling whole clusters.

    Each resample draws as many clusters as there are, with replacement, and takes the mean
    over every value in the drawn clusters (a cluster drawn twice counts twice)."""
    values = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(pd.Series(list(clusters)), sort=True)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 2:
        return float("nan"), float("nan")
    sums = np.bincount(codes, weights=values, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    draws = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    means = (draws @ sums) / (draws @ counts)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def metric_table(rows: pd.DataFrame, policies=None, *, metric: str = "regret_s", cluster: str = "driver",
                 common: bool = False, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Mean of `metric` per policy with a cluster-bootstrap interval. `policies` defaults to
    every policy with a value; `common` restricts all of them to the scenarios they all have."""
    vals = _values(rows, metric)
    policies = list(policies) if policies is not None else sorted(vals["policy"].unique())
    vals = vals[vals["policy"].isin(policies)]
    if common:
        per = vals.groupby("scenario_id")["policy"].nunique()
        vals = vals[vals["scenario_id"].isin(per[per == len(policies)].index)]
    out = []
    for pol in policies:
        g = vals[vals["policy"] == pol]
        ids = _cluster_ids(g, cluster)
        low, high = cluster_bootstrap(g["value"], ids, n_boot=n_boot, level=level, seed=seed)
        out.append({"metric": metric, "policy": pol, "scenarios": len(g), "clusters": len(set(ids)),
                    "mean": float(g["value"].mean()) if len(g) else np.nan, "ci_low": low, "ci_high": high})
    return pd.DataFrame(out)


def paired_table(rows: pd.DataFrame, reference: str, others=None, *, metric: str = "regret_s",
                 cluster: str = "driver", tie: float = REGRET_TOL_S, n_boot: int = 2000,
                 level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """`reference` against each policy in `others` (default: every other policy with a value)
    on the scenarios both have a value for. diff = reference - other, in the metric's units."""
    if reference not in set(rows["policy"]):
        raise KeyError(f"no rows for reference policy {reference!r}")
    vals = _values(rows, metric)
    ref = vals[vals["policy"] == reference].set_index("scenario_id")
    others = list(others) if others is not None else sorted(set(vals["policy"]) - {reference})
    out = []
    for pol in others:
        other = vals[vals["policy"] == pol].set_index("scenario_id")
        both = ref.index.intersection(other.index)
        diff = ref.loc[both, "value"] - other.loc[both, "value"]
        ids = _cluster_ids(ref.loc[both].reset_index(), cluster)
        low, high = cluster_bootstrap(diff, ids, n_boot=n_boot, level=level, seed=seed)
        n = len(both)
        out.append({
            "metric": metric, "reference": reference, "policy": pol, "paired": n, "clusters": len(set(ids)),
            "reference_mean": float(ref.loc[both, "value"].mean()) if n else np.nan,
            "policy_mean": float(other.loc[both, "value"].mean()) if n else np.nan,
            "mean_diff": float(diff.mean()) if n else np.nan, "ci_low": low, "ci_high": high,
            "reference_wins": int((diff < -tie).sum()), "ties": int((diff.abs() <= tie).sum()),
            "policy_wins": int((diff > tie).sum()),
        })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.eval.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", help="per-row csv from python -m src.eval score")
    ap.add_argument("--reference", required=True, help="policy the others are compared against, e.g. agent")
    ap.add_argument("--others", nargs="+", help="default: every other policy with a value")
    ap.add_argument("--metric", choices=METRICS, default="regret_s")
    ap.add_argument("--cluster", choices=sorted(CLUSTERS), default="driver",
                    help="resample (race, driver) groups or whole races")
    ap.add_argument("--tie", "--tie-s", dest="tie", type=float, default=REGRET_TOL_S,
                    help="gap counted as a tie, in the metric's units")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="prefix: writes <out>_<metric>_paired.csv and <out>_<metric>_ci.csv")
    args = ap.parse_args(argv)

    rows = pd.read_csv(args.rows)
    kw = {"metric": args.metric, "cluster": args.cluster, "n_boot": args.n_boot, "level": args.level,
          "seed": args.seed}
    paired = paired_table(rows, args.reference, args.others, tie=args.tie, **kw)
    policies = [args.reference, *paired["policy"]]
    ci = pd.concat([metric_table(rows, policies, **kw).assign(subset="own"),
                    metric_table(rows, policies, common=True, **kw).assign(subset="common")],
                   ignore_index=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        paired.to_csv(out.with_name(f"{out.name}_{args.metric}_paired.csv"), index=False)
        ci.to_csv(out.with_name(f"{out.name}_{args.metric}_ci.csv"), index=False)
    pct = round(args.level * 100)
    print(f"## {args.reference} against each policy: {args.metric}, paired, {pct}% CI, "
          f"clusters by {args.cluster}\n")
    print(to_markdown(paired))
    print(f"\n## Mean {args.metric}, own scenarios and common scenarios ({pct}% CI)\n")
    print(to_markdown(ci))
    return 0


if __name__ == "__main__":
    sys.exit(main())
