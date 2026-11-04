"""Paired comparison tables with confidence intervals, from the rows `score` writes.

    python -m src.eval.compare reports/eval/v1_baselines_split.csv --reference fixed_one_stop
    python -m src.eval.compare rows.csv --reference agent --out reports/eval/v1_agent

Two tables:

- `regret_table`: each policy's mean regret with a confidence interval. With `common=True`
  only the scenarios every listed policy scored count, so the means are over the same
  scenarios. The plain summary (score.summarize) averages each policy over whatever it
  scored, which differs between policies.
- `paired_table`: the reference policy against each other policy on the scenarios both
  scored. diff_s is reference regret minus other regret per scenario, so a negative mean
  means the reference loses less time. Wins, ties and losses count scenarios where the
  reference is better, within `tie_s`, or worse.

Intervals are a cluster bootstrap: a race's scenarios for one driver are strongly correlated
(the same car at laps 0, 5, 10, ...), so (race_id, driver) clusters are resampled whole rather
than single scenarios, which would make the intervals too narrow. With one cluster there is
nothing to resample and the interval is NaN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .score import REGRET_TOL_S, to_markdown

CLUSTER = ("race_id", "driver")


def _ok(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["status"] == "ok"]


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


def _cluster_ids(df: pd.DataFrame) -> list[tuple]:
    return list(df[list(CLUSTER)].itertuples(index=False, name=None))


def regret_table(rows: pd.DataFrame, policies=None, *, common: bool = False, n_boot: int = 2000,
                 level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Mean regret per policy with a cluster-bootstrap interval. `policies` defaults to every
    policy with a scored row; `common` restricts all of them to the scenarios they all scored."""
    ok = _ok(rows)
    policies = list(policies) if policies is not None else sorted(ok["policy"].unique())
    ok = ok[ok["policy"].isin(policies)]
    if common:
        per = ok.groupby("scenario_id")["policy"].nunique()
        ok = ok[ok["scenario_id"].isin(per[per == len(policies)].index)]
    out = []
    for pol in policies:
        g = ok[ok["policy"] == pol]
        low, high = cluster_bootstrap(g["regret_s"], _cluster_ids(g), n_boot=n_boot, level=level, seed=seed)
        out.append({"policy": pol, "scenarios": len(g), "clusters": g.groupby(list(CLUSTER)).ngroups,
                    "mean_regret_s": float(g["regret_s"].mean()) if len(g) else np.nan,
                    "ci_low_s": low, "ci_high_s": high})
    return pd.DataFrame(out)


def paired_table(rows: pd.DataFrame, reference: str, others=None, *, tie_s: float = REGRET_TOL_S,
                 n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """`reference` against each policy in `others` (default: every other policy with a scored
    row) on the scenarios both scored. diff_s = reference regret - other regret."""
    ok = _ok(rows)
    if reference not in set(rows["policy"]):
        raise KeyError(f"no rows for reference policy {reference!r}")
    ref = ok[ok["policy"] == reference].set_index("scenario_id")
    others = list(others) if others is not None else sorted(set(ok["policy"]) - {reference})
    out = []
    for pol in others:
        other = ok[ok["policy"] == pol].set_index("scenario_id")
        both = ref.index.intersection(other.index)
        diff = ref.loc[both, "regret_s"] - other.loc[both, "regret_s"]
        clusters = _cluster_ids(ref.loc[both].reset_index())
        low, high = cluster_bootstrap(diff, clusters, n_boot=n_boot, level=level, seed=seed)
        n = len(both)
        out.append({
            "reference": reference, "policy": pol, "paired": n,
            "clusters": len(set(clusters)),
            "reference_regret_s": float(ref.loc[both, "regret_s"].mean()) if n else np.nan,
            "policy_regret_s": float(other.loc[both, "regret_s"].mean()) if n else np.nan,
            "mean_diff_s": float(diff.mean()) if n else np.nan, "ci_low_s": low, "ci_high_s": high,
            "reference_wins": int((diff < -tie_s).sum()), "ties": int((diff.abs() <= tie_s).sum()),
            "policy_wins": int((diff > tie_s).sum()),
        })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.eval.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", help="per-row csv from python -m src.eval score")
    ap.add_argument("--reference", required=True, help="policy the others are compared against, e.g. agent")
    ap.add_argument("--others", nargs="+", help="default: every other policy with a scored row")
    ap.add_argument("--tie-s", type=float, default=REGRET_TOL_S, help="regret gap counted as a tie")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="prefix: writes <out>_paired.csv and <out>_regret_ci.csv")
    args = ap.parse_args(argv)

    rows = pd.read_csv(args.rows)
    kw = {"n_boot": args.n_boot, "level": args.level, "seed": args.seed}
    paired = paired_table(rows, args.reference, args.others, tie_s=args.tie_s, **kw)
    policies = [args.reference, *paired["policy"]]
    ci = pd.concat([regret_table(rows, policies, **kw).assign(subset="own"),
                    regret_table(rows, policies, common=True, **kw).assign(subset="common")],
                   ignore_index=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        paired.to_csv(out.with_name(out.name + "_paired.csv"), index=False)
        ci.to_csv(out.with_name(out.name + "_regret_ci.csv"), index=False)
    pct = round(args.level * 100)
    print(f"## {args.reference} against each policy (paired, {pct}% cluster-bootstrap CI)\n")
    print(to_markdown(paired))
    print(f"\n## Mean regret, own scenarios and common scenarios ({pct}% CI)\n")
    print(to_markdown(ci))
    return 0


if __name__ == "__main__":
    sys.exit(main())
