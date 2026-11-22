"""Tables for the rule baselines (docs/eval.md "Rule baselines"), from committed result files.

    python scripts/rule_baseline_tables.py --combined reports/eval/d4_combined.csv \
        --rules reports/eval/d4_rule_baselines.csv --scenarios data/scenarios/d4.json --prefix reports/eval/d4

Reads csv and scenario json only: no race is loaded, nothing is simulated or rescored. Joins
the new rule-baseline rows to an existing combined file (after checking that both carry the
same reference per scenario) and writes, beside `--prefix`:

    <prefix>_combined_rules.csv                       the existing rows plus the new policies
    <prefix>_rules_compare_regret_s_{paired,ci}.csv   agent against each policy, (race, driver) clusters
    <prefix>_rules_compare_race_regret_s_{paired,ci}.csv          the same with race clusters
    <prefix>_rules_compare_per_race_regret_s_paired.csv           per race, (race, driver) clusters
    <prefix>_rules_compare_stopped_regret_s_paired.csv            split by whether the driver has
                                                                  already stopped (snapshot.stints)

and prints the summary, paired and split tables as markdown. The committed combined file is
left as it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.compare import metric_table, paired_table  # noqa: E402
from src.eval.score import to_markdown  # noqa: E402

POLICIES = ("agent", "belief_optimizer", "fixed_one_stop", "actual_strategy", "run_to_flag", "one_stop_total")
REFERENCE_COLUMNS = ("opt_plan", "opt_time_s", "sim_code_version", "world_config_hash", "n_samples", "seed")


def stopped_ids(scenarios_path) -> set[str]:
    """Scenario ids whose snapshot has a completed stint: the driver has already stopped."""
    items = json.loads(Path(scenarios_path).read_text())["items"]
    return {it["scenario_id"] for it in items if it["snapshot"]["stints"]}


def join(combined: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    new = set(rules["policy"])
    if new & set(combined["policy"]):
        raise SystemExit(f"policies {sorted(new & set(combined['policy']))} are already in the combined file")
    ref_old = combined.drop_duplicates("scenario_id").set_index("scenario_id")
    ref_new = rules.drop_duplicates("scenario_id").set_index("scenario_id")
    if set(ref_old.index) != set(ref_new.index):
        raise SystemExit("the two files cover different scenarios")
    for col in REFERENCE_COLUMNS:
        a, b = ref_old.loc[ref_new.index, col], ref_new[col]
        same = np.isclose(a, b, rtol=0, atol=1e-9) if col == "opt_time_s" else (a.astype(str) == b.astype(str))
        if not bool(np.all(same)):
            raise SystemExit(f"reference column {col} differs between the files")
    return pd.concat([combined, rules], ignore_index=True)


def counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (race, policy) and pooled: scored / invalid / unavailable, regret mean, median, p90."""
    def one(g):
        ok = g.loc[g["status"] == "ok", "regret_s"].astype(float)
        st = g["status"].value_counts()
        return {"scored": int(st.get("ok", 0)), "invalid": int(st.get("invalid", 0)),
                "unavailable": int(st.get("unavailable", 0)), "mean": ok.mean(), "median": ok.median(),
                "p90": ok.quantile(0.9) if len(ok) else np.nan}
    pols = [p for p in POLICIES if p in set(rows["policy"])]
    out = []
    for race in ["ALL", *sorted(rows["race_id"].unique())]:
        g = rows if race == "ALL" else rows[rows["race_id"] == race]
        out += [{"race_id": race, "policy": p, **one(g[g["policy"] == p])} for p in pols]
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--combined", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. reports/eval/d4")
    args = ap.parse_args(argv)

    rows = join(pd.read_csv(args.combined), pd.read_csv(args.rules))
    prefix = Path(args.prefix)
    rows.to_csv(f"{prefix}_combined_rules.csv", index=False)
    others = [p for p in POLICIES[1:] if p in set(rows["policy"])]

    print("## Summary (ok rows of each policy)\n")
    print(to_markdown(counts(rows)))

    for cluster, tag in (("driver", "rules_compare"), ("race", "rules_compare_race")):
        paired = paired_table(rows, "agent", others, cluster=cluster)
        pols = ["agent", *others]
        ci = pd.concat([metric_table(rows, pols, cluster=cluster).assign(subset="own"),
                        metric_table(rows, pols, cluster=cluster, common=True).assign(subset="common")],
                       ignore_index=True)
        paired.to_csv(f"{prefix}_{tag}_regret_s_paired.csv", index=False)
        ci.to_csv(f"{prefix}_{tag}_regret_s_ci.csv", index=False)
        print(f"\n## agent minus each policy, paired, 95% CI, clusters by {cluster}\n")
        print(to_markdown(paired))

    per_race = pd.concat([paired_table(g, "agent", others).assign(race=race, cluster="driver")
                          for race, g in rows.groupby("race_id")], ignore_index=True)
    per_race = per_race[["race", "cluster", *[c for c in per_race.columns if c not in ("race", "cluster")]]]
    per_race.to_csv(f"{prefix}_rules_compare_per_race_regret_s_paired.csv", index=False)
    print("\n## Per race, agent minus each policy, (race, driver) clusters\n")
    print(to_markdown(per_race.drop(columns=["metric", "reference", "cluster"])))

    done = stopped_ids(args.scenarios)
    rows["stopped"] = rows["scenario_id"].isin(done)
    split = []
    for flag, g in rows.groupby("stopped"):
        label = "already stopped" if flag else "not yet stopped"
        split.append(paired_table(g, "agent", others).assign(subset=label, scenarios=g["scenario_id"].nunique()))
    split = pd.concat(split, ignore_index=True)
    split = split[["subset", "scenarios", *[c for c in split.columns if c not in ("subset", "scenarios")]]]
    split.to_csv(f"{prefix}_rules_compare_stopped_regret_s_paired.csv", index=False)
    print("\n## Split by whether the driver has already stopped, agent minus each policy, (race, driver) clusters\n")
    print(to_markdown(split.drop(columns=["metric", "reference"])))
    print("\n## Split: summary per policy\n")
    print(to_markdown(pd.concat([counts(g).query("race_id == 'ALL'").assign(
        subset="already stopped" if flag else "not yet stopped") for flag, g in rows.groupby("stopped")],
        ignore_index=True).drop(columns="race_id")))
    rtf = rows[(rows["policy"] == "run_to_flag") & (rows["status"] == "ok")]
    print("\nrun_to_flag stops per scored row, by subset:")
    print(rtf.groupby("stopped")["num_stops"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
