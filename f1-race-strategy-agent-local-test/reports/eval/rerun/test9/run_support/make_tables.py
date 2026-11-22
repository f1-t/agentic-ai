"""Render the test9 report tables as markdown from the pipeline's output CSVs (reads outputs only)."""
import json
import sys

import pandas as pd

OUT = sys.argv[1]
DEV = sys.argv[2]
T = OUT + "/tables/"
POL = ["agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy",
       "fixed_one_stop", "fixed_two_stop", "greedy_crossover"]
LEADS = ["run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer"]


def f(x, d=2, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def paired(df, groupcol, groups):
    rows = ["| group | X | paired (clusters) | agent mean | X mean | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in groups:
        for _, r in df[df[groupcol] == g].iterrows():
            rows.append(f"| {g} | {r.policy} | {r.paired} ({r.race_driver_clusters}) | {f(r.reference_mean)} | {f(r.policy_mean)} | "
                        f"{f(r.mean_diff, sign=True)} | ({f(r.ci_low_race_driver, sign=True)}, {f(r.ci_high_race_driver, sign=True)}) | "
                        f"({f(r.ci_low_race, sign=True)}, {f(r.ci_high_race, sign=True)}) | {r.reference_wins} / {r.ties} / {r.policy_wins} | {r.verdict} |")
    return "\n".join(rows)


p = pd.read_csv(T + "reporting_paired.csv")
print("## PAIRED\n")
print(paired(p, "group", ["fresh five", "earlier four", "all nine"]))
e = pd.read_csv(T + "reporting_paired_by_era.csv")
e = e[e.policy.isin(LEADS)]
print("\n## BY ERA\n")
print(paired(e.assign(group=e.era + " (" + e.era_races.astype(str) + ")"), "group",
             [f"{a} ({b})" for a, b in e[["era", "era_races"]].drop_duplicates().itertuples(index=False)]))

c = pd.read_csv(T + "reporting_counts.csv").fillna(0)
print("\n## COUNTS (scored / invalid / unavailable / error / no_reference)\n")
print("| policy | fresh five | earlier four | all nine |\n| --- | --- | --- | --- |")
for pol in POL:
    cells = []
    for g in ["fresh five", "earlier four", "all nine"]:
        r = c[(c.policy == pol) & (c.group == g)].iloc[0]
        cells.append(f"{int(r.get('ok', 0))} / {int(r.get('invalid', 0))} / {int(r.get('unavailable', 0))} / "
                     f"{int(r.get('error', 0))} / {int(r.get('no_reference', 0))}")
    print(f"| {pol} | " + " | ".join(cells) + " |")

s = pd.read_csv(T + "summary.csv")
print("\n## REGRET mean / median / p90 / worst (s), scored n\n")
races = [r for r in s.race_id.unique() if r != "ALL"] + ["ALL"]
print("| race | " + " | ".join(POL) + " |\n| --- |" + " --- |" * len(POL))
for race in races:
    cells = []
    for pol in POL:
        r = s[(s.race_id == race) & (s.policy == pol)].iloc[0]
        cells.append(f"{f(r.mean_regret_s, 1)} / {f(r.median_regret_s, 1)} / {f(r.p90_regret_s, 1)} / {f(r.max_regret_s, 1)} (n {r.scored})")
    print(f"| {race} | " + " | ".join(cells) + " |")
ch = pd.read_csv(T + "chart_regret_s.csv").set_index("policy")
print("\nPooled mean with 95% (race, driver) interval: " + "; ".join(
    f"{pol} {f(ch.loc[pol, 'mean'])} ({f(ch.loc[pol, 'ci_low'])}, {f(ch.loc[pol, 'ci_high'])})" for pol in POL))

t = pd.read_csv(T + "reporting_tail.csv")
print("\n## TAIL\n")
print("| group | policy | scored | over 20 s (stated) | over 10 s | over 30 s | median s | worst s |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
for g in ["fresh five", "earlier four", "all nine"]:
    for pol in POL:
        r = t[(t.group == g) & (t.policy == pol)].iloc[0]
        print(f"| {g} | {pol} | {r.scored} | {f(r['share_over_20s (stated)'], 3)} | {f(r.share_over_10s, 3)} | "
              f"{f(r.share_over_30s, 3)} | {f(r.median_s)} | {f(r.worst_s)} |")

dp = pd.read_csv(DEV + "/tables/reporting_paired.csv").set_index("policy")
dt = pd.read_csv(DEV + "/tables/reporting_tail.csv").set_index("policy")
print("\n## DEV BESIDE (agent - X, primary CI, verdict)\n")
print("| row | dev20 (development, 19 races) | fresh five | earlier four | all nine |\n| --- | --- | --- | --- | --- |")
for pol in LEADS:
    cells = [f"{f(dp.loc[pol, 'mean_diff'], sign=True)} ({f(dp.loc[pol, 'ci_low_race_driver'], sign=True)}, {f(dp.loc[pol, 'ci_high_race_driver'], sign=True)}) {dp.loc[pol, 'verdict']}"]
    for g in ["fresh five", "earlier four", "all nine"]:
        r = p[(p.group == g) & (p.policy == pol)].iloc[0]
        cells.append(f"{f(r.mean_diff, sign=True)} ({f(r.ci_low_race_driver, sign=True)}, {f(r.ci_high_race_driver, sign=True)}) {r.verdict}")
    print(f"| agent - {pol} | " + " | ".join(cells) + " |")
ag = {g: p[(p.group == g) & (p.policy == "run_to_flag")].iloc[0].reference_mean for g in ["fresh five", "earlier four", "all nine"]}
print(f"| agent mean regret s | {f(dp.loc['run_to_flag', 'reference_mean'])} | " + " | ".join(f(ag[g]) for g in ag) + " |")
for pol in ["agent", "belief_optimizer", "run_to_flag", "one_stop_total"]:
    cells = [f(dt.loc[pol, 'share_over_20s (stated)'], 3)]
    for g in ["fresh five", "earlier four", "all nine"]:
        cells.append(f(t[(t.group == g) & (t.policy == pol)].iloc[0]['share_over_20s (stated)'], 3))
    print(f"| {pol} share over 20 s | " + " | ".join(cells) + " |")
