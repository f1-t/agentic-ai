"""Report numbers for a finished rerun pass directory (read-only).
Usage (from the project dir, PYTHONPATH=.): python analyse.py <pass dir> [<previous pass dir>]"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.compare import cluster_bootstrap

D = Path(sys.argv[1])
PREV = Path(sys.argv[2]) if len(sys.argv) > 2 else None
LOW_WEAR = ["2019_10_united_kingdom", "2021_09_austrian", "2021_22_united_arab_emirates", "2022_08_azerbaijan"]
COMP = ["run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def load(d):
    a = pd.read_csv(d / "agent.csv")
    b = pd.read_csv(d / "baselines.csv")
    rows = pd.concat([a, b], ignore_index=True)
    rows["regret_s"] = pd.to_numeric(rows["regret_s"], errors="coerce")
    return a, b, rows


def section(t):
    print(f"\n===== {t} =====")


a, b, rows = load(D)
ok = rows[rows.status == "ok"]

section("counts per policy and status")
print(rows.groupby(["policy", "status"]).size().unstack(fill_value=0))
section("non-ok reasons")
print(rows[rows.status != "ok"].groupby(["policy", "status", "reason"]).size().to_string())
section("negative regret rows")
print(int((ok.regret_s < -1e-6).sum()))


def stats(g):
    r = g.regret_s.dropna()
    return pd.Series({"n": len(r), "mean": r.mean(), "median": r.median(), "p90": r.quantile(0.9), "worst": r.max()})


section("pooled regret per policy")
print(ok.groupby("policy").apply(stats).round(2).sort_values("mean"))
section("per race mean regret")
print(ok.pivot_table(index="race_id", columns="policy", values="regret_s", aggfunc="mean").round(2)
      [["agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy"]])
section("per race median / p90 / worst (agent)")
print(ok[ok.policy == "agent"].groupby("race_id").apply(stats).round(2))


def paired(sub, x):
    p = sub[sub.policy == "agent"][["scenario_id", "race_id", "driver", "regret_s"]].merge(
        sub[sub.policy == x][["scenario_id", "regret_s"]], on="scenario_id", suffixes=("_a", "_x"))
    p = p.dropna()
    if p.empty:
        return None
    d = p.regret_s_a - p.regret_s_x
    lo, hi = cluster_bootstrap(d, list(zip(p.race_id, p.driver)))
    rlo, rhi = cluster_bootstrap(d, p.race_id)
    w, l = int((d < -1e-6).sum()), int((d > 1e-6).sum())
    verdict = "ahead" if hi < 0 else "behind" if lo > 0 else "level"
    return dict(x=x, n=len(p), agent=p.regret_s_a.mean(), x_mean=p.regret_s_x.mean(), diff=d.mean(), ci=(lo, hi),
                race_ci=(rlo, rhi), wtl=(w, len(p) - w - l, l), verdict=verdict)


def show_split(name, sub):
    section(f"split: {name} ({sub[sub.policy == 'agent'].shape[0]} agent rows)")
    print(sub.groupby("policy").apply(stats).round(2).loc[[p for p in ["agent"] + COMP if p in set(sub.policy)]])
    for x in COMP:
        r = paired(sub, x)
        if r:
            print(f"  agent - {x}: {r['diff']:+.2f} ({r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}) race ({r['race_ci'][0]:+.2f}, "
                  f"{r['race_ci'][1]:+.2f}) n {r['n']} W/T/L {r['wtl']} {r['verdict']}")


show_split("all", ok)

# --- trajectories ------------------------------------------------------------------------------------------
def parse_plan(s):
    return [[c, int(n)] for c, n in (part.strip().split() for part in s.split(">"))]


def traj(sid):
    p = D / "trajectories" / (sid.replace(":", "_") + ".jsonl")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


REASONS = [
    ("neutralisation (SC/VSC)", r"\b(sc|vsc|safety car|virtual safety|neutrali)"),
    ("observed wear vs model", r"(observed|actual).{0,40}(wear|deg)|(wear|deg).{0,40}(observed|slower than model|faster than model|than modell?ed)"),
    ("tyre life / cliff", r"tyre.?life|cliff|life (estimate|prior)|beyond .{0,20}life|longest run"),
    ("traffic / position / gaps", r"traffic|position|gap|rejoin|undercut|overtak|car [a-z]\b"),
    ("pit loss", r"pit.?loss"),
    ("fewer stops / risk", r"risk|fewer stop|extra stop|avoid.{0,20}stop"),
]


def reason_of(text):
    t = (text or "").lower()
    for name, pat in REASONS:
        if re.search(pat, t):
            return name
    return "other / none stated"


bo = ok[ok.policy == "belief_optimizer"].set_index("scenario_id")
recs = []
for _, r in a.iterrows():
    tr = traj(r.scenario_id)
    tools = [x.get("tool") for x in tr]
    ranks = [x for x in tr if x.get("tool") == "rank_plans" and isinstance(x.get("output"), dict) and x["output"].get("plans")]
    subs = [x for x in tr if x.get("tool") == "submit_recommendation" and isinstance(x.get("output"), dict) and x["output"].get("accepted")]
    sub = subs[-1]["input"] if subs else {}
    top = ranks[-1]["output"]["plans"][0] if ranks else None
    top_plan = parse_plan(top["plan"]) if top else None
    plan = json.loads(r.plan) if isinstance(r.plan, str) else None
    imm_top = top.get("immediate") if top else None
    imm_agent = None
    if isinstance(r.decision, str):
        imm_agent = "STAY_OUT" if r.decision == "STAY_OUT" else f"PIT {sub.get('compound') or (plan[1][0] if plan and len(plan) > 1 else '')}".strip()
    bo_reg = bo.regret_s.get(r.scenario_id, np.nan) if r.scenario_id in bo.index else np.nan
    bo_plan = json.loads(bo.plan[r.scenario_id]) if r.scenario_id in bo.index else None
    recs.append(dict(scenario_id=r.scenario_id, race_id=r.race_id, driver=r.driver, lap=r.lap, status=r.status,
                     regret=r.regret_s, bo_regret=bo_reg, ranked=bool(ranks), n_rank=len(ranks), named_plan="plan" in sub,
                     departs_plan=(top_plan is not None and plan is not None and plan != top_plan),
                     departs_action=(imm_top is not None and imm_agent is not None and imm_top.split()[0] != imm_agent.split()[0]),
                     departs_immediate=(imm_top is not None and imm_agent is not None and imm_top != imm_agent),
                     top_eq_bo=(top_plan == bo_plan) if top_plan is not None else None,
                     rationale=sub.get("rationale", ""), reason=reason_of(sub.get("rationale", "")),
                     decision=r.decision, top_immediate=imm_top, agent_immediate=imm_agent,
                     plan=plan, top_plan=top_plan, tools=len(tools), grace=r.submitted_in_grace))
T = pd.DataFrame(recs)
T["delta_vs_bo"] = T.regret - T.bo_regret
T["outcome"] = np.select([T.delta_vs_bo < -0.05, T.delta_vs_bo > 0.05], ["helped", "hurt"], "same")
T.to_csv(Path("<scratch>/v3")
         / (D.name + "_departures.csv"), index=False)

section("look-ahead departures")
print("episodes with a rank_plans reading:", int(T.ranked.sum()), "of", len(T))
print("rank_plans plan 1 equals belief_optimizer's plan:", int((T.top_eq_bo == True).sum()), "of", int(T.top_eq_bo.notna().sum()))
print("submissions naming a whole plan:", int(T.named_plan.sum()))
dep = T[T.departs_plan & (T.status == "ok")]
print("scored plan differs from plan 1:", len(dep), "; action (PIT/STAY) differs:", int(T.departs_action.sum()),
      "; immediate (incl. compound) differs:", int(T.departs_immediate.sum()))
print("departures vs belief_optimizer (0.05 s): helped / hurt / same:", (dep.outcome == "helped").sum(), (dep.outcome == "hurt").sum(),
      (dep.outcome == "same").sum(), " net sum", round(dep.delta_vs_bo.sum(), 2), "mean", round(dep.delta_vs_bo.mean(), 3))
print("non-departures: agent - belief_optimizer mean", round(T[~T.departs_plan & (T.status == "ok")].delta_vs_bo.mean(), 3))
g = dep.groupby("reason").agg(n=("delta_vs_bo", "size"), mean=("delta_vs_bo", "mean"), total=("delta_vs_bo", "sum"),
                              helped=("outcome", lambda s: (s == "helped").sum()), hurt=("outcome", lambda s: (s == "hurt").sum()))
print(g.round(2).sort_values("n", ascending=False))
print("worst departures:")
print(dep.sort_values("delta_vs_bo", ascending=False)[["scenario_id", "decision", "top_immediate", "agent_immediate", "plan", "top_plan",
                                                         "delta_vs_bo", "reason"]].head(8).to_string())
print("best departures:")
print(dep.sort_values("delta_vs_bo")[["scenario_id", "plan", "top_plan", "delta_vs_bo", "reason"]].head(5).to_string())

# --- reliability --------------------------------------------------------------------------------------------
section("reliability")
ag = a.copy()
sub = ag.decision.notna()
print("rows", len(ag), "submitted", int(sub.sum()), "voluntary", int((sub & ~ag.submitted_in_grace.fillna(False).astype(bool)).sum()),
      "grace", int((sub & ag.submitted_in_grace.fillna(False).astype(bool)).sum()))
print("status", ag.status.value_counts().to_dict(), "api_retries", int(pd.to_numeric(ag.api_retries, errors="coerce").fillna(0).sum()))
print("decisions", ag.decision.value_counts().to_dict())
for c in ["latency_s", "tool_calls", "evidence_calls", "repeat_calls", "input_tokens", "output_tokens"]:
    v = pd.to_numeric(ag[c], errors="coerce")
    print(f"  {c}: mean {v.mean():.1f} median {v.median():.1f} p90 {v.quantile(0.9):.1f} total {v.sum():.0f}")

# --- splits ----------------------------------------------------------------------------------------------------
show_split("four low-wear races", ok[ok.race_id.isin(LOW_WEAR)])
show_split("rest (not the low-wear four)", ok[~ok.race_id.isin(LOW_WEAR)])
show_split("first 15 laps (lap <= 15)", ok[ok.lap <= 15])
show_split("after lap 15", ok[ok.lap > 15])

# races without circuit history: every tyre-life source the race_picture showed is a generic tier (>= 3)
hist = {}
for _, r in a.iterrows():
    for x in traj(r.scenario_id):
        if x.get("tool") == "race_picture" and isinstance(x.get("output"), dict):
            for c, v in (x["output"].get("compounds") or {}).items():
                src = ((v or {}).get("tyre_life") or {}).get("source") or ""
                m = re.search(r"tier(\d+)", src)
                if m:
                    hist.setdefault(r.race_id, set()).add(int(m.group(1)))
no_hist = sorted(r for r, tiers in hist.items() if min(tiers) >= 3)
print("\nrace_picture tyre-life tiers per race:", {r: sorted(t) for r, t in sorted(hist.items())})
print("races without circuit history (all tiers >= 3):", no_hist)
show_split("races without circuit history", ok[ok.race_id.isin(no_hist)])
show_split("races with circuit history", ok[~ok.race_id.isin(no_hist)])

# track status split
sc = json.loads((D / "scenarios.json").read_text())
status = {it["scenario_id"]: it["snapshot"].get("track_status", "green") for it in sc["items"]}
ok2 = ok.assign(track=ok.scenario_id.map(status))
print("\ntrack status of scenarios:", pd.Series(status).value_counts().to_dict())
for st in ["green", "sc", "vsc", "SC", "VSC"]:
    s2 = ok2[ok2.track == st]
    if len(s2):
        show_split(f"track status {st}", s2)

# --- previous pass ----------------------------------------------------------------------------------------------
if PREV is not None:
    pa, pb, prow = load(PREV)
    pok = prow[prow.status == "ok"]
    section("previous pass (committed dev20): pooled")
    print(pok.groupby("policy").apply(stats).round(2).sort_values("mean"))
    m = a[["scenario_id", "decision", "regret_s", "race_id", "driver"]].merge(pa[["scenario_id", "decision", "regret_s"]], on="scenario_id",
                                                                             suffixes=("", "_prev"))
    print("same decision as previous pass:", int((m.decision == m.decision_prev).sum()), "of", len(m))
    print("previous decisions", pa.decision.value_counts().to_dict())
