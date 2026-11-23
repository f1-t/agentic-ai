"""Classify each episode's scored plan against the last rank_plans plan 1 (read-only).
Usage: python departures.py <pass dir> <departures csv from analyse.py>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, CSV = Path(sys.argv[1]), sys.argv[2]
T = pd.read_csv(CSV)
out = []
for _, r in T.iterrows():
    tr = [json.loads(x) for x in (D / "trajectories" / (r.scenario_id.replace(":", "_") + ".jsonl")).read_text().splitlines() if x.strip()]
    ranks = [x for x in tr if x.get("tool") == "rank_plans" and isinstance(x.get("output"), dict) and x["output"].get("plans")]
    subs = [x for x in tr if x.get("tool") == "submit_recommendation" and isinstance(x.get("output"), dict) and x["output"].get("accepted")]
    top = ranks[-1]["output"]["plans"][0]["submit"]
    sub = subs[-1]["input"]
    kind = "same plan"
    if r.departs_plan:
        if sub.get("decision") != top.get("decision"):
            kind = "stated: other action"
        elif sub.get("plan") is not None:
            kind = "stated: named plan"
        elif any(sub.get(k) is not None and top.get(k) is not None and sub.get(k) != top.get(k) for k in ("compound", "first_stop_lap")) \
                or (sub.get("first_stop_lap") is not None and top.get("first_stop_lap") is None):
            kind = "stated: other compound / stop lap"
        elif sub.get("decision") == "STAY_OUT" and sub.get("reassess_after_laps") != top.get("reassess_after_laps") \
                and sub.get("first_stop_lap") is None:
            kind = "bare STAY_OUT, other reassess"
        else:
            kind = "bare submission, continuation differs"
    out.append(kind)
T["kind"] = out
pd.set_option("display.width", 250)
print(T.groupby("kind").agg(n=("delta_vs_bo", "size"), mean=("delta_vs_bo", "mean"), total=("delta_vs_bo", "sum"),
                            helped=("outcome", lambda s: ((s == "helped") & True).sum()),
                            hurt=("outcome", lambda s: (s == "hurt").sum())).round(3))
stated = T[T.kind.str.startswith("stated")]
print("\nstated departures by reason:")
print(stated.groupby("reason").agg(n=("delta_vs_bo", "size"), mean=("delta_vs_bo", "mean"), total=("delta_vs_bo", "sum"),
                                   helped=("outcome", lambda s: (s == "helped").sum()), hurt=("outcome", lambda s: (s == "hurt").sum())).round(2))
print(stated[["scenario_id", "kind", "top_immediate", "agent_immediate", "plan", "top_plan", "delta_vs_bo", "reason"]].to_string())
bare = T[T.kind.str.startswith("bare")]
print("\nbare-submission differences by reason:")
print(bare.groupby("reason").agg(n=("delta_vs_bo", "size"), mean=("delta_vs_bo", "mean"), total=("delta_vs_bo", "sum"),
                                 helped=("outcome", lambda s: (s == "helped").sum()), hurt=("outcome", lambda s: (s == "hurt").sum())).round(2))
T.to_csv(CSV, index=False)
