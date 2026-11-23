"""Print a compact reading of trajectories (read-only). Usage: python view.py <pass dir> <scenario_id>..."""
import json
import sys
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1])
a = pd.read_csv(D / "agent.csv").set_index("scenario_id")
b = pd.read_csv(D / "baselines.csv")
bo = b[b.policy == "belief_optimizer"].set_index("scenario_id")
sc = {it["scenario_id"]: it["snapshot"] for it in json.loads((D / "scenarios.json").read_text())["items"]}
for sid in sys.argv[2:]:
    r = a.loc[sid]
    s = sc[sid]
    print(f"\n#### {sid}  lap {r.lap} {s['compound']} age {s['tyre_age']} left {r.remaining_laps} track {s['track_status']} "
          f"| decision {r.decision} p_pit {r.p_pit} grace {r.submitted_in_grace} | plan {r.plan} regret {r.regret_s:.2f} "
          f"| ref {r.opt_plan} | belief_opt {bo.plan.get(sid)} {bo.regret_s.get(sid):.2f}")
    tr = [json.loads(x) for x in (D / "trajectories" / (sid.replace(":", "_") + ".jsonl")).read_text().splitlines() if x.strip()]
    for x in tr:
        t, o = x.get("tool"), x.get("output")
        if t == "rank_plans" and isinstance(o, dict):
            ps = o.get("plans") or []
            print("  rank_plans:", " | ".join(f"{p['rank']}. {p['plan']} ({p.get('family')}) gap {p.get('gap_s')} "
                                             f"{p.get('gap_90', '')} best_share {p.get('best_share')}" for p in ps[:4]))
            extra = {k: v for k, v in o.items() if k != "plans"}
            if extra:
                print("    extra:", json.dumps(extra)[:400])
        elif t == "race_picture" and isinstance(o, dict):
            c = o.get("car", {})
            print("  race_picture: pos", c.get("position"), "status", c.get("track_status"), "window", c.get("pit_window"),
                  "set", c.get("current_set"), "wear", json.dumps(c.get("set_wear")))
        elif t == "submit_recommendation":
            print("  SUBMIT", json.dumps(x.get("input"))[:700], "->", json.dumps(o)[:300])
        elif t == "evaluate_plan" and isinstance(o, dict):
            print("  evaluate_plan", json.dumps(x.get("input")), "->", o.get("plan"), json.dumps(o.get("vs_best")), o.get("flags"))
        elif t:
            print("  ", t, json.dumps(x.get("input"))[:120], ("ERROR " + str(x.get("error"))) if x.get("error") else "")
        else:
            print("  (record)", json.dumps(x)[:300])
