"""Print a compact view of one agent trajectory (pipeline output; reads nothing else)."""
import ast
import json
import sys

KEYS = ("plan", "continuation", "note", "valid", "errors", "gap_reading", "probability_improves_outcome",
        "p_best", "pit_loss_now_s", "track_status", "pit_window", "same_plan_as", "accepted", "error")


def short(o, n):
    s = o if isinstance(o, str) else json.dumps(o)
    return s[:n]


def main(path, n=260):
    for line in open(path):
        e = json.loads(line)
        if "tool" not in e:
            print(f"  [{e.get('event')}] finish={e.get('finish_reason')} parse_error={short(e.get('parse_error'), 120)}")
            continue
        t, inp, out = e["tool"], e.get("input"), e.get("output")
        rep = f" (repeat of {e['repeat_of']})" if e.get("repeat_of") else ""
        head = f"s{e['step']} {e['phase'][:4]} {t} {short(inp, 400 if t == 'submit_recommendation' else 160)}{rep}"
        print(head)
        if t in ("get_model_report", "estimate_tyre_degradation"):
            continue
        if isinstance(out, dict):
            picked = {k: out[k] for k in KEYS if k in out}
            if "candidates" in out:
                picked["candidates"] = [(c.get("plan") if isinstance(c, dict) else c) for c in out["candidates"]][:6] \
                    if isinstance(out["candidates"], list) else out["candidates"]
            if "alternative" in out:
                picked["alternative"] = out["alternative"]
            if "remaining" in out:
                picked["remaining"] = out["remaining"]
            print("     -> " + short(picked, n))
        else:
            print("     -> " + short(out, n))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 260)
