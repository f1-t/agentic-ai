"""Grade an agent recommendation against the deterministic optimum.

Success = the agent's recommended strategy is valid and its simulated race time
is within `tolerance_pct` of the optimum (data/ground_truth.json). This rewards a
near-optimal plan reached via the tools rather than one exact hard-coded plan.
"""
from . import legacy_tools as tools


def grade(rec, gt):
    plan = (rec or {}).get("stint_plan")
    if not plan:
        return {"valid": False, "passed": False, "reason": "no stint_plan submitted"}

    sim = tools.simulate_strategy(plan)
    if "error" in sim:
        return {"valid": False, "passed": False, "reason": sim["error"]}

    opt = gt["optimal_total_time_s"]
    agent_time = sim["total_time_s"]
    gap_pct = (agent_time - opt) / opt * 100.0
    passed = gap_pct <= gt["tolerance_pct"]
    return {
        "valid": True,
        "passed": bool(passed),
        "agent_time_s": agent_time,
        "optimal_time_s": opt,
        "gap_pct": round(gap_pct, 3),
        "tolerance_pct": gt["tolerance_pct"],
        "agent_num_stops": sim["num_stops"],
        "agent_first_pit_lap": sim["first_pit_lap"],
    }
