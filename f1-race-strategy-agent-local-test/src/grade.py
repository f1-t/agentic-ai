"""Grade an agent recommendation.

grade(rec, gt): the legacy whole-race grade. Success = the recommended stint plan is valid
and its simulated race time is within `tolerance_pct` of the optimum
(data/ground_truth.json), on the frozen legacy simulator. This rewards a near-optimal plan
reached via the tools rather than one exact hard-coded plan.

grade_decision(snapshot, submission): the snapshot loop's grade. The submission (PIT or
STAY_OUT, src/agent.py submit_recommendation) is expanded into an action plus a
continuation on the eval's belief simulator (src/eval/decisions.py) and scored by the world
against the optimizer's plan from the same snapshot, exactly as the eval scores an agent
row. It reports validity and regret; whether a test asserts more than a valid submission
within budget is the test's call.
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


GRADE_KEYS = ("decision", "p_pit", "plan", "num_stops", "first_pit_lap", "time_s", "regret_s", "regret_pct",
              "opt_plan", "opt_num_stops", "opt_first_pit_lap", "opt_time_s", "pit_now_opt", "n_samples")


def grade_decision(snapshot, submission, *, sim_for=None, profile=None, seed=0, n_samples=None, max_stops=None):
    """Validity and regret of one agent submission at a contracts.Snapshot.

    `sim_for(race_id, role)` gives the belief and world simulators (eval.score.race_simulators
    over the committed cache by default, with `profile`); n_samples and max_stops default to
    the profile's. Returns {"valid", "reason", **GRADE_KEYS}: valid is the scorer's status
    "ok" (a well-formed decision whose expanded plan the world accepts), reason the scorer's
    message otherwise; regret_s is the expanded plan's mean remaining time less the
    optimizer's, under shared draws. Reported, never asserted here."""
    import math

    from .eval.policies import Decision
    from .eval.scenarios import Scenario
    from .eval.score import race_simulators, score_scenario
    from .sim.compute import load_profile

    if sim_for is None or n_samples is None or max_stops is None:
        profile = profile or load_profile()
    sim_for = sim_for or race_simulators(profile)
    race_id = snapshot.race.race_id
    scenario = Scenario(scenario_id=f"{race_id}:{snapshot.driver}:{snapshot.laps_completed:02d}", race_id=race_id,
                        driver=snapshot.driver, lap=snapshot.laps_completed, snapshot=snapshot, actual_stints=[])
    (row,) = score_scenario(
        scenario, sim_for(race_id, "world"), {"agent": lambda *_: Decision(None, submission=submission)},
        belief=sim_for(race_id, "belief"), seed=seed,
        n_samples=profile.sim.n_samples if n_samples is None else n_samples,
        max_stops=profile.sim.max_stops if max_stops is None else max_stops)

    def clean(v):
        return None if isinstance(v, float) and math.isnan(v) else v
    return {"valid": row["status"] == "ok", "reason": row.get("reason"),
            **{k: clean(row.get(k)) for k in GRADE_KEYS}}
