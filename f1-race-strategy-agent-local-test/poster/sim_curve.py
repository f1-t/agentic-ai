"""Per-lap predicted lap times for the poster's simulator figure (writes poster/sim_curve.csv).

    python poster/sim_curve.py

The example decision's snapshot (poster_stats.EXAMPLE, 2022_06_spain:MAG:25, a development-set race) is read from
the committed development set's scenarios.json. The belief simulator the agent's tools use
(agent_tools.belief_simulator, configs/sim/belief.yaml) is built at that snapshot, and the two plans the example's
rank_plans showed first are simulated: plan 1 "MEDIUM 1 > SOFT 20 > SOFT 20" and plan 2 "MEDIUM 2 > MEDIUM 15 > SOFT 24".
For each remaining lap the CSV holds the point estimate's lap time (Simulator.point_estimate: no parameter draw and
no lap noise; seed 0, one sample). The lap times carry no pit loss: the simulator adds it once per stop, recorded per
plan (both stops summed). sc_free says whether that sample drew no safety car. The CSV also records the number of
plans the look-ahead scores at this snapshot (optimize_batch with the profile's n_samples and max_stops) and its best
plan.

This is the only poster script that imports `src`; build_poster.py reads its CSV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from src.eval.scenarios import ScenarioSet  # noqa: E402
from src.sim import agent_tools  # noqa: E402
from src.sim.compute import load_profile  # noqa: E402
from src.sim.optimizer_bridge import optimize_batch  # noqa: E402

SCENARIO = "2022_06_spain:MAG:25"
SET = PROJECT / "reports" / "eval" / "rerun" / "v3" / "dev20" / "scenarios.json"
PLANS = {"plan1": [("MEDIUM", 1), ("SOFT", 20), ("SOFT", 20)],
         "plan2": [("MEDIUM", 2), ("MEDIUM", 15), ("SOFT", 24)]}
OUT = HERE / "sim_curve.csv"


def main():
    sc = next(s for s in ScenarioSet.load(SET).items if s.scenario_id == SCENARIO)
    if sc.race_id in json.loads((PROJECT / "data" / "eval_set.json").read_text())["races"]:
        raise SystemExit("the example is a test-set race")
    snap = sc.snapshot
    prof = load_profile()
    sim = agent_tools.belief_simulator(snap, prof).fitted_for(snap)
    plans = [[{"compound": c, "laps": n} for c, n in PLANS[k]] for k in PLANS]
    plans = [{"stints": p} for p in plans]
    pt = sim.point_estimate().simulate_batch(snap, plans, seed=0, n_samples=1, return_laps=True)
    if not pt.valid.all():
        raise SystemExit(f"plan rejected: {pt.errors}")
    laps = np.arange(snap.laps_completed + 1, snap.race.total_laps + 1)
    opt = optimize_batch(snap, sim, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops)
    n_searched = opt.candidates_evaluated + opt.invalid
    best = " > ".join(f"{s.compound} {s.laps}" for s in opt.plan.stints)
    df = pd.DataFrame({"lap": laps, "plan1_s": pt.lap_times[0], "plan2_s": pt.lap_times[1]})
    df["sc_free"] = bool(np.all(np.isnan(pt.sc_field_delta))) if pt.sc_field_delta is not None else None
    for k, stints in PLANS.items():
        df[k] = " > ".join(f"{c} {n}" for c, n in stints)
    df["plan1_pit_loss_s"] = float(pt.pit_loss[0, 0])
    df["plan2_pit_loss_s"] = float(pt.pit_loss[1, 0])
    df["n_samples_profile"] = prof.sim.n_samples
    df["plans_scored"] = opt.candidates_evaluated
    df["plans_searched"] = n_searched
    df["best_plan"] = best
    df.round(3).to_csv(OUT, index=False)
    print(df.round(2).to_string(), "\nplans scored", opt.candidates_evaluated, "invalid", opt.invalid, "best", best,
          "\ndiagnostics", {k: v for k, v in opt.diagnostics.items() if k not in ("times", "searched", "runner_up")})


if __name__ == "__main__":
    main()
