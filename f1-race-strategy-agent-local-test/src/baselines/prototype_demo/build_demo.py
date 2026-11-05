import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ...contracts import Plan
from ...eval.policies import BASELINE_POLICIES
from ...eval.reference import compute_reference
from ...eval.scenarios import Scenario
from ...eval.score import score_scenario
from ...sim import SimConfig, Simulator, calibration, jobs
from ...sim.compute import load_profile
from ...sim.data import race_stints
from ...sim.optimizer_bridge import from_contract, score_batch
from .. import candidate_plans
from ..runner import run_pipeline

HERE = Path(__file__).parent
PROJECT = HERE.parents[2]
SAMPLES = 32
SEED = 0


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_payload():
    profile = load_profile("local")
    race = jobs.load_race("2021_04_spain", profile)
    belief_config = SimConfig.from_file(PROJECT / "configs/sim/belief.yaml")
    seeded_config = belief_config.model_copy(update={"fit_window": "full"})
    belief = Simulator(race.frame, belief_config, race.practice)
    worlds = {
        "default": Simulator(race.frame, race.config, race.practice),
        "seeded": Simulator(race.frame, seeded_config, race.practice),
    }
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "race": "2021 Spanish Grand Prix",
        "race_id": race.race_id,
        "driver": "HAM",
        "seed": SEED,
        "n_samples": SAMPLES,
        "live_agent_called": False,
        "command": "python -m src.baselines.runner --driver HAM --lap 30 --no-agent --n-samples 32 --seed 0 --json",
        "worlds": {
            name: {
                "config": world.config.model_dump(mode="json"),
                "config_hash": calibration.config_hash(world.config),
            }
            for name, world in worlds.items()
        },
        "belief_config": belief_config.model_dump(mode="json"),
        "cases": {},
    }
    actual = [(c, n) for c, n, *_ in race_stints(race.frame, "HAM")]
    for lap in (0, 30, 42, 60, 65):
        snapshot = calibration.snapshot_at(race.frame, race.race, "HAM", lap)
        scenario = Scenario(
            scenario_id=f"{race.race_id}:HAM:{lap:02d}",
            race_id=race.race_id,
            driver="HAM",
            lap=lap,
            snapshot=snapshot,
            actual_stints=actual,
        )
        case = {
            "snapshot": snapshot.model_dump(mode="json"),
            "actual_stints": actual,
            "runs": {},
        }
        for world_name, world in worlds.items():
            case["runs"][world_name] = {}
            for budget in (1, 2):
                reference = compute_reference(
                    scenario, world, seed=SEED, n_samples=SAMPLES, max_stops=budget
                )
                rows = score_scenario(
                    scenario,
                    world,
                    BASELINE_POLICIES,
                    belief=belief,
                    seed=SEED,
                    n_samples=SAMPLES,
                    max_stops=budget,
                    reference=reference,
                )
                state = from_contract(snapshot, world)
                raw_plans = list(candidate_plans(state, budget))
                plans = [p for p in raw_plans if p.stints[0].laps > 0]
                evaluations, contracts, _, candidate_batch = score_batch(
                    snapshot, plans, world, seed=SEED, n_samples=SAMPLES, state=state
                )
                frontier = {}
                for evaluation, contract in zip(evaluations, contracts):
                    if evaluation is None:
                        continue
                    pits = contract.pit_laps(snapshot)
                    first_pit = pits[0] if pits else snapshot.race.total_laps
                    key = (contract.num_stops, first_pit)
                    candidate = {
                        "stops": contract.num_stops,
                        "first_pit": first_pit,
                        "mean_s": evaluation.mean_time_s,
                        "plan": contract.as_pairs(),
                    }
                    if (
                        key not in frontier
                        or candidate["mean_s"] < frontier[key]["mean_s"]
                    ):
                        frontier[key] = candidate
                selected = [reference.plan]
                selected.extend(
                    Plan.of(json.loads(row["plan"]))
                    for row in rows
                    if row["status"] == "ok"
                )
                replay = world.simulate_batch(
                    snapshot, selected, seed=SEED, n_samples=SAMPLES, return_laps=True
                )
                reference_row = {
                    "policy": "optimizer",
                    "status": "ok",
                    "plan": reference.plan.as_pairs(),
                    "time_s": reference.mean_s,
                    "regret_s": 0,
                    "regret_se_s": 0,
                    "std_s": reference.sd_s,
                    "num_stops": reference.plan.num_stops,
                    "first_pit_lap": next(
                        iter(reference.plan.pit_laps(snapshot)), None
                    ),
                }
                display_rows = [reference_row]
                replay_index = 1
                for row in rows:
                    shown = dict(row)
                    if "plan" in shown:
                        shown["plan"] = json.loads(shown["plan"])
                    if shown["status"] == "ok":
                        shown["draw_times"] = replay.times[replay_index].tolist()
                        shown["paired_deltas"] = (
                            replay.times[replay_index] - replay.times[0]
                        ).tolist()
                        shown["lap_times"] = replay.lap_times[replay_index].tolist()
                        shown["pit_loss_s"] = float(
                            replay.pit_loss[replay_index].mean()
                        )
                        shown["stint_seconds"] = (
                            replay.stint_seconds[replay_index].mean(axis=1).tolist()
                        )
                        shown["outside_reference_budget"] = shown["num_stops"] > budget
                        replay_index += 1
                    display_rows.append(shown)
                reference_row.update(
                    draw_times=replay.times[0].tolist(),
                    paired_deltas=[0.0] * SAMPLES,
                    lap_times=replay.lap_times[0].tolist(),
                    pit_loss_s=float(replay.pit_loss[0].mean()),
                    stint_seconds=replay.stint_seconds[0].mean(axis=1).tolist(),
                )
                case["runs"][world_name][str(budget)] = {
                    "rows": display_rows,
                    "frontier": sorted(
                        frontier.values(), key=lambda x: (x["stops"], x["first_pit"])
                    ),
                    "enumerated": len(raw_plans),
                    "zero_lap_filtered": len(raw_plans) - len(plans),
                    "scored": int(candidate_batch.valid.sum()),
                    "rejected": int((~candidate_batch.valid).sum()),
                }
                print(
                    f"HAM lap {lap:02d}, {world_name}, {budget} stops: {candidate_batch.valid.sum()} legal plans",
                    flush=True,
                )
        result["cases"][str(lap)] = case
    result["pipeline_example"] = run_pipeline(
        driver="HAM",
        lap=30,
        run_llm_agent=False,
        seed=SEED,
        n_samples=SAMPLES,
        verbose=False,
    )
    return clean(result)


def main():
    payload = build_payload()
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (HERE / "evidence.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    template = (HERE / "prototype_template.html").read_text()
    (HERE / "index.html").write_text(
        template.replace("__DEMO_PAYLOAD__", text.replace("</", "<\\/"))
    )
    print(HERE / "index.html")


if __name__ == "__main__":
    main()
