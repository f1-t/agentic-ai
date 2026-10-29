# Baselines and optimizer

This package supplies Quan's four reference policies and an exhaustive optimizer for remaining race plans.
It uses no language model or model API key.
The optimizer accepts a simulator callback so Jason can change physics without changing the search.
The included adapter uses the existing deterministic Spanish Grand Prix simulator model.

## Run the offline comparison

Use Python 3.10 or newer from the project directory.
Install the baseline dependencies once with internet access.
The dependencies inherit the project's current simulator requirements.
Use an isolated environment because the committed schedule response expects gzip and deflate request headers.
Additional compression packages can make the cache library try to refresh an archived response even in offline mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baselines.txt
python -m src.baselines --driver HAM --lap 0
python -m src.baselines --driver HAM --lap 30 --seeds 0 1 2
```

Each command prints a JSON report with the snapshot, optimizer result, and four baseline results.
Add `--scorer batch` to score every plan through the simulator in one `simulate_batch` call (`src/sim/optimizer_bridge.py`); the seeds must then be `0 1 ... n-1`, and the report has the same shape.
The command uses the committed race cache and makes no language model calls.
The adapter currently supports only the race configured in the existing tools.
Drivers with incomplete race histories are unavailable rather than extended with invented laps.

Run the checks from the same directory.

```bash
python -m pytest tests/test_baselines.py tests/test_baselines_offline.py -q
ruff check src/baselines tests/test_baselines.py tests/test_baselines_offline.py
ruff format --check src/baselines tests/test_baselines.py tests/test_baselines_offline.py
```

The offline integration tests block network connections.
They compare every legal full-race candidate against the extracted simulator using matched legacy coefficients and per-stint rounding.
They also reproduce the frozen optimum with fresh starting tyres and exercise mid-race snapshots.
The existing agent test still requires its own backend dependencies and model access.

## Snapshot and plan contract

The plan rules shared with the simulator are R1 to R6 in `docs/rules.md`.

The records in `src/baselines/domain.py` are a proposed integration contract.
The team has shared race contracts in `src/contracts.py` and Jason's extracted simulator in `src/sim`.
The baseline records describe the inputs needed for Quan's search and policies, rather than replacing the team's richer race state.
This foundation still needs a bridge to those richer contracts before it can optimize all of Jason's current simulator features.

| Snapshot field | Meaning |
| --- | --- |
| `total_laps` | Scheduled race length. |
| `completed_laps` | Laps completed at the decision boundary, before an optional stop. |
| `current_compound` | Compound currently fitted to the car. |
| `tyre_age` | Completed laps on the tyre set, including earlier use before its current stint. |
| `current_stint_laps` | Laps on this set since the most recent pit stop in this race. |
| `used_compounds` | Compounds already fitted during the race, including the current compound. |
| `available_compounds` | Compounds with usable simulator models for future stints. |
| `min_stint_laps` | Minimum duration of a new planned stint. |
| `pit_loss_s` | Pit time loss used by the legacy adapter. |
| `require_two_compounds` | Whether the complete race must include at least two compounds. |

All records are immutable, and compound names normalize to uppercase.
A plan is an ordered tuple of `Stint` records containing a compound and a remaining lap count.
The first stint always continues the current tyre set at its existing age.
Each later stint starts with a new set at age zero and incurs one future pit stop.
The plan covers exactly the remaining race laps.
Repeating the same compound in adjacent stints represents fitting a fresh set of that compound.

The first stint can contain zero laps when an immediate pit stop is legal.
The current stint must reach the minimum before a future pit stop, including its already completed laps.
Every new stint must meet the minimum, including the last stint.
Continuing an existing stint to the finish requires no minimum number of additional laps.
The two-compound rule counts prior race history, so a late snapshot can finish without another stop.
A finished race has a single zero-lap stint and no future stops.
The contract models dry compounds and unlimited fresh sets, without tyre inventory limits.

Pit laps are absolute race boundaries.
A stop at lap 28 occurs after completing lap 28, before lap 29.
For a snapshot after lap 30, a first remaining stint of five laps means a first pit at lap 35.
`num_stops` counts future stops, not stops already completed.

## Search and stochastic objective

`candidate_plans` enumerates every legal plan with zero, one, or two future stops up to `max_stops`.
It includes immediate stops, repeated compounds, and every integer lap split satisfying the contract.
The search deliberately supports at most two future stops.
With race length L, C compounds, and K seeds, two-stop scoring requires order L squared times C squared times K simulator calls.
The legacy adapter fits each compound once before search and reuses those fits.

`optimize` minimizes the arithmetic mean of remaining simulated time over an explicit seed tuple.
Every candidate and baseline receives exactly the same seeds.
The optimizer chooses one fixed plan before observing stochastic outcomes.
It does not choose the best plan separately for each seed and average those times.
This is an empirical optimum for the supplied seeds, not a guarantee about the entire outcome distribution.

Seeds must be nonnegative unique integers, and at least one seed is required.
The simulator callback receives `(snapshot, plan, seed)` and returns finite nonnegative remaining time in seconds.
It must reset its random state from the supplied seed on every call and must not mutate shared state.
To share safety-car futures across plans, event generation must depend on race state and seed rather than candidate order.
An invalid simulator value or exception fails the search instead of silently dropping that candidate.
Use new, held-out seeds to estimate generalization after selecting a plan.

Exact ties prefer fewer future stops, then earlier absolute pit laps, then alphabetical compound order.
The returned `OptimizationResult` contains the selected `Evaluation` and the number of candidates scored.
`Evaluation` exposes the plan, seeds, individual times, mean, population standard deviation, and sample standard error.
A single-seed result reports zero standard error and provides no estimate of stochastic uncertainty.
Search comparisons use unrounded times returned by the simulator.

`Evaluation.to_dict(snapshot)` produces the shared result fields for Manya's evaluator.
The fields include `stint_plan`, `num_stops`, `pit_laps`, `first_pit_lap`, `expected_remaining_time_s`, `seeds`, and `seed_times_s`.
The command adds uncertainty fields, candidate count, and each baseline's regret against the optimizer on those same seeds.
Unavailable policies retain a reason and do not receive a numerical score.
`NoFeasiblePlan` means the allowed search space has no legal plan.
`PolicyUnavailable` means a particular reference policy cannot apply to that snapshot.

## Reference policies

| Policy | Fixed behavior |
| --- | --- |
| Actual strategy | Crop the driver's recorded full-race stints at the snapshot boundary and retain every remaining recorded stop and compound. |
| Fixed one-stop | Split remaining laps as evenly as legality allows, then switch to HARD, MEDIUM, or SOFT in that preference order, choosing a different available compound. |
| Fixed two-stop | Split remaining laps as evenly as legality allows, switch to MEDIUM, HARD, or SOFT in that order, then return to the snapshot's current compound on a fresh set. |
| Greedy crossover | At each lap boundary, compare next-lap pace on the current tyre against the fastest fresh different compound, and pit when degradation makes the current tyre slower. |

The fixed policies specify one or two future stops relative to the snapshot.
They never search simulated total times.
If the exact stop count cannot fit, they report unavailable rather than changing stop count.
If only one compound is available, a fresh set of that compound is permitted when past race history already meets the compound rule.
Unknown compound names fall back to alphabetical preference after the named dry compounds.

The greedy rule accepts a deterministic expected-pace callback `(compound, tyre_age)` and an optional nonnegative crossover margin.
It observes the minimum stint length and the future-stop budget.
It forces a required compound switch at the last legal boundary when necessary.
It compares lap pace without amortizing pit loss, which deliberately makes it a simple reference policy.
The optimizer alone considers total remaining simulated time across all legal plans.

The actual-strategy policy requires full-race coverage and a matching observed compound history and current stint length.
It reports unavailable for counterfactual snapshots whose history differs from the actual driver.
Actual stints that violate the simulator's minimum duration also report unavailable, without moving recorded stops.

## Teammate integration

Sarah supplies completed laps, current tyre age, current stint length, and compounds already used.
Map those values into the proposed snapshot and keep safety-car history and other simulator-specific fields in the team's richer snapshot.

Jason supplies a callback that translates the remaining plan into his simulator's plan format.
The callback can retain the richer snapshot in its closure while accepting the small search snapshot.
Keep past elapsed time outside the score because every candidate starts at the same decision boundary.
Do not include completed pit losses again.
The existing adapter preserves the original linear tyre model and its per-stint and total rounding.
The extracted simulator computes unrounded totals, so its current tool wrapper can differ by a tenth of a second from legacy per-stint rounding.
The adapter reads actual tyre age from cached timing data, so used starting tyres can differ from the older fresh-tyre ground truth.
These fitted models use the full historical race and are a demonstration adapter, not evidence of leakage-free forecasting or physical calibration.

Manya can import `optimize`, `score_plan`, and the four policies from `src.baselines`.
Use the same snapshot, simulator version, legality constraints, future-stop limit, and seed tuple for all compared plans.
Record the scenario-set version and simulator revision alongside results in the evaluation harness.
The command labels its simulator as deterministic even when multiple seeds are supplied.
Stochastic search is verified with synthetic seeded simulators.
Direct use of Jason's simulator, tyre inventory, and shared snapshots remains an integration step.

The joint definition of stochastic optimality remains a team decision.
This implementation proposes expected remaining time on a fixed shared seed set.
The sprint-three rival strategist is not implemented because its scope remains conditional in the team plan.
