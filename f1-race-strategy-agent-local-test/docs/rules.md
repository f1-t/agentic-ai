# Plan rules shared by the simulator and the baseline optimizer

The simulator (`src/sim`, rules in `src/sim/rules.py`) and the baseline optimizer
(`src/baselines`) apply the same rules to a remaining-race plan. The bridge between them
is `src/sim/optimizer_bridge.py`. Cite these rules by number.

**R1 Decision point.** A decision is made at the start of lap `laps_completed + 1`. A stop
happens at the end of a lap. The first stint of any plan has at least one lap; an
"immediate stop" is a snapshot taken one lap earlier.

**R2 Minimum stint.** The minimum counts the laps already run on the set in this race. A
stop may be taken only when `current_stint_laps + first.laps >= min`; every later stint
has at least `min` laps; a first stint that runs to the flag has no minimum. A stint on
either side of a change of tyre class (for example inters to slicks) is exempt
(class-change waiver). The value is `RaceParams.min_stint_laps`: default 3, set per race
by the race registry (the profile's `sim.min_stint` overrides it for a whole run). Real
races have 3-lap stints, and the simulator prices short stints honestly, so the rule no
longer needs a 10-lap floor to keep plans sensible. The agent's tools read it from the
snapshot (`race.min_stint_laps`, shown to the agent as `RaceSnapshot.min_stint_laps`),
not from a tool-layer constant (src/sim/agent_tools.py, SPEC 0015). `src/legacy_tools.py`,
the frozen baseline, pins 10. A scenario set keeps the value inside its snapshots: v1
carries 10 and keeps scoring under 10; sets sampled from now on carry 3.

**R3 Compounds.** A dry race uses at least two dry compounds over the whole race, history
included. The rule is waived when a non-slick class appears.

**R4 Sets.** A new set has `start_age` 0; a used set carries its `laps_run` as
`start_age`. With an inventory, each later stint draws a distinct available set. The
scorer substitutes the least-used available set of the compound when no new one exists,
and records the substitution.

**R5 Physics.** Pit loss, safety-car factors, the lap-1 extra and tyre wear come from the
simulator and its fit window, never from the snapshot record. The Milestone's optimizer
uses the `pre_race` window.

**R6 Randomness.** Sample `i` of seed `s` is `numpy.random.default_rng([s, i])` (the main
stream) for every candidate; its SC/VSC scenario comes from the events substream
`default_rng([s, i, 1])`. The degradation parameter draw (`degradation.params.draw:
posterior`) and the residual lap noise (`lap_noise: residual`) each come from one generator
per batch call, `default_rng([s, K, 2])` and `default_rng([s, K, 3])` with K = 2^32 - 1
(`engine.STREAM_KEY`, never a sample index, so these keys never equal a main or event key):
an (n, d) matrix of normals, d the parameter count or the remaining laps, whose row `i` is
sample `i`'s draw. Rows are drawn in order, so row `i` depends on (s, i, d) only and sample
`i` is the same in a batch of 10 or 200 (`engine.batch_normals`). Every draw is shared by
every candidate, and turning one on leaves the other streams' draws as they were. Under a
wear bound (`degradation.params.wear_lower_bound`) a rejected parameter proposal is
followed by the next one from `default_rng([s, K, 4])`, row `i` holding sample `i`'s 49
further proposals, so a sample whose first proposal is accepted draws what it draws
without the bound. A degradation model without a parameter vector gets `default_rng([s, i, 2])` per sample in
its `draw(rng)`. The baseline optimizer's seed tuple `(0, ..., n-1)` and the simulator's
`(seed=0, n_samples=n)` describe the same draws. Seed mapping in the bridge:
`optimize_batch(seed=s, n_samples=n)` reports `Evaluation.seeds = (0, ..., n-1)`, the
sample indices, and records the batch seed `s` in the result's `diagnostics["seed"]`;
sample `i` of those times was drawn from `default_rng([s, i])`, its events from
`default_rng([s, i, 1])`. With `s = 0` the
evaluation's seeds are exactly the baseline optimizer's seed tuple. Belief-side simulations
(the agent's tools, the continuation of a decision in `eval.decisions.to_plan`) run at
`s + agent_tools.BELIEF_SEED_OFFSET` (1 000 003), so the belief never plans on the samples
the world scores with seed `s`.

**R7 Ties.** Ties on mean time break by fewer stops, then earlier pit laps, then the
compound tuple (Quan's key). The additive model scores stint orders identically (on
Spain, S28/S28/M10 and S28/M10/S28 tie exactly), so this is the convention that makes the
optimizer return one plan.

**R8 Stint limit.** Per compound, a stint ends at most at the oldest tyre age any car ran on
that compound in this race plus `SimConfig.stint_limit_margin_laps` laps (default 3; `null`
turns R8 off). `SimConfig.stint_limit_rank` (default 1) picks which car sets the age: the
limit is the age at least that many cars reached, the rank-th largest of the per-car maximum
ages on the compound, so 1 is the oldest tyre any car ran and 3 the age three cars reached
(2023 Spain: HARD 50 at rank 1, 32 at rank 3). A compound fewer cars ran than the rank takes
its smallest per-car maximum; a rank above 1 needs the frame's driver column. The end age is `tyre_age + laps` for the current set and `start_age + laps`
for a later one. The limits are computed once per race from the full race lap frame
(`rules.stint_limits`, `Simulator.stint_limits`, every lap counted, clean or not), which the
belief and the world simulators are both built on, so both apply the same limits whatever
their fit window. A compound no car ran in the race has no limit. There is no exemption for
a degradation model with a hinge: past the last observed age a hinge is still a straight
line. If the current set cannot keep inside its limit (it cannot reach its soonest legal
stop, R1 and R2, inside the limit, or no legal stop fits in the laps left and the flag is
past the limit), the first stint may run to that soonest stop, or to the flag when no legal
stop fits; `optimize_batch` records this
as `diagnostics["stint_limit_fallback"]`. The limit is a hindsight race constant, in the
same class as the observed pit loss: it reads the whole race, including laps after the
snapshot. An as-of source (earlier seasons at the circuit, Pirelli guidance) can replace
it later without changing where the rule is enforced.

## Where each rule lives

| Rule | Simulator | Baseline optimizer | Bridge |
| --- | --- | --- | --- |
| R1 | plan stints have at least one lap (`contracts.PlanStint`) | `candidate_plans` can still emit a zero-lap first stint | `optimize_batch` filters those plans; `to_contract` rejects them |
| R2 | `min_stint`, laps on the set from `rules.laps_on_set` | `validate_plan` | `from_contract` sets `current_stint_laps = laps_on_set(snapshot)` |
| R3 | `two_compounds` | `validate_plan` with `require_two_compounds` | `from_contract` clears `require_two_compounds` after a non-slick stint |
| R4 | `inventory`, `apply_inventory` | no inventory (fresh sets) | `to_contract` assigns the sets |
| R5 | `Simulator`, `SimConfig` | `Snapshot.pit_loss_s` is a legacy field | `optimize_batch` scores with the simulator only |
| R6 | `simulate_batch` | `optimize(seeds=...)` | `optimize_batch` reports `seeds = (0, ..., n-1)`, batch seed in `diagnostics["seed"]` |
| R7 | `compare` ranks by mean only (`best` is the lowest mean) | `optimize`'s tie-break key | `optimize_batch` uses the same key (`_key`) |
| R8 | `stint_limit` (a validity rule, as R1-R4: a plan past a limit is invalid, with an error the agent's tools return) | not modelled; the eval marks a baseline plan past a limit unavailable (`eval.policies`) | `optimize_batch` filters the candidates, counted in `diagnostics["stint_limit_dropped"]` |

`laps_on_set` is the laps since the last recorded stop, capped by the set's age when the
stint history does not add up.
