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

**R8 Stint limit: RETIRED.** R8 said a stint may end at most at the oldest tyre age any car ran
on its compound in the whole race plus 3 laps. That number is computed from laps after the
decision snapshot, and it was handed to every policy: the reference and the belief optimizer
searched inside it, the continuation of an agent decision, the limit-aware baselines and the
forced stops of `run_to_flag` / `one_stop_total` planned around it, `simulate_batch` rejected a
plan past it, and the agent's tools returned an error stating it. It was a hindsight leak, and
it is no longer a rule: no rule a policy sees may depend on a lap after the snapshot
(docs/eval.md "Information contract").

What replaced it is physics, not a rule: **the wear cliff** (docs/eval.md "The wear cliff",
`src/sim/cliff.py`). Past a compound's cliff age every lap costs `penalty_s_per_lap2 x (age -
cliff age)` more, so an over-long stint is slow rather than forbidden. The scorer (the world
simulator and the hindsight reference) places the cliff from the whole race; a policy's belief
places it only from what is known at the snapshot, as an estimate with uncertainty, and the
agent's tools report it as such ("longest run seen so far on this compound: N laps; ... wear
beyond that is uncertain"). A plan is `invalid` only for a published-regulation violation (R1-R4,
the laps, the future-stop limit), never for exceeding a tyre life.

The legacy setting reproduces the results produced under R8 exactly:
`SimConfig.stint_limit_margin_laps: 3` with `wear_cliff.enabled: false`
(`configs/sim/legacy/world_r8.yaml` and `configs/sim/legacy/belief_r8.yaml`; default `null`, off).
With it the old machinery runs unchanged: `rules.stint_limits` (rank `stint_limit_rank`, from the
full race frame), `proxy.unraced_limits` for an unraced compound, the `stint_limit` validity rule,
`optimize_batch`'s candidate filter (`diagnostics["stint_limit_dropped"]`,
`["stint_limit_fallback"]`) and the limit-aware baseline wrappers in `src/eval/policies.py`
(`fixed_one_stop` moves its first stop earlier, `greedy_crossover` reads a prohibitive lap time
past the limit). `tests/test_tyre_life_future_blindness.py` shows that this setting is the leak:
with it, scrambling the laps after the snapshot changes what a policy sees.

Not modelled: a race's own mandated maximum stint (2023 and 2025 Qatar: 25 laps per set), a
published regulation that would be a legitimate rule; R8 happened to stand in for it at 2025
Qatar, a calibration race that no scenario set holds.

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
| R8 (retired; legacy setting only) | `stint_limit`, a no-op unless `stint_limit_margin_laps` is set; tyre life is priced by the wear cliff (`cliff.py`, `Simulator.cliffs_for`) | not modelled; the eval wrapper's limit-aware adjustments run only under the legacy setting | `optimize_batch`'s candidate filter, legacy setting only |

`laps_on_set` is the laps since the last recorded stop, capped by the set's age when the
stint history does not add up.
