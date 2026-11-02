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
not from a tool-layer constant (SPEC 0015 points the tools at it). `src/legacy_tools.py`,
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
`default_rng([s, i, 1])`. The baseline optimizer's seed tuple `(0, ..., n-1)` and the simulator's
`(seed=0, n_samples=n)` describe the same draws. Seed mapping in the bridge:
`optimize_batch(seed=s, n_samples=n)` reports `Evaluation.seeds = (0, ..., n-1)`, the
sample indices, and records the batch seed `s` in the result's `diagnostics["seed"]`;
sample `i` of those times was drawn from `default_rng([s, i])`, its events from
`default_rng([s, i, 1])`. With `s = 0` the
evaluation's seeds are exactly the baseline optimizer's seed tuple.

**R7 Ties.** Ties on mean time break by fewer stops, then earlier pit laps, then the
compound tuple (Quan's key). The additive model scores stint orders identically (on
Spain, S28/S28/M10 and S28/M10/S28 tie exactly), so this is the convention that makes the
optimizer return one plan.

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

`laps_on_set` is the laps since the last recorded stop, capped by the set's age when the
stint history does not add up.
