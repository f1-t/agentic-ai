# Evaluation harness

`src/eval` scores any strategy policy against the optimizer from the same snapshot, over a
fixed, versioned set of decision points. No model is needed except for the `agent` policy.

```bash
python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json
python -m src.eval reference --scenarios data/scenarios/v1.json      # optional: cache the references
python -m src.eval score --scenarios data/scenarios/v1.json --policies baselines \
    --recorded data/eval/baseline_agent_sample_run.json --out reports/eval/v1_baselines.csv
python -m src.eval report reports/eval/v1_baselines.csv
```

`score` writes one row per (scenario, policy) to `--out` and the summary table beside it
(`*_summary.csv`), and prints the table. `--policies agent` adds the live agent through
whatever `LLM_BACKEND` is configured (see "The agent policy"). `--belief-config` and
`--world-config` (alias `--config`) choose the two simulators below; `--world-configs a b`
is the optional model-disagreement flag. `--max-stops 0` with the agent or a recorded run
in the policies warns (`NoStopReferenceWarning`: "agent decisions are compared against a
no-stop reference"), so that comparison is a deliberate choice.

## Who sees what

Each race has two simulators over the same lap frame and practice
(`score.race_simulators(profile, belief_config, world_config)` gives
`sim_for(race_id, role)`, role `belief` or `world`):

| role | default config | fit | used by |
| --- | --- | --- | --- |
| belief | `configs/sim/belief.yaml` | `to_snapshot`: race laps up to the snapshot, shrunk toward the practice prior; traffic off; SC and VSC periods from the rate model (EventModel defaults, provisional, Spain-derived); samples carry posterior parameter draws and residual lap noise, drawn from seed + `agent_tools.BELIEF_SEED_OFFSET` so the belief never plans on the world's samples (docs/rules.md R6) | every policy: the baselines plan on it through the bridge (`from_contract(snapshot, belief)`, `pace_for_crossover(belief, snapshot)`), and agent decisions are expanded on it |
| world | the profile's SimConfig | `full`: every race lap (hindsight); traffic off | the reference `optimize_batch` and every scoring call |

A policy is called as `policy(scenario, belief, max_stops)` and never sees the world. The
regret definition is unchanged: the policy's plan and the world optimizer's plan, both
scored by the world. With the default world, v1 stays comparable with the numbers before
the split. `configs/sim/world.yaml` is the world with ghost traffic on, for position regret
(below); pass it with `--world-config`. Ghost traffic replays the recorded field, so it
belongs to the world only, and only under the recorded events: an event model other than
`none` (or a fixed scenario other than the record) adds `traffic_recorded_with_drawn_events`
to every plan's support warnings (docs/traffic.md, limit 5).

Rows carry `belief_config_hash`, `world_config_hash` and `belief_window`. The config hash
leaves `fit_window` out, so the two hashes are equal under the defaults and
`belief_window` says which window the policies planned on.

## Decisions

The agent submits one decision at a snapshot, not a whole plan: PIT with a compound, or
STAY_OUT with `reassess_after_laps`. A policy returns it as `Decision(None,
submission=...)`, and `eval.decisions.to_plan(snapshot, decision, belief, max_stops)`
turns it into the plan the world scores:

- PIT c is `{"action": "pit", "compound": c}`, a stop at the end of this lap;
- STAY_OUT with `reassess_after_laps` n is `{"action": "stay", "laps": n}`;
- the continuation from the snapshot after the action is
  `optimizer_bridge.optimize_batch(advanced, belief, stops left)`, on the belief's fit at
  the decision snapshot, so no lap after the decision enters it;
- `max_stops` counts future stops from the decision snapshot, as the reference optimizer
  does: after a PIT the continuation gets `max_stops - 1` (at least 0), after a STAY_OUT
  `max_stops`. Every plan from a decision lies inside the space the reference searched, so
  an agent row with negative regret (beyond 1e-6 s) is a bug, and `summarize` warns
  (`AgentRegretWarning`). Baselines keep their exception: an actual three-stop can beat a
  two-stop reference. The R8 stint limit (docs/rules.md) binds every plan alike: the
  reference and the continuation search only inside it, and the world rejects a plan past
  it as invalid, a baseline's as an agent's. The eval wrapper (`eval.policies._bridged`, not
  `src/baselines`, which is unchanged) makes `fixed_one_stop` and `greedy_crossover`
  limit-aware first (docs/rules.md R8): fixed_one_stop moves its first stop to the last
  legal lap, greedy_crossover takes a forced stop on the lap its set would pass the limit;

- `Simulator.expand_action` joins the two. A stay of n at least the laps left runs the
  current set to the flag.

A submission that cannot become a legal plan (no compound, a compound the belief does not
model, n below 1, no legal continuation) is an invalid row.

## The belief_optimizer policy

`--policies baselines belief_optimizer agent` adds a baseline with no model call: at each
snapshot it takes the plan `optimizer_bridge.optimize_batch` picks on the belief simulator (the
one the agent's tools use, fitted on practice plus race laps up to the snapshot), with no
decision imposed, and the world scores that plan like any other. It answers what the language
model adds over running the optimizer on what it can see. It is planned by
`eval.decisions.belief_optimum`, with the same belief fit, seed (seed + `BELIEF_SEED_OFFSET`),
samples and future-stop limit as the agent continuation in `to_plan`. It follows every rule
(R1-R8); a plan the world rejects is an `invalid` row with the reason. Rows carry the usual
plan, stops, first pit, regret and R8 stamps; the agent-only columns (`decision`, `p_pit`,
tool calls, tokens) are blank. It is not in the `baselines` group, so default commands and
existing result files are unchanged; select it by name.

## Proxied compounds

A strategy is not banned because nobody ran it. The world's full-window fit models a compound
only when it has `min_samples` (5) clean race laps (`degradation._fuel_fit` keeps
`_usable` compounds; the rest go to `fit_report["unmodelled"]` with the lap count), and it
never reads practice: the practice prior enters only the belief's `to_snapshot` and
`pre_race` fits (the shrinkage prior of `_shrunk`, whose compounds are the prior's plus the
raced ones). So at 2022 Saudi Arabia, where SOFT was run in practice but never raced, the
belief priced SOFT and the world did not: every plan with a later SOFT stint was "Unknown
compound 'SOFT'" (20 of the 21 invalid agent rows of dev6 pass 1). With
`SimConfig.compound_proxy.enabled` (the default; `false` restores the simulator as it was,
bit for bit) every fit prices HARD, MEDIUM and SOFT, each parameter from the best tier that
has data (`src/sim/proxy.py`):

| tier | source | pace (fresh base) | wear | prior sd floor (pace, wear) |
| --- | --- | --- | --- | --- |
| 1 | race laps, `min_samples` clean rows (belief: up to its calibration point) | the fit, unchanged | the fit | the fit's own |
| 2 | practice long runs this weekend (FP sessions, plus the Sprint on a sprint weekend) | anchor + (practice base_c - practice base_anchor) | anchor + (practice wear_c - practice wear_anchor) | 0.5 s, 0.03 s/lap |
| 3 | qualifying this weekend | anchor + the qualifying gap (below) | none | 0.75 s |
| 4 | same circuit, other seasons | not built (`proxy._tier4_pace`, `_tier4_wear` are the extension point) | | |
| 5 | cross-race neighbour step (`configs/sim/compound_steps.json`) | anchor x (1 + pace_frac) per step | anchor x wear_ratio per step | 1.0 s and at least the step's spread, 0.045 s/lap and at least the spread |

**Anchoring.** Every borrowed number is a step from an anchor compound with tier-1 data in
that simulator (the nearest in SOFT < MEDIUM < HARD; ties to the one with more race rows),
so the race sets the level. Tier 2 also needs the anchor in the practice fit, tier 3 in
qualifying; otherwise the next tier is tried. A fit with no tier-1 compound (the belief
before its first race laps, `pre_race`) keeps its practice compounds and anchors the missing
ones on them. The belief's tier 1 uses only the laps up to its calibration point, and
tiers 2, 3 and 5 are known before the race, so the belief gains no hindsight; a belief
compound with fewer than 5 race rows by then is re-priced from practice relative to a raced
one (before, it stayed at practice plus the config base delta, whatever the race said).
Tier-2 wear is a difference, not a ratio: an error in the fixed practice fuel prior biases
every compound's practice wear by the same amount, which the difference cancels, and a
practice wear can be negative.

**The qualifying correction (tier 3).** A qualifying lap is a low-fuel push lap on an evolving
track, so it is never compared with a race lap directly. Each driver's best timed lap per
(segment Q1/Q2/Q3, compound) (`data.qualifying_frame`: a lap time, no pit in or out, a dry
compound, track status green) is one row of a least-squares fit lap = driver + segment +
compound, slow rows trimmed against the fit as `data.robust_filter` trims practice (2.5
robust sd, two passes; fast rows never cut). Only the compound term is used, as the gap
q_c - q_anchor added to the anchor's race base. In that gap fuel cancels (every qualifying
lap carries a few laps of fuel, so the race fit's fuel slope is not needed: the race base the
gap is added to already carries the race's fuel level); car and driver pace cancel (the
driver term); track evolution and engine modes between segments cancel (the segment term,
which takes the pre-2020 Q3 modes too); evolution within a segment does not, nor does push
against managed driving: in the race nobody drives a qualifying lap, so a qualifying gap is a
push-lap gap. A compound enters only with at least 2 drivers who also set a row on another
compound. When the anchor (the raced compound) has no qualifying laps there is nothing in
qualifying to tie it to, and tier 3 is not used: pace falls to tier 5. In practice HARD is
never run in qualifying, so tier 3 can price only SOFT against MEDIUM or MEDIUM against
SOFT, and at 2022 Saudi Arabia (2 drivers on MEDIUM, one of them trimmed) it prices nothing.

**The tier-5 table** (`python -m src.sim.compound_steps estimate`; calibration races only,
the four held-out races never loaded; the per-race values and the command are in the file).
Per race, the full-window world fit with the proxy off; a pair counts when both compounds
are tier 1; the value is the median over races, sd the sample sd across them:

| pair | pace_frac (softer / harder base - 1) | wear_ratio (softer / harder) | life_ratio (R8 limits) |
| --- | --- | --- | --- |
| MEDIUM-HARD | -0.0035, sd 0.0038, n 6 (about -0.33 s on a 95 s lap) | 1.53, sd 0.85, n 6 | 0.64, sd 0.26, n 5 |
| SOFT-MEDIUM | +0.0019, sd 0.0061, n 6 (SOFT about 0.17 s slower) | 0.60, sd 0.88, n 6 | 0.76, sd 0.88, n 5 |

Races: 2019 United Kingdom, 2022 Saudi Arabia (MEDIUM-HARD only), 2023 Bahrain, 2023 Spain,
2024 Emilia Romagna and 2025 Qatar, and 2021 Spain (SOFT-MEDIUM only); 2025 Qatar is left
out of the life ratio (a mandated 25-lap maximum stint). Compound names are relative to each
race's allocation, and the race fits do not show the textbook order between SOFT and MEDIUM:
in five of six races the fitted SOFT is no faster fresh than MEDIUM (by at most 1.1 s), and
in four of six it wears less. So a tier-5 SOFT is, after the ordering constraint, a copy of its MEDIUM anchor.
That is what the race data say, not a tuning choice.

**Ordering.** A softer compound is never slower fresh and never wears less than a harder one.
The constraint moves proxied values only, never a tier-1 one; where the tier-1 values are
themselves out of order the proxied value is put between them ("conflict"). A proxied wear is
also floored at 0 (and at the belief's wear bound): a tyre does not get faster with age. This
floor is the one constraint not in the original design; without it, tier 2 gave HARD a wear
of -0.15 s/lap at 2019 United Kingdom and 2022 Saudi Arabia (practice HARD long runs that
got faster).

**Uncertainty.** A proxied parameter is g x anchor + a step, so `fit_report["cov"]` carries it
as T cov T' + its tier's tau^2: posterior draws move it with its anchor plus its own spread,
and its shrinkage weight is 0 (model report source "prior"). tau is the tier's floor above,
or the tier's own estimate when larger (the practice fit's sd of the difference, the
qualifying fit's sd of the gap, a tier-5 step's spread across races times the anchor).

**Reporting.** `fit_report["proxy"]` holds, per compound, `pace_tier`, `pace_source`,
`pace_anchor`, `wear_tier`, `wear_source`, `wear_anchor`, `ordering` and the step. The model
report flags `pace_from_<source>:<compound>` and `wear_from_<source>:<compound>`, and the
agent's model report and `estimate_tyre_degradation` carry `pace_source`, `wear_source` and
`price_basis` ("SOFT: pace from practice long runs (anchor MEDIUM), wear from practice long
runs (anchor MEDIUM)"). Score rows carry `proxied_compounds` (the plan's compounds the world
prices by proxy, ";"-joined, empty for none) and `opt_proxied_compounds` (the reference's),
only when the world's proxy is on. `calibrators.hindsight_targets` leaves a proxied compound
out: it has no race laps to be a target.

**R8.** A compound no car raced gets a stint limit: its oldest practice tyre age, else the
nearest raced compound's limit times the life ratio, plus the margin (docs/rules.md R8). On
the calibration races: 2021 Spain HARD, 18 laps (FP1, LEC; longest practice run 13 laps),
cap 21; 2022 Saudi Arabia SOFT, 18 laps (FP1, GAS; longest run 12 laps), cap 21.

### Validation

`python -m src.sim.compound_steps validate` (calibration races only; outputs under
`reports/calibration/compound_proxy/`). For each calibration race and each compound that was
raced, its race laps are hidden (its rows dropped from the race frame), the world refitted,
and the compound priced again through one tier at a time (tier 5 leave-one-race-out: the
table without that race), then compared with the real tier-1 fit: the fresh pace at
mid-race, the wear, and the time of a 15- and a 25-lap stint on a new set centred on
mid-race. Errors are proxy minus truth (negative: the proxy is faster):

| tier | n | fresh pace, s: mean / MAE | wear, s/lap per lap: mean / MAE | 15-lap stint, s: mean / MAE | 25-lap stint, s: mean / MAE |
| --- | --- | --- | --- | --- | --- |
| 2 practice | 16 | +0.18 / 1.06 | +0.020 / 0.067 | +5.1 / 10.0 | +10.9 / 17.7 |
| 3 qualifying (pace only) | 6 | -0.15 / 0.85 | | | |
| 5 cross-race step, leave-one-race-out | 19 | +0.03 / 0.38 | +0.004 / 0.043 | +0.9 / 4.4 | +2.0 / 10.8 |
| pace from 3, wear from 5 | 6 | -0.15 / 0.85 | +0.018 / 0.032 | -0.1 / 9.0 | +2.0 / 11.1 |
| the chain as built (best tier available) | 19 | +0.06 / 0.99 | +0.019 / 0.064 | +3.3 / 9.6 | +7.8 / 16.4 |

n counts the raced compounds a tier could price (19 in all: tier 2 needs a practice long run
on the compound and its anchor, tier 3 qualifying laps on both, which only SOFT and MEDIUM
ever have). On the same compounds, tier 5 is better than tier 2 (16 compounds: fresh-pace MAE
0.36 against 1.06 s, wear 0.043 against 0.067, 25-lap stint 11.3 against 17.7 s) and better
than tier 3 (6 compounds: 0.38 against 0.85 s), and tier 3 is better than tier 2 on pace. So
**the tier order of the design (practice before qualifying before the cross-race step) is the
reverse of the order of accuracy here**, and the chain inherits tier 2's errors. Part of tier
5's edge is the ordering constraint: its steps are small, so after clipping a tier-5 compound
is often its anchor's copy, and the race fits put neighbouring compounds close together.
Tier 2's large errors come from practice long runs that are not race-like: HARD at 2019
United Kingdom and 2022 Saudi Arabia got faster in practice (wear floored at 0, pace off by
+3.6 and +3.1 s), SOFT at 2023 Bahrain was 2.9 s too fast fresh and wore 0.32 s/lap per lap
too much. Tier 3 has a consistent sign: qualifying makes SOFT about 1 s faster than MEDIUM
(SOFT hidden: -1.0 s on average; MEDIUM hidden: +0.7 s), where the race fits put them within
0.3 s: the push-lap gap is wider than the race gap. These numbers were computed once with the
design as specified; the tiers were not tuned to them, and the order has not been changed
(it would be a choice made on the validation set itself).

Decision level: for each race, each raced compound hidden in turn and one lap-0 snapshot
per starting compound (a finisher's), the reference (`optimize_batch`, seed 0, 50 samples,
two stops) on the world with that compound hidden and proxied by the chain, against the
reference on the true world (`validation_decisions.csv`). Cost is the true world's time of
the proxied world's plan less the true optimum:

| hidden compound priced by | rows | plan changed | invalid in the true world | cost, s (valid rows): mean / median / max | laps on it: more / fewer / same |
| --- | --- | --- | --- | --- | --- |
| tier 2 (practice) | 38 | 34 | 1 | 2.57 / 0.69 / 23.56 | 7 / 21 / 10 |
| tier 3 pace + tier 5 wear | 3 | 3 | 3 | | 3 / 0 / 0 |
| tier 5 | 6 | 5 | 0 | 4.02 / 3.56 / 9.36 | 2 / 2 / 2 |
| all | 47 | 42 | 4 | 2.77 / 0.69 / 23.56 | 12 / 23 / 12 |

Most plans move (often by a lap or two: the other compounds' fits move too once a compound's
laps are gone). The proxied plan uses the hidden compound more than the true plan in 12 of
47 and less in 23: on these races the search does not systematically pick the proxied
compound for being priced too kindly. The 4 invalid rows put the hidden compound past its
true race limit (2023 Bahrain MEDIUM, raced to 10 laps; 2024 Emilia Romagna SOFT, raced to 8),
because its proxied limit is its practice age. The worst cost (23.6 s, 2022 Saudi Arabia,
HARD hidden) is tier 2 again: practice HARD got faster with age.

### What changed in the references

References rebuilt at seed 0, 50 samples, two stops (`python -m src.eval reference`), against
main's. With the proxy off every record of every set is identical to main's (v1, pilot1, d2,
dev6); with it on, the sets move only where a compound is proxied (2021 Spain: HARD, tier 5
for pace and wear, R8 cap 21; 2022 Saudi Arabia: SOFT, tier 2, R8 cap 21):

| set | references | plan changed | plan on a proxied compound | stops 0 / 1 / 2, main | stops 0 / 1 / 2, proxy on |
| --- | --- | --- | --- | --- | --- |
| v1 (2021 Spain) | 264 | 50 | 50 | 40 / 216 / 8 | 40 / 174 / 50 |
| pilot1 (2021 Spain) | 25 | 7 | 7 | 3 / 22 / 0 | 3 / 15 / 7 |
| d2 (2023 Spain) | 25 | 0 | 0 | 8 / 17 / 0 | 8 / 17 / 0 |
| dev6 (six races) | 150 | 32 | 32 (Spain 7 of 25, Saudi 25 of 25) | 56 / 85 / 9 | 44 / 69 / 37 |

**2022 Saudi Arabia: the proxy is probably too kind there.** Tier 2 prices SOFT 2.57 s faster
fresh than MEDIUM (practice step) with 0.32 s/lap per lap of wear, and every one of the 25
dev6 references now ends on SOFT (one or two stints of 7 to 16 laps), 2 to 15 s faster than
main's plan.
Re-scored in a scratch directory, actual_strategy's mean regret at Saudi moves from 0.72 to
7.71 s (median 0.17 to 6.60 s); nobody raced SOFT there, and the validation above puts tier
2's SOFT about 0.9 s too fast fresh on average. **2021 Spain: HARD wins through the ordering
constraint.** The race fit has SOFT wearing less than MEDIUM (0.033 against 0.068 s/lap per
lap), so a HARD that may not wear more than either softer compound gets SOFT's 0.033 with a
base 0.3 s above MEDIUM's, and references switch a MEDIUM stint for a 21-lap HARD one (gain
about 0.3 s). actual_strategy's mean regret at Spain moves from 8.91 to 9.00 s. Elsewhere the
world prices nothing new and the references are unchanged, but the belief does price more:
before its first race laps it now also models a compound with no practice long run (tier 5),
so baseline plans move at lap 0. At 2023 Spain two actual_strategy rows that were unavailable
are scored and fixed_one_stop and fixed_two_stop take HARD in two scenarios; at 2023 Bahrain
greedy_crossover has 4 unavailable rows turned invalid and 4 plans with MEDIUM for SOFT; at
2024 Emilia Romagna greedy_crossover changes 3 plans, one ok row turning invalid. At 2021
Spain fixed_one_stop takes the proxied HARD and is invalid on 14 rows past its practice cap
(27 invalid fixed_one_stop rows in dev6, from 14). Over dev6 the mean regret moves from 6.97
to 8.32 s (actual_strategy), 8.21 to 10.20 (fixed_one_stop), 24.38 to 25.46 (fixed_two_stop)
and 22.01 to 20.90 s (greedy_crossover). Regret is at least 0 on every scored baseline row. Replaying the dev6 pass-1 agent submissions (their
recorded `submit_recommendation`, no agent run) on the new simulators: Saudi Arabia goes from
21 invalid rows (20 "Unknown compound 'SOFT'") to 1 (a PIT the minimum-stint rule refuses),
and no row anywhere is invalid for an unknown compound.

**Limitation.** A proxied compound's price has no race laps behind it. Its pace and wear are
the anchor's plus a step measured somewhere else (practice, qualifying, other races), with
the errors above; its stint limit is a practice age, not a race one. The optimizer will use a
proxied compound wherever that price says so, and the validation shows it can be wrong in
either direction. Read `proxied_compounds` beside any result on a race with a proxied compound.

## The agent policy

`--policies agent` is `eval.policies.AgentPolicy`, the snapshot loop of `src/agent.py`, on
any scenario of any race (the lap-0 same-race guard is gone):

- the agent's view is `RaceSnapshot.from_contract(scenario.snapshot)` (`src/snapshot.py`):
  lap, tyre, gaps, `field.nearby`, `recent_lap_times_s`, weather, `race.min_stint_laps`
  and the inventory-legal compounds; the stint history and the sets stay out of it;
- the tools keep the contract itself (`tools.set_snapshot`) and answer from the belief
  simulator at the snapshot (`src/sim/agent_tools.py`, SPEC 0015): degradation and pit
  loss ranges, the model report, plan comparisons, and an action against its alternative,
  with `within_sim_error` judged against the calibration error; traffic is not modelled
  there;
- `run_agent(scenario.snapshot, verbose=False, max_steps=profile.agent.max_steps, log_path=...)` logs
  the trajectory to `reports/eval/trajectories/<set version>/<scenario_id>.jsonl` (":" in
  the id written "_"; a rerun replaces the file);
- the submission is scored as a decision (above); `tool_calls` is the trajectory length
  (every attempted call: refused ones, repeats and submission attempts included),
  `evidence_calls` the evidence calls that actually ran (no submissions, no calls refused
  for the budget or in a grace turn, no repeats served from the cache; a repeat that was
  run again counts, since it ran), `repeat_calls` the evidence calls identical to an
  earlier one in the episode (same tool and arguments), whether run again or served from
  the cache, so looping stays measurable in either repeat mode (a repeat refused for the
  budget or in a grace turn is not counted), `input_tokens` and `output_tokens` the
  backend's reported usage (blank when it reports none); a backend exception is an `error`
  row, not a crash;
- `AGENT_REPEAT_MODE` (also `run_agent(repeat_mode=...)`) says what a repeat gets: `run`
  (the default) runs it again and counts it against the evidence budget, as before #72;
  `cache` gives it the earlier call's cached result with `repeat_of` added and uses no
  budget, also once the budget is spent (a grace turn still refuses it). Any other value
  is an error. In both modes the trajectory event carries `repeat_of` (the earlier
  call's `call_index`); in `run` mode the model's tool result does not;
- `AGENT_TURN_STATUS` (`1`, `true` or `on`; also `run_agent(turn_status=...)`) ends each
  evidence turn's tool-results message with "Turn s of N. Evidence calls used u of B."
  and, on turn N-1, "One evidence turn is left after this one." It is off by default:
  the tool-results messages are then what they were before #72;
- each trajectory event has a `phase`: `evidence` for the `max_steps` turns, `grace` for
  the up to `AGENT_SUBMIT_GRACE_STEPS` (default 2) turns that follow when no submission was
  accepted, which offer only `submit_recommendation`; `submitted_in_grace` marks a
  submission made there. The grace turns are on whatever the two options above say.

Why those defaults: in live runs on the same 50 scenarios, unprompted submissions went 43
with neither the status line nor repeat handling, 31 with the status line and refused
repeats, and 27 with the status line and cached repeats, while input
tokens per episode went 53k, 63k, 85k. The decisions barely changed. The grace turns are
what guarantee a submission.

`src.grade.grade_decision(snapshot, submission)` grades one submission the same way (valid,
regret against the optimizer from that snapshot) for `tests/test_case.py`, which reports
the regret and does not assert it.

## Scenario sets

A scenario is one (race, driver, lap) snapshot plus the driver's actual full-race stints. A
set is sampled once and written with every snapshot in full, so changes to the snapshot
builder, the cache or the lap filter never move a published set. The file carries a sha256
of its items and `ScenarioSet.load` refuses a file whose items no longer match: build a new
version instead of editing one.

**v1** (`data/scenarios/v1.json`, 264 scenarios): 2021 Spain; the 19 drivers with laps 1..n
and at most two laps down (Tsunoda's DNF is out); the decision laps of
`calibration.default_laps_k` (every 5th lap from 0, plus the laps either side of each SC and
VSC period) that leave at least one minimum stint to run. Snapshots come from Jason's
stand-in builder `src.sim.calibration.snapshot_at`.

v1 snapshots carry `race.min_stint_laps` 10 (the rule before R2 became 3), so v1 keeps
scoring under 10: the committed split table reproduces row for row. Sets sampled from now
on carry the registry's value, 3 by default (docs/rules.md R2).

Next versions: v2 once Sarah's snapshot builder and the new races land (pass it as
`builder=` to `build_set`; it has the same signature as `snapshot_at`).

### Development and report sets

`data/eval_set.json` is the registered eval split, written by `register_eval_set` (Sarah's
item): the races held out for reporting, `{"races": [...], ...}` or a bare list of ids.
While it exists, `build_set` (and `scenarios`, `--eval-set` for another file) refuses any
race outside it, records the file's sha256 as `sampling["eval_set_digest"]`, and marks the
set `development: false`. Without it a set is `development: true`.

- **v1 is the development set**: sampled before the split, from the race we tuned on.
  Its numbers are for debugging the harness and the baselines, not for the report.
- **A set sampled from the eval races is the report set**: the numbers the report quotes.
  Today that is d4, still to be built (see "The registered split" below).

`scenarios --development` (`build_set(..., development=True)`) still samples a development
set once the split exists, a Spain debug set say: no race is refused, the set is marked
`development: true`, and `sampling["eval_set_digest"]` is recorded anyway so the set shows
the split existed. Report numbers come only from sets with `development: false`; `score`
prints one line naming which kind of set it scored.

### The registered split

The cache holds 2021 Spain plus a 2018-25 sample (dataset 3): the 2023 and 2024 races already
there and eight drawn with `random.seed(2026)` from the 173 races of 2018-25, 14 race weekends
in all. Of these, 2024 Monaco (lap-1 red flag), 2024 Britain (wet) and 2024 Sao Paulo (wet and
red flag) are not eligible.

`data/eval_set.json` holds out four races: **2019 Monaco, 2020 Sakhir, 2025 Austria and 2024
Austria**. Nothing loads them until the final run: no exploratory commands, no fits, no test
that asserts on their content. The eligibility record for 2024 Austria below is the one
exception.

| race id | role |
|---|---|
| 2019_06_monaco, 2020_16_sakhir, 2025_11_austria, 2024_11_austria | held out (`data/eval_set.json`) |
| 2021_04_spain | development and calibration: the simulator was tuned on it |
| 2023_07_spain | calibration (held out under the previous split, see below) |
| 2023_01_bahrain, 2019_10_united_kingdom, 2022_02_saudi_arabia, 2024_07_emilia_romagna | calibration |
| 2025_23_qatar | calibration only, never sampled into an eval scenario set |
| 2024_08_monaco, 2024_12_united_kingdom, 2024_21_brazil | not eligible |

How the four were chosen:

- **Sarah's draw, extended.** The pool is the six newly cached eligible races, in season and
  round order. Sarah's registered draw took one race from it; the same draw with k raised
  from 1 to 3 gives three:

  ```python
  pool = ["2019_06_monaco", "2019_10_united_kingdom", "2020_16_sakhir",
          "2022_02_saudi_arabia", "2024_07_emilia_romagna", "2025_11_austria"]
  random.seed(2026); random.sample(pool, 3)   # ['2019_06_monaco', '2020_16_sakhir', '2025_11_austria']
  ```

  With k=1 the same seed returns `['2019_06_monaco']`, Sarah's draw. 2019 Monaco is kept
  although track position dominates there.
- **2024 Austria is added** because it is the one eligible race no run has ever loaded. It is a
  sprint weekend, so real-run discovery left it out until sprint weekends were supported.
  Eligibility facts only, from the race session, with no model run: dry, no red flag, 19
  eligible drivers, HARD, MEDIUM and SOFT raced, 71 laps.
- **2023 Spain moves to calibration.** It was the held-out report race of d2 and d3, and it was
  used to diagnose the simulator after Milestone 2, so it can no longer be held out.

Disclosure: what each held-out race has been exposed to.

- **2019 Monaco:** references and the four baselines were scored once on Nov 11 under the
  pre-R8 simulator (Sarah's dataset 3 commit 4a8477c; `reports/eval/d3_baselines.csv` has 100
  Monaco rows at sim_code_version b62094155bde and no agent rows) and reported. No agent run.
  No modelling choice used those numbers.
- **2020 Sakhir and 2025 Austria:** two of the seven races in the leave-one-race-out comparison
  of Nov 12 that pointed to driver effects with the hinge and to rule R8, so they informed
  those choices before being held out. No eval scenarios, no baselines, no agent run.
- **2024 Austria:** nothing at all. Eligibility facts only.
- **All four:** the agent has never been run on any of them.

Scenario sets under the split:

- **d2** (`data/scenarios/d2.json`, 25 scenarios of 2023 Spain) and **d3**
  (`data/scenarios/d3.json`, 50 scenarios: 25 each of 2019 Monaco and 2023 Spain, seed 0) were
  sampled under the earlier splits, and each keeps the `eval_set_digest` of the file it was
  sampled under (d2 the one-race split `{"races": ["2023_07_spain"]}`, d3 the two-race split
  of 2019 Monaco and 2023 Spain). Both are records: they still load and are not regenerated.
  d3 is the Milestone-era set and is no longer the report set. Since half of it is a held-out
  race, no new references, baselines or agent runs are made on d3. Its Monaco baseline rows
  stay in the repo as they are, are not regenerated, and are superseded by d4 at the final run.
- **d4** is the report set: the four held-out races, 25 scenarios each, seed 0,
  `development: false`. Its scenario file is built only after sprint-weekend support lands
  (2024 Austria is a sprint weekend), and no references, baselines or agent runs are made on
  it until the final run. d4's Monaco scenarios may coincide with d3's, since both sample 25
  per race at seed 0; the final run will state whether they do.
- **dev6** (`data/scenarios/dev6.json`, 150 scenarios, digest `8823b7b73c87`) is a
  **development set on calibration races, not a held-out result**: 25 scenarios each of the six
  non-sprint calibration races 2021_04_spain, 2023_07_spain, 2023_01_bahrain,
  2019_10_united_kingdom, 2022_02_saudi_arabia and 2024_07_emilia_romagna, seed 0,
  `--development` (`scenarios --races <the six> --version dev6 --max-per-race 25 --seed 0
  --development`). The simulator was calibrated on these races, so its numbers say how the agent
  and the baselines behave where the model fits best; they are not a report-set estimate. Its 25
  2021 Spain scenarios are pilot1's and its 25 2023 Spain scenarios are d2's (same ids and the
  same snapshots, item for item). References: `data/references/dev6/72bebbb40278.parquet`
  (sim_code_version `fc338d72f763`, seed 0, 50 samples, two stops, default world with the
  compound proxy). Pass 1 and `dev6_baselines.csv` were scored on the earlier
  `4ecc62a30937` (sim_code_version `1dcf46ad75aa`), before the compound proxy, where 2022 Saudi
  Arabia had a known issue: the belief modelled SOFT (from the practice prior) but the world,
  fitted on the race laps, modelled only HARD and MEDIUM, so a decision whose belief
  continuation used SOFT was scored invalid with "Unknown compound 'SOFT'" (20 of the agent's
  21 invalid rows on pass 1, and 5 greedy_crossover rows). The proxy prices SOFT in the world
  now ("Proxied compounds"); the committed pass-1 and baseline files are not rescored here.

### Sprint weekends

A sprint weekend (2023 on) has one practice session, FP1, and a Sprint on Saturday. The practice prior
reads the FP sessions a weekend has plus the Sprint (`calibrate.practice_sessions`, from the
cached event schedule): FP1 and the Sprint on a sprint weekend, for both the degradation prior
(`calibrate._practice`) and the pace offsets (`pace.load_practice`, as `cross_race` and the
`calibrate` commands call it). A conventional weekend reads what it always did, FP1-FP3 for
the prior and FP2-FP3 for the pace offsets, and its practice frame is unchanged (checked frame
for frame on 2021 Spain and 2023 Bahrain). No physics module changed, so `SIM_CODE_VERSION`
and the cached references are untouched. Real-run
discovery (`real_run.cached_races`) now lists the three cached sprint weekends, 2024 Austria,
2024 Sao Paulo and 2025 Qatar, where it used to leave them out with a note. Listing is not
loading: 2024 Austria is held out, and 2024 Sao Paulo is not eligible.

- On a sprint weekend both the prior and the pace offsets read FP1 + Sprint: FP1 is the only
  practice session, and the Sprint alone would tie a driver's pace to where they ran in the
  train. This choice was fixed before any sprint race was evaluated.
- 2021-2022 sprint weekends read FP1, FP2 and the Sprint by the same rule. The session
  selection is unit-tested, but no such weekend is cached, so the loaders have not been run on
  one.

**2025 Qatar is used for calibration only and is never sampled into an eval set.** The
weekend had a mandated maximum of 25 laps per tyre set. Race control messages do not state
it (it was set before the event), but the stint data does: no car ran a lap on a tyre older
than 25 laps on any compound in FP1, the Sprint or the race; 29 of the 61 race stints end at
exactly 25 laps of tyre life; 16 cars pitted under the lap-7 safety car and 15 of them pitted
again on lap 32, 25 laps later, to finish the 57 laps on a second 25-lap set. Rules R1-R8 have
no rule for a stint cap, so a recommendation scored there could break a rule the race had.
`build_set` refuses it (`scenarios.CALIBRATION_ONLY`) in every mode, `--development`
included.

## Metrics

For each scenario the optimizer (`optimizer_bridge.optimize_batch`, Quan's search scored by
Jason's simulator) finds the world's best plan with at most `max_stops` future stops. Every
policy's plan is scored by the world under the same draws (docs/rules.md R6).

| column | meaning |
| --- | --- |
| `regret_s` | mean remaining time of the policy's plan minus the optimizer's. A plan with more stops than the optimizer searches can come in below it, so this can be negative. |
| `regret_pct` | `regret_s` as a percentage of the optimizer's remaining time |
| `invalid_rate` | invalid plans / (ok + invalid). Invalid: a rule broken (docs/rules.md) or no plan submitted. |
| `stop_agreement` | share of scored plans with the optimizer's number of future stops |
| `first_pit_error` | \|first pit lap - optimizer's first pit lap\|; 0 when neither stops, blank when only one does |
| `latency_s`, `tool_calls`, `input_tokens`, `output_tokens` | per decision, when the policy reports them. `tool_calls` counts every call the agent attempted, refused calls, repeats and submission attempts included |
| `evidence_calls` | agent rows: the evidence calls that ran (no submissions, no refused calls, no repeats served from the cache; a repeat run again counts); summary `mean_evidence_calls` beside `mean_tool_calls` |
| `repeat_calls` | agent rows: the evidence calls identical to an earlier one in the episode, run again (`AGENT_REPEAT_MODE=run`, the default) or answered with its cached result and no budget (`cache`); blank for other policies and for rows from before this column; summary `mean_repeat_calls` |
| `pit_now_opt` | the world optimizer's first stop is at the end of this lap (`opt_first_pit_lap == laps_completed + 1`) |
| `submitted_in_grace` | agent rows: the submission came in a grace turn, after the evidence turns ran out (blank for other policies) |
| `decision`, `p_pit` | agent rows only: PIT or STAY_OUT, and the stated probability that pitting now is right (`confidence` for PIT, `1 - confidence` for STAY_OUT) |
| `brier` | summary, agent rows: mean of `(p_pit - pit_now_opt)^2`. Only in tables of rows that have `p_pit`. |
| `pit_now_agreement` | summary, agent rows: share of decisions where PIT matches `pit_now_opt` |
| `final_position_mean`, `opt_final_position_mean` | world traffic on: the plan's and the optimizer's mean finishing position over the samples, 1 + the ghosts that finished the last lap before the plan's world time (`traffic.place`); a DNF sample is behind every finisher, ordered by `dnf_lap` against the ghosts that retired |
| `position_regret` | `final_position_mean - opt_final_position_mean`; positive means finishing behind the reference. NaN with world traffic off (the default world) or when traffic cannot be placed (no `elapsed_s`): without ghosts there is no field to finish in |
| `mean_position_regret`, `mean_final_position` | summary, only when some row has a position |
| `ref_models_agree` | optional (`--world-configs`): every alternative world's reference has the same number of stops and a first pit lap within 2 of the world's. `score` then also writes `*_summary_agree.csv`, the table over the agreeing rows |
| `regret_se_s` | the paired standard error of `regret_s` under the shared draws: `sd(policy_times - reference_times) / sqrt(N)`, sd with N - 1, over the N per-draw remaining times (the reference's in float32, from the reference cache). 0 at N = 1 |
| `within_mc` | `\|regret_s\| <= 2 * regret_se_s`: at this N the draws cannot tell the plan from the reference. At N = 1 (or in a world without drawn events, where every draw is the same) the standard error is 0, to rounding, and this is the tie rule, `regret_s` 0 |
| `share_within_mc` | summary: share of scored rows with `within_mc` |
| `opt_set_size` | near ties (information only): the searched plans whose mean is within 2 paired standard errors of the reference's (`gap <= 2 * SE`, the reference included), the optimal set. NaN from a format-1 reference cache |
| `best_pit_now_s`, `best_stay_s` | near ties: the mean of the best searched plan that pits at the end of this lap, and of the best one that does not; NaN where there is no such plan (pitting now would leave a stint under the minimum, say) |
| `pit_now_gap_s`, `pit_now_gap_se_s` | near ties: `best_pit_now_s - best_stay_s` and the paired standard error of that difference over the shared draws |
| `pit_now_decisive` | near ties: `\|pit_now_gap_s\| > 2 * pit_now_gap_se_s`; empty where one of the two plans does not exist |
| `stop_agree_set`, `first_pit_error_set` | near ties, ok rows: `stop_agree` and `first_pit_error` against the closest member of the optimal set (some member makes the same number of stops; the smallest first pit error over the members) |

Row statuses: `ok`, `invalid`, `unavailable` (the policy does not apply here, e.g. the actual
strategy of a driver whose real stint broke the minimum), `error` (the policy itself failed,
e.g. an API error; not counted as invalid), `no_reference` (no legal plan exists).

Each row is stamped with the scenario set version and digest, `belief_config_hash`,
`world_config_hash`, `belief_window`, `sim_code_version`, the seed and `n_samples` (and
`ref_world_config_hashes` with `--world-configs`). Rows scored before the split carry one
`config_hash` instead. Rows also carry the world's R8 settings, `stint_limit_rank` and
`stint_limit_margin_laps` (blank: R8 off), and `summarize` copies both into every summary
row (`score.SUMMARY_STAMPS`; values joined by `;` if rows of two settings are summarized
together), so a rank-1 and a rank-3 table can be shown side by side. Files scored before
the stamps have neither column.

## Reference cache

The reference of a scenario (the world optimizer's plan and its per-draw times) depends only
on the scenario set, the world config, the simulator code, the seed, `n_samples`,
`max_stops` and whether the events are fixed to the record, so it is computed once
(`src/eval/reference.py`):

```bash
python -m src.eval reference --scenarios data/scenarios/v1.json [--world-config W] [--profile P] \
    [--hindsight] [--seed 0] [--n-samples N] [--max-stops K]
python -m src.eval reference --check        # cached references from another sim_code_version
```

**Expected exception: d3.** d3 is rebuilt at the final evaluation. Until then its cached
reference (`data/references/d3/9f933a611bea.parquet`, sim_code_version `b62094155bde`) is
stale on purpose and is neither rebuilt nor rescored. `reference --check` lists it apart,
under "expected stale (docs/eval.md)" with that reason, and does not count it as stale
(`reference.EXPECTED_STALE`). `stale_references` still returns it, so a real run's
references stage rebuilds it when that run is made.

It writes `data/references/<set version>/<key>.parquet`, with the inputs in `<key>.json`
beside it, where `key = sha256(set digest, world config hash, sim_code_version, seed,
n_samples, max_stops, hindsight)[:12]` (`reference.reference_key`; `--n-samples` and
`--max-stops` default to the profile's, as in `score`). Per scenario the table holds the
reference plan, its per-draw remaining times (float32), their mean, sd and `cvar_90`
(mean of the worst 10 percent), `pit_now_opt`, the runner-up's mean (the next plan in the
optimizer's tie-break order) and the paired standard error between the two. Since format 2
(`reference.REFERENCE_FORMAT`, `"format"` in the sidecar) it also holds, in tie-break order,
the searched plans whose gap to the reference is at most max(10 s, 5 x their paired SE)
(`STORE_GAP_S`, `STORE_SE`), plus the best pit-now and best stay-out plans always, so
neither the optimal set (gap <= 2 SE, inside that bound) nor `pit_now_gap_s` depends on the
bound. For each: its mean minus the reference's (`plan_gap_s`) and
its paired standard error against the reference (`plan_se_s`), both float32, its number of
stops and first pit lap, and the paired standard error of the best pit-now plan against the
best stay-out plan. Those give the near-tie columns. A format-1 cache still loads (its
near-tie columns are NaN) and `reference --check` lists it as stale. The command
prints the reference's precision: how far the runner-up is above the reference and whether
the gap is within 2 standard errors.

`score` looks up the reference for its own inputs under `--reference-dir` (default
`data/references`). A hit is used as it is; a miss computes the reference the same way and
prints `reference computed, not cached` once per run. Cached and computed references give
identical rows. With `--require-reference` a miss is an error: report runs use it. The
`jobs` runner reads the cache the same way (`python -m src.eval jobs --reference-dir D
--require-reference ...`); both travel in the runner spec, so on Sol a report run's
workers fail their first chunk on a miss instead of computing. `--hindsight` fixes every draw's SC and VSC laps to
the ones the driver recorded over the remaining laps; `score` never reads a hindsight
reference, since a regret needs both plans under the same draws.

The key carries `sim_code_version`, a physics version: a hash of the modules listed in
`calibration.PHYSICS_MODULES` (engine, degradation, events, hazard, track, traffic, rules,
config, data, tyre_sets, weather, pace, calibrators, optimizer_bridge, registry, event_rates,
fuel_prior, proxy) plus `src/contracts.py`. Every other module of `src/sim` is on
`calibration.NON_PHYSICS` (the package `__init__`, agent_tools, jobs, compute, calibrate,
calibration, model_report, race_ids, cross_race, real_run, compound_steps), and a test fails on a module on neither list, so a new module is classified on
purpose. calibration is non-physics because the scenario sets it builds are frozen by their
digest. A change to a physics module orphans the cache: `reference --check` lists what is
stale, and `reference` rebuilds it; a change to the tooling (agent tools, jobs) does not.
Moving to the allowlist moved every `sim_code_version` stamp once; the committed reports
keep their old stamps (stamp-only drift, reports/calibration/README.md).
`optimizer_bridge.optimize_batch(..., return_times=True)` (off by default) is how the cache
gets the per-draw times: `diagnostics["times"]`, `diagnostics["runner_up"]` and
`diagnostics["searched"]` (every legal plan in tie-break order with its mean and times).

Committed (SPEC 0023, the R8 stint limit at rank 1, local profile: seed 0, 50 samples, two
stops, default world with the compound proxy), at sim_code_version `fc338d72f763`:
`data/references/v1/78bd08512180.parquet`, `data/references/d2/9ebe6bc5491f.parquet`,
`data/references/pilot1/3a4a0322b1a0.parquet` and `data/references/dev6/72bebbb40278.parquet`.
They replace `820f58b3bce2`, `763a6d2cfb45`, `c87fdfd26007` and `4ecc62a30937` (sim_code_version
`1dcf46ad75aa`), on which the committed `*_r8` and dev6 result files were scored; what the
proxy changed in them is in "Proxied compounds". With the proxy off, every record of the four
sets is identical to those (rebuilt in a scratch directory and compared column by column).

**Rebuilt at the current simulator.** The first R8 references (`441a798fbebb`, `182013ff1f42`,
`a4241208bfb7`, at sim_code_version `983fd6f4f2bf`, built after main was merged in for #61)
went stale when #69 (the `_ParameterDraws._plan` fix in degradation.py), #60 and the driver
effects code (off by default) moved the physics version. They were rebuilt once with the
sidecars' inputs and the stale files removed. The new parquet files are byte-identical to the
ones they replace (every scenario's plan, per-draw times and near-tie columns), so under the
default world none of those changes moves a v1, pilot1 or d2 reference, and the `*_r8` result
files below, scored on the old keys, hold for the new ones. `reference --check` now lists only
d3, the expected exception.

v1 scored on these references (baselines with the limit-aware wrapper, and the recorded run):
`reports/eval/v1_baselines_r8.csv`; pilot1 and d2 baselines: `pilot1_baselines_r8.csv` and
`d2_baselines_r8.csv` ("Result files" below).

The record below is of the reference before R8.

Before: `data/references/v1/396dff32d244.parquet` (135 KB, format 2; the format-1 file
was 16 KB, and keeping every searched plan would be 595 KB), v1 under the default world (local profile: seed 0, 50 samples, two stops) at
sim_code_version `21b81a57ea42`. From
it, `score --require-reference` reproduced `reports/eval/v1_baselines_split.csv` row for
row (every value column; the stamp columns `latency_s`, `sim_code_version`,
`belief_config_hash` and `world_config_hash` are excluded, stamp-only drift under
reports/calibration/README.md); the rows gain `regret_se_s` and
`within_mc`. That world draws no events, so every `regret_se_s` is 0 and `within_mc` is
the tie rule there. The rows also gain the near-tie columns, and in that world every
optimal set is the reference alone (`opt_set_size` 1 on all 1320 rows), every pit-now
call with both kinds of plan is decisive, and the set-based agreement equals the plain one.

Under R8 at the defaults (`reports/eval/v1_baselines_r8.csv`) that no longer holds: 6 of the
264 v1 scenarios (30 of 1320 rows), all HAM, laps 0, 10, 11, 15, 20 and 25, have exact ties,
`opt_set_size` 2 or 4. The references there are two-stops, and in this deterministic linear
world two kinds of plan cost exactly the same: the later stints swapped (S22 M17 S27 against
S22 S27 M17), and one SOFT lap moved between two SOFT stints that end at the same tyre age
(from lap 0 on a 4-lap-old set, S22 M17 S27 against S23 M17 S26). R7 picks one: equal stop
counts, then the earlier pit laps, so S22 M17 S27 (stops after laps 22 and 39). Every pit-now
call is still decisive and the set-based stop agreement equals the plain one; the set-based
first pit error is one lap lower on 8 rows.

### Near ties (information only)

`python -m src.eval report` prints, below the existing tables, a near-tie view when the rows
carry the near-tie columns (`score.summarize_near_ties`), one row per policy over every
race: the share of scenarios whose optimal set has more than one plan, the share where the
pit-now call is decisive (of the scenarios with both a pit-now and a stay-out plan), and
the stop agreement and mean first pit error against the reference and against the closest
member of the optimal set. Nothing in the summary table or the headline metrics changes.
Under the deterministic default world the view shows every call decisive and sets of size
1, except the exact ties R8 brings (above: 6 v1 scenarios with sets of 2 or 4).

v1 under the default world with `events: {name: rate}` (default rates), 200 samples, belief
default, baselines and the recorded run, from a cached reference:

| policy | scenarios | share_opt_set_gt1 | share_pit_now_decisive | scored | stop_agreement | stop_agreement_set | mean_first_pit_error | mean_first_pit_error_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actual_strategy | 264 | 0.78 | 0.84 | 199 | 0.59 | 0.60 | 9.99 | 8.22 |
| baseline_agent_recorded | 264 | 0.78 | 0.84 | 1 | 0.00 | 0.00 | 14.00 | 10.00 |
| fixed_one_stop | 264 | 0.78 | 0.84 | 250 | 0.86 | 0.86 | 8.14 | 6.12 |
| fixed_two_stop | 264 | 0.78 | 0.84 | 211 | 0.05 | 0.05 | 9.28 | 7.03 |
| greedy_crossover | 264 | 0.78 | 0.84 | 185 | 0.64 | 0.64 | 15.54 | 13.31 |

Per scenario the optimal set has median 5 plans (max 9; 58 of 264 have one). 138 scenarios
have both a pit-now and a stay-out plan; on 116 the call is decisive (median |gap| 1.69 s
against a median paired SE of 0.24 s), and of the 22 that are not, 13 have the reference
pitting now. The near ties sit within one strategy: the set-based stop agreement moves by
at most 0.01, while the first pit error drops by about 2 laps for every policy.

### Precision under drawn events (v1, N = 200)

The default world with `events: {name: rate}` (default rates), 200 samples, belief
default, baselines and the recorded run, from a cached reference:

| policy | scored | mean regret s | share_within_mc | mean regret_se_s |
| --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.90 | 0.17 | 0.27 |
| actual_strategy | 199 | 6.52 | 0.22 | 0.24 |
| greedy_crossover | 185 | 11.69 | 0.04 | 0.29 |
| fixed_two_stop | 211 | 12.85 | 0.00 | 0.33 |
| baseline agent (recorded) | 1 | 7.33 | 0.00 | 0.39 |

Reference against runner-up on the 250 scenarios with more than one legal plan: paired
standard error mean 0.19 s (median 0.19, max 0.29); the runner-up's gap has median 0.07 s
(mean 1.45), and on 82% of them it is within 2 standard errors. At 200 draws the reference
is not told apart from the next plan on most scenarios, while regrets of a few seconds are
well outside the noise (a regret_se_s of about 0.3 s).

## physics-v1

`physics-v1` is the simulator release the report numbers are computed against: a git tag
at the end of sprint 2, with the sim_code_version below. Report runs score with
`--require-reference` against references cached at that sim_code_version, so a report
number cannot come from another simulator version without failing.
`data/ground_truth.json` stays the frozen legacy check (`legacy_tools.optimal_strategy`,
`tests/test_case.py`) and is not regenerated from this simulator.

| tag | sim_code_version | commit |
| --- | --- | --- |
| physics-v1 | tagged at the end of sprint 2 (Nov 20); its SIM_CODE_VERSION is recorded here when the tag is made | not tagged yet |

TODO (end of sprint 2, the user): apply the tag with `git tag physics-v1 <commit>` on the
commit that freezes the simulator, put the commit here, and record the sim_code_version
above from `python -c "from src.sim.calibration import SIM_CODE_VERSION as v; print(v)"` at
that commit. Then
rebuild the report references (`python -m src.eval reference ...`) and check
`python -m src.eval reference --check` lists nothing.

## Result files

Which committed table is current. Every current file has its `*_summary.csv` beside it.

| set | current | superseded (kept as the record) |
| --- | --- | --- |
| v1, baselines + recorded run | `reports/eval/v1_baselines_r8.csv` (R8 rank 1, margin 3, limit-aware baselines) | `v1_baselines_split.csv` (Milestone 2: belief/world split, before R8); `v1_baselines.csv` (before the split) |
| v1, world traffic on | none under R8 yet | `v1_world_baselines.csv` (before R8) |
| pilot1, baselines | `reports/eval/pilot1_baselines_r8.csv` | `pilot1.csv` baseline rows (Milestone 2, before R8) |
| pilot1, live agent | `pilot1.csv` agent rows (Milestone 2, before R8; the cited ones) | `pilot1_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d2, baselines | `reports/eval/d2_baselines_r8.csv` | `d2.csv` baseline rows (Milestone 2, before R8) |
| d2, live agent | `d2.csv` agent rows (Milestone 2, before R8; the cited ones) | `d2_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d3 | none: d3 is rebuilt and scored at the final evaluation | `d3_baselines.csv` (before R8; its reference is the expected stale one) |
| dev6, baselines (development, calibration races) | `reports/eval/dev6_baselines.csv` (R8 rank 1, margin 3, limit-aware baselines) | |
| dev6, live agent pass 1 (development, calibration races) | `reports/eval/dev6_agent_pass1.csv`, trajectories in `reports/eval/trajectories/dev6_pass1/`; `dev6_pass1_combined.csv` (agent + baseline rows, the input to `dev6_pass1_compare_regret_s_*` with driver clusters and `dev6_pass1_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_pass1_compare_per_race_regret_s_paired.csv`; figure table `dev6_pass1_chart.csv` | |

Only the files whose rows carry `stint_limit_rank` and `stint_limit_margin_laps` were scored
under R8.

## Current numbers (v1, R8, deterministic)

`reports/eval/v1_baselines_r8.csv`: `python -m src.eval score --scenarios data/scenarios/v1.json
--policies baselines --recorded data/eval/baseline_agent_sample_run.json --require-reference
--out reports/eval/v1_baselines_r8.csv`, at R8 rank 1 and margin 3, with the limit-aware
baselines, against the Milestone 2 table (`v1_baselines_split.csv`, before R8). Overall:

| policy | scored / invalid / unavailable, M2 | R8 | mean regret s, M2 | R8 | median, M2 | R8 | p90, M2 | R8 | stop agreement, M2 | R8 | first-pit error, M2 | R8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 / 0 / 14 | 250 / 0 / 14 | 4.95 | 4.33 | 3.01 | 2.14 | 13.60 | 13.45 | 0.90 | 0.86 | 6.80 | 4.67 |
| actual_strategy | 199 / 0 / 65 | 199 / 0 / 65 | 6.72 | 5.49 | 7.11 | 5.33 | 14.52 | 12.06 | 0.57 | 0.61 | 9.98 | 6.66 |
| fixed_two_stop | 211 / 0 / 53 | 197 / 14 / 53 | 12.95 | 11.55 | 9.25 | 8.46 | 21.18 | 21.18 | 0.00 | 0.03 | 8.77 | 4.96 |
| greedy_crossover | 185 / 0 / 79 | 165 / 20 / 79 | 11.91 | 8.97 | 11.54 | 8.40 | 20.14 | 20.14 | 0.70 | 0.15 | 13.73 | 2.77 |
| baseline agent (recorded) | 1 / 0 / 263 | 1 / 0 / 263 | 6.29 | 2.05 | | | | | 0.00 | 1.00 | 13.00 | 6.00 |

- The unavailable rows are exactly Milestone 2's (a zero-lap first stint, a real stint
  under v1's 10-lap minimum, no recorded decision); R8 makes none. The rows the first
  version of #61 marked unavailable for R8 now come out as follows. fixed_one_stop: 91 of 91
  scored. greedy_crossover: 109 of 129 scored and 20 invalid (past the limit with no stop
  left). fixed_two_stop: 14 invalid (not limit-aware).
- HAM lap 0: fixed_one_stop's S33 > M33 ends the 4-lap-old SOFT at 37, past 35; the wrapper
  gives S31 > M35 (regret 0.40 s). greedy_crossover's S10 > M10 > S46 becomes
  S4 > M3 > S59, still past the limit with no stop left: invalid.
- greedy_crossover's stop agreement falls (0.70 to 0.15) because the forced stop is
  usually followed by a second crossover stop, a two-stop against mostly one-stop references.
- No row has negative regret.

On the rows scored ok both times, Milestone 2 being `v1_baselines_split.csv` (not the older
`v1_baselines.csv`) (mean of R8 minus M2, 95% cluster-bootstrap interval over
(race, driver), `eval.compare.cluster_bootstrap`, 2000 draws):

| policy | n | mean regret M2 | R8 | R8 - M2 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.95 | 4.33 | -0.63 | [-0.84, -0.51] |
| actual_strategy | 199 | 6.72 | 5.49 | -1.23 | [-1.58, -0.85] |
| fixed_two_stop | 197 | 13.19 | 11.55 | -1.64 | [-1.95, -1.34] |
| greedy_crossover | 165 | 11.47 | 8.97 | -2.51 | [-3.93, -0.95] |
| baseline agent (recorded) | 1 | 6.29 | 2.05 | -4.24 | |

pilot1 and d2 baselines under R8 (`pilot1_baselines_r8.csv`, `d2_baselines_r8.csv`) against the
baseline rows of their Milestone 2 files, scored / invalid / unavailable and mean regret s:

| set | policy | M2 | R8 | mean regret M2 | R8 |
| --- | --- | --- | --- | --- | --- |
| pilot1 | fixed_one_stop | 25 / 0 / 0 | 25 / 0 / 0 | 5.28 | 4.66 |
| pilot1 | actual_strategy | 25 / 0 / 0 | 25 / 0 / 0 | 10.68 | 8.91 |
| pilot1 | fixed_two_stop | 25 / 0 / 0 | 22 / 3 / 0 | 16.19 | 15.65 |
| pilot1 | greedy_crossover | 14 / 0 / 11 | 12 / 2 / 11 | 23.77 | 11.87 |
| d2 | fixed_one_stop | 25 / 0 / 0 | 24 / 1 / 0 | 8.13 | 8.09 |
| d2 | actual_strategy | 23 / 0 / 2 | 23 / 0 / 2 | 7.95 | 7.59 |
| d2 | fixed_two_stop | 25 / 0 / 0 | 25 / 0 / 0 | 23.02 | 22.58 |
| d2 | greedy_crossover | 0 / 4 / 21 | 1 / 3 / 21 | | 47.70 |

d2's fixed_one_stop invalid row breaks the limit in its second (SOFT) stint, which moving
the stop earlier cannot fix. d2's four greedy_crossover invalid rows on main were
TEST_UNKNOWN plans; under R8 three are invalid for the limit. No pilot1 or d2 row has
negative regret.

## Milestone 2 numbers (v1, deterministic, before R8)

The Milestone 2 table is `reports/eval/v1_baselines_split.csv` (summary
`reports/eval/v1_baselines_split_summary.csv`; `python -m src.eval report
reports/eval/v1_baselines_split.csv` prints it): belief `to_snapshot`, world the default
SimConfig with the full window. `reports/eval/v1_baselines.csv` is kept as the record from
before the split (default SimConfig, full window for both planning and scoring). Both,
overall:

| policy | scored before | scored after | mean regret s before | after | median before | after | p90 before | after | stop agreement before | after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 250 | 4.95 | 4.95 | 3.01 | 3.01 | 13.60 | 13.60 | 0.90 | 0.90 |
| actual_strategy | 199 | 199 | 6.72 | 6.72 | 7.11 | 7.11 | 14.52 | 14.52 | 0.57 | 0.57 |
| fixed_two_stop | 211 | 211 | 12.95 | 12.95 | 9.25 | 9.25 | 21.18 | 21.18 | 0.00 | 0.00 |
| greedy_crossover | 123 | 185 | 18.77 | 11.91 | 18.61 | 11.54 | 23.85 | 20.14 | 0.06 | 0.70 |
| baseline agent (recorded) | 1 | 1 | 6.29 | 6.29 | | | | | 0.00 | 0.00 |

Only greedy_crossover moves: it is the one baseline that reads the fitted pace. The fixed
policies and the actual strategy depend only on the snapshot and the modelled compounds,
which are the same in both fits, and the recorded plan is fixed. Every reference plan is
the same. On the 123 scenarios greedy_crossover scored both times, 60 have lower regret
under the belief and none higher; on 62 more it no longer stops immediately. Under the
hindsight fit, the race-fitted linear MEDIUM wear (0.068 s/lap, the curvature #26
explained) brought its crossover early; on the to_snapshot fit it does not. A baseline
planned on hindsight doing worse than one planned on the belief is the direction an honest
eval should show.

Before the split, from `reports/eval/v1_baselines_summary.csv`:

| policy | scored | mean regret s | median | p90 | stop agreement |
| --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.95 | 3.01 | 13.60 | 0.90 |
| actual_strategy | 199 | 6.72 | 7.11 | 14.52 | 0.57 |
| fixed_two_stop | 211 | 12.95 | 9.25 | 21.18 | 0.00 |
| greedy_crossover | 123 | 18.77 | 18.61 | 23.85 | 0.06 |
| baseline agent (recorded, 1 scenario) | 1 | 6.29 | | | 0.00 |

The baseline agent row is the recorded run in `docs/sample_run.txt` (HAM from lap 0), scored
under this simulator rather than the legacy linear one it was graded on. The live agent
(`--policies agent`) now answers every scenario through the snapshot loop.

Things the numbers show that the owners may want to look at:
- `greedy_crossover` stops immediately (a zero-lap first stint, R1) on 141 of 264 scenarios
  before the split and on 79 under it, so it is unavailable there.
- `actual_strategy` is unavailable on 65 scenarios where a real stint was shorter than the
  10-lap minimum v1 carries. Sets sampled under R2's default of 3 keep those strategies.

## Position regret on v1 (world traffic on)

`reports/eval/v1_world_baselines.csv` (summary beside it): v1, the baselines, belief
default, world `configs/sim/world.yaml` (full window, ghost traffic, events none, so no
drawn-events warning). The world model changes, so the reference and the time regrets move
too:

| policy | scored | mean regret s | median | stop agreement | mean position regret | mean final position |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 5.74 | 3.68 | 0.88 | 0.34 | 6.88 |
| actual_strategy | 199 | 7.45 | 7.99 | 0.58 | 0.32 | 7.46 |
| greedy_crossover | 185 | 12.85 | 11.78 | 0.68 | 0.64 | 7.02 |
| fixed_two_stop | 211 | 13.93 | 10.65 | 0.02 | 0.82 | 7.15 |

The policy order by position regret is the order by time regret except that actual_strategy
and fixed_one_stop swap (0.32 against 0.34). Per row, the signs of `position_regret` and
`regret_s` agree on 332 of 845 scored rows (actual_strategy 86/199, fixed_one_stop 65/250,
fixed_two_stop 101/211, greedy_crossover 80/185). All 513 disagreements are one kind: time
lost (`regret_s` > 0) with the same finishing position (`position_regret` 0), median 7.8 s
of regret against 14.1 s on the rows that lose a place. There is no row with opposite
signs, no row that loses a place without losing time, and no negative position regret. The
world is deterministic (one draw), so positions are whole numbers: a few seconds rarely
cross a gap to the next car.

That shape is expected, not a defect. With one draw a position is an integer, and in a
spread-out field a few seconds rarely move a place, so most rows that lose time finish where
the reference does: 513 of 845 lose time at the same position, and the signs agree on 332.
Position regret becomes informative with `n_samples` above 1 (the mean over draws moves in
fractions of a place) and where the field is close, after an SC, which is exactly where the
SC-bunching limit applies (docs/traffic.md, limits 1 and 5). Read it beside
`final_position_mean`, not instead of the time regret.

## Limitations

- **The world is fitted on hindsight.** It reads every race lap (`full` window); only the
  belief is held to the laps up to the snapshot (above, "Who sees what").
- **The R8 stint limit reads the whole race, so the belief side sees that much hindsight.**
  `rules.stint_limits` takes the oldest tyre age run on each compound over the full race lap
  frame, laps after the snapshot included, and the belief applies the same limits as the
  world. So the baselines' limit-aware adjustment, the belief's continuation search and the
  agent's rule errors all know a race-wide constant that nobody at the snapshot could know.
  It is the same class of hindsight as the observed pit loss; an as-of source (earlier
  seasons, Pirelli guidance) can replace it later (docs/rules.md R8).
- **A proxied compound's price has no race laps behind it** ("Proxied compounds"): its pace
  and wear are a step from a raced compound measured in practice, qualifying or other races,
  with the validation errors listed there, and its R8 limit is a practice age. At 2022 Saudi
  Arabia (SOFT, from practice) it makes every dev6 reference end on SOFT and is probably too
  kind; rows carry `proxied_compounds` and `opt_proxied_compounds` so results resting on one
  can be counted and set apart.
- **The R8 rank moves the results.** At rank 3 instead of 1, 9 of d2's 13 references at laps
  0 to 20 are two-stops instead of 0. Rows and summaries carry `stint_limit_rank` and
  `stint_limit_margin_laps` so both can be shown; the committed tables are rank 1.
- **v1 scores under a 10-lap minimum stint** (its snapshots predate R2's 3), which makes
  actual_strategy unavailable on 65 scenarios.
- **The belief's SC and VSC rates are provisional** and Spain-derived (configs/sim/belief.yaml);
  the default world draws no events. Position regret has the traffic limits of
  docs/traffic.md.
- **d3 is rebuilt at the final evaluation**; its committed reference and baselines predate R8
  ("Reference cache").

## Running on Sol

`python -m src.eval jobs` installs the eval runner and hands everything after `--` to
`python -m src.sim.jobs`, so `eval_chunk` tasks score slices of a scenario set:

```bash
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    manifest eval_chunk --profile sol --n-items 264 --manifest $OUT/manifest.json --out $OUT
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    eval_chunk --profile sol --manifest $OUT/manifest.json --task-id 0 --out $OUT
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    aggregate eval_chunk --profile sol --manifest $OUT/manifest.json --out $OUT
```

`slurm/eval_chunk.sbatch` uses the `python -m src.eval jobs ... -- eval_chunk ...` form
(set `EVAL_SCENARIOS`, optionally `EVAL_POLICIES`, `EVAL_WORLD_CONFIG`,
`EVAL_BELIEF_CONFIG`). `run-all` works with worker processes too:
`install_job_runner` registers its spec (scenarios path, policy names, recorded files,
config paths) with `jobs.set_eval_runner`, `run-all` carries it in every `eval_chunk`
task's payload, and each worker installs the runner before its task, under any start
method:

```bash
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines \
    --world-config configs/sim/world.yaml -- run-all eval_chunk --manifest M --out O
```

The runner reads cached references under `--reference-dir` (default `data/references`)
when they match the seed (0) and the profile's `n_samples` and `max_stops`; a report run adds
`--require-reference` so a missing reference fails the task (see "Reference cache").

LLM policies stay on the desktop.

## Tests

`tests/test_eval.py` runs offline: synthetic races for sampling and scoring, plus the
committed Spain cache for the v1 set and an end-to-end baseline run.
`tests/test_eval_integration.py` covers the belief and world split and the decision unit,
`tests/test_eval_part2.py` positions, the drawn-events warning, the disagreement flag, the
eval split, `run-all` with two spawned workers, the agent policy, the snapshot contract and
the minimum-stint rule. `tests/test_eval_reference.py` covers the reference cache (key
stability, cached against computed rows, `--check` across code versions, hindsight
events, the d3 exception), `regret_se_s` on a world with known per-draw differences,
`within_mc` at N = 1 against the tie rule, and the R8 stamps on rows and summaries.
`tests/test_optimizer_bridge.py` covers the limit-aware baselines (fixed_one_stop's moved
stop, greedy_crossover's forced stop, a plan still past the limit scored invalid, and every
baseline unchanged with R8 off). `tests/test_sim_proxy.py` covers the compound proxy (tier
selection and mixing, anchoring, the qualifying gap, the ordering constraint and the wear
floor, the proxied covariance and draws, the belief's information boundary, R8 for an
unraced compound, the off switch, the committed step table) on synthetic weekends, and on
2022 Saudi Arabia that the world prices SOFT and accepts the belief's continuations. Modules
that test other mechanics on synthetic two-compound races carry `pytest.mark.compound_proxy_off`
(tests/conftest.py): their configs default to the proxy off, so they pin those mechanics as
they were. No test calls a model.

## Live agent runs for the Milestone 2 report

The live-agent outputs in `reports/eval/case_b/`, `reports/eval/pilot1*.csv` and
`reports/eval/trajectories/` were produced on ASU RC Voyager with `LLM_BACKEND=voyager`,
`VOYAGER_BASE_URL=https://openai.rc.asu.edu/v1`, `VOYAGER_MODEL=qwen3-30b-a3b` and the default
`VOYAGER_TEMPERATURE` (1e-9), all set through `.env` (the API key is never committed). Case B is
`python -m src.baselines.runner --driver HAM --lap 30 --json --quiet`, run ten times
(`reports/eval/case_b/run_01..10`); the pilot is `python -m src.eval score --scenarios
data/scenarios/pilot1.json --policies baselines agent`, and the paired intervals are
`python -m src.eval.compare reports/eval/pilot1.csv --reference agent --others fixed_one_stop
actual_strategy`. The model is not deterministic at this temperature: identical inputs gave 5 of
10 submissions on Case B. The pilot set is a development set, not a report-set estimate.

**Reruns after the submission-budget fix (#59).** `reports/eval/pilot1_fix*.csv`,
`reports/eval/d2_fix*.csv`, `reports/eval/case_b/fix_run_01..10.json` and
`reports/eval/trajectories/{pilot1_fix,d2_fix}/` and `trajectories/case_b/fix_run_*.jsonl` are the
same runs with the same settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2), repeated after #59 made a submission always accepted and
outside the 24-call evidence budget. The `*_fix.csv` files hold only the agent rows; the
`*_fix_combined.csv` files add the unchanged baseline rows of the Milestone 2 files (the
reference columns match row for row) and are the input to `*_fix_compare_regret_s_*`. The
Milestone 2 files above stay the cited ones; the reruns are a follow-up measurement.

**Reruns after the turn-limit grace (#72).** `reports/eval/pilot1_fix2*.csv`,
`reports/eval/d2_fix2*.csv` and `reports/eval/trajectories/{pilot1_fix2,d2_fix2}/` repeat the
pilot1 and d2 runs once with the same settings. The two sets ran one after the other, so the
latency figures are not affected by contention. Since #72 the agent loop prints a turn-status
line, refuses an evidence call identical to one already run, and follows the 10 evidence turns
with up to 2 grace turns that offer only `submit_recommendation`. The rows carry
`submitted_in_grace` and `evidence_calls`. Loop settings of these runs: status line on,
repeats refused (a mode that no longer exists; their `evidence_calls` excludes the refused
repeats and they have no `repeat_calls`), grace on. The current defaults (status line off,
repeats run again) are the loop of the `*_fix` runs plus the grace. The `*_fix2_combined.csv` files are built the same way
as the `_fix` ones: the new agent rows plus the Milestone 2 baseline rows, whose reference
columns match row for row. They are the input to `*_fix2_compare_regret_s_*`. The Milestone 2
files stay the cited ones.

**dev6, agent pass 1.** `reports/eval/dev6_agent_pass1.csv` is one pass of the agent over the 150
dev6 scenarios with the same settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2, main's world defaults: R8 at rank 1 and margin 3, driver effects
off), run as one sequential job (`score --scenarios data/scenarios/dev6.json --policies agent
--seed 0 --require-reference`) at commit 3223131, 94 minutes wall time, with no other agent
run at the same time. `score` logs trajectories under the set version; they were moved to
`trajectories/dev6_pass1/`. dev6 is a development set on calibration races, not a held-out
result. Loop settings of this pass: status line on, repeats served from the cache (today
`AGENT_TURN_STATUS=1 AGENT_REPEAT_MODE=cache`), grace on; its `repeat_calls` are cached
repeats, which its `evidence_calls` exclude.

To print a Case B episode as a terminal transcript (for screenshots), replay a committed run with
`python scripts/show_case.py --replay reports/eval/case_b/run_01.json --commit 29d164c` (the commit
the run was produced at), or run one fresh episode with `--live`, which writes `case_b/live_<time>.*`.
