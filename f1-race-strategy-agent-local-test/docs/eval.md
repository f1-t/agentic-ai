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

## Information contract

A policy decides at a snapshot: race R, driver D, after lap L. Let t be the session time at
which D completed lap L. **Everything a policy can read must be a deterministic function only
of**

- (a) the raw timing records of race R with timestamps at or before t;
- (b) this weekend's sessions before the race start;
- (c) published regulations and the tyre allocation;
- (d) artifacts built only from races whose start precedes R's start, each with a committed
  manifest of its source races.

"What a policy can read" is the snapshot and the agent's view, every tool output and error
string, the belief simulator's fit, draws, constants and rules, the continuation behind an
agent decision, and every baseline's input. The world simulator and the hindsight reference
are the scorer: they keep the whole race.

`tests/test_eval_information_contract.py` checks it. It builds one synthetic race three ways,
each as `lap_frame` would (the whole-race derived columns recomputed on the rows given): as
run, truncated at t, and with everything after t altered (lap times, compounds including rain
to come, tyre ages, stops, track status, positions and clock, a retirement, the snapshot
driver lapped, a longer race). The snapshot, the anonymised view, the belief's fit, every
agent tool's output and error strings, `belief_optimum`, `to_plan` (the continuation), the
bridged baselines, the rule baselines and the sampler's decision points up to t must be
identical across the three. Each legacy switch below is shown to fail the same check on its
own. The R8 stint limit and the pit loss (items 1 and 2 of the leakage audit) are fixed on
other branches; the harness runs the belief with R8 off and a fixed pit loss until they land.
`tests/test_future_blindness.py` (the future-blindness harness: real races through
`jobs.load_race`'s own code path, truncated or re-raced after the snapshot) keeps
`test_leak_frame_filter`, `test_leak_race_length` and `test_leak_compound_list` as regression
checks (`future_blindness.FIXED_LEAKS`) and observes snapshots from `snapshot_as_of`.

What each fix does (the legacy switch reproduces the old behaviour, so old results stay
reproducible):

| item | was | now | legacy switch |
| --- | --- | --- | --- |
| 3 clean flag and the other derived columns | the frame's `clean`, `pace_resid_s`, `driver_offset_s` and gap columns, from a robust filter fitted on every lap of the race, chose the belief's fit rows, `ref_lap_s` (the SC lap) and the lap-noise sd | `data.as_of_columns` recomputes them on the rows the window can see (`Simulator.window_rows`); the world keeps the whole-race ones | `SimConfig.policy_view.clean: race` |
| 6 window cut by lap number | `lap <= L` for the fit and the calibrator's `laps_so_far`: another car's lap L that ended up to minutes later was in | `end_s <= t` (`Simulator.cut_time`: D's lap-L end; a car a lap ahead's lap L + 1 that ended earlier is in) | `SimConfig.policy_view.window_cut: lap` |
| 4 race length | `snapshot_at` set `total_laps` to D's own lap count: a lapped or retiring car showed it | `calibration.snapshot_as_of` keeps the scheduled distance (`session.total_laps`) | the builder `snapshot_at` |
| 5 compounds | `RaceParams.compounds`: every compound raced in the whole race (2021 Spain showed MEDIUM, SOFT; a wet race would show INTERMEDIATE) | the nominated dry three, HARD, MEDIUM and SOFT (`calibrate.race_compounds`), plus any compound a car had run by t (a wet start) | `calibrate --compounds raced`, `calibrate.DEFAULT_COMPOUND_SOURCE = "raced"` |
| 6 snapshot reads | positions, gaps and the car behind came from the lap L + 1 rows | `snapshot_as_of` reads rows completed by t: the car ahead and its gap at the end of lap L, the car behind at the previous crossing (the last one it had made by t), each car's position at its last lap by t, the stint history from D's laps up to L; on an in-lap the set going on is the lap L + 1 row's start-of-lap compound and tyre age, as before | `snapshot_at` |
| 7 race identity | the view showed season, event slug and real driver codes | the model sees `"event": "anonymised race"`, no season, its own car as "Car A" and the nearby cars as "Car B", "Car C", ...; `tools.present` applies the same mapping, plus any other code of the field to "another car" and any year to `<year>`, to every tool output and error string. Lap count and pit loss stay (the tools give them). The mapping stays in the harness: `tools.identity()`, `run_agent`'s `stats["identity"]` and the agent row's `meta["identity"]`; scoring and the trajectory file names keep the real ids, the trajectory events hold what the model saw | `AGENT_SHOW_IDENTITY=1` (ablation only, default off) |
| S scenario selection | finishers at most two laps down only, and the lap before each SC or VSC period | `scenarios --rule as_of` (the default for new sets): every driver still running at the decision lap (laps 1..L recorded), at a fixed grid (every 5th lap to the scheduled distance less 10; `neutralised=True` adds the laps D's own record shows a neutralisation in progress or just ended, off by default because a car behind decides after the others at the same lap number), capped to leave one minimum stint of the scheduled distance | `scenarios --rule legacy` |

`SimConfig.policy_view` is left out of `config_hash` while it holds its default, so a config
that does not set it keeps its hash; the behaviour change under the default shows in
`sim_code_version`. Sets sampled before these fixes (v1, pilot1, d2, d3, d4, dev6) have no
`sampling["rule"]`: they were sampled under the legacy rule with `snapshot_at`, and their
snapshots embed the driver's own lap count and the raced compounds. They load and score as
before; a new set records `"rule": "as_of"` and the builder `snapshot_as_of`.

**Scoring a car that was lapped or later retired.** Every plan covers the scheduled
distance, the reference's included, and the world scores all of it with its whole-race fit
(the world has no per-car pace and draws no retirement). Policy and reference then cover the
same laps under the same draws, so regret stays a like-for-like difference. Truncating to the
laps the car really ran would need the scorer to cut every plan, the reference's too, at a
number no policy may know, and a retired car has no such number at all. The
`actual_strategy` comparator runs its last recorded set on to the flag when the car covered
fewer laps.

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

## Rule baselines: run_to_flag and one_stop_total

Two deterministic rule policies (`src/eval/policies.py`), selected by name
(`--policies run_to_flag one_stop_total`) and, like `belief_optimizer`, not in the `baselines`
group, so default commands and existing result files are unchanged. They were added because
`fixed_one_stop` always makes exactly one more stop from the snapshot, also after the driver
has made their stop, when the sensible plan is usually to run to the flag.

**run_to_flag.** Make no further stop when the rules allow it; otherwise the fewest stops
that make the plan legal, each as late as the rules allow:

1. The plan is the current set to the flag. If it passes every rule (R1-R4, R8 and the
   belief's known compounds, checked with `belief.validate` on the plan as a contract, sets
   assigned by `to_contract` as for every baseline), that is the plan.
2. Otherwise, for k = 1, 2, ... up to `max_stops` (the future-stop limit every policy gets,
   2 in the committed runs), take the k-stop plans that pass every rule and submit the first
   one in this order: the latest first stop, then the latest second stop (pit laps compared
   first stop first), then the compounds of the new stints in the preference order SOFT,
   MEDIUM, HARD, then any other modelled compound alphabetically, compared stint by stint.
   The new stints may use any compound the belief models, the current one included (a new set).
   The first k with a legal plan wins.
3. If no plan with at most `max_stops` stops is legal, the plan is the current set to the
   flag; the world rejects it and the row is `invalid` with the world's reason.

The compound rule is fixed: since the forced stops are placed as late as the rules allow, the
stint after one is as short as they allow, and the softest legal compound is the usual choice
for a short final stint. Placing the stops first and choosing the compound second is part of
the rule: when a later stop needs a harder compound (an R8 limit), the later stop wins.

**one_stop_total.** A one-stop race in the ordinary sense, one stop in the whole race:

- If the driver has not stopped yet in this race (the snapshot's completed-stint history,
  `snapshot.stints`, is empty), the decision is `fixed_one_stop`'s, through the same eval
  wrapper: the stop at the middle of the remaining laps (moved to respect the minimum stint,
  R2), onto the first of HARD, MEDIUM, SOFT the belief models that differs from the current
  compound, and moved earlier to the last legal lap when the plan breaks R8
  (`_one_stop_within_limit`). Where `fixed_one_stop` is unavailable or invalid, so is this row.
- If the driver has stopped at least once (`snapshot.stints` is not empty), it is
  `run_to_flag`.

What they read. Both are functions of the snapshot and the published rules only: the laps
completed and left, the current compound, its age and laps on the set, the stint history, the
tyre sets, the set of compounds the belief models at the snapshot (fitted only on what is known
there), the minimum stint (R2) and the future-stop limit. Neither reads a lap time, a pace or
wear fit, the belief optimizer, the reference or any tuned constant, and neither reads the pit
loss. The one thing they know that nobody at the snapshot could is what every policy already
gets: the race-wide R8 stint limits, a hindsight race constant (docs/rules.md R8, "Limitations"
below). A plan the world rejects is an `invalid` row with its reason, as for any policy.

## Proxied compounds

A strategy is not banned because nobody ran it. The world's full-window fit models a compound
only when it has `min_samples` (5) clean race laps (`degradation._fuel_fit` keeps `_usable`
compounds; the rest go to `fit_report["unmodelled"]` with the lap count), and it never reads
practice: the practice prior enters only the belief's `to_snapshot` and `pre_race` fits (the
shrinkage prior of `_shrunk`, whose compounds are the prior's plus the raced ones). So at
2022 Saudi Arabia, where SOFT was run in practice but never raced, the belief priced SOFT and
the world did not: every plan with a later SOFT stint was "Unknown compound 'SOFT'" (20 of the
21 invalid agent rows of dev6 pass 1). With `SimConfig.compound_proxy.enabled` (the default;
`false` restores the simulator as it was, bit for bit) every fit prices HARD, MEDIUM and SOFT
(`src/sim/proxy.py`).

**The rule.** A compound with `min_samples` clean race rows in that simulator keeps its fit
(the belief: rows up to its calibration point). For any other, pace (the fresh base) and wear
are each a precision-weighted blend, from the nearest raced compound (the anchor), of every
source that exists for it:

| source | pace | wear | exists when |
| --- | --- | --- | --- |
| practice long runs this weekend (FP sessions, plus the Sprint on a sprint weekend) | anchor + (practice base_c - practice base_anchor) | anchor + (practice wear_c - practice wear_anchor) | the compound and the anchor have practice long runs |
| qualifying this weekend | anchor + the qualifying gap (below) | none | the compound and the anchor have qualifying laps |
| the cross-race neighbour step (`configs/sim/compound_steps.json`) | anchor x (1 + pace_frac) per step | anchor x wear_ratio per step | always |

value = sum_i w_i v_i / sum_i w_i, with fixed weights, one set for all races, w = 1 / that
source's mean squared error alone in the hidden-compound validation below (all calibration
races). Pace and wear have separate weights: qualifying informs pace only, and a stint error
mixes the two. The pace weights use each source's fresh-pace error, the wear weights its wear
error:

| | practice | qualifying | cross-race step |
| --- | --- | --- | --- |
| pace: fresh-pace MSE, s^2 (n) | 2.671 (16) | 1.091 (6) | 0.247 (19) |
| pace weight (share when all three exist) | 0.374 (7%) | 0.917 (17%) | 4.055 (76%) |
| wear: wear MSE, (s/lap per lap)^2 (n) | 0.0139 (16) | | 0.0058 (19) |
| wear weight (share) | 71.8 (29%) | | 172.7 (71%) |
| for comparison, 15-lap stint MSE, s^2 (n) | 322.4 (16) | | 31.8 (19) |

(Weights from the 15-lap stint error would give practice 9% against the step's 91%; they
cannot weigh qualifying, which has no wear.) The blended value's prior sd is the larger of the
blend's own leave-one-race-out error (0.585 s pace, 0.079 s/lap per lap wear,
`compound_proxy.sd_floor`) and the cross-race step's spread times the anchor value. Tier 4
(the same circuit in other seasons) is not built: `proxy._tier4_pace` and `_tier4_wear` are
the extension point. A belief compound with fewer than 5 race rows by its calibration point is
re-priced the same way, relative to a raced compound: **this changes what the agent sees on
any race where a compound has under 5 race rows at the snapshot** (before, it stayed at
practice plus the config base delta). A fit with no raced compound yet (the belief before its
first race laps, `pre_race`) keeps its practice compounds and anchors the missing ones on them.
Weights, steps and sd floor are read through the config, so `config_hash` covers them.

**The qualifying correction.** A qualifying lap is a low-fuel push lap on an evolving track, so
it is never compared with a race lap directly. Each driver's best timed lap per (segment
Q1/Q2/Q3, compound) (`data.qualifying_frame`: a lap time, no pit in or out, a dry compound,
track status green) is one row of a least-squares fit lap = driver + segment + compound, slow
rows trimmed against the fit as `data.robust_filter` trims practice (2.5 robust sd, two
passes; fast rows never cut). Only the compound gap q_c - q_anchor is used, added to the
anchor's race base. Fuel cancels in it (every qualifying lap carries little fuel, and the race
base the gap is added to carries the race's fuel level, so the race fit's fuel slope is not
needed); car and driver pace cancel (the driver term); evolution and engine modes between
segments cancel (the segment term, which takes the pre-2020 Q3 modes too). Not corrected:
evolution within a segment, and push against managed driving. A compound needs 2 drivers who
also set a row on another compound. Without qualifying laps on the anchor, qualifying does not
enter. HARD is never run in qualifying, so it only ever informs SOFT against MEDIUM, and at
2022 Saudi Arabia it informs nothing (2 drivers on MEDIUM, one trimmed). The measured gap is
shown to the agent for every compound that has one (`qualifying_gap_s` in the model report,
e.g. 2021 Spain "SOFT: ...; qualifying gap to MEDIUM -0.94 s on fresh tyres (3 drivers)"), raced
or not, since fresh-tyre pace is the datapoint the race fit cannot give.

**The step table** (`python -m src.sim.compound_steps estimate`; calibration races only, the
four held-out races never loaded). Per race, the full-window world fit with the proxy off; a
pair counts when both compounds have at least 50 clean race rows (`MIN_PAIR_ROWS`); the
value is the median over races, sd the sample sd across them. Before and after that filter:

| pair | key | all raced pairs (before) | at least 50 rows (shipped) |
| --- | --- | --- | --- |
| MEDIUM-HARD | pace_frac | -0.0035, sd 0.0038, n 6 | -0.0027, sd 0.0033, n 5 |
| MEDIUM-HARD | wear_ratio | 1.53, sd 0.85, n 6 | 1.40, sd 0.65, n 5 |
| MEDIUM-HARD | life_ratio | 0.64, sd 0.26, n 5 | 0.67, sd 0.24, n 4 |
| SOFT-MEDIUM | pace_frac | +0.0019, sd 0.0061, n 6 | +0.0001, sd 0.0026, n 3 |
| SOFT-MEDIUM | wear_ratio | 0.60, sd 0.88, n 6 | 0.72, sd 0.33, n 3 |
| SOFT-MEDIUM | life_ratio | 0.76, sd 0.88, n 5 | 0.76, sd 0.14, n 3 |

The filter drops 2023 Bahrain (MEDIUM, 8 rows) from both pairs and 2024 Emilia Romagna (SOFT,
11 rows) and 2025 Qatar (SOFT, 34 rows) from SOFT-MEDIUM, which rests on 2019 United Kingdom,
2021 Spain and 2023 Spain only. **Its pace step is +0.0001, zero to within 0.01 s on a 90 s
lap: on these race fits a fresh SOFT is no faster than a fresh MEDIUM, and its wear ratio
(0.72) says SOFT wears less.** After the ordering constraint a step-priced SOFT is a copy of
MEDIUM. The yardstick matters here: the step reflects the simulator's own race fits, which
under-separate SOFT from MEDIUM (the pooled fit confounds compound with when and by whom it was
run). A proxied compound has to sit where the world's own fits would put it, or the search
exploits the difference; the table is re-estimated when driver effects are adopted.

**Ordering, R8, uncertainty, reporting.** A softer compound is never slower fresh and never
wears less than a harder one: the constraint moves proxied values only, never a raced one
(where raced values are out of order the proxied one is put between them, "conflict"); a
proxied wear is also floored at 0 and at the belief's wear bound. A compound no car raced gets
an R8 limit: its oldest practice tyre age, else the nearest raced compound's limit times the
life ratio, plus the margin (docs/rules.md R8; 2021 Spain HARD: 18 in FP1, longest practice run
13 laps, cap 21; 2022 Saudi Arabia SOFT: 18 in FP1, longest run 12 laps, cap 21). A blended
parameter is linear in its anchors, so `fit_report["cov"]` carries it as T cov T' + tau^2 and
posterior draws move it with them. `fit_report["proxy"]` holds, per compound, `pace_sources` and
`wear_sources` ([{source, weight, anchor, value, step}]), `pace_source` / `wear_source` (the
sources joined by "+", or "race"), the steps, the sds, the ordering notes and
`qualifying_gap_s`. The model report flags `pace_from_<sources>:<compound>`; the agent's model
report and `estimate_tyre_degradation` carry the sources, their weights, the qualifying gap and
`price_basis` ("SOFT: pace from practice long runs 8%, the cross-race step 92% (anchor MEDIUM);
wear from practice long runs 29%, the cross-race step 71% (anchor MEDIUM)" at Saudi). Score rows
carry `proxied_compounds` and `opt_proxied_compounds` (the plan's and the reference's compounds
the world prices by proxy, ";"-joined, empty for none) when the world's proxy is on. **Results
can be read with and without the rows where either is non-empty.**
`calibrators.hindsight_targets` and `cross_race.wear_check` leave proxied compounds out.

### Validation

`python -m src.sim.compound_steps validate` (calibration races only; outputs under
`reports/calibration/compound_proxy/`; it writes the weights and the sd floor into the table).
For each calibration race and each compound that was raced, its race laps are hidden (its rows
dropped from the race frame), the world refitted, and the compound priced again, then compared
with the real race fit: the fresh pace at mid-race, the wear, and the time of a 15- and a
25-lap stint on a new set centred on mid-race. Each source alone uses the step table without
that race; **the blend is leave-one-race-out throughout: for each race its weights come from the
other races' source errors and its step table from the other races** (the shipped constants
use every race). Proxy minus truth, mean signed / mean absolute error:

| priced by | n | fresh pace, s | wear, s/lap per lap | 15-lap stint, s | 25-lap stint, s |
| --- | --- | --- | --- | --- | --- |
| practice alone | 16 | +0.18 / 1.06 | +0.020 / 0.067 | +5.1 / 10.0 | +10.9 / 17.7 |
| qualifying alone (pace) | 6 | -0.15 / 0.85 | | | |
| cross-race step alone | 19 | +0.01 / 0.38 | +0.004 / 0.043 | +0.6 / 4.4 | +1.6 / 10.4 |
| **blend (leave-one-race-out)** | 19 | +0.02 / 0.47 | +0.007 / 0.048 | +1.1 / 4.9 | +2.6 / 10.6 |
| (the old best-single-tier chain, old step table) | 19 | +0.06 / 0.99 | +0.019 / 0.064 | +3.3 / 9.6 | +7.8 / 16.4 |

The blend's root mean squared errors (0.585 s, 0.079 s/lap per lap) are the sd floor. The
step alone is slightly better than the blend on every column: practice and qualifying, even
at 7% and 17% of the pace weight, add more error than they remove on these races (practice
long runs that are not race-like: HARD got faster in practice at 2019 United Kingdom and 2022
Saudi Arabia; qualifying makes SOFT about 1 s faster than MEDIUM where the race fits put them
within 0.3 s). Both are far better than the chain, which put practice first.

**Decision level**: for each race, each raced compound hidden in turn and one lap-0 snapshot
per starting compound (a finisher's), the reference (`optimize_batch`, seed 0, 50 samples,
two stops) on the world with that compound hidden and proxied, against the reference on the
true world (`validation_decisions.csv`, `validation_decisions_summary.csv`). Cost is the true
world's time of the proxied world's plan less the true optimum (valid rows):

| priced by | rows | plan changed | invalid in the true world | cost, s: mean / median / max | laps on the hidden compound: more / fewer / same |
| --- | --- | --- | --- | --- | --- |
| (i) the chain as built (old step table) | 47 | 42 | 4 | 2.77 / 0.69 / 23.56 | 12 / 23 / 12 |
| (ii) cross-race step alone | 47 | 40 | 3 | 3.45 / 2.01 / 13.52 | 12 / 24 / 11 |
| (iii) the blend, leave-one-race-out (shipped) | 47 | 41 | 3 | 3.57 / 2.01 / 13.52 | 11 / 26 / 10 |

**The step alone beats the blend at decision level too, by 0.12 s mean cost (within 0.3 s),
and the chain as built had the lowest mean (2.77 s) and median (0.69 s) cost but the worst
tail (23.6 s, practice HARD at Saudi) and one more invalid row.** The three are within about
1 s of each other on 47 cases; the parameter errors above separate them more clearly. In all
three the search uses the hidden compound less than the truth more often than more, so it does
not systematically exploit a proxied compound. The invalid rows run the hidden compound past
its true race limit, because its proxied limit is its practice age (2023 Bahrain MEDIUM,
raced to 10 laps; 2024 Emilia Romagna SOFT, raced to 8; for the step alone also 2022 Saudi
Arabia MEDIUM, raced to 16).

### What changed in the references

References rebuilt with the blend at seed 0, 50 samples, two stops (`python -m src.eval
reference`), against main's. With the proxy off every record of every set is identical to
main's (rebuilt in a scratch directory and compared column by column); with it on, only races
with a proxied compound move:

| set | references | plan changed | plan on a proxied compound | stops 0 / 1 / 2, main | stops 0 / 1 / 2, blend |
| --- | --- | --- | --- | --- | --- |
| v1 (2021 Spain) | 264 | 66 | 66 (HARD) | 40 / 216 / 8 | 40 / 174 / 50 |
| pilot1 (2021 Spain) | 25 | 7 | 7 | 3 / 22 / 0 | 3 / 15 / 7 |
| d2 (2023 Spain) | 25 | 0 | 0 | 8 / 17 / 0 | 8 / 17 / 0 |
| dev6 (six races) | 150 | 7 | 7 (Spain 7 of 25, Saudi 0 of 25) | 56 / 85 / 9 | 56 / 78 / 16 |

The two real cases, on main, with the chain as built (#81's first version) and with the blend
(dev6 baselines re-scored in a scratch directory; regret is at least 0 on every scored row):

| | unraced compound | blended fresh-pace step from the anchor | blended wear | dev6 references using it | actual_strategy mean regret, s: main / chain / blend |
| --- | --- | --- | --- | --- | --- |
| 2022 Saudi Arabia | SOFT | -0.21 s from MEDIUM (practice 8%, step 92%) | 0.127 s/lap per lap (MEDIUM 0.064; practice 29%, step 71%) | 0 of 25 (chain: 25 of 25) | 0.72 / 7.71 / 0.72 |
| 2021 Spain | HARD | +0.28 s from MEDIUM (step only; raised to SOFT's base) | 0.033 (lowered to SOFT's) | 7 of 25 | 8.91 / 9.00 / 9.08 |

At Saudi the blend leaves the references where main had them: SOFT, 0.2 s faster than MEDIUM
fresh but wearing twice as much, is not worth a stop. At 2021 Spain HARD is priced by the step
alone (no HARD long run, no HARD in qualifying), and the ordering constraint makes it a copy of
SOFT: the race fit has SOFT both slower than MEDIUM fresh and wearing less, so HARD may be
neither faster than SOFT nor wear more than it. The references then use HARD as a second
SOFT-like compound (SOFT-HARD-SOFT two-stops, capped at 21 laps by R8), which a SOFT-SOFT-SOFT
plan could not do under R3: 66 of v1's 264 references and 7 of dev6's 25 Spain ones. Over dev6
the mean regret moves from 6.97 to 7.15 s (actual_strategy), 8.21 to 9.00 (fixed_one_stop, which
takes the proxied HARD at Spain and is invalid on 13 more rows past its practice cap),
24.38 to 24.44 (fixed_two_stop) and 22.01 to 20.86 s (greedy_crossover). Replaying the dev6
pass-1 agent submissions (no agent run) on the blend: no row is invalid for an unknown compound;
Saudi Arabia goes from 21 invalid rows to 1 (a PIT the minimum-stint rule refuses).

**Limitation.** A proxied compound's price has no race laps behind it: its pace and wear are
the anchor's plus a step measured elsewhere (practice, qualifying, other races), with the
errors above, and its stint limit is a practice age. Read `proxied_compounds` and
`opt_proxied_compounds` beside any result on a race with a proxied compound.

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

Every set listed here was sampled under the legacy rule, which reads the future (above,
"Information contract"). New sets default to `--rule as_of` and the builder
`calibration.snapshot_as_of`; `--rule legacy` reproduces the old sampling.

### Development and report sets

`data/eval_set.json` is the registered eval split, written by `register_eval_set` (Sarah's
item): the races held out for reporting, `{"races": [...], ...}` or a bare list of ids.
While it exists, `build_set` (and `scenarios`, `--eval-set` for another file) refuses any
race outside it, records the file's sha256 as `sampling["eval_set_digest"]`, and marks the
set `development: false`. Without it a set is `development: true`.

- **v1 is the development set**: sampled before the split, from the race we tuned on.
  Its numbers are for debugging the harness and the baselines, not for the report.
- **A set sampled from the eval races is the report set**: the numbers the report quotes.
  Today that is d4, built and scored once at tag `d4-final-eval` (see "The registered split"
  below and "d4, the held-out result").

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
  same snapshots, item for item). References: `data/references/dev6/fbb33ef732dd.parquet`
  (sim_code_version `58b50a00e0e3`, seed 0, 50 samples, two stops, default world with the
  compound proxy's blend). Pass 1 and `dev6_baselines.csv` were scored on the earlier
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
stops, default world with the compound proxy's blend), at sim_code_version `58b50a00e0e3`:
`data/references/v1/c0c7247b95b7.parquet`, `data/references/d2/dbc5b90ac885.parquet`,
`data/references/pilot1/fda18509e2f7.parquet` and `data/references/dev6/fbb33ef732dd.parquet`.
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
| d3 | none: its Monaco rows are superseded by d4 (below); d3 is not rescored | `d3_baselines.csv` (Nov 11, before R8; its reference is the expected stale one) |
| d4, baselines + belief_optimizer (**report set, the held-out result**) | `reports/eval/d4_baselines.csv` (reference `data/references/d4/39bf1bbb4aa9.parquet`) | supersedes d3's 2019 Monaco baseline rows |
| d4, live agent, one pass (**report set, the held-out result**) | `reports/eval/d4_agent.csv`, trajectories in `reports/eval/trajectories/d4/`; `d4_combined.csv` (agent + baseline + belief_optimizer rows, the input to `d4_compare_regret_s_*` with driver clusters and `d4_compare_race_regret_s_*` with race clusters); per-race paired table `d4_compare_per_race_regret_s_paired.csv`; figure table `d4_chart.csv` | |
| dev6, baselines (development, calibration races) | `reports/eval/dev6_baselines.csv` (R8 rank 1, margin 3, limit-aware baselines) | |
| dev6, live agent pass 1 (development, calibration races) | `reports/eval/dev6_agent_pass1.csv`, trajectories in `reports/eval/trajectories/dev6_pass1/`; `dev6_pass1_combined.csv` (agent + baseline rows, the input to `dev6_pass1_compare_regret_s_*` with driver clusters and `dev6_pass1_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_pass1_compare_per_race_regret_s_paired.csv`; figure table `dev6_pass1_chart.csv` | |
| dev6, baselines + belief_optimizer, run 2 (development, calibration races) | `reports/eval/dev6_run2_baselines.csv` (compound proxy on, reference `fbb33ef732dd`) | `dev6_baselines.csv` (before the proxy) is the record pass 1 was compared with |
| d4, rule baselines, scored once after the d4 results (**report set**) | `reports/eval/d4_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, frozen at `5a2bc41`); `d4_combined_rules.csv` (`d4_combined.csv`'s rows plus these) and `d4_rules_compare_*` ("Rule baselines added after the held-out run") | `d4_combined.csv` stays as committed |
| dev6, rule baselines (development, calibration races) | `reports/eval/dev6_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, reference `fbb33ef732dd`); `dev6_run2_combined_rules.csv` (run 2's rows plus these) and `dev6_run2_rules_compare_*` ("Rule baselines added after the held-out run") | |
| dev6, live agent run 2 (development, calibration races) | `reports/eval/dev6_run2_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run2/`; `dev6_run2_combined.csv` (agent + baseline + belief_optimizer rows, the input to `dev6_run2_compare_regret_s_*` with driver clusters and `dev6_run2_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run2_compare_per_race_regret_s_paired.csv`; figure table `dev6_run2_chart.csv` | pass 1 stays as the record of the earlier loop and world |
| dev6, live agent run 3 (variance repeat of run 2; development, calibration races) | `reports/eval/dev6_run3_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run3/`; `dev6_run3_combined.csv` (run 3's agent rows + run 2's baseline and belief_optimizer rows, the input to `dev6_run3_compare_regret_s_*` with driver clusters and `dev6_run3_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run3_compare_per_race_regret_s_paired.csv`; figure table `dev6_run3_chart.csv` | not a replacement: run 2 stays the cited dev6 result; run 3 only measures run-to-run variation |

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
  with the validation errors listed there, and its R8 limit is a practice age. At 2021 Spain
  the ordering constraint makes the proxied HARD a copy of SOFT, which 66 of v1's references
  use as a second SOFT-like compound; rows carry `proxied_compounds` and
  `opt_proxied_compounds` so results can be read with and without the rows that rest on one.
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
baseline unchanged with R8 off). `tests/test_eval_rule_baselines.py` covers the rule
baselines on synthetic races and on 2021 Spain: the default `baselines` group unchanged, the
already-stopped and not-yet-stopped branches, a forced stop for R3 and for R8 (one stop and two),
the invalid row when no legal plan exists within `max_stops`, legality, determinism and regret
>= 0. `tests/test_sim_proxy.py` covers the compound proxy (which
sources enter the blend and with what weight, the weighted mean, anchoring, the qualifying gap
and its place in the model report, the ordering constraint and the wear floor, the proxied
covariance and draws, the belief's information boundary, R8 for an unraced compound, the off
switch, the committed table and weights, the leave-one-race-out weight helper) on synthetic
weekends, and on
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

**dev6, run 2.** `reports/eval/dev6_run2_agent.csv` is a second single pass over the same 150
scenarios with the same model settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2, R8 at rank 1 and margin 3, driver effects off), one sequential
job (`score --scenarios data/scenarios/dev6.json --policies agent --seed 0 --n-samples 50
--max-stops 2 --require-reference`) at commit 55a67ee, 54.6 minutes wall time, with no other
agent run at the same time; trajectories were moved from `trajectories/dev6/` to
`trajectories/dev6_run2/`. `dev6_run2_baselines.csv` is `--policies baselines
belief_optimizer` on the same references. Three things differ from pass 1: (1) the world prices
unraced compounds by the weighted blend ("Proxied compounds"; reference
`data/references/dev6/fbb33ef732dd.parquet`, sim_code_version `58b50a00e0e3`, world config
`ccb08edf391b`), so 2022 Saudi Arabia's SOFT continuations are scored rather than rejected;
(2) the agent loop runs at main's defaults (status line off, repeats run again and counted
against the budget, grace on; `AGENT_TURN_STATUS` and `AGENT_REPEAT_MODE` unset), where pass 1
had the status line on and cached repeats, so run 2's `evidence_calls` include repeats and
pass 1's do not; (3) the `belief_optimizer` baseline is added. Pooled, the agent scored 150 of
150 (pass 1: 129, with 21 Saudi rows invalid for the unknown compound), mean regret 3.95 s
(median 1.68), against belief_optimizer 4.11, actual_strategy 7.15 and fixed_one_stop 9.00;
agent minus belief_optimizer is -0.16 s, 95% interval [-0.58, 0.20] with driver clusters (88
ties of 150). 119 submissions were voluntary and 31 came in a grace turn (pass 1: 74 and 76),
at 72k mean input tokens (pass 1: 100k). The PIT / STAY_OUT decision is the same as pass 1's on
142 of 150 scenarios. No row of any policy is invalid for anything but its own plan's R8 stint
limit. dev6 is a development set on calibration races, not a held-out result.

**dev6, run 3 (variance repeat).** `reports/eval/dev6_run3_agent.csv` is an identical repeat of
run 2, made only to measure run-to-run variation; **run 2 stays the cited development result**.
Same command, scenarios, reference (`fbb33ef732dd`), sim_code_version `58b50a00e0e3`, world and
belief config (`ccb08edf391b` / `0002e90cb6fe`), model (qwen3-30b-a3b on Voyager, default
temperature, seed 0) and agent loop (main's defaults, the four `AGENT_*` / `VOYAGER_TEMPERATURE`
overrides checked unset), at commit c685349 (no change under `src/` or `configs/` since run 2's
55a67ee or tag `d4-final-eval`), one sequential pass, 60.9 minutes wall time, nothing else calling
the model. Trajectories were moved from `trajectories/dev6/` to `trajectories/dev6_run3/`;
`dev6_run3_combined.csv` is the new agent rows plus run 2's baseline and belief_optimizer rows
(30 reference and stamp columns match row for row). Every episode submitted (110 voluntary, 40 in
a grace turn; run 2: 119 and 31) and there were no API errors; one row is invalid
(2021_04_spain:OCO:00, a lap-0 PIT whose plan has a 1-lap first stint, R2), so 149 of 150 scored.
Between the two passes the PIT / STAY_OUT decision agrees on 145 of 150 scenarios and the final
plan on 119; on the 149 scenarios scored in both, run 3 minus run 2 regret is +0.41 s per
scenario (sd 2.62, 95% interval [0.05, 0.90] with driver clusters), 121 of them identical. Pooled
mean regret moves 3.95 to 4.36 s, and the paired agent-minus-X differences move by +0.27
(fixed_one_stop: -5.28 to -5.01), +0.46 (actual_strategy: -3.25 to -2.79) and +0.40 s
(belief_optimizer: -0.16 to +0.24, interval [-0.21, 0.79]). So a headline number moves by about
0.3 to 0.5 s between identical passes: small against the agent's lead over fixed_one_stop and
actual_strategy, but as large as the agent-versus-belief_optimizer difference, which is a tie in
both passes and changes sign. Per race the movement is concentrated in 2023 Bahrain (+1.66 s),
2023 Spain (+0.60) and 2021 Spain (+0.39); four episodes there (the invalid one and three
STAY_OUT to PIT flips worth +4.4, +15.1 and +21.3 s) made three tool calls in the first turn, no
logged call in the next nine, and then pitted onto MEDIUM in a grace turn without evaluating a
plan. Without those four the per-scenario difference is +0.13 s [-0.06, 0.42].

### d4, the held-out result

d4 is the report set and this is the held-out result, declared in advance and run once at tag
`d4-final-eval` (commit cd1b937, main after #82), with no code or config change. Steps, each run
once: `scenarios --races 2019_06_monaco 2020_16_sakhir 2025_11_austria 2024_11_austria --version
d4 --max-per-race 25 --seed 0` (report set, `development: false`, 100 scenarios, 25 per race,
none skipped, digest `ae878eb91c28`, eval_set digest `98edff4380e1`); `reference` (seed 0, 50
samples, two stops: `data/references/d4/39bf1bbb4aa9.parquet`, sim_code_version
`58b50a00e0e3`, world config `ccb08edf391b`; `reference --check` clean apart from d3);
`score --policies baselines belief_optimizer --require-reference` (`d4_baselines.csv`); then,
after the baseline rows passed the gate (no row invalid for anything but an R8 breach of its own
plan, none for an unknown compound, no negative regret, all four races loaded and fitted), one
sequential agent pass (`score --policies agent --seed 0 --n-samples 50 --max-stops 2
--require-reference`, `d4_agent.csv`): qwen3-30b-a3b on Voyager, default temperature, seed 0,
main's agent loop (status line off, repeats run again, 2 grace turns, 10 evidence turns, 24-call
budget; `AGENT_TURN_STATUS`, `AGENT_REPEAT_MODE`, `AGENT_SUBMIT_GRACE_STEPS` and
`VOYAGER_TEMPERATURE` unset), R8 at rank 1 and margin 3, compound proxy on, 47.2 minutes wall
time, nothing else calling the model. It is one pass: the model is not deterministic at this
temperature, and no episode was rerun. d4's 25 Monaco scenario ids are d3's (same seed and per-race
count), so **d3's 2019 Monaco baseline rows of November 11 (`d3_baselines.csv`, pre-R8 simulator)
are superseded by d4's**; d3 itself is not rescored. What each race had been exposed to before
this run is the disclosure under "The registered split"; this run adds nothing to it beyond the
run itself.

Pooled over the 100 scenarios, the agent scored 100 of 100 (71 voluntary submissions, 29 in a
grace turn), mean regret 8.93 s (median 6.53), against belief_optimizer 8.95, actual_strategy
8.87 and fixed_one_stop 12.87. Paired, agent minus belief_optimizer is -0.03 s, 95% interval
[-0.29, 0.22] with driver clusters (39 ties of 100); agent minus actual_strategy -0.19 [-1.88,
1.43]; agent minus fixed_one_stop -4.28 [-6.73, -1.90]. Per race the agent's mean regret is 8.58
(2019 Monaco), 8.41 (2020 Sakhir), 2.55 (2025 Austria) and 16.17 s (2024 Austria). No row of any
policy is invalid for anything but its own plan's R8 stint limit, and the world prices every
compound from race laps at all four races (`proxied_compounds` and `opt_proxied_compounds` are
empty on every row), so the with-and-without-proxy reading is the same table. The belief does
price compounds without race laps at some snapshots (read from the model reports in the
trajectories): at 2024 Austria the belief prices SOFT from practice and the cross-race step at
24 of 25 snapshots (the 25th is lap 0, practice only), up to lap 67, while 23 of the 25 world
references, fitted on every race lap, use SOFT; every policy's regret there is about 16 s. Against dev6 run 2 (development, calibration races) the regrets roughly double for the agent
(3.95 to 8.93 s) and belief_optimizer (4.11 to 8.95), and the agent's lead over actual_strategy
(-3.25 [-5.50, -1.25] on dev6) is not there on d4.

### Rule baselines added after the held-out run

**Why.** `fixed_one_stop` always makes exactly one more stop from the snapshot, also after the
driver has made their stop, when the best plan is usually to run to the flag. Split by whether
the driver had already stopped, the agent was 7.4 s worse than fixed_one_stop on d4 before the
driver's first stop and 8.5 s better after it, so the headline "the agent beats a fixed
one-stop rule by 4.3 s" came from scenarios where the baseline makes a pointless stop. Two
sensible rule baselines were added beside it, `run_to_flag` and `one_stop_total` (rules in
"Rule baselines: run_to_flag and one_stop_total" above). They were added after the d4 results
were known.

**Order.** The rules were written in this file first, then implemented and tested, then scored
on the development set dev6 (below). No rule was changed after it was scored. They were then
frozen (code, docs, tests and dev6 outputs committed; the commit is given in the d4 part below)
and scored once on d4. No existing d4 or dev6 row was rescored.

**dev6 (development, calibration races).** `score --scenarios data/scenarios/dev6.json --policies
run_to_flag one_stop_total --seed 0 --n-samples 50 --max-stops 2 --require-reference`
(`reports/eval/dev6_rule_baselines.csv`, reference `fbb33ef732dd`, sim_code_version
`58b50a00e0e3`), compared with the committed dev6 run 2 rows
(`dev6_run2_combined.csv`) by `scripts/rule_baseline_tables.py`, which reads csv and scenario
json only and writes `dev6_run2_combined_rules.csv` (the run 2 rows plus the new ones; the
reference columns are checked equal first) and `dev6_run2_rules_compare_*`. Pooled:

| policy | scored / invalid / unavailable | mean regret s | median | p90 |
| --- | --- | --- | --- | --- |
| agent | 150 / 0 / 0 | 3.95 | 1.68 | 11.82 |
| belief_optimizer | 150 / 0 / 0 | 4.11 | 1.20 | 13.26 |
| fixed_one_stop | 123 / 27 / 0 | 9.00 | 7.28 | 20.57 |
| actual_strategy | 147 / 0 / 3 | 7.15 | 3.10 | 20.98 |
| run_to_flag | 150 / 0 / 0 | 4.27 | 0.05 | 10.93 |
| one_stop_total | 135 / 15 / 0 | 4.10 | 0.05 | 10.93 |

Agent minus each policy, paired on the rows both scored, 95% interval with (race, driver)
clusters (`eval.compare`, 2000 draws), wins / ties / losses from the agent's side:

| X | n | agent - X | 95% CI | W / T / L |
| --- | --- | --- | --- | --- |
| belief_optimizer | 150 | -0.16 | [-0.58, 0.20] | 26 / 88 / 36 |
| fixed_one_stop | 123 | -5.28 | [-7.25, -3.30] | 88 / 2 / 33 |
| actual_strategy | 147 | -3.25 | [-5.50, -1.25] | 70 / 37 / 40 |
| run_to_flag | 150 | -0.31 | [-2.10, 1.35] | 49 / 53 / 48 |
| one_stop_total | 135 | -0.24 | [-2.24, 1.56] | 46 / 47 / 42 |

Split by whether the driver has already stopped (`snapshot.stints` not empty), agent minus X:

| subset | X | n | agent - X | 95% CI |
| --- | --- | --- | --- | --- |
| not yet stopped (60) | fixed_one_stop | 45 | +3.96 | [1.86, 6.04] |
| not yet stopped | run_to_flag | 60 | +2.73 | [0.62, 4.59] |
| not yet stopped | one_stop_total | 45 | +3.96 | [1.86, 6.04] |
| already stopped (90) | fixed_one_stop | 78 | -10.61 | [-12.44, -8.75] |
| already stopped | run_to_flag | 90 | -2.34 | [-4.75, -0.17] |
| already stopped | one_stop_total | 90 | -2.34 | [-4.75, -0.17] |

On dev6 the agent ties both rule baselines pooled (intervals across 0); it loses to them before
the driver's first stop and beats them after it. `one_stop_total`'s 15 invalid rows are
`fixed_one_stop`'s own, on not-yet-stopped drivers (its plan there is fixed_one_stop's, row for
row: status, plan and regret), all R8 breaches of the second stint (12 at 2021 Spain, on the
proxied HARD and its practice limit of 21, 2 at 2022 Saudi Arabia, 1 at 2023 Bahrain), which
moving the stop earlier cannot fix. `run_to_flag` has no invalid row; it makes a stop on all 60
not-yet-stopped rows (the two-compound rule) and on 23 of the 90 already-stopped rows (20 where
the current set would pass its R8 limit before the flag, 3 where the driver had stopped onto the
same compound, so the two-compound rule still needed a stop), and runs to the flag on the other
67. No row of either has negative regret. Nothing was changed after this run:
it found no bug. Per race (`dev6_run2_rules_compare_per_race_regret_s_paired.csv`), the agent is
worse than both at 2022 Saudi Arabia (+3.3 and +3.7 s) and better than `one_stop_total` at 2021
Spain on its 13 scored rows; dev6 is a development set on calibration races, not a held-out
result.

**Freeze.** Commit `5a2bc41` (branch `feat/fair-rule-baselines`) holds the rules, the code, the
tests and the dev6 outputs above. No file under `src/`, `configs/` or `tests/` changed after it.

**d4 (held out), scored once, after the d4 results were known.** One command at the freeze
commit, the only command that loaded the four held-out races: `score --scenarios
data/scenarios/d4.json --policies run_to_flag one_stop_total --seed 0 --n-samples 50 --max-stops 2
--require-reference --out reports/eval/d4_rule_baselines.csv` (reference
`data/references/d4/39bf1bbb4aa9.parquet`, sim_code_version `58b50a00e0e3`). It ran without error
and was not rerun. No existing d4 row was rescored: the comparison rows are the committed
`d4_combined.csv`, and `scripts/rule_baseline_tables.py` (csv and scenario json only, the
reference columns checked equal) wrote `d4_combined_rules.csv` and `d4_rules_compare_*`.
`d4_combined.csv` is unchanged. Pooled:

| policy | scored / invalid / unavailable | mean regret s | median | p90 |
| --- | --- | --- | --- | --- |
| agent | 100 / 0 / 0 | 8.93 | 6.53 | 22.13 |
| belief_optimizer | 100 / 0 / 0 | 8.95 | 6.36 | 22.67 |
| fixed_one_stop | 94 / 6 / 0 | 12.87 | 16.76 | 26.28 |
| actual_strategy | 98 / 0 / 2 | 8.87 | 2.56 | 28.03 |
| run_to_flag | 100 / 0 / 0 | 9.26 | 6.63 | 24.59 |
| one_stop_total | 95 / 5 / 0 | 7.02 | 2.94 | 20.57 |

Agent minus each policy, paired, 95% interval with (race, driver) clusters, wins / ties / losses
from the agent's side:

| X | n | agent - X | 95% CI | W / T / L |
| --- | --- | --- | --- | --- |
| belief_optimizer | 100 | -0.03 | [-0.29, 0.22] | 29 / 39 / 32 |
| fixed_one_stop | 94 | -4.28 | [-6.73, -1.90] | 59 / 0 / 35 |
| actual_strategy | 98 | -0.19 | [-1.88, 1.43] | 30 / 28 / 40 |
| run_to_flag | 100 | -0.33 | [-2.14, 1.47] | 41 / 26 / 33 |
| one_stop_total | 95 | +1.66 | [-0.33, 3.65] | 35 / 26 / 34 |

Split by whether the driver has already stopped, agent minus X:

| subset | X | n | agent - X | 95% CI |
| --- | --- | --- | --- | --- |
| not yet stopped (30) | fixed_one_stop | 25 | +7.44 | [3.59, 11.42] |
| not yet stopped | run_to_flag | 30 | -0.18 | [-3.93, 2.96] |
| not yet stopped | one_stop_total | 25 | +7.44 | [3.59, 11.42] |
| already stopped (70) | fixed_one_stop | 69 | -8.53 | [-10.66, -6.37] |
| already stopped | run_to_flag | 70 | -0.40 | [-2.55, 1.81] |
| already stopped | one_stop_total | 70 | -0.40 | [-2.55, 1.81] |

Per race, agent minus run_to_flag / one_stop_total (`d4_rules_compare_per_race_regret_s_paired.csv`):
2019 Monaco +3.55 / +6.88, 2020 Sakhir +4.30 / +5.49, 2024 Austria -2.82 / -1.21, 2025 Austria
-6.37 / -5.42. The rule baselines are much better than the agent at Monaco and Sakhir and much
worse at 2025 Austria; a t interval over the four race means is [-8.51, 7.84] for run_to_flag
and [-7.76, 10.63] for one_stop_total.

What this says, plainly. On the held-out races **the agent does not beat a sensible rule**: it
ties `run_to_flag` (-0.33 s, interval across 0) and is 1.66 s worse than `one_stop_total` on the
95 scenarios that baseline scored, an interval that just reaches 0, so the agent is not shown
to be worse either. The agent's 4.3 s lead over `fixed_one_stop` is the pointless extra stop
that baseline makes after the driver's first stop: against a rule that does not make it, the
lead is gone, after the first stop (-0.40 s) as before it. Before the first stop, the agent
loses to the one-stop rule by 7.4 s, as it lost to `fixed_one_stop`, since the two are the same
plan there. `one_stop_total` has 5 invalid rows, `fixed_one_stop`'s own on not-yet-stopped
drivers (its plan there is fixed_one_stop's, row for row): the first stint passes its R8 limit
(2025 Austria SOFT, limit 19; 2024 Austria MEDIUM, limit 32) and no earlier stop keeps the one
remaining stint inside the second compound's limit. Those 5 rows are left out of its paired
comparison; the agent's regret on them is 1.9, 4.9, 24.8, 20.5 and 15.8 s. `run_to_flag` has
no invalid row and stops in 57 of 100 scenarios (25 one-stops and 5 two-stops before the first
stop, 26 one-stops and 1 two-stop after it). No row of either has negative regret. These are
one scoring of deterministic rules on four races; with the large race-to-race spread, the
pooled intervals are the ones to quote.

To print a Case B episode as a terminal transcript (for screenshots), replay a committed run with
`python scripts/show_case.py --replay reports/eval/case_b/run_01.json --commit 29d164c` (the commit
the run was produced at), or run one fresh episode with `--live`, which writes `case_b/live_<time>.*`.
