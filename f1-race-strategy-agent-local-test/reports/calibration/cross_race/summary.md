# Cross-race calibration report (SPEC 0016)

sim_code_version `d517750065d4`, profile `local` (`a0d0d34f2faa`), 1 race(s) considered, 1 eligible.

**PROVISIONAL: 1 eligible race(s), fewer than 8. The selection below is provisional and the defaults stay** (belief.yaml, world.yaml and the SimConfig defaults are unchanged).

All G and stop-gain numbers are held out: each race's rows come from its own leave-one-race-out fold, and Spain is one race among the rest.

## Held-out races

data/eval_set.json holds out 2019_06_monaco, 2023_07_spain. Excluded from this run for that reason: 2019_06_monaco, 2023_07_spain.

## Eligibility

Drivers: eval.scenarios.eligible_drivers (laps 1..n with no gap, at most two laps down), the eval sampler's set.

| race | eligible | drivers | wet share | red flags | SC | VSC | dry compounds | reasons |
|---|---|---|---|---|---|---|---|---|
| 2021_04_spain | yes | 19 | 0.00 | - | 8-10 | - | MEDIUM SOFT | - |


## Arms

| arm | file | file_sha256 | config_hash | role |
|---|---|---|---|---|
| baseline | configs/sim/arms/baseline.yaml | 766d61c22308 | 07dc9587709d | baseline |
| none | configs/sim/arms/none.yaml | d77eeb468f4e | 5a22044bf2a0 | candidate |
| hinge | configs/sim/arms/hinge.yaml | b9085ee335a7 | 42527df9536f | candidate |
| free_air | configs/sim/arms/free_air.yaml | 009c6574e84e | 5a1e15595f9c | candidate |
| hinge_free | configs/sim/arms/hinge_free.yaml | c4ca78e1e412 | 672fa97a83df | candidate |
| posterior | configs/sim/arms/posterior.yaml | c3d0f8d509a6 | cd7e99df6e50 | coverage only |
| wear_nonneg | configs/sim/arms/wear_nonneg.yaml | 3bfbe0f44583 | e7c0d34bdef9 | candidate |
| collinear_hold | configs/sim/arms/collinear_hold.yaml | 7d4bd15481f1 | 2fc84b635d2e | candidate |
| collinear_hold_hinge | configs/sim/arms/collinear_hold_hinge.yaml | 9df26c36bcde | ca1c878f0c90 | candidate |
| collinear_hold_nonneg | configs/sim/arms/collinear_hold_nonneg.yaml | 3a9c5a90f3ee | ca9375a538eb | candidate |
| collinear_hold_nonneg_hinge | configs/sim/arms/collinear_hold_nonneg_hinge.yaml | 3b25f5df12ff | ad2158a91ab3 | candidate |
| driver_effects | configs/sim/arms/driver_effects.yaml | 07bba5f9196b | e74729616414 | candidate |
| driver_effects_nonneg | configs/sim/arms/driver_effects_nonneg.yaml | e3086edc1423 | d1d8ab79e239 | candidate |
| driver_effects_hinge | configs/sim/arms/driver_effects_hinge.yaml | ac2b6bd2d5da | 092feb106391 | candidate |


## Current defaults

Each window's incumbent: the arm its config carries now. Every comparison is incumbent-first, so a change in either direction has to pass the window's rule.

| window | incumbent | config_hash | source | set by | matches the config |
|---|---|---|---|---|---|
| to_snapshot | wear_nonneg | e7c0d34bdef9 | a priori | configs/sim/belief.yaml: wear bound 0 (SPEC 0017b, #40: the physical argument and the stop-gain check) | yes |
| full | baseline | 07dc9587709d | default | SimConfig defaults (configs/sim/world.yaml adds traffic only) | yes |
| pre_race | baseline | 07dc9587709d | default | SimConfig defaults | yes |


## Selection

G: mae_offset_s at horizons 10-49; SG: stop-gain mae_s at real pit stops, and SG bias its signed mean (predicted less real gain: below 0 when the model under-predicts the gain from fresh tyres), race-weighted. mae and diff are race-weighted (the mean over races of each race's value, the statistic the race-cluster 95% CI in brackets resamples); diff is the per-race paired difference, challenger less the stage's comparator (the incumbent); 'pooled' is the row-pooled difference, secondary. Belief rule (to_snapshot): SG diff CI below 0 and G diff CI upper end under +2 s, in two stages: the bound (the baseline, no bound, challenging the incumbent wear_nonneg; the bound stays unless removing it passes), then every other candidate against the stage-1 result. World (full) and pre_race: G diff below 0 with its CI excluding 0, against the SimConfig default. Ties to the simpler arm. n/a: the same as the comparator in that window by construction (in the full window there is no calibrator and no shrinkage fit).

### full (G mae_offset 10-49; incumbent baseline)

| arm | role | races | G mae | G bias | G diff | G diff pooled | SG mae | SG bias | SG diff | SG diff pooled | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | comparator | 1 | 10.93 [-, -] | -7.03 | 0.00 [-, -] | 0.00 | 0.53 [-, -] | -0.34 | 0.00 [-, -] | 0.00 | no |
| none | n/a (same by construction) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| hinge | candidate | 1 | 9.32 [-, -] | -6.68 | -1.61 [-, -] | -1.61 | 0.55 [-, -] | -0.45 | 0.02 [-, -] | 0.02 | no |
| free_air | candidate | 1 | 11.40 [-, -] | -5.95 | 0.47 [-, -] | 0.47 | 0.47 [-, -] | -0.16 | -0.06 [-, -] | -0.06 | no |
| hinge_free | candidate | 1 | 10.29 [-, -] | -5.52 | -0.64 [-, -] | -0.64 | 0.51 [-, -] | -0.29 | -0.02 [-, -] | -0.02 | no |
| posterior | coverage only | 1 | 10.93 [-, -] | -7.06 | -0.00 [-, -] | -0.00 | - [-, -] | - | - [-, -] | - | no |
| wear_nonneg | candidate | 1 | 10.93 [-, -] | -7.03 | 0.00 [-, -] | 0.00 | 0.53 [-, -] | -0.34 | 0.00 [-, -] | 0.00 | no |
| collinear_hold | n/a (same by construction) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| collinear_hold_hinge | candidate | 1 | 9.32 [-, -] | -6.68 | -1.61 [-, -] | -1.61 | 0.55 [-, -] | -0.45 | 0.02 [-, -] | 0.02 | no |
| collinear_hold_nonneg | candidate | 1 | 10.93 [-, -] | -7.03 | 0.00 [-, -] | 0.00 | 0.53 [-, -] | -0.34 | 0.00 [-, -] | 0.00 | no |
| collinear_hold_nonneg_hinge | candidate | 1 | 9.32 [-, -] | -6.68 | -1.61 [-, -] | -1.61 | 0.55 [-, -] | -0.45 | 0.02 [-, -] | 0.02 | no |
| driver_effects | candidate | 1 | 10.68 [-, -] | -8.16 | -0.25 [-, -] | -0.25 | 0.46 [-, -] | -0.10 | -0.07 [-, -] | -0.07 | no |
| driver_effects_nonneg | candidate | 1 | 10.68 [-, -] | -8.16 | -0.25 [-, -] | -0.25 | 0.46 [-, -] | -0.10 | -0.07 [-, -] | -0.07 | no |
| driver_effects_hinge | candidate | 1 | 9.71 [-, -] | -7.77 | -1.22 [-, -] | -1.22 | 0.38 [-, -] | -0.16 | -0.15 [-, -] | -0.15 | no |


Decision for full: **provisional choice baseline; the defaults stay**.

### to_snapshot (belief: stop-gain, G non-inferiority; incumbent wear_nonneg)

Stage 1, the bound: baseline (no bound) challenging the incumbent wear_nonneg:

| arm | role | races | G mae | G bias | G diff | G diff pooled | SG mae | SG bias | SG diff | SG diff pooled | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wear_nonneg | comparator | 1 | 19.12 [-, -] | 1.37 | 0.00 [-, -] | 0.00 | 0.99 [-, -] | -0.90 | 0.00 [-, -] | 0.00 | no |
| baseline | candidate | 1 | 20.08 [-, -] | 2.50 | 0.95 [-, -] | 0.95 | 1.15 [-, -] | -1.07 | 0.16 [-, -] | 0.16 | no |


Stage 2, against the stage-1 result wear_nonneg:

| arm | role | races | G mae | G bias | G diff | G diff pooled | SG mae | SG bias | SG diff | SG diff pooled | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wear_nonneg | comparator | 1 | 19.12 [-, -] | 1.37 | 0.00 [-, -] | 0.00 | 0.99 [-, -] | -0.90 | 0.00 [-, -] | 0.00 | no |
| none | candidate | 1 | 10.18 [-, -] | -0.27 | -8.94 [-, -] | -8.94 | 0.83 [-, -] | -0.77 | -0.17 [-, -] | -0.17 | no |
| hinge | candidate | 1 | 19.37 [-, -] | 3.14 | 0.24 [-, -] | 0.24 | 1.14 [-, -] | -1.06 | 0.14 [-, -] | 0.14 | no |
| free_air | candidate | 1 | 19.03 [-, -] | 0.74 | -0.10 [-, -] | -0.10 | 1.03 [-, -] | -0.70 | 0.03 [-, -] | 0.03 | no |
| hinge_free | candidate | 1 | 18.23 [-, -] | 1.48 | -0.89 [-, -] | -0.89 | 1.01 [-, -] | -0.69 | 0.02 [-, -] | 0.02 | no |
| posterior | coverage only | 1 | 19.82 [-, -] | 1.85 | 0.69 [-, -] | 0.69 | - [-, -] | - | - [-, -] | - | no |
| collinear_hold | candidate | 1 | 16.70 [-, -] | -8.31 | -2.42 [-, -] | -2.42 | 1.10 [-, -] | -0.91 | 0.11 [-, -] | 0.11 | no |
| collinear_hold_hinge | candidate | 1 | 15.90 [-, -] | -7.44 | -3.22 [-, -] | -3.22 | 1.09 [-, -] | -0.90 | 0.09 [-, -] | 0.09 | no |
| collinear_hold_nonneg | candidate | 1 | 16.65 [-, -] | -8.36 | -2.48 [-, -] | -2.48 | 0.96 [-, -] | -0.76 | -0.03 [-, -] | -0.03 | no |
| collinear_hold_nonneg_hinge | candidate | 1 | 15.85 [-, -] | -7.49 | -3.27 [-, -] | -3.27 | 0.94 [-, -] | -0.75 | -0.05 [-, -] | -0.05 | no |
| driver_effects | candidate | 1 | 12.44 [-, -] | -4.82 | -6.68 [-, -] | -6.68 | 1.00 [-, -] | -0.97 | 0.01 [-, -] | 0.01 | no |
| driver_effects_nonneg | candidate | 1 | 12.44 [-, -] | -4.82 | -6.68 [-, -] | -6.68 | 0.98 [-, -] | -0.95 | -0.01 [-, -] | -0.01 | no |
| driver_effects_hinge | candidate | 1 | 11.94 [-, -] | -4.32 | -7.18 [-, -] | -7.18 | 0.99 [-, -] | -0.96 | -0.00 [-, -] | -0.00 | no |


Stage 1 result: wear_nonneg (the incumbent stays, provisional).

Decision for to_snapshot: **provisional choice wear_nonneg; the defaults stay**.

### pre_race (G mae_offset 10-49; incumbent baseline)

| arm | role | races | G mae | G bias | G diff | G diff pooled | SG mae | SG bias | SG diff | SG diff pooled | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | comparator | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| none | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| hinge | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| free_air | candidate | 1 | 11.72 [-, -] | -4.98 | 1.53 [-, -] | 1.53 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| hinge_free | candidate | 1 | 11.72 [-, -] | -4.98 | 1.53 [-, -] | 1.53 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| posterior | coverage only | 1 | 10.52 [-, -] | 0.47 | 0.34 [-, -] | 0.34 | - [-, -] | - | - [-, -] | - | no |
| wear_nonneg | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| collinear_hold | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| collinear_hold_hinge | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| collinear_hold_nonneg | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| collinear_hold_nonneg_hinge | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| driver_effects | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| driver_effects_nonneg | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |
| driver_effects_hinge | candidate | 1 | 10.18 [-, -] | -0.27 | 0.00 [-, -] | 0.00 | 0.83 [-, -] | -0.77 | 0.00 [-, -] | 0.00 | no |


Decision for pre_race: **provisional choice baseline; the defaults stay**.

### Prior-sd fix (SPEC 0017 G), measured alone

wear_nonneg__s2 against wear_nonneg under the belief rule: G diff 0.10 [-, -], SG diff -0.00 [-, -]; passes: no (provisional).

### Tau grid on wear_nonneg (belief rule against the x1 taus: stop-gain mae primary, G at horizons 20-40 non-inferior within +2 s)

| arm | x_base_s | x_wear_s_per_lap | x_fuel_s_per_lap | x_hinge_s_per_lap | SG mae | SG bias | SG diff | G mae 20-40 | G diff | non_inferior | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wear_nonneg__tau_0.5_0.5_1_1 | 0.50 | 0.50 | 1.00 | 1.00 | 0.90 | -0.81 | -0.09 [-, -] | 15.87 | 0.37 [-, -] | no | no |
| wear_nonneg__tau_0.5_0.5_2_1 | 0.50 | 0.50 | 2.00 | 1.00 | 0.91 | -0.79 | -0.08 [-, -] | 18.91 | 3.41 [-, -] | no | no |
| wear_nonneg__tau_1_0.5_1_1 | 1.00 | 0.50 | 1.00 | 1.00 | 0.92 | -0.83 | -0.08 [-, -] | 15.30 | -0.20 [-, -] | no | no |
| wear_nonneg__tau_0.5_0.5_0.5_1 | 0.50 | 0.50 | 0.50 | 1.00 | 0.92 | -0.86 | -0.07 [-, -] | 12.60 | -2.91 [-, -] | no | no |
| wear_nonneg__tau_1_0.5_2_1 | 1.00 | 0.50 | 2.00 | 1.00 | 0.92 | -0.80 | -0.07 [-, -] | 18.85 | 3.35 [-, -] | no | no |

27 combinations; the current taus (x1) rank 14; provisional, nothing adopted.

## Stop-gain per race

Per window, race and arm, over the held-out stop-gain rows (one per real stop and snapshot k before it): mean real and predicted gain (s/lap, the last two clean laps before the stop less the first two after it), the signed bias (predicted less real) and the mae. stop_gain_by_race.csv holds the rows.

### full

| race | arm | n | stops | real | pred | SG bias | SG mae |
|---|---|---|---|---|---|---|---|
| 2021_04_spain | baseline | 336 | 37 | 1.73 | 1.39 | -0.34 | 0.53 |
| 2021_04_spain | none | 336 | 37 | 1.73 | 1.39 | -0.34 | 0.53 |
| 2021_04_spain | hinge | 336 | 37 | 1.73 | 1.28 | -0.45 | 0.55 |
| 2021_04_spain | free_air | 336 | 37 | 1.73 | 1.57 | -0.16 | 0.47 |
| 2021_04_spain | hinge_free | 336 | 37 | 1.73 | 1.44 | -0.29 | 0.51 |
| 2021_04_spain | wear_nonneg | 336 | 37 | 1.73 | 1.39 | -0.34 | 0.53 |
| 2021_04_spain | collinear_hold | 336 | 37 | 1.73 | 1.39 | -0.34 | 0.53 |
| 2021_04_spain | collinear_hold_hinge | 336 | 37 | 1.73 | 1.28 | -0.45 | 0.55 |
| 2021_04_spain | collinear_hold_nonneg | 336 | 37 | 1.73 | 1.39 | -0.34 | 0.53 |
| 2021_04_spain | collinear_hold_nonneg_hinge | 336 | 37 | 1.73 | 1.28 | -0.45 | 0.55 |
| 2021_04_spain | driver_effects | 336 | 37 | 1.73 | 1.63 | -0.10 | 0.46 |
| 2021_04_spain | driver_effects_nonneg | 336 | 37 | 1.73 | 1.63 | -0.10 | 0.46 |
| 2021_04_spain | driver_effects_hinge | 336 | 37 | 1.73 | 1.57 | -0.16 | 0.38 |


### to_snapshot

| race | arm | n | stops | real | pred | SG bias | SG mae |
|---|---|---|---|---|---|---|---|
| 2021_04_spain | baseline | 336 | 37 | 1.73 | 0.66 | -1.07 | 1.15 |
| 2021_04_spain | none | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | hinge | 336 | 37 | 1.73 | 0.67 | -1.06 | 1.14 |
| 2021_04_spain | free_air | 336 | 37 | 1.73 | 1.03 | -0.70 | 1.03 |
| 2021_04_spain | hinge_free | 336 | 37 | 1.73 | 1.04 | -0.69 | 1.01 |
| 2021_04_spain | wear_nonneg | 336 | 37 | 1.73 | 0.83 | -0.90 | 0.99 |
| 2021_04_spain | collinear_hold | 336 | 37 | 1.73 | 0.83 | -0.91 | 1.10 |
| 2021_04_spain | collinear_hold_hinge | 336 | 37 | 1.73 | 0.84 | -0.90 | 1.09 |
| 2021_04_spain | collinear_hold_nonneg | 336 | 37 | 1.73 | 0.98 | -0.76 | 0.96 |
| 2021_04_spain | collinear_hold_nonneg_hinge | 336 | 37 | 1.73 | 0.99 | -0.75 | 0.94 |
| 2021_04_spain | driver_effects | 336 | 37 | 1.73 | 0.76 | -0.97 | 1.00 |
| 2021_04_spain | driver_effects_nonneg | 336 | 37 | 1.73 | 0.78 | -0.95 | 0.98 |
| 2021_04_spain | driver_effects_hinge | 336 | 37 | 1.73 | 0.77 | -0.96 | 0.99 |


### pre_race

| race | arm | n | stops | real | pred | SG bias | SG mae |
|---|---|---|---|---|---|---|---|
| 2021_04_spain | baseline | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | none | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | hinge | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | free_air | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | hinge_free | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | wear_nonneg | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | collinear_hold | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | collinear_hold_hinge | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | collinear_hold_nonneg | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | collinear_hold_nonneg_hinge | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | driver_effects | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | driver_effects_nonneg | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |
| 2021_04_spain | driver_effects_hinge | 336 | 37 | 1.73 | 0.97 | -0.77 | 0.83 |


## Stop-count check (lap 0)

Per race: the field's stop counts (pit-in laps of the cars within 2 laps of the distance, 'stops:cars') and their mode, against the reference's from lap 0 for the same cars (optimizer_bridge.optimize_batch on the arm's world: full window, one sample, up to 2 stops) and its mode; 2-1 is the best two-stop less the best one-stop remaining time over the cars (s; above 0 when one stop is quicker), median [min, max]. stop_counts.csv holds the rows.

| race | field (cars) | field stops | field mode | arm | reference stops | reference mode | match | 2-1 (s) |
|---|---|---|---|---|---|---|---|---|
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | baseline | 1:19 | 1 | no | 5.41 [4.24, 7.63] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | none | 1:19 | 1 | no | 5.41 [4.24, 7.63] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | hinge | 1:19 | 1 | no | 5.77 [3.91, 9.04] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | free_air | 2:19 | 2 | yes | -4.39 [-5.98, -1.34] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | hinge_free | 1:19 | 1 | no | 24.32 [21.22, 24.61] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | wear_nonneg | 1:19 | 1 | no | 5.41 [4.24, 7.63] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | collinear_hold | 1:19 | 1 | no | 5.41 [4.24, 7.63] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | collinear_hold_hinge | 1:19 | 1 | no | 5.77 [3.91, 9.04] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | collinear_hold_nonneg | 1:19 | 1 | no | 5.41 [4.24, 7.63] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | collinear_hold_nonneg_hinge | 1:19 | 1 | no | 5.77 [3.91, 9.04] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | driver_effects | 1:9 2:10 | 2 | yes | -0.17 [-1.48, 2.34] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | driver_effects_nonneg | 1:9 2:10 | 2 | yes | -0.17 [-1.48, 2.34] |
| 2021_04_spain | 19 | 1:2 2:16 3:1 | 2 | driver_effects_hinge | 1:9 2:10 | 2 | yes | -3.62 [-6.47, 1.48] |


## Coverage (intervals at the profile's n_samples; only the posterior arm has a width)

| arm | window | horizon_bucket | n | coverage_68 | coverage_90 | mean_width_90_s | mae_offset_s |
|---|---|---|---|---|---|---|---|
| baseline | full | 0 | 2 | 0.00 | 0.00 | 0.00 | 13.45 |
| baseline | full | 10 | 38 | 0.00 | 0.00 | 0.00 | 10.37 |
| baseline | full | 20 | 38 | 0.00 | 0.00 | 0.00 | 10.66 |
| baseline | full | 30 | 38 | 0.00 | 0.00 | 0.00 | 9.93 |
| baseline | full | 40 | 38 | 0.00 | 0.00 | 0.00 | 12.76 |
| baseline | full | 50 | 69 | 0.00 | 0.00 | 0.00 | 18.98 |
| baseline | full | 60 | 43 | 0.00 | 0.00 | 0.00 | 26.42 |
| baseline | pre_race | 0 | 2 | 0.00 | 0.00 | 0.00 | 10.02 |
| baseline | pre_race | 10 | 38 | 0.00 | 0.00 | 0.00 | 7.89 |
| baseline | pre_race | 20 | 38 | 0.00 | 0.00 | 0.00 | 8.65 |
| baseline | pre_race | 30 | 38 | 0.00 | 0.00 | 0.00 | 8.89 |
| baseline | pre_race | 40 | 38 | 0.00 | 0.00 | 0.00 | 15.31 |
| baseline | pre_race | 50 | 69 | 0.00 | 0.00 | 0.00 | 22.34 |
| baseline | pre_race | 60 | 43 | 0.00 | 0.00 | 0.00 | 18.10 |
| baseline | to_snapshot | 0 | 2 | 0.00 | 0.00 | 0.00 | 14.19 |
| baseline | to_snapshot | 10 | 38 | 0.00 | 0.00 | 0.00 | 12.62 |
| baseline | to_snapshot | 20 | 38 | 0.00 | 0.00 | 0.00 | 16.14 |
| baseline | to_snapshot | 30 | 38 | 0.00 | 0.00 | 0.00 | 15.05 |
| baseline | to_snapshot | 40 | 38 | 0.00 | 0.00 | 0.00 | 36.50 |
| baseline | to_snapshot | 50 | 69 | 0.00 | 0.00 | 0.00 | 27.34 |
| baseline | to_snapshot | 60 | 43 | 0.00 | 0.00 | 0.00 | 16.26 |
| posterior | full | 0 | 2 | 0.00 | 0.00 | 9.83 | 13.36 |
| posterior | full | 10 | 38 | 0.24 | 0.37 | 11.58 | 10.24 |
| posterior | full | 20 | 38 | 0.21 | 0.26 | 14.56 | 10.77 |
| posterior | full | 30 | 38 | 0.21 | 0.37 | 17.07 | 10.01 |
| posterior | full | 40 | 38 | 0.24 | 0.37 | 20.26 | 12.70 |
| posterior | full | 50 | 69 | 0.16 | 0.33 | 22.96 | 18.77 |
| posterior | full | 60 | 43 | 0.43 | 0.43 | 23.45 | 25.75 |
| posterior | pre_race | 0 | 2 | 0.50 | 0.50 | 20.88 | 9.65 |
| posterior | pre_race | 10 | 38 | 0.76 | 0.84 | 35.49 | 7.84 |
| posterior | pre_race | 20 | 38 | 0.89 | 1.00 | 57.46 | 8.59 |
| posterior | pre_race | 30 | 38 | 0.92 | 1.00 | 74.12 | 9.41 |
| posterior | pre_race | 40 | 38 | 0.87 | 0.97 | 87.16 | 16.24 |
| posterior | pre_race | 50 | 69 | 0.79 | 0.91 | 103.75 | 23.76 |
| posterior | pre_race | 60 | 43 | 1.00 | 1.00 | 102.90 | 18.55 |
| posterior | to_snapshot | 0 | 2 | 0.00 | 0.00 | 9.12 | 14.11 |
| posterior | to_snapshot | 10 | 38 | 0.18 | 0.26 | 11.83 | 12.55 |
| posterior | to_snapshot | 20 | 38 | 0.11 | 0.16 | 16.02 | 16.46 |
| posterior | to_snapshot | 30 | 38 | 0.32 | 0.53 | 25.95 | 14.88 |
| posterior | to_snapshot | 40 | 38 | 0.05 | 0.29 | 38.80 | 35.38 |
| posterior | to_snapshot | 50 | 69 | 0.49 | 0.70 | 77.69 | 27.17 |
| posterior | to_snapshot | 60 | 43 | 0.86 | 1.00 | 88.41 | 15.75 |


## Gate and pairs

Per arm and window, over every held-out G row: the finishing-position gate (simulated order) and team-mate sign agreement on remaining time.

| arm | window | within_2_rate_sim | mae_pos_sim | n_pairs | sign_agree_rate |
|---|---|---|---|---|---|
| baseline | full | 0.74 | 1.60 | 112 | 0.49 |
| baseline | to_snapshot | 0.68 | 1.81 | 112 | 0.42 |
| baseline | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| none | full | 0.74 | 1.60 | 112 | 0.49 |
| none | to_snapshot | 0.78 | 1.52 | 112 | 0.50 |
| none | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| hinge | full | 0.77 | 1.64 | 112 | 0.36 |
| hinge | to_snapshot | 0.67 | 1.83 | 112 | 0.44 |
| hinge | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| free_air | full | 0.77 | 1.58 | 112 | 0.49 |
| free_air | to_snapshot | 0.66 | 1.86 | 112 | 0.47 |
| free_air | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| hinge_free | full | 0.82 | 1.48 | 112 | 0.32 |
| hinge_free | to_snapshot | 0.66 | 1.87 | 112 | 0.45 |
| hinge_free | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| posterior | full | 0.74 | 1.61 | 112 | 0.49 |
| posterior | to_snapshot | 0.68 | 1.82 | 112 | 0.41 |
| posterior | pre_race | 0.76 | 1.53 | 112 | 0.53 |
| wear_nonneg | full | 0.74 | 1.60 | 112 | 0.49 |
| wear_nonneg | to_snapshot | 0.71 | 1.74 | 112 | 0.39 |
| wear_nonneg | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| collinear_hold | full | 0.74 | 1.60 | 112 | 0.49 |
| collinear_hold | to_snapshot | 0.67 | 1.85 | 112 | 0.44 |
| collinear_hold | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| collinear_hold_hinge | full | 0.77 | 1.64 | 112 | 0.36 |
| collinear_hold_hinge | to_snapshot | 0.67 | 1.88 | 112 | 0.45 |
| collinear_hold_hinge | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| collinear_hold_nonneg | full | 0.74 | 1.60 | 112 | 0.49 |
| collinear_hold_nonneg | to_snapshot | 0.70 | 1.78 | 112 | 0.40 |
| collinear_hold_nonneg | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| collinear_hold_nonneg_hinge | full | 0.77 | 1.64 | 112 | 0.36 |
| collinear_hold_nonneg_hinge | to_snapshot | 0.70 | 1.81 | 112 | 0.42 |
| collinear_hold_nonneg_hinge | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| driver_effects | full | 0.77 | 1.56 | 112 | 0.50 |
| driver_effects | to_snapshot | 0.76 | 1.57 | 112 | 0.50 |
| driver_effects | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| driver_effects_nonneg | full | 0.77 | 1.56 | 112 | 0.50 |
| driver_effects_nonneg | to_snapshot | 0.77 | 1.56 | 112 | 0.51 |
| driver_effects_nonneg | pre_race | 0.78 | 1.52 | 112 | 0.50 |
| driver_effects_hinge | full | 0.79 | 1.58 | 112 | 0.38 |
| driver_effects_hinge | to_snapshot | 0.77 | 1.54 | 112 | 0.50 |
| driver_effects_hinge | pre_race | 0.78 | 1.52 | 112 | 0.50 |


Pace gate per race (calibration.positions, simulated order, baseline config):

| race | pace | n_drivers | within_2_rate | mae_pos | exact_rate |
|---|---|---|---|---|---|
| 2021_04_spain | none | 19 | 0.53 | 2.63 | 0.16 |
| 2021_04_spain | practice | 19 | 0.90 | 1.16 | 0.32 |
| 2021_04_spain | team | 19 | 0.84 | 1.68 | 0.10 |
| 2021_04_spain | insample | 19 | 0.95 | 0.63 | 0.58 |
| 2021_04_spain | grid | 19 | 0.74 | 1.58 | 0.37 |


## Stop-gain decomposition (to_snapshot, mean per stop and k)

err_s = pred - real gain; the terms sum to it against each arm's own full-window fit (t_floor is that fit's own error, which includes the behavioural part no lap-time model of tyre age captures: drivers manage the last laps of a stint and push the first laps of the next).

| arm | n | err_s | t_base_old | t_base_new | t_wear_old | t_wear_new | t_hinge_old | t_hinge_new | t_fuel | t_floor |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 336 | -1.072 | 0.456 | -0.484 | -0.810 | 0.156 | 0.000 | 0.000 | -0.051 | -0.338 |
| none | 336 | -0.766 | 0.413 | -0.310 | -0.496 | 0.022 | 0.000 | 0.000 | -0.057 | -0.338 |
| hinge | 336 | -1.059 | 0.209 | -0.290 | -0.243 | 0.099 | -0.330 | 0.000 | -0.051 | -0.452 |
| free_air | 336 | -0.704 | 0.484 | -0.530 | -0.605 | 0.125 | 0.000 | 0.000 | -0.020 | -0.158 |
| hinge_free | 336 | -0.692 | 0.267 | -0.388 | -0.078 | 0.088 | -0.269 | 0.000 | -0.017 | -0.294 |
| wear_nonneg | 336 | -0.904 | 0.406 | -0.438 | -0.605 | 0.116 | 0.000 | 0.000 | -0.046 | -0.338 |
| collinear_hold | 336 | -0.905 | 0.494 | -0.432 | -0.715 | 0.115 | 0.000 | 0.000 | -0.030 | -0.338 |
| collinear_hold_hinge | 336 | -0.896 | 0.245 | -0.238 | -0.150 | 0.058 | -0.330 | 0.000 | -0.030 | -0.452 |
| collinear_hold_nonneg | 336 | -0.756 | 0.445 | -0.389 | -0.526 | 0.079 | 0.000 | 0.000 | -0.027 | -0.338 |
| collinear_hold_nonneg_hinge | 336 | -0.746 | 0.196 | -0.195 | 0.039 | 0.023 | -0.330 | 0.000 | -0.027 | -0.452 |
| driver_effects | 336 | -0.973 | 0.440 | -0.445 | -0.968 | 0.145 | 0.000 | 0.000 | -0.047 | -0.098 |
| driver_effects_nonneg | 336 | -0.953 | 0.434 | -0.439 | -0.943 | 0.140 | 0.000 | 0.000 | -0.047 | -0.098 |
| driver_effects_hinge | 336 | -0.960 | 0.258 | -0.297 | -0.567 | 0.102 | -0.245 | 0.000 | -0.049 | -0.161 |


## Factor summary

factors.csv holds every race's row; the per-circuit fuel prior and the 0011b training table come from it.

| factor | races | mean | sd | min | max |
|---|---|---|---|---|---|
| total_laps | 1 | 66.000 | - | 66.000 | 66.000 |
| pit_loss_s | 1 | 23.850 | - | 23.850 | 23.850 |
| observed_pit_loss_s | 1 | 23.850 | - | 23.850 | 23.850 |
| observed_sc_factor | 1 | 1.361 | - | 1.361 | 1.361 |
| observed_sc_factor_median | 1 | 1.398 | - | 1.398 | 1.398 |
| observed_lap1_extra_s | 1 | 9.510 | - | 9.510 | 9.510 |
| observed_sc_pit_loss_factor | 1 | 0.632 | - | 0.632 | 0.632 |
| n_sc_stops | 1 | 3.000 | - | 3.000 | 3.000 |
| observed_pack_penalty_s | 1 | 0.240 | - | 0.240 | 0.240 |
| observed_pack_penalty_median_s | 1 | 0.114 | - | 0.114 | 0.114 |
| free_air_share | 1 | 0.676 | - | 0.676 | 0.676 |
| sc_count | 1 | 1.000 | - | 1.000 | 1.000 |
| sc_laps | 1 | 3.000 | - | 3.000 | 3.000 |
| vsc_count | 1 | 0.000 | - | 0.000 | 0.000 |
| vsc_laps | 1 | 0.000 | - | 0.000 | 0.000 |
| hindsight_fuel_s_per_lap | 1 | -0.058 | - | -0.058 | -0.058 |
| base_delta_s[MEDIUM] | 1 | 0.367 | - | 0.367 | 0.367 |
| base_delta_s[SOFT] | 1 | 1.005 | - | 1.005 | 1.005 |


No race features file (race_features.csv): no circuit covariates yet.

## What changed from Spain

The constants set from 2021 Spain against what the eligible races measure (Spain's own value included in the mean).

| constant | config (from Spain) | Spain measured | races | mean | sd | min | max |
|---|---|---|---|---|---|---|---|
| sc_lap_factor | 1.360 | 1.361 | 1 | 1.361 | - | 1.361 | 1.361 |
| lap1_extra_s | 9.000 | 9.510 | 1 | 9.510 | - | 9.510 | 9.510 |
| sc_pit_loss_factor | 0.500 | 0.632 | 1 | 0.632 | - | 0.632 | 0.632 |
| pre_race_base_delta_s | 1.000 | 0.686 | 1 | 0.686 | - | 0.686 | 0.686 |
| pre_race_fuel_slope_s_per_lap | -0.045 | -0.058 | 1 | -0.058 | - | -0.058 | -0.058 |
| p_sc (per green lap) | 0.015 | 0.016 | 1 | 0.016 | - | 0.016 | 0.016 |
| d_sc (laps) | 3.000 | 3.000 | 1 | 3.000 | - | 3.000 | 3.000 |


Only one eligible race: nothing has changed from Spain yet; every number here is Spain's.
