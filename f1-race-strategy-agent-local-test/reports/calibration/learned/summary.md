# Learned calibrator on the calibration races

Written by `python -m src.sim.learned_calibrator train` (src/sim/learned_calibrator.py). Races (leave-one-race-out, calibration races only): 2021_04_spain, 2023_07_spain, 2023_01_bahrain, 2019_10_united_kingdom, 2022_02_saudi_arabia, 2024_07_emilia_romagna, 2025_23_qatar. sim_code_version 2ccedd3638d5.

## Per-factor adoption

Per factor kind (pooled over compounds), every race's rows from its own fold; ridge_all for the event kinds. Adopted when in both belief windows the learned rmse is below shrinkage's and |coverage - nominal| is no larger at 68 and 90 percent.

| factor | window | rmse shrinkage | rmse learned | cov68 shr | cov68 learned | cov90 shr | cov90 learned | adopted |
|---|---|---|---|---|---|---|---|---|
| base_delta_s | pre_race | 1.7019 | 1.9209 | 0.33 | 0.76 | 0.40 | 0.83 | no |
| base_delta_s | to_snapshot | 0.6929 | 0.7707 | 0.25 | 0.31 | 0.45 | 0.49 | no |
| fuel_s_per_lap | pre_race | 0.0304 | 0.0195 | 0.13 | 0.58 | 0.29 | 0.71 | yes |
| fuel_s_per_lap | to_snapshot | 0.0222 | 0.0211 | 0.24 | 0.35 | 0.41 | 0.55 | yes |
| lap1_extra_s | pre_race | 1.8313 | 2.8747 | 0.00 | 0.68 | 0.00 | 0.68 | no |
| lap1_extra_s | to_snapshot | 1.8313 | 2.8256 | 0.00 | 0.56 | 0.00 | 0.68 | no |
| pit_loss_s | pre_race | 0.7227 | 0.7923 | 0.00 | 0.87 | 0.00 | 1.00 | no |
| pit_loss_s | to_snapshot | 0.7227 | 1.0028 | 0.00 | 0.99 | 0.00 | 1.00 | no |
| sc_lap_factor | pre_race | 0.1458 | 0.1458 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| sc_lap_factor | to_snapshot | 0.1458 | 0.1458 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| sc_pit_loss_factor | pre_race | 0.5238 | 0.5238 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| sc_pit_loss_factor | to_snapshot | 0.5238 | 0.5238 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| vsc_lap_factor | pre_race | 0.0918 | 0.0918 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| vsc_lap_factor | to_snapshot | 0.0918 | 0.0918 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| wear_s_per_lap | pre_race | 0.0693 | 0.0858 | 0.42 | 0.64 | 0.63 | 0.64 | no |
| wear_s_per_lap | to_snapshot | 0.0596 | 0.0691 | 0.41 | 0.54 | 0.61 | 0.67 | no |

## Factor error and coverage by calibrator

| window | factor | calibrator | n | rmse | bias | mean sd | cov68 | cov90 |
|---|---|---|---|---|---|---|---|---|
| pre_race | base_delta_s | hybrid | 193 | 1.7019 | 0.2358 | 0.5000 | 0.33 | 0.40 |
| pre_race | base_delta_s | none | 193 | 1.7019 | 0.2358 | 0.0000 | 0.00 | 0.00 |
| pre_race | base_delta_s | ridge | 193 | 1.9209 | 0.0789 | 1.8706 | 0.76 | 0.83 |
| pre_race | base_delta_s | ridge_all | 193 | 1.9209 | 0.0789 | 1.8706 | 0.76 | 0.83 |
| pre_race | base_delta_s | shrinkage | 193 | 1.7019 | 0.2358 | 0.5000 | 0.33 | 0.40 |
| pre_race | fuel_s_per_lap | hybrid | 85 | 0.0195 | 0.0034 | 0.0195 | 0.58 | 0.71 |
| pre_race | fuel_s_per_lap | none | 85 | 0.0304 | 0.0250 | 0.0000 | 0.00 | 0.00 |
| pre_race | fuel_s_per_lap | ridge | 85 | 0.0195 | 0.0034 | 0.0195 | 0.58 | 0.71 |
| pre_race | fuel_s_per_lap | ridge_all | 85 | 0.0195 | 0.0034 | 0.0195 | 0.58 | 0.71 |
| pre_race | fuel_s_per_lap | shrinkage | 85 | 0.0304 | 0.0250 | 0.0080 | 0.13 | 0.29 |
| pre_race | lap1_extra_s | hybrid | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| pre_race | lap1_extra_s | none | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| pre_race | lap1_extra_s | ridge | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| pre_race | lap1_extra_s | ridge_all | 85 | 2.8747 | 0.5475 | 1.7592 | 0.68 | 0.68 |
| pre_race | lap1_extra_s | shrinkage | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| pre_race | pit_loss_s | hybrid | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| pre_race | pit_loss_s | none | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| pre_race | pit_loss_s | ridge | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| pre_race | pit_loss_s | ridge_all | 85 | 0.7923 | 0.0472 | 1.7175 | 0.87 | 1.00 |
| pre_race | pit_loss_s | shrinkage | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_lap_factor | hybrid | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_lap_factor | none | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_lap_factor | ridge | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_lap_factor | ridge_all | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_lap_factor | shrinkage | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_pit_loss_factor | hybrid | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_pit_loss_factor | none | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_pit_loss_factor | ridge | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_pit_loss_factor | ridge_all | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| pre_race | sc_pit_loss_factor | shrinkage | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| pre_race | vsc_lap_factor | hybrid | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| pre_race | vsc_lap_factor | none | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| pre_race | vsc_lap_factor | ridge | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| pre_race | vsc_lap_factor | ridge_all | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| pre_race | vsc_lap_factor | shrinkage | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| pre_race | wear_s_per_lap | hybrid | 228 | 0.0693 | 0.0045 | 0.0254 | 0.42 | 0.63 |
| pre_race | wear_s_per_lap | none | 228 | 0.0693 | 0.0045 | 0.0000 | 0.00 | 0.00 |
| pre_race | wear_s_per_lap | ridge | 228 | 0.0858 | 0.0153 | 0.0935 | 0.64 | 0.64 |
| pre_race | wear_s_per_lap | ridge_all | 228 | 0.0858 | 0.0153 | 0.0935 | 0.64 | 0.64 |
| pre_race | wear_s_per_lap | shrinkage | 228 | 0.0693 | 0.0045 | 0.0254 | 0.42 | 0.63 |
| to_snapshot | base_delta_s | hybrid | 193 | 0.7989 | 0.2790 | 0.2076 | 0.24 | 0.40 |
| to_snapshot | base_delta_s | none | 193 | 1.7019 | 0.2358 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | base_delta_s | ridge | 193 | 0.7707 | 0.2935 | 0.3456 | 0.31 | 0.49 |
| to_snapshot | base_delta_s | ridge_all | 193 | 0.7707 | 0.2935 | 0.3456 | 0.31 | 0.49 |
| to_snapshot | base_delta_s | shrinkage | 193 | 0.6929 | 0.1238 | 0.1955 | 0.25 | 0.45 |
| to_snapshot | fuel_s_per_lap | hybrid | 85 | 0.0384 | -0.0074 | 0.0105 | 0.33 | 0.46 |
| to_snapshot | fuel_s_per_lap | none | 85 | 0.0304 | 0.0250 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | fuel_s_per_lap | ridge | 85 | 0.0211 | -0.0011 | 0.0108 | 0.35 | 0.55 |
| to_snapshot | fuel_s_per_lap | ridge_all | 85 | 0.0211 | -0.0011 | 0.0108 | 0.35 | 0.55 |
| to_snapshot | fuel_s_per_lap | shrinkage | 85 | 0.0222 | 0.0138 | 0.0055 | 0.24 | 0.41 |
| to_snapshot | lap1_extra_s | hybrid | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | lap1_extra_s | none | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | lap1_extra_s | ridge | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | lap1_extra_s | ridge_all | 85 | 2.8256 | 0.8405 | 2.1972 | 0.56 | 0.68 |
| to_snapshot | lap1_extra_s | shrinkage | 85 | 1.8313 | 1.0940 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | pit_loss_s | hybrid | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | pit_loss_s | none | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | pit_loss_s | ridge | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | pit_loss_s | ridge_all | 85 | 1.0028 | 0.1055 | 2.8711 | 0.99 | 1.00 |
| to_snapshot | pit_loss_s | shrinkage | 85 | 0.7227 | 0.5478 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_lap_factor | hybrid | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_lap_factor | none | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_lap_factor | ridge | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_lap_factor | ridge_all | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_lap_factor | shrinkage | 50 | 0.1458 | -0.1070 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_pit_loss_factor | hybrid | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_pit_loss_factor | none | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_pit_loss_factor | ridge | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_pit_loss_factor | ridge_all | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | sc_pit_loss_factor | shrinkage | 50 | 0.5238 | -0.2952 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | vsc_lap_factor | hybrid | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | vsc_lap_factor | none | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | vsc_lap_factor | ridge | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | vsc_lap_factor | ridge_all | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | vsc_lap_factor | shrinkage | 25 | 0.0918 | 0.0576 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | wear_s_per_lap | hybrid | 228 | 0.0594 | -0.0039 | 0.0225 | 0.38 | 0.61 |
| to_snapshot | wear_s_per_lap | none | 228 | 0.0693 | 0.0045 | 0.0000 | 0.00 | 0.00 |
| to_snapshot | wear_s_per_lap | ridge | 228 | 0.0691 | -0.0157 | 0.0440 | 0.54 | 0.67 |
| to_snapshot | wear_s_per_lap | ridge_all | 228 | 0.0691 | -0.0157 | 0.0440 | 0.54 | 0.67 |
| to_snapshot | wear_s_per_lap | shrinkage | 228 | 0.0596 | -0.0075 | 0.0221 | 0.41 | 0.61 |

## Belief measures

Replay of the teams' real remaining plans from each calibration point (G, horizons 10-49, pace-corrected, race-weighted; 95% race-bootstrap CIs), the paired difference against shrinkage, G's interval coverage, and the stop-gain error at the real pit stops.

| window | calibrator | G mae s [CI] | G bias s | G diff vs shrinkage [CI] | G cov68 | G cov90 | stop-gain mae s [CI] | stop-gain bias s | stop-gain diff [CI] |
|---|---|---|---|---|---|---|---|---|---|
| pre_race | none | 26.12 [16.92, 35.13] | 15.39 | -1.48 [-3.19, 0.10] | 0.14 | 0.21 | 0.73 [0.57, 0.94] | -0.26 | 0.00 [0.00, 0.00] |
| pre_race | shrinkage | 27.59 [18.15, 36.83] | 18.72 | 0.00 [0.00, 0.00] | 0.49 | 0.68 | 0.73 [0.57, 0.94] | -0.26 | 0.00 [0.00, 0.00] |
| pre_race | ridge | 45.00 [34.41, 60.46] | 38.37 | 17.40 [1.94, 33.85] | 0.83 | 0.96 | 1.19 [0.73, 1.82] | 0.08 | 0.45 [0.05, 1.15] |
| pre_race | ridge_all | 44.56 [33.86, 60.21] | 37.79 | 16.96 [1.65, 33.29] | 0.83 | 0.96 | 1.19 [0.73, 1.82] | 0.08 | 0.45 [0.05, 1.15] |
| pre_race | hybrid | 23.61 [19.16, 27.98] | 1.40 | -3.98 [-11.98, 4.67] | 0.65 | 0.89 | 0.70 [0.56, 0.87] | -0.14 | -0.03 [-0.08, 0.01] |
| to_snapshot | none | 26.12 [16.92, 35.13] | 15.39 | 0.71 [-4.25, 5.67] | 0.14 | 0.21 | 0.73 [0.57, 0.94] | -0.26 | -0.10 [-0.19, -0.01] |
| to_snapshot | shrinkage | 25.40 [18.58, 31.50] | 13.06 | 0.00 [0.00, 0.00] | 0.18 | 0.27 | 0.84 [0.68, 0.99] | -0.37 | 0.00 [0.00, 0.00] |
| to_snapshot | ridge | 31.47 [21.34, 41.30] | 12.17 | 6.07 [0.49, 12.28] | 0.22 | 0.34 | 1.01 [0.90, 1.11] | -0.34 | 0.17 [-0.01, 0.33] |
| to_snapshot | ridge_all | 31.16 [21.06, 41.00] | 11.38 | 5.75 [0.08, 12.21] | 0.23 | 0.36 | 1.01 [0.90, 1.11] | -0.34 | 0.17 [-0.01, 0.33] |
| to_snapshot | hybrid | 25.05 [18.29, 31.01] | 0.86 | -0.35 [-3.56, 2.93] | 0.22 | 0.34 | 0.84 [0.72, 0.96] | -0.13 | 0.00 [-0.04, 0.05] |

## Per race (to_snapshot)

| race | calibrator | G mae s | G bias s | G cov90 | stop-gain mae s | stop-gain bias s |
|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | hybrid | 25.60 | -13.89 | 0.47 | 1.07 | -0.02 |
| 2019_10_united_kingdom | none | 39.30 | 31.44 | 0.06 | 1.28 | -0.77 |
| 2019_10_united_kingdom | ridge | 53.42 | 41.16 | 0.54 | 0.88 | -0.09 |
| 2019_10_united_kingdom | ridge_all | 53.25 | 40.70 | 0.56 | 0.88 | -0.09 |
| 2019_10_united_kingdom | shrinkage | 32.32 | 24.07 | 0.36 | 1.16 | -0.81 |
| 2021_04_spain | hybrid | 22.63 | -4.27 | 0.30 | 1.06 | -0.90 |
| 2021_04_spain | none | 10.19 | -0.20 | 0.38 | 0.83 | -0.77 |
| 2021_04_spain | ridge | 31.03 | 5.81 | 0.27 | 1.08 | -0.80 |
| 2021_04_spain | ridge_all | 30.98 | 5.60 | 0.28 | 1.08 | -0.80 |
| 2021_04_spain | shrinkage | 18.74 | 0.46 | 0.32 | 1.09 | -1.01 |
| 2022_02_saudi_arabia | hybrid | 35.65 | -23.34 | 0.28 | 0.63 | 0.17 |
| 2022_02_saudi_arabia | none | 31.77 | 3.94 | 0.07 | 0.48 | -0.13 |
| 2022_02_saudi_arabia | ridge | 33.20 | -18.59 | 0.30 | 1.11 | -0.21 |
| 2022_02_saudi_arabia | ridge_all | 33.35 | -18.77 | 0.30 | 1.11 | -0.21 |
| 2022_02_saudi_arabia | shrinkage | 28.24 | -0.87 | 0.24 | 0.60 | -0.22 |
| 2023_01_bahrain | hybrid | 29.95 | 20.75 | 0.26 | 0.87 | -0.02 |
| 2023_01_bahrain | none | 43.15 | 42.78 | 0.05 | 0.79 | -0.02 |
| 2023_01_bahrain | ridge | 30.12 | 19.52 | 0.28 | 1.18 | -0.96 |
| 2023_01_bahrain | ridge_all | 29.11 | 16.98 | 0.29 | 1.18 | -0.96 |
| 2023_01_bahrain | shrinkage | 34.57 | 29.91 | 0.19 | 0.85 | -0.15 |
| 2023_07_spain | hybrid | 10.17 | 3.33 | 0.54 | 0.69 | 0.10 |
| 2023_07_spain | none | 18.27 | 17.48 | 0.57 | 0.60 | 0.14 |
| 2023_07_spain | ridge | 11.14 | 5.02 | 0.48 | 0.83 | -0.11 |
| 2023_07_spain | ridge_all | 10.68 | 3.78 | 0.53 | 0.83 | -0.11 |
| 2023_07_spain | shrinkage | 11.58 | 6.79 | 0.43 | 0.67 | 0.02 |
| 2024_07_emilia_romagna | hybrid | 16.95 | -5.09 | 0.39 | 0.71 | 0.54 |
| 2024_07_emilia_romagna | none | 11.75 | -9.79 | 0.30 | 0.49 | 0.24 |
| 2024_07_emilia_romagna | ridge | 17.47 | -6.87 | 0.39 | 0.83 | 0.58 |
| 2024_07_emilia_romagna | ridge_all | 17.50 | -6.92 | 0.39 | 0.83 | 0.58 |
| 2024_07_emilia_romagna | shrinkage | 17.01 | 1.06 | 0.26 | 0.59 | 0.36 |
| 2025_23_qatar | hybrid | 34.41 | 28.55 | 0.15 | 0.87 | -0.75 |
| 2025_23_qatar | none | 28.37 | 22.05 | 0.07 | 0.66 | -0.52 |
| 2025_23_qatar | ridge | 43.95 | 39.16 | 0.16 | 1.14 | -0.83 |
| 2025_23_qatar | ridge_all | 43.22 | 38.27 | 0.16 | 1.14 | -0.83 |
| 2025_23_qatar | shrinkage | 35.36 | 30.02 | 0.12 | 0.89 | -0.78 |

## Default

Rule (to_snapshot, race-weighted point estimates): G mae below shrinkage's, stop-gain mae no worse, |G coverage - nominal| no larger at 68 and 90 percent; hybrid first, then plain ridge.

- hybrid: G mae 25.051 s vs 25.404 (lower); stop-gain mae 0.8417 s vs 0.8368 (worse); G coverage 68/90 0.216/0.341 vs 0.175/0.274 (no worse): does not pass
- ridge: G mae 31.475 s vs 25.404 (not lower); stop-gain mae 1.0088 s vs 0.8368 (worse); G coverage 68/90 0.220/0.343 vs 0.175/0.274 (no worse): does not pass

**Not adopted: shrinkage stays the belief's default; the learned calibrator ships as an option, `--belief-config configs/sim/belief_learned.yaml` (calibrator `trained`).**

Artifacts: `models/calibrators/ridge_hybrid/` (sha256 f57c93918dfe): all.json and loo/<race_id>.json, manifest.json.
