# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_04_spain, 2022_02_saudi_arabia, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version 58b50a00e0e3-d2360f4838af. World config 872d3a71bc43 (race default); belief config 8f32db455cfb. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "seed": 0, "max_stops": 2}. Run time 2 s.

There are two simulators. The JUDGE (the eval's "world") scores every plan and builds the
reference: it should be as close as possible to what really happened while using as little
hindsight as it can. The PREDICTIVE simulator (the "belief") is what the agent, its tools and the
rule baselines plan with at a snapshot: it uses no hindsight, and should be as close as possible to
the judge under that restriction. Finishing ORDER leads every section: close-competitor pair order
first, then Spearman and position error, then relative gaps (common offset removed), then absolute
time. Selection rule for the ablation, stated before the results: primary = close-pair order
accuracy of the judge out of sample (A_close_pair_acc); then relative gap error (A_rel_gap_mae_s);
then plan-ranking agreement between predictive and judge (B2_rank_rho, B2_regret_s); absolute time
error is a tie-breaker. A factor that improves order or relative gaps while worsening absolute time
counts as an improvement.

## Headline

| measure | value | definition |
|---|---|---|
| A_close_pair_acc | 0.644 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.967 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 8.883 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.748 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 2.002 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.846 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.602 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.888 | same, Spearman, mean over k |
| A_abs_mae_s | 27.056 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -16.967 | same, mean error |
| A_stop_gain_bias_s | -0.336 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 95.350 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 95.341 | same, mean |
| B1_green_mae_s | 21.419 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.766 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.302 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.533 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 27.821 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.650 | 0.953 | 13.168 | 19.247 | 0.550 | 0.949 | 13.599 | 19.714 | 0.869 | 2.112 | 107.274 | 107.274 | 0.480 | 0.859 | 18.616 | 33.609 |
| 2021_04_spain | 0.714 | 0.981 | 4.775 | 15.058 | 0.714 | 0.981 | 4.976 | 15.505 | 0.791 | 0.511 | 92.068 | 92.068 | 0.571 | 0.940 | 11.472 | 21.826 |
| 2022_02_saudi_arabia | 0.615 | 0.934 | 12.059 | 85.224 | 0.615 | 0.930 | 14.276 | 81.306 | 0.868 | 0.258 | 113.352 | 113.352 | 0.538 | 0.814 | 17.425 | 25.018 |
| 2023_01_bahrain | 0.583 | 0.978 | 9.095 | 25.375 | 0.583 | 0.978 | 8.790 | 25.471 | 0.430 | 3.116 | 96.755 | 96.700 | 0.683 | 0.958 | 17.821 | 45.991 |
| 2023_07_spain | 0.640 | 0.974 | 3.729 | 7.113 | 0.720 | 0.980 | 5.535 | 8.427 | 0.706 | 3.450 | 92.197 | 92.197 | 0.728 | 0.953 | 15.882 | 21.158 |
| 2024_07_emilia_romagna | 0.700 | 0.988 | 6.173 | 11.923 | 0.650 | 0.984 | 6.122 | 11.911 | 0.840 | 2.375 | 78.160 | 78.160 | 0.580 | 0.803 | 12.613 | 21.459 |
| pooled | 0.654 | 0.968 | 8.167 | 27.323 | 0.644 | 0.967 | 8.883 | 27.056 | 0.748 | 2.002 | 95.350 | 95.341 | 0.602 | 0.888 | 15.638 | 27.821 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.654 | 104.000 | 0.968 | 0.820 | 0.791 | 8.167 | 27.323 | -17.340 | 17.667 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.606 | 104.000 | 0.689 | 2.897 | 0.373 | 24.477 | 43.839 | -17.884 | 17.667 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.529 | 104.000 | 0.783 | 2.125 | 0.541 | 21.797 | 36.443 | -22.824 | 17.667 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.644 | 104.000 | 0.967 | 0.828 | 0.779 | 8.883 | 27.056 | -16.967 | 17.667 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.596 | 104.000 | 0.632 | 3.210 | 0.309 | 26.805 | 45.541 | -17.438 | 17.667 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.538 | 104.000 | 0.761 | 2.219 | 0.522 | 23.322 | 37.266 | -22.379 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.529 | 104.000 | 0.933 | 1.283 | 0.654 | 10.097 | 26.014 | -14.938 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.510 | 104.000 | 0.630 | 3.047 | 0.382 | 25.370 | 76.874 | 31.120 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.548 | 104.000 | 0.763 | 2.301 | 0.509 | 23.435 | 72.474 | 26.180 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.538 | 104.000 | 0.928 | 1.359 | 0.644 | 10.245 | 25.809 | -14.966 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.548 | 104.000 | 0.631 | 3.031 | 0.371 | 27.241 | 75.613 | 32.010 | 17.667 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.510 | 104.000 | 0.742 | 2.442 | 0.476 | 25.139 | 71.100 | 27.070 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.683 | 104.000 | 0.968 | 0.809 | 0.781 | 8.461 | 27.469 | -21.847 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.606 | 104.000 | 0.686 | 2.879 | 0.381 | 24.588 | 43.739 | -22.391 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.529 | 104.000 | 0.781 | 2.198 | 0.532 | 21.923 | 37.136 | -27.332 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.702 | 104.000 | 0.970 | 0.757 | 0.817 | 8.929 | 27.649 | -21.938 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.615 | 104.000 | 0.627 | 3.195 | 0.345 | 27.022 | 46.030 | -22.410 | 17.667 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.538 | 104.000 | 0.759 | 2.269 | 0.494 | 23.378 | 38.498 | -27.350 | 17.667 |
| 2 judge, substituted | r8_off | 0 | in-sample | insample | yes | 0.654 | 104.000 | 0.968 | 0.820 | 0.791 | 8.167 | 27.323 | -17.340 | 17.667 |
| 2 judge, substituted | r8_off | 0 | in-sample | none | yes | 0.606 | 104.000 | 0.689 | 2.897 | 0.373 | 24.477 | 43.839 | -17.884 | 17.667 |
| 2 judge, substituted | r8_off | 0 | in-sample | practice | yes | 0.529 | 104.000 | 0.783 | 2.125 | 0.541 | 21.797 | 36.443 | -22.824 | 17.667 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | insample | yes | 0.644 | 104.000 | 0.967 | 0.828 | 0.779 | 8.883 | 27.056 | -16.967 | 17.667 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | none | no | 0.596 | 104.000 | 0.632 | 3.210 | 0.309 | 26.805 | 45.541 | -17.438 | 17.667 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | practice | no | 0.538 | 104.000 | 0.761 | 2.219 | 0.522 | 23.322 | 37.266 | -22.379 | 17.667 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.654 | 104.000 | 0.968 | 0.820 | 0.791 | 8.167 | 27.323 | -17.340 | 17.667 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.606 | 104.000 | 0.689 | 2.897 | 0.373 | 24.477 | 43.839 | -17.884 | 17.667 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.529 | 104.000 | 0.783 | 2.125 | 0.541 | 21.797 | 36.443 | -22.824 | 17.667 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.644 | 104.000 | 0.966 | 0.835 | 0.778 | 8.883 | 26.724 | -16.175 | 17.500 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.596 | 104.000 | 0.622 | 3.238 | 0.302 | 26.805 | 45.326 | -16.666 | 17.500 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.538 | 104.000 | 0.758 | 2.235 | 0.516 | 23.322 | 36.931 | -21.674 | 17.500 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.573 | 75.000 | 0.934 | 1.152 | 0.703 | 11.378 | 29.012 | -20.351 | 16.800 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.547 | 75.000 | 0.599 | 3.040 | 0.413 | 26.533 | 81.242 | 27.845 | 16.800 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.520 | 75.000 | 0.741 | 2.190 | 0.527 | 24.891 | 76.363 | 21.800 | 16.800 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.573 | 75.000 | 0.925 | 1.249 | 0.677 | 11.433 | 29.030 | -20.718 | 16.800 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.560 | 75.000 | 0.583 | 3.148 | 0.367 | 28.554 | 80.053 | 28.152 | 16.800 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.493 | 75.000 | 0.728 | 2.279 | 0.531 | 26.389 | 74.656 | 22.106 | 16.800 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.510 | 104.000 | 0.760 | 2.359 | 0.461 | 23.699 | 75.803 | 41.197 | 17.667 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.615 | 104.000 | 0.916 | 1.536 | 0.621 | 18.914 | 51.380 | 37.989 | 17.667 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.606 | 104.000 | 0.902 | 1.589 | 0.611 | 16.173 | 33.678 | 26.304 | 17.667 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.606 | 104.000 | 0.923 | 1.310 | 0.680 | 11.332 | 16.613 | 6.632 | 17.667 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.673 | 104.000 | 0.938 | 1.135 | 0.771 | 8.073 | 11.037 | -1.707 | 17.667 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.519 | 104.000 | 0.666 | 3.014 | 0.363 | 25.428 | 80.091 | 46.137 | 17.667 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.587 | 104.000 | 0.792 | 2.349 | 0.411 | 21.731 | 57.227 | 35.712 | 17.667 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.615 | 104.000 | 0.798 | 2.097 | 0.505 | 19.888 | 35.220 | 24.808 | 17.667 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.615 | 104.000 | 0.865 | 1.635 | 0.639 | 16.695 | 24.117 | 5.729 | 17.667 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.702 | 104.000 | 0.926 | 1.187 | 0.726 | 12.487 | 18.034 | -1.862 | 17.667 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 106.000 | 42.333 | -15.576 | 24.591 | -14.812 | 11.034 | 0.351 | 4.281 | -3.542 | 296 | 9.532 | -6.168 | 0.459 | 183.000 | 0.591 | -0.337 | 0.138 |
| pooled | lodo | 106.000 | 44.286 | -15.315 | 24.489 | -14.577 | 11.627 | 0.523 | 4.061 | -3.546 | 296 | 9.740 | -6.192 | 0.470 | 183.000 | 0.616 | -0.336 | 0.141 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 294 | 10.996 | -6.232 | 0.533 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 29.45 | -10.78 | 19.25 | -11.87 | 19.28 | 0.39 | 6.27 | -1.57 | 44 | 8.53 | -5.51 | 0.47 | 25.00 | 0.35 | -0.05 | 0.10 |
| 2019_10_united_kingdom | lodo | 17.00 | 34.28 | -11.05 | 19.71 | -12.01 | 19.98 | 0.63 | 6.84 | -1.55 | 44 | 8.86 | -5.71 | 0.48 | 25.00 | 0.37 | -0.05 | 0.13 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.75 | -6.48 | 0.51 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 52.33 | -17.76 | 15.06 | -13.90 | 6.99 | -0.12 | 2.33 | -0.13 | 56 | 7.14 | -4.67 | 0.39 | 37.00 | 0.54 | -0.31 | 0.19 |
| 2021_04_spain | lodo | 19.00 | 55.79 | -18.19 | 15.50 | -14.49 | 7.17 | -0.13 | 2.43 | -0.13 | 56 | 7.38 | -4.87 | 0.40 | 37.00 | 0.55 | -0.30 | 0.20 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 8.79 | -4.62 | 0.46 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 82.30 | -82.30 | 85.22 | -85.22 | 11.49 | -9.89 | 11.88 | -8.93 | 29 | 41.22 | -37.86 | 1.53 | 15.00 | 0.41 | 0.23 | 0.15 |
| 2022_02_saudi_arabia | lodo | 14.00 | 77.56 | -77.56 | 81.31 | -81.31 | 13.29 | -5.82 | 9.31 | -8.94 | 29 | 41.47 | -38.08 | 1.55 | 15.00 | 0.45 | 0.27 | 0.16 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 41.39 | -37.71 | 1.56 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 31.99 | 19.93 | 25.38 | 19.65 | 14.04 | 3.93 | 5.24 | 10.64 | 62 | 5.58 | 1.65 | 0.41 | 42.00 | 0.81 | -0.51 | 0.20 |
| 2023_01_bahrain | lodo | 17.00 | 33.86 | 19.59 | 25.47 | 18.73 | 14.06 | 3.80 | 5.38 | 10.59 | 62 | 5.53 | 1.49 | 0.42 | 42.00 | 0.87 | -0.52 | 0.22 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.10 | 1.88 | 0.53 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 23.99 | -5.52 | 7.11 | -3.86 | 7.02 | 0.66 | 1.25 | - | 63 | 3.66 | -1.81 | 0.21 | 42.00 | 0.72 | -0.62 | 0.18 |
| 2023_07_spain | lodo | 20.00 | 26.63 | -6.19 | 8.43 | -4.03 | 8.05 | 0.05 | 1.11 | - | 63 | 4.00 | -1.44 | 0.22 | 42.00 | 0.74 | -0.64 | 0.16 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 5.33 | -1.59 | 0.29 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 42.96 | -10.87 | 11.92 | -8.84 | 8.91 | 0.17 | 1.19 | - | 42 | 6.54 | -5.05 | 0.25 | 22.00 | 0.40 | -0.22 | -0.05 |
| 2024_07_emilia_romagna | lodo | 19.00 | 45.15 | -11.22 | 11.91 | -8.70 | 8.98 | 0.04 | 1.26 | - | 42 | 6.71 | -4.91 | 0.26 | 22.00 | 0.38 | -0.20 | -0.07 |
| 2024_07_emilia_romagna | loso | - | - | - | - | - | - | - | - | - | 42 | 8.29 | -5.02 | 0.32 | - | - | - | - |

Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:

| variant | compound | 1-5 | 11-15 | 16-20 | 21-25 | 26-30 | 31-35 | 36-40 | 41+ | 6-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| in | HARD | -0.21 | 0.03 | 0.06 | 0.10 | 0.06 | -0.03 | -0.17 | -0.31 | 0.02 |
| in | MEDIUM | -0.03 | 0.02 | -0.03 | -0.16 | 0.04 | 0.19 | 0.79 | 2.35 | -0.01 |
| in | SOFT | -0.04 | 0.14 | 0.14 | 0.21 | 0.45 | 0.21 | - | - | -0.06 |
| lodo | HARD | -0.21 | 0.03 | 0.06 | 0.11 | 0.06 | -0.05 | -0.24 | -0.53 | 0.01 |
| lodo | MEDIUM | -0.04 | 0.02 | -0.03 | -0.15 | 0.09 | 0.33 | 1.15 | 2.58 | 0.00 |
| lodo | SOFT | -0.04 | 0.13 | 0.14 | 0.21 | 0.46 | 0.15 | - | - | -0.06 |

Neutralised laps:

| variant | kind | n | mean_resid_s | mae_s |
|---|---|---|---|---|
| in | sc | 155 | 7.28 | 15.46 |
| in | vsc | 62 | -5.80 | 9.19 |
| lodo | sc | 155 | 7.28 | 15.46 |
| lodo | vsc | 62 | -5.78 | 9.17 |
| loso | sc | 155 | 7.31 | 15.45 |
| loso | vsc | 62 | -5.74 | 9.18 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 1.00 | 0.00 | 54 | 0.33 | 0.00 | 0.51 |
| 2021_04_spain | 2 | 1 | 1:12 2:7 | 0.37 | 0.47 | 1.00 | 0.37 | 72 | 0.44 | 0.24 | 1.46 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 1.00 | 0.00 | 54 | 0.39 | 0.00 | 0.26 |
| 2023_01_bahrain | 2 | 2 | 1:1 2:16 | 0.94 | 0.53 | 0.06 | 0.00 | 60 | 0.13 | 0.00 | 1.29 |
| 2023_07_spain | 2 | 1 | 1:20 | 0.00 | 0.00 | 1.00 | 0.00 | 72 | 0.49 | 0.00 | 1.46 |
| 2024_07_emilia_romagna | 1 | 1 | 1:17 2:2 | 0.89 | 0.89 | 0.11 | 0.00 | 66 | 0.00 | 0.00 | 0.35 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 104 | 104 | -0.125 | -0.035 | 1.930 | -0.787 | 1.828 | -0.146 | -0.081 | 1.821 | 0.115 | -0.501 | -0.026 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 104 | 104 | -0.106 | -0.039 | 1.362 | -0.866 | 1.521 | -0.130 | -0.136 | 1.634 | - | - | - |
| pit_loss_prior | in | pit_loss | the code's constant prior, 22.0 s (no circuit history on this branch) |  | 104 | 104 | 0.029 | 0.000 | 0.294 | 0.553 | -4.885 | 0.000 | -2.725 | 0.000 | -0.043 | -0.010 | 0.026 |
| pit_loss_prior | lodo | pit_loss | the code's constant prior, 22.0 s (no circuit history on this branch) |  | 104 | 104 | 0.058 | 0.004 | 0.046 | 0.841 | -5.164 | 0.000 | -2.881 | 0.000 | - | - | - |
| r8_off | in | stint_limit | no stint limit (A4 only: replays do not apply R8) |  | 104 | 104 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.044 | 0.149 | 0.782 |
| r8_off | lodo | stint_limit | no stint limit (A4 only: replays do not apply R8) |  | 104 | 104 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 104 | 104 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.053 | -0.009 | -0.061 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 104 | 104 | 0.000 | -0.001 | 0.000 | -0.312 | 0.407 | 0.002 | 0.270 | 0.164 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 75 | 104 | -0.081 | -0.034 | 3.211 | 2.331 | -3.547 | -0.209 | -2.567 | 3.219 | -0.034 | -0.352 | 0.281 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 75 | 104 | -0.071 | -0.042 | 2.550 | 2.378 | -4.076 | -0.191 | -2.733 | 3.038 | - | - | - |

A6: hindsight ledger (judge and predictive):

| role | id | input | tag | substitute | source |
|---|---|---|---|---|---|
| judge | fit_window | degradation fit rows (fit_window full) | this race, whole | pre_race_fit | derived |
| judge | clean_flag | lap filter (data.lap_frame clean flag) | this race, whole | - | derived |
| judge | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, whole | - | derived |
| judge | pit_loss | pit loss (RaceParams.pit_loss_s) | this race, whole | pit_loss_prior | derived |
| judge | race_length | race length (snapshot total_laps = the driver's own completed laps) | this race, whole | - | derived |
| judge | compound_list | compound list (RaceParams.compounds: every compound in the race frame) | this race, whole | - | derived |
| judge | stint_limit | R8 stint limit (rank-1 oldest age per compound + 3) | this race, whole | r8_off | derived |
| judge | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| judge | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| judge | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| judge | events | neutralisations (events none) | a priori | - | derived |
| judge | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| judge | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| judge | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |
| predictive | fit_window | degradation fit rows (fit_window to_snapshot) | this race, to snapshot | - | derived |
| predictive | clean_flag | lap filter (data.lap_frame clean flag) | this race, whole | - | derived |
| predictive | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, to snapshot | - | derived |
| predictive | pit_loss | pit loss (RaceParams.pit_loss_s) | this race, whole | pit_loss_prior | derived |
| predictive | race_length | race length (snapshot total_laps = the driver's own completed laps) | this race, whole | - | derived |
| predictive | compound_list | compound list (RaceParams.compounds: every compound in the race frame) | this race, whole | - | derived |
| predictive | stint_limit | R8 stint limit (rank-1 oldest age per compound + 3) | this race, whole | r8_off | derived |
| predictive | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| predictive | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| predictive | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| predictive | events | neutralisations (events rate) | history | - | derived |
| predictive | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| predictive | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| predictive | priors | pre_race base delta, fuel prior, shrinkage taus | history | - | derived |
| predictive | calibrator | calibrator shrinkage | a priori | - | derived |
| predictive | wear_lower_bound | wear bounded below at 0.0 | a priori | - | derived |
| predictive | draws | parameter draws (posterior) and lap noise (residual) | this race, to snapshot | - | derived |
| predictive | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |

## B. Predictive versus judge

Plan panel per snapshot: the real remaining strategy, stay out, one stop at 4 laps spread from the soonest legal stop on each compound both model, two stops at 3 first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) over the plans legal under both configs' own rules. Snapshots every 5 laps for 6 sampled drivers per race. 20 draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.

| race_id | snapshots | rank_rho | regret_s | same_favourite | fav_b_illegal_w | pit_now_agree | rank_rho_green | regret_green_s | plans | b1_mae_s | b1_green_mae_s | b1_green_bias_s | b1_bias_s | b1_mae_per_lap_s | cov_68 | cov_90 | gap_cov_68 | gap_cov_90 | factor_cov_68 | factor_cov_90 | b4_mae_s | b4_bias_s | b4_mae_offset_s | b4_bias_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 54 | 0.869 | 2.112 | 0.815 | 0.000 | 0.898 | 0.904 | 2.079 | 1906 | 107.274 | 30.113 | 30.113 | 107.274 | 3.750 | 0.001 | 0.066 | 0.488 | 0.653 | 0.365 | 0.587 | 71.908 | 65.846 | 33.609 | 24.497 |
| 2021_04_spain | 72 | 0.791 | 0.511 | 0.708 | 0.000 | 0.857 | 0.822 | 0.557 | 2568 | 92.068 | 15.669 | 11.777 | 92.068 | 2.540 | 0.363 | 0.539 | 0.518 | 0.766 | 0.393 | 0.595 | 38.390 | 7.859 | 21.826 | 7.804 |
| 2022_02_saudi_arabia | 54 | 0.868 | 0.258 | 0.926 | 0.000 | 0.938 | 0.889 | 0.124 | 1806 | 113.352 | 33.213 | 33.213 | 113.352 | 4.522 | 0.006 | 0.054 | 0.854 | 0.937 | 0.571 | 0.683 | 35.948 | 3.707 | 25.018 | 5.782 |
| 2023_01_bahrain | 60 | 0.430 | 3.116 | 0.417 | 0.000 | 0.885 | 0.426 | 2.987 | 2141 | 96.755 | 23.680 | 16.302 | 96.700 | 3.142 | 0.335 | 0.479 | 0.432 | 0.742 | 0.329 | 0.586 | 47.320 | 42.841 | 45.991 | 42.659 |
| 2023_07_spain | 72 | 0.706 | 3.450 | 0.583 | 0.000 | 0.831 | 0.709 | 3.726 | 2593 | 92.197 | 19.242 | 19.170 | 92.197 | 2.538 | 0.164 | 0.271 | 0.338 | 0.589 | 0.262 | 0.417 | 28.318 | 22.268 | 21.158 | 19.272 |
| 2024_07_emilia_romagna | 66 | 0.840 | 2.375 | 0.545 | 0.000 | 0.712 | 0.861 | 2.438 | 2423 | 78.160 | 12.215 | 11.550 | 78.160 | 2.295 | 0.129 | 0.297 | 0.146 | 0.265 | 0.273 | 0.377 | 35.743 | -9.892 | 21.459 | 3.813 |
| pooled | 378 | 0.748 | 2.002 | 0.656 | 0.000 | 0.846 | 0.766 | 2.030 | 13437 | 95.350 | 21.419 | 19.366 | 95.341 | 3.029 | 0.178 | 0.302 | 0.444 | 0.642 | 0.358 | 0.533 | 42.500 | 21.758 | 27.821 | 17.253 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 493.00 | 15.61 | 15.61 |
| 10.00 | 2880.00 | 41.43 | 41.43 |
| 20.00 | 2875.00 | 60.45 | 60.41 |
| 30.00 | 2775.00 | 108.87 | 108.87 |
| 40.00 | 2557.00 | 139.31 | 139.31 |
| 50.00 | 1453.00 | 174.73 | 174.72 |
| 60.00 | 404.00 | 168.82 | 168.82 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 4355 | 156.83 | 156.83 |
| late | 3859 | 38.80 | 38.80 |
| mid | 5223 | 85.87 | 85.85 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 3950 | 89.78 | 89.77 |
| real | 368 | 95.23 | 95.23 |
| stay | 98 | 102.43 | 101.93 |
| two | 9021 | 97.71 | 97.71 |

B5 future blindness: {"status": "not checked", "detail": "tests/future_blindness.py is not on this branch"}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | belief support edge @k0 | oldest age run | 16 | 6 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k0 | median stint-end age | 16 | 6 | 10.406 | 10.406 | - | - |
| tyre_life | as-of data (race to k0, practice) | oldest age run | 16 | 6 | 14.250 | -11.000 | - | - |
| tyre_life | as-of data (race to k0, practice) | median stint-end age | 16 | 6 | 7.844 | -0.594 | - | - |
| tyre_life | belief support edge @k10 | oldest age run | 16 | 6 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k10 | median stint-end age | 16 | 6 | 10.406 | 10.406 | - | - |
| tyre_life | as-of data (race to k10, practice) | oldest age run | 16 | 6 | 14.062 | -10.812 | - | - |
| tyre_life | as-of data (race to k10, practice) | median stint-end age | 16 | 6 | 7.656 | -0.406 | - | - |
| tyre_life | belief support edge @k20 | oldest age run | 16 | 6 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k20 | median stint-end age | 16 | 6 | 10.406 | 10.406 | - | - |
| tyre_life | as-of data (race to k20, practice) | oldest age run | 16 | 6 | 13.000 | -9.750 | - | - |
| tyre_life | as-of data (race to k20, practice) | median stint-end age | 16 | 6 | 6.969 | 0.656 | - | - |
| tyre_life | belief support edge @k(mid) | oldest age run | 16 | 6 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k(mid) | median stint-end age | 16 | 6 | 10.406 | 10.406 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | oldest age run | 16 | 6 | 9.625 | -6.375 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | median stint-end age | 16 | 6 | 6.969 | 4.031 | - | - |
| wear_rate | belief @k0 | within-stint slope less fuel | 15 | 6 | 0.033 | -0.008 | 0.600 | 0.933 |
| wear_rate | belief @k10 | within-stint slope less fuel | 15 | 6 | 0.042 | -0.016 | 0.400 | 0.467 |
| wear_rate | belief @k20 | within-stint slope less fuel | 15 | 6 | 0.042 | -0.024 | 0.333 | 0.467 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 15 | 6 | 0.034 | -0.022 | 0.333 | 0.533 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 15 | 6 | 0.016 | -0.013 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 183 | 6 | 0.772 | -0.310 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 169 | 6 | 1.126 | -0.534 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 113 | 6 | 0.872 | -0.314 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 71 | 6 | 0.627 | 0.029 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 14 | 6 | 0.934 | -0.598 | - | - |
| compound_gap | belief @k0 | qualifying | 3 | 3 | 0.362 | 0.362 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 14 | 6 | 0.724 | -0.467 | - | - |
| compound_gap | belief @k10 | qualifying | 3 | 3 | 0.750 | 0.698 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 14 | 6 | 0.514 | -0.123 | - | - |
| compound_gap | belief @k20 | qualifying | 3 | 3 | 0.641 | 0.578 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 14 | 6 | 0.317 | -0.030 | - | - |
| compound_gap | belief @k(mid) | qualifying | 3 | 3 | 0.897 | 0.826 | - | - |
| fuel | prior (config) | race (judge full fit) | 6 | 6 | 0.023 | 0.023 | - | - |
| fuel | belief @k0 | race (judge full fit) | 6 | 6 | 0.023 | 0.023 | 0.167 | 0.333 |
| fuel | belief @k10 | race (judge full fit) | 6 | 6 | 0.021 | 0.020 | 0.167 | 0.333 |
| fuel | belief @k20 | race (judge full fit) | 6 | 6 | 0.015 | 0.015 | 0.333 | 0.333 |
| fuel | belief @k(mid) | race (judge full fit) | 6 | 6 | 0.008 | 0.004 | 0.500 | 0.667 |
| pit_loss | prior (code constant) | whole race, robust (median of green stops) | 6 | 6 | 2.282 | -1.637 | - | - |
| pit_loss | prior (code constant) | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 4.233 | -1.917 | - | - |
| pit_loss | belief charged @k0 | whole race, robust (median of green stops) | 6 | 6 | 2.435 | 0.280 | - | - |
| pit_loss | belief charged @k0 | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 0.000 | 0.000 | - | - |
| pit_loss | belief charged @k0 | whole race, SC stops | 3 | 3 | 6.280 | -5.049 | - | - |
| pit_loss | as-of data (green stops to k0) | whole race, robust (median of green stops) | 0 | 0 | - | - | - | - |
| pit_loss | as-of data (green stops to k0) | whole race, mean (RaceParams.pit_loss_s) | 0 | 0 | - | - | - | - |
| pit_loss | belief charged @k10 | whole race, robust (median of green stops) | 6 | 6 | 2.435 | 0.280 | - | - |
| pit_loss | belief charged @k10 | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 0.000 | 0.000 | - | - |
| pit_loss | belief charged @k10 | whole race, SC stops | 3 | 3 | 6.280 | -5.049 | - | - |
| pit_loss | as-of data (green stops to k10) | whole race, robust (median of green stops) | 3 | 3 | 0.627 | -0.627 | - | - |
| pit_loss | as-of data (green stops to k10) | whole race, mean (RaceParams.pit_loss_s) | 3 | 3 | 1.901 | -1.901 | - | - |
| pit_loss | belief charged @k20 | whole race, robust (median of green stops) | 6 | 6 | 2.435 | 0.280 | - | - |
| pit_loss | belief charged @k20 | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 0.000 | 0.000 | - | - |
| pit_loss | belief charged @k20 | whole race, SC stops | 3 | 3 | 6.280 | -5.049 | - | - |
| pit_loss | as-of data (green stops to k20) | whole race, robust (median of green stops) | 6 | 6 | 0.459 | 0.077 | - | - |
| pit_loss | as-of data (green stops to k20) | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 2.685 | -0.203 | - | - |
| pit_loss | belief charged @k(mid) | whole race, robust (median of green stops) | 6 | 6 | 2.435 | 0.280 | - | - |
| pit_loss | belief charged @k(mid) | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 0.000 | 0.000 | - | - |
| pit_loss | belief charged @k(mid) | whole race, SC stops | 3 | 3 | 6.280 | -5.049 | - | - |
| pit_loss | as-of data (green stops to k(mid)) | whole race, robust (median of green stops) | 6 | 6 | 0.257 | -0.091 | - | - |
| pit_loss | as-of data (green stops to k(mid)) | whole race, mean (RaceParams.pit_loss_s) | 6 | 6 | 2.651 | -0.370 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 6 | 6 | 1.725 | 1.535 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 6 | 6 | 0.339 | 0.075 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 190 | 6 | 0.584 | 0.121 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 190 | 6 | 0.711 | 0.384 | - | - |
| neutralisations | judge expected (none) | occurred | 24 | 6 | 0.917 | -0.917 | - | - |
| neutralisations | judge config | observed (mean SC lap / ref lap) | 3 | 3 | 0.074 | -0.074 | - | - |
| neutralisations | judge config | observed (median VSC lap / green prediction) | 2 | 2 | 0.072 | 0.061 | - | - |
| neutralisations | belief expected (rate) | occurred | 24 | 6 | 1.347 | 0.853 | - | - |
| neutralisations | belief config | observed (mean SC lap / ref lap) | 3 | 3 | 0.074 | -0.074 | - | - |
| neutralisations | belief config | observed (median VSC lap / green prediction) | 2 | 2 | 0.072 | 0.061 | - | - |

Component status:

| component | status | races |
|---|---|---|
| compound_gap | ok | 6 |
| fresh_tyre_gain | ok | 6 |
| fuel | ok | 6 |
| lap1_and_warmup | ok | 6 |
| neutralisations | ok | 6 |
| pit_loss | ok | 6 |
| traffic_rejoin | ok | 6 |
| tyre_life | not present | 6 |
| tyre_life | ok | 6 |
| wear_rate | ok | 6 |

