# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_11_austria, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version a2ede471f2ed-90d1fa9a6662. World config fec8ad022c3d (race default); belief config 28b37a277a44. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": false, "components": true, "field_replay": false, "seed": 0, "max_stops": 2}. Run time 389 s.

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
| A_close_pair_acc | 0.747 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.960 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 7.600 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.886 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 2.932 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.865 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.666 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.876 | same, Spearman, mean over k |
| A_abs_mae_s | 13.375 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -4.795 | same, mean error |
| A_stop_gain_bias_s | -0.191 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 57.072 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 56.072 | same, mean |
| B1_green_mae_s | 17.674 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.908 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.972 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.485 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 31.982 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |
| E_real_regret_s | 4.772 | expert consistency (secondary): judge time of the team's real remaining strategy less the judge's best panel plan, mean over snapshots; should fall where passing is hard, never chased to zero |
| E_real_over_20s | 0.057 | same, share of snapshots where the judge says the real strategy lost more than 20 s |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.650 | 0.961 | 3.737 | 12.838 | 0.650 | 0.961 | 3.375 | 12.657 | 0.737 | 4.435 | 39.867 | 36.920 | 0.540 | 0.873 | 17.532 | 23.644 |
| 2021_01_bahrain | 0.909 | 0.992 | 5.122 | 11.887 | 0.909 | 0.992 | 5.000 | 12.110 | 0.934 | 1.369 | 64.340 | 64.340 | 0.691 | 0.940 | 13.902 | 15.940 |
| 2021_04_spain | 0.714 | 0.974 | 4.901 | 12.988 | 0.714 | 0.972 | 5.102 | 13.145 | 0.773 | 6.963 | 64.323 | 64.323 | 0.757 | 0.974 | 10.488 | 17.917 |
| 2021_09_austrian | 0.750 | 0.958 | 5.516 | 12.286 | 0.750 | 0.970 | 5.720 | 12.291 | 0.827 | 3.506 | 62.057 | 62.049 | 0.512 | 0.911 | 19.871 | 25.320 |
| 2021_20_qatar | 0.462 | 0.973 | 6.811 | 13.731 | 0.462 | 0.973 | 6.668 | 14.150 | 0.826 | 3.256 | 60.622 | 60.621 | 0.769 | 0.939 | 23.145 | 46.908 |
| 2021_22_united_arab_emirates | 0.857 | 0.975 | 27.618 | 26.080 | 0.893 | 0.982 | 27.649 | 25.962 | 0.887 | 4.708 | 47.357 | 46.883 | 0.693 | 0.854 | 33.623 | 50.958 |
| 2022_01_bahrain | 0.761 | 0.796 | 9.529 | 19.074 | 0.761 | 0.821 | 9.598 | 19.067 | 0.937 | 2.025 | 62.291 | 62.153 | 0.713 | 0.736 | 48.079 | 104.577 |
| 2022_02_saudi_arabia | 0.692 | 0.960 | 6.116 | 10.246 | 0.692 | 0.943 | 6.575 | 10.672 | 0.912 | 0.478 | 47.914 | 46.292 | 0.615 | 0.849 | 17.002 | 32.458 |
| 2022_06_spain | 0.500 | 0.967 | 12.332 | 12.887 | 0.500 | 0.967 | 12.540 | 13.539 | 0.909 | 2.490 | 93.998 | 93.998 | 0.800 | 0.968 | 17.218 | 23.634 |
| 2022_08_azerbaijan | 0.364 | 0.938 | 9.094 | 12.229 | 0.364 | 0.944 | 9.014 | 12.280 | 0.788 | 5.452 | 48.181 | 46.992 | 0.455 | 0.813 | 23.085 | 19.675 |
| 2022_09_canada | 0.800 | 0.978 | 5.331 | 7.178 | 0.800 | 0.973 | 5.596 | 7.148 | 0.932 | 0.806 | 53.949 | 49.062 | 0.627 | 0.811 | 42.192 | 41.974 |
| 2022_11_austria | 1.000 | 0.964 | 6.431 | 11.652 | 1.000 | 0.964 | 5.845 | 11.334 | 0.916 | 3.300 | 48.809 | 48.792 | 0.467 | 0.750 | 15.795 | 19.581 |
| 2022_12_france | 0.727 | 0.979 | 5.045 | 19.828 | 0.773 | 0.982 | 5.008 | 20.309 | 0.946 | 1.836 | 43.975 | 43.970 | 0.745 | 0.884 | 12.397 | 27.843 |
| 2022_20_mexico | 0.714 | 0.967 | 8.809 | 12.580 | 0.786 | 0.972 | 8.682 | 12.338 | 0.913 | 4.354 | 49.452 | 49.062 | 0.386 | 0.818 | 28.851 | 34.442 |
| 2022_21_brazil | 0.854 | 0.971 | 3.380 | 7.890 | 0.875 | 0.973 | 3.556 | 8.044 | 0.937 | 3.876 | 49.950 | 43.889 | 0.729 | 0.876 | 29.953 | 33.070 |
| 2022_22_united_arab_emirates | 0.643 | 0.989 | 4.170 | 14.413 | 0.643 | 0.986 | 5.100 | 14.803 | 0.945 | 1.832 | 57.312 | 57.312 | 0.614 | 0.944 | 11.376 | 17.565 |
| 2023_01_bahrain | 0.583 | 0.978 | 5.805 | 13.520 | 0.500 | 0.975 | 5.807 | 13.714 | 0.822 | 1.547 | 85.524 | 85.524 | 0.733 | 0.962 | 19.782 | 28.689 |
| 2023_07_spain | 0.800 | 0.983 | 7.120 | 9.557 | 0.800 | 0.983 | 8.827 | 10.335 | 0.947 | 1.687 | 59.806 | 59.806 | 0.760 | 0.941 | 21.218 | 17.338 |
| 2024_07_emilia_romagna | 0.750 | 0.914 | 4.839 | 10.228 | 0.750 | 0.914 | 4.729 | 10.218 | 0.910 | 1.307 | 36.946 | 36.867 | 0.650 | 0.805 | 11.987 | 20.379 |
| pooled | 0.739 | 0.959 | 7.458 | 13.215 | 0.747 | 0.960 | 7.600 | 13.375 | 0.886 | 2.932 | 57.072 | 56.072 | 0.666 | 0.876 | 21.973 | 31.982 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.739 | 368 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.739 | 368 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.568 | 368 | 0.878 | 1.574 | 0.597 | 26.247 | 28.982 | -7.900 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.747 | 368 | 0.960 | 0.828 | 0.828 | 7.600 | 13.375 | -4.795 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.747 | 368 | 0.960 | 0.828 | 0.828 | 7.600 | 13.375 | -4.795 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.565 | 368 | 0.875 | 1.596 | 0.589 | 26.348 | 29.000 | -7.882 | 17.053 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.546 | 368 | 0.773 | 2.270 | 0.496 | 28.370 | 79.943 | -15.119 | 17.053 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.641 | 368 | 0.870 | 1.629 | 0.613 | 27.397 | 48.612 | 14.777 | 17.053 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.652 | 368 | 0.885 | 1.449 | 0.653 | 22.922 | 32.443 | 8.045 | 17.053 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.715 | 368 | 0.917 | 1.121 | 0.737 | 17.207 | 25.253 | 6.650 | 17.053 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.777 | 368 | 0.934 | 0.919 | 0.821 | 13.971 | 21.033 | 4.688 | 17.053 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.514 | 368 | 0.695 | 2.877 | 0.383 | 25.728 | 81.797 | -12.031 | 17.053 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.541 | 368 | 0.784 | 2.421 | 0.409 | 22.186 | 51.164 | 18.837 | 17.053 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.606 | 368 | 0.817 | 1.954 | 0.533 | 17.954 | 37.125 | 9.045 | 17.053 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.690 | 368 | 0.886 | 1.450 | 0.677 | 16.764 | 30.150 | 7.587 | 17.053 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.755 | 368 | 0.914 | 1.129 | 0.747 | 13.662 | 24.441 | 5.297 | 17.053 |

## Noise floor (the judge handed the real race, step by step)

The judge's whole-race replay (version 1, leave-one-driver-out fit, whole-race pace offset: level 0 is the headline) with each level's real values in place of the model's, cumulative from 1 to 4, then each alone: stint_pace: each stint's own pace offset (mean clean-lap residual of that stint, five or more clean laps; else the car's), on every lap of the stint; pit_laps: in- and out-laps as recorded, no charged pit loss (the real stop durations and pit-lane driving); neutral_laps: SC and VSC laps as recorded; lap1: lap 1 as recorded (the start). What level 4 leaves is lap-to-lap noise around each stint's pace, incidents, errors and unmodelled effects. Pooled with race-cluster bootstrap 95% intervals:

| level | close_pair_acc | close_pair_lo | close_pair_hi | spearman | rel_gap_mae_s | rel_gap_lo | rel_gap_hi | abs_mae_s | close_pairs | races |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 judge | 0.747 | 0.683 | 0.798 | 0.960 | 7.600 | 5.748 | 10.223 | 13.375 | 368 | 19 |
| 1 +stint pace | 0.658 | 0.596 | 0.709 | 0.948 | 6.375 | 4.459 | 9.045 | 12.814 | 368 | 19 |
| 2 +pit laps | 0.652 | 0.606 | 0.701 | 0.941 | 6.900 | 4.886 | 9.345 | 11.767 | 368 | 19 |
| 3 +neutral laps | 0.611 | 0.557 | 0.673 | 0.941 | 5.241 | 4.226 | 6.339 | 8.680 | 368 | 19 |
| 4 +lap 1 | 0.745 | 0.682 | 0.800 | 0.967 | 3.166 | 2.589 | 3.885 | 7.704 | 368 | 19 |
| alone: pit laps | 0.712 | 0.667 | 0.755 | 0.943 | 8.074 | 6.360 | 10.236 | 12.226 | 368 | 19 |
| alone: neutral laps | 0.606 | 0.550 | 0.675 | 0.890 | 8.964 | 7.137 | 11.214 | 13.683 | 368 | 19 |
| alone: lap 1 | 0.728 | 0.661 | 0.778 | 0.953 | 7.466 | 5.779 | 9.887 | 14.054 | 368 | 19 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 324.000 | 13.171 | -4.661 | 13.171 | -4.661 | 11.966 | -0.632 | 3.437 | 0.681 | 925 | 6.354 | -1.988 | 0.441 | 565.000 | 0.490 | -0.187 | -0.014 |
| pooled | lodo | 324.000 | 13.348 | -4.634 | 13.348 | -4.634 | 12.144 | -0.593 | 3.498 | 0.678 | 925 | 6.545 | -2.003 | 0.451 | 565.000 | 0.513 | -0.191 | -0.013 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 917 | 11.122 | -1.176 | 0.701 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 12.84 | -3.36 | 12.84 | -3.36 | 14.72 | -2.56 | 4.16 | 2.92 | 44 | 5.50 | -0.36 | 0.38 | 25.00 | 0.34 | 0.13 | -0.08 |
| 2019_10_united_kingdom | lodo | 17.00 | 12.66 | -3.79 | 12.66 | -3.79 | 14.80 | -2.52 | 4.36 | 2.88 | 44 | 5.36 | -0.55 | 0.38 | 25.00 | 0.38 | 0.12 | -0.07 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.23 | -3.65 | 0.52 | - | - | - | - |
| 2021_01_bahrain | in | 18.00 | 11.89 | -4.26 | 11.89 | -4.26 | 12.84 | -1.37 | 2.41 | -0.24 | 56 | 3.95 | -3.03 | 0.30 | 35.00 | 0.40 | -0.14 | 0.10 |
| 2021_01_bahrain | lodo | 18.00 | 12.11 | -4.43 | 12.11 | -4.43 | 12.97 | -1.37 | 2.43 | -0.24 | 56 | 4.06 | -3.09 | 0.31 | 35.00 | 0.42 | -0.14 | 0.10 |
| 2021_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 55 | 4.83 | -2.81 | 0.37 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 12.99 | -12.12 | 12.99 | -12.12 | 7.63 | -0.71 | 3.45 | 0.60 | 56 | 5.15 | -3.62 | 0.30 | 37.00 | 0.47 | -0.04 | -0.17 |
| 2021_04_spain | lodo | 19.00 | 13.15 | -12.08 | 13.15 | -12.08 | 7.43 | -0.69 | 3.49 | 0.60 | 56 | 5.18 | -3.63 | 0.30 | 37.00 | 0.48 | -0.04 | -0.17 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 7.11 | -4.36 | 0.39 | - | - | - | - |
| 2021_09_austrian | in | 19.00 | 12.29 | -10.19 | 12.29 | -10.19 | 10.81 | -3.35 | 2.88 | -0.73 | 46 | 6.36 | -5.96 | 0.26 | 26.00 | 0.44 | -0.25 | 0.16 |
| 2021_09_austrian | lodo | 19.00 | 12.29 | -9.98 | 12.29 | -9.98 | 10.76 | -3.31 | 2.85 | -0.73 | 46 | 6.49 | -5.90 | 0.27 | 26.00 | 0.45 | -0.24 | 0.16 |
| 2021_09_austrian | loso | - | - | - | - | - | - | - | - | - | 45 | 8.68 | -5.84 | 0.34 | - | - | - | - |
| 2021_20_qatar | in | 18.00 | 13.73 | -11.28 | 13.73 | -11.28 | 12.23 | -1.92 | 2.98 | 0.35 | 46 | 5.44 | -2.94 | 0.34 | 27.00 | 0.71 | -0.34 | 0.07 |
| 2021_20_qatar | lodo | 18.00 | 14.15 | -11.61 | 14.15 | -11.61 | 12.42 | -1.94 | 3.03 | 0.35 | 46 | 5.74 | -3.05 | 0.35 | 27.00 | 0.74 | -0.36 | 0.08 |
| 2021_20_qatar | loso | - | - | - | - | - | - | - | - | - | 46 | 7.71 | -2.80 | 0.44 | - | - | - | - |
| 2021_22_united_arab_emirates | in | 15.00 | 26.08 | -17.98 | 26.08 | -17.98 | 27.49 | 8.32 | 7.92 | -4.18 | 41 | 17.46 | -12.76 | 2.37 | 19.00 | 0.45 | -0.10 | 0.02 |
| 2021_22_united_arab_emirates | lodo | 15.00 | 25.96 | -17.99 | 25.96 | -17.99 | 27.50 | 8.32 | 7.92 | -4.19 | 41 | 17.80 | -12.76 | 2.39 | 19.00 | 0.48 | -0.10 | 0.01 |
| 2021_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 40 | 25.03 | -18.34 | 3.20 | - | - | - | - |
| 2022_01_bahrain | in | 19.00 | 19.07 | 3.94 | 19.07 | 3.94 | 16.11 | -1.30 | 2.90 | 2.91 | 75 | 8.93 | 1.32 | 0.81 | 54.00 | 0.50 | -0.24 | 0.13 |
| 2022_01_bahrain | lodo | 19.00 | 19.07 | 4.01 | 19.07 | 4.01 | 16.07 | -1.29 | 2.90 | 2.87 | 75 | 9.07 | 1.34 | 0.82 | 54.00 | 0.52 | -0.24 | 0.13 |
| 2022_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 75 | 19.23 | 3.24 | 1.67 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 10.25 | -9.87 | 10.25 | -9.87 | 6.68 | -4.50 | 7.82 | 1.03 | 29 | 7.01 | -4.21 | 0.32 | 15.00 | 0.31 | 0.02 | 0.13 |
| 2022_02_saudi_arabia | lodo | 14.00 | 10.67 | -9.97 | 10.67 | -9.97 | 7.24 | -4.58 | 7.82 | 1.03 | 29 | 7.19 | -4.22 | 0.33 | 15.00 | 0.35 | 0.01 | 0.13 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 13.18 | -7.50 | 0.55 | - | - | - | - |
| 2022_06_spain | in | 18.00 | 12.89 | -9.95 | 12.89 | -9.95 | 13.23 | 0.05 | 1.32 | - | 69 | 3.77 | -2.34 | 0.27 | 49.00 | 0.51 | -0.14 | 0.12 |
| 2022_06_spain | lodo | 18.00 | 13.54 | -10.51 | 13.54 | -10.51 | 14.17 | 0.15 | 1.35 | - | 69 | 4.23 | -2.56 | 0.29 | 49.00 | 0.53 | -0.14 | 0.15 |
| 2022_06_spain | loso | - | - | - | - | - | - | - | - | - | 67 | 21.00 | 13.11 | 0.94 | - | - | - | - |
| 2022_08_azerbaijan | in | 16.00 | 12.23 | -2.12 | 12.23 | -2.12 | 12.15 | 1.30 | 5.68 | 0.78 | 40 | 4.41 | -1.67 | 0.31 | 23.00 | 0.49 | -0.25 | 0.08 |
| 2022_08_azerbaijan | lodo | 16.00 | 12.28 | -1.76 | 12.28 | -1.76 | 12.23 | 1.27 | 5.77 | 0.78 | 40 | 4.51 | -1.52 | 0.31 | 23.00 | 0.51 | -0.27 | 0.07 |
| 2022_08_azerbaijan | loso | - | - | - | - | - | - | - | - | - | 40 | 5.74 | -1.20 | 0.38 | - | - | - | - |
| 2022_09_canada | in | 17.00 | 7.18 | 6.96 | 7.18 | 6.96 | 5.33 | -0.40 | 3.20 | 2.02 | 47 | 7.19 | 3.37 | 0.34 | 30.00 | 0.43 | -0.03 | 0.08 |
| 2022_09_canada | lodo | 17.00 | 7.15 | 6.85 | 7.15 | 6.85 | 5.60 | -0.41 | 3.32 | 2.03 | 47 | 7.38 | 3.34 | 0.35 | 30.00 | 0.46 | -0.04 | 0.09 |
| 2022_09_canada | loso | - | - | - | - | - | - | - | - | - | 47 | 14.38 | 2.99 | 0.62 | - | - | - | - |
| 2022_11_austria | in | 7.00 | 11.65 | -10.08 | 11.65 | -10.08 | 6.16 | -2.27 | 2.33 | -0.46 | 24 | 3.51 | -1.91 | 0.19 | 9.00 | 0.41 | -0.13 | 0.04 |
| 2022_11_austria | lodo | 7.00 | 11.33 | -10.20 | 11.33 | -10.20 | 6.43 | -2.29 | 2.36 | -0.46 | 24 | 3.84 | -1.92 | 0.21 | 9.00 | 0.44 | -0.10 | 0.02 |
| 2022_11_austria | loso | - | - | - | - | - | - | - | - | - | 23 | 5.81 | -1.57 | 0.30 | - | - | - | - |
| 2022_12_france | in | 16.00 | 19.83 | 14.04 | 19.83 | 14.04 | 10.47 | -0.47 | 5.65 | 7.06 | 36 | 10.57 | 7.15 | 0.44 | 20.00 | 0.32 | -0.11 | -0.05 |
| 2022_12_france | lodo | 16.00 | 20.31 | 14.43 | 20.31 | 14.43 | 10.46 | -0.08 | 5.79 | 7.06 | 36 | 10.72 | 7.11 | 0.45 | 20.00 | 0.36 | -0.12 | -0.05 |
| 2022_12_france | loso | - | - | - | - | - | - | - | - | - | 36 | 11.96 | 6.39 | 0.52 | - | - | - | - |
| 2022_20_mexico | in | 19.00 | 12.58 | -3.45 | 12.58 | -3.45 | 13.02 | 1.54 | 2.42 | -2.19 | 41 | 10.46 | -2.78 | 0.38 | 21.00 | 0.46 | -0.26 | -1.06 |
| 2022_20_mexico | lodo | 19.00 | 12.34 | -3.54 | 12.34 | -3.54 | 12.91 | 1.51 | 2.41 | -2.19 | 41 | 10.77 | -2.80 | 0.39 | 21.00 | 0.48 | -0.26 | -1.06 |
| 2022_20_mexico | loso | - | - | - | - | - | - | - | - | - | 41 | 14.00 | -1.55 | 0.49 | - | - | - | - |
| 2022_21_brazil | in | 17.00 | 7.89 | -4.09 | 7.89 | -4.09 | 9.42 | -1.25 | 2.45 | -0.30 | 59 | 5.52 | -2.68 | 0.33 | 39.00 | 0.39 | -0.18 | 0.07 |
| 2022_21_brazil | lodo | 17.00 | 8.04 | -4.07 | 8.04 | -4.07 | 9.51 | -1.30 | 2.49 | -0.29 | 59 | 5.56 | -2.64 | 0.34 | 39.00 | 0.40 | -0.18 | 0.07 |
| 2022_21_brazil | loso | - | - | - | - | - | - | - | - | - | 59 | 11.59 | -2.19 | 0.66 | - | - | - | - |
| 2022_22_united_arab_emirates | in | 19.00 | 14.41 | -12.29 | 14.41 | -12.29 | 14.50 | -3.95 | 4.43 | - | 49 | 4.30 | -3.22 | 0.23 | 30.00 | 0.43 | -0.15 | 0.13 |
| 2022_22_united_arab_emirates | lodo | 19.00 | 14.80 | -12.21 | 14.80 | -12.21 | 14.65 | -3.72 | 4.51 | - | 49 | 4.64 | -3.33 | 0.25 | 30.00 | 0.45 | -0.17 | 0.15 |
| 2022_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 49 | 6.10 | -3.48 | 0.31 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 13.52 | 0.58 | 13.52 | 0.58 | 13.28 | 0.31 | 2.73 | 2.75 | 62 | 4.85 | -0.89 | 0.40 | 42.00 | 0.78 | -0.26 | 0.12 |
| 2023_01_bahrain | lodo | 17.00 | 13.71 | 0.67 | 13.71 | 0.67 | 13.44 | 0.37 | 2.85 | 2.75 | 62 | 4.97 | -0.90 | 0.41 | 42.00 | 0.83 | -0.26 | 0.11 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.01 | -0.89 | 0.56 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 9.56 | -4.75 | 9.56 | -4.75 | 8.79 | -0.53 | 1.05 | - | 63 | 4.10 | -1.28 | 0.22 | 42.00 | 0.63 | -0.48 | 0.03 |
| 2023_07_spain | lodo | 20.00 | 10.34 | -3.97 | 10.34 | -3.97 | 9.62 | -0.51 | 1.14 | - | 63 | 4.47 | -1.04 | 0.23 | 42.00 | 0.66 | -0.49 | 0.01 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 6.15 | -1.60 | 0.32 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 10.23 | -1.18 | 10.23 | -1.18 | 10.26 | 0.36 | 1.44 | - | 42 | 5.47 | -1.66 | 0.23 | 22.00 | 0.44 | -0.34 | -0.42 |
| 2024_07_emilia_romagna | lodo | 19.00 | 10.22 | -0.96 | 10.22 | -0.96 | 10.35 | 0.48 | 1.55 | - | 42 | 5.59 | -1.63 | 0.23 | 22.00 | 0.43 | -0.34 | -0.43 |
| 2024_07_emilia_romagna | loso | - | - | - | - | - | - | - | - | - | 42 | 7.50 | -1.59 | 0.30 | - | - | - | - |

Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:

| variant | compound | 1-5 | 11-15 | 16-20 | 21-25 | 26-30 | 31-35 | 36-40 | 41+ | 6-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| in | HARD | -0.04 | 0.06 | 0.04 | 0.01 | 0.02 | 0.02 | -0.04 | -1.38 | 0.04 |
| in | MEDIUM | 0.01 | 0.03 | 0.01 | -0.02 | 0.06 | -0.12 | -1.26 | -3.19 | 0.03 |
| in | SOFT | 0.01 | 0.13 | 0.04 | -0.02 | -0.65 | -2.55 | -4.08 | - | 0.03 |
| lodo | HARD | -0.04 | 0.06 | 0.04 | 0.02 | 0.03 | 0.04 | -0.05 | -1.45 | 0.04 |
| lodo | MEDIUM | 0.01 | 0.03 | 0.00 | -0.02 | 0.06 | -0.11 | -1.21 | -3.17 | 0.03 |
| lodo | SOFT | 0.00 | 0.12 | 0.04 | -0.03 | -0.65 | -2.57 | -4.14 | - | 0.02 |

Neutralised laps:

| variant | kind | n | mean_resid_s | mae_s |
|---|---|---|---|---|
| in | sc | 682 | -1.04 | 9.06 |
| in | vsc | 360 | -0.00 | 4.33 |
| lodo | sc | 682 | -1.04 | 9.07 |
| lodo | vsc | 360 | -0.00 | 4.33 |
| loso | sc | 682 | -0.53 | 9.60 |
| loso | vsc | 360 | 0.04 | 4.40 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.53 |
| 2021_01_bahrain | 2 | 2 | 2:16 | 1.00 | 0.81 | 0.94 | 0.94 | 60 | 0.47 | 0.47 | 1.81 |
| 2021_04_spain | 2 | 2 | 1:9 2:10 | 0.53 | 0.53 | 1.00 | 1.00 | 72 | 0.81 | 0.81 | 1.51 |
| 2021_09_austrian | 1 | 1 | 1:19 | 1.00 | 0.58 | 0.26 | 0.26 | 78 | 0.14 | 0.14 | 0.74 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 0.00 | 0.00 | 60 | 0.23 | 0.23 | 0.74 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 0.00 | 0.00 | 60 | 0.02 | 0.02 | 0.62 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 2.52 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 0.93 | 0.93 | 54 | 0.30 | 0.30 | 0.60 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.44 | 0.44 | 72 | 0.07 | 0.07 | 2.69 |
| 2022_08_azerbaijan | 2 | 1 | 1:14 2:1 | 0.07 | 0.40 | 0.33 | 0.33 | 54 | 0.28 | 0.28 | 0.53 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 1.00 | 1.00 | 78 | 0.37 | 0.37 | 1.14 |
| 2022_11_austria | 2 | 2 | 2:17 | 1.00 | 0.82 | 0.76 | 0.76 | 55 | 0.40 | 0.40 | 2.21 |
| 2022_12_france | 1 | 1 | 1:13 2:2 | 0.87 | 0.67 | 0.13 | 0.13 | 54 | 0.13 | 0.13 | 0.68 |
| 2022_20_mexico | 1 | 2 | 2:18 | 0.00 | 0.17 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 2.06 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.06 | 0.06 | 78 | 0.03 | 0.03 | 2.13 |
| 2022_22_united_arab_emirates | 2 | 2 | 1:5 2:12 | 0.71 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.72 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.10 | 0.10 | 1.79 |
| 2023_07_spain | 2 | 1 | 1:14 2:6 | 0.30 | 0.30 | 0.20 | 0.20 | 72 | 0.06 | 0.06 | 0.93 |
| 2024_07_emilia_romagna | 1 | 1 | 1:19 | 1.00 | 0.84 | 0.95 | 0.95 | 66 | 0.33 | 0.33 | 1.03 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

_(none)_

A6: hindsight ledger (judge and predictive):

| role | id | input | tag | substitute | source |
|---|---|---|---|---|---|
| judge | fit_window | degradation fit rows (fit_window full) | this race, whole | pre_race_fit | derived |
| judge | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, whole | - | derived |
| judge | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, whole | - | derived |
| judge | pit_loss | pit loss (Simulator.race_pit_loss_s, eval.score.world_pit_loss) | this race, whole | pit_loss_prior | derived |
| judge | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| judge | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
| judge | cliff_edge | wear-cliff support edge (world_edge as_of) | this race, to snapshot | cliff_history_only | derived |
| judge | cliff_prior | tyre-life history prior (tyre_history) | history | - | derived |
| judge | cliff_shape | cliff cost min(0.05 o^2, 4.0) s/lap past the edge | a priori | - | derived |
| judge | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| judge | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| judge | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| judge | events | neutralisations (events hybrid) | this race, whole | events_rate | derived |
| judge | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| judge | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| judge | driver_effects | driver effects (race) in the degradation fit | this race, whole | - | derived |
| judge | sc_queue_spacing | safety-car queue spacing 0.5 s | history | - | derived |
| judge | sc_queue | safety-car queue: the car placed among the other cars' recorded race | this race, whole | - | derived |
| judge | sc_queue_placement | safety-car queue: the car placed by its recorded place at the deployment | this race, whole | - | derived |
| judge | wear_guard | thin-fit wear guard (under 3 drivers or 40 rows: shrunk toward the historical wear) | history | - | derived |
| judge | tyre_inventory | tyre sets at the snapshot (sim/tyre_inventory.py) | this race, to snapshot | - | derived |
| judge | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |
| predictive | fit_window | degradation fit rows (fit_window to_snapshot) | this race, to snapshot | - | derived |
| predictive | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, to snapshot | - | derived |
| predictive | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, to snapshot | - | derived |
| predictive | pit_loss | pit loss (pit_loss.source asof) | this race, to snapshot | - | derived |
| predictive | pit_loss_prior | pit loss prior table (configs/sim/pit_loss_prior.json, earlier races only) | history | - | derived |
| predictive | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| predictive | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
| predictive | cliff_edge | wear-cliff support edge (as of the snapshot) | this race, to snapshot | - | derived |
| predictive | cliff_prior | tyre-life history prior (tyre_history) | history | - | derived |
| predictive | cliff_shape | cliff cost min(0.05 o^2, 4.0) s/lap past the edge | a priori | - | derived |
| predictive | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| predictive | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| predictive | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| predictive | events | neutralisations (events rate) | history | - | derived |
| predictive | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| predictive | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| predictive | priors | pre_race base delta, fuel prior, shrinkage taus | history | - | derived |
| predictive | calibrator | calibrator as_of_history | a priori | - | derived |
| predictive | driver_effects | driver effects (race) in the degradation fit | this race, to snapshot | - | derived |
| predictive | wear_lower_bound | wear bounded below at 0.0 | a priori | - | derived |
| predictive | draws | parameter draws (posterior) and lap noise (persistent) | this race, to snapshot | - | derived |
| predictive | belief_guard | belief thin-fit guard (under 3 drivers or 40 visible rows: level and wear shrunk toward the well-covered compounds and the historical wear) | this race, to snapshot | - | derived |
| predictive | tyre_inventory | tyre sets at the snapshot (sim/tyre_inventory.py) | this race, to snapshot | - | derived |
| predictive | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |
| predictive | field:belief_cliff_edge | config field belief_cliff_edge = "point" | unclassified | - | derived |
| predictive | field:fuel_anchor | config field fuel_anchor = {"prior": "anchored", "practice": "anchored", "full_fit": false} | unclassified | - | derived |

## B. Predictive versus judge

Plan panel per snapshot: the real remaining strategy, stay out, one stop at 4 laps spread from the soonest legal stop on each compound both model, two stops at 3 first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) over the plans legal under both configs' own rules. Snapshots every 5 laps for 6 sampled drivers per race. 20 draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.

| race_id | snapshots | rank_rho | regret_s | same_favourite | fav_b_illegal_w | pit_now_agree | rank_rho_green | regret_green_s | plans | b1_mae_s | b1_green_mae_s | b1_green_bias_s | b1_bias_s | b1_mae_per_lap_s | cov_68 | cov_90 | gap_cov_68 | gap_cov_90 | factor_cov_68 | factor_cov_90 | b4_mae_s | b4_bias_s | b4_mae_offset_s | b4_bias_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 54 | 0.737 | 4.435 | 0.500 | 0.000 | 0.830 | 0.792 | 4.138 | 1931 | 39.867 | 14.226 | 5.551 | 36.920 | 1.290 | 0.927 | 0.989 | 0.537 | 0.644 | 0.228 | 0.476 | 52.894 | 38.890 | 23.644 | 2.058 |
| 2021_01_bahrain | 60 | 0.934 | 1.369 | 0.617 | 0.000 | 0.879 | 0.951 | 1.573 | 1886 | 64.340 | 12.705 | 5.053 | 64.340 | 2.154 | 0.934 | 0.958 | 0.650 | 0.881 | 0.186 | 0.271 | 31.644 | -16.993 | 15.940 | -8.514 |
| 2021_04_spain | 72 | 0.773 | 6.963 | 0.236 | 0.000 | 0.803 | 0.814 | 7.454 | 2535 | 64.323 | 14.026 | 5.179 | 64.323 | 1.852 | 0.861 | 0.993 | 0.723 | 0.877 | 0.413 | 0.583 | 62.248 | -41.592 | 17.917 | -13.825 |
| 2021_09_austrian | 78 | 0.827 | 3.506 | 0.654 | 0.000 | 0.868 | 0.873 | 3.827 | 2542 | 62.057 | 24.540 | 18.287 | 62.049 | 1.438 | 0.858 | 0.916 | 0.717 | 0.890 | 0.335 | 0.546 | 49.036 | -17.771 | 25.320 | 1.660 |
| 2021_20_qatar | 60 | 0.826 | 3.256 | 0.783 | 0.000 | 0.914 | 0.862 | 3.872 | 2033 | 60.622 | 20.501 | 19.438 | 60.621 | 1.794 | 0.855 | 0.926 | 0.710 | 0.862 | 0.390 | 0.557 | 52.246 | 33.599 | 46.908 | 44.846 |
| 2021_22_united_arab_emirates | 60 | 0.887 | 4.708 | 0.667 | 0.000 | 0.695 | 0.924 | 5.943 | 2063 | 47.357 | 12.975 | 3.859 | 46.883 | 1.577 | 0.952 | 0.961 | 0.795 | 0.907 | 0.276 | 0.581 | 58.640 | 53.329 | 50.958 | 37.322 |
| 2022_01_bahrain | 60 | 0.937 | 2.025 | 0.550 | 0.000 | 0.852 | 0.959 | 1.514 | 2126 | 62.291 | 12.147 | 7.909 | 62.153 | 2.188 | 0.946 | 1.000 | 0.826 | 0.938 | 0.431 | 0.605 | 86.476 | 83.594 | 104.577 | 88.444 |
| 2022_02_saudi_arabia | 54 | 0.912 | 0.478 | 0.926 | 0.000 | 1.000 | 0.942 | 0.319 | 1713 | 47.914 | 14.724 | 5.240 | 46.292 | 1.915 | 0.949 | 0.988 | 0.799 | 0.900 | 0.582 | 0.696 | 41.260 | -25.659 | 32.458 | -23.500 |
| 2022_06_spain | 72 | 0.909 | 2.490 | 0.639 | 0.000 | 0.870 | 0.916 | 2.964 | 2603 | 93.998 | 41.304 | 38.191 | 93.998 | 2.434 | 0.896 | 0.939 | 0.754 | 0.950 | 0.456 | 0.621 | 67.969 | -42.441 | 23.634 | -3.036 |
| 2022_08_azerbaijan | 54 | 0.788 | 5.452 | 0.537 | 0.000 | 0.923 | 0.811 | 5.892 | 1698 | 48.181 | 14.942 | -10.170 | 46.992 | 1.941 | 1.000 | 1.000 | 0.709 | 0.833 | 0.404 | 0.544 | 29.803 | 8.409 | 19.675 | 5.796 |
| 2022_09_canada | 78 | 0.932 | 0.806 | 0.718 | 0.000 | 0.853 | 0.960 | 0.294 | 2322 | 53.949 | 12.498 | -1.612 | 49.062 | 1.263 | 0.953 | 1.000 | 0.934 | 0.981 | 0.471 | 0.577 | 30.149 | 11.477 | 41.974 | -4.115 |
| 2022_11_austria | 55 | 0.916 | 3.300 | 0.600 | 0.000 | 0.889 | 0.930 | 1.680 | 1766 | 48.809 | 11.421 | -4.949 | 48.792 | 1.193 | 0.984 | 0.999 | 0.665 | 0.773 | 0.483 | 0.714 | 34.304 | 4.749 | 19.581 | 1.010 |
| 2022_12_france | 54 | 0.946 | 1.836 | 0.722 | 0.000 | 0.943 | 0.948 | 2.446 | 1879 | 43.975 | 9.759 | 7.284 | 43.970 | 1.614 | 0.981 | 1.000 | 0.763 | 0.921 | 0.328 | 0.431 | 32.290 | 8.385 | 27.843 | 19.959 |
| 2022_20_mexico | 78 | 0.913 | 4.354 | 0.474 | 0.000 | 0.827 | 0.919 | 4.954 | 2702 | 49.452 | 24.768 | -4.645 | 49.062 | 1.385 | 0.920 | 0.956 | 0.769 | 0.847 | 0.214 | 0.273 | 55.054 | 26.904 | 34.442 | 20.881 |
| 2022_21_brazil | 78 | 0.937 | 3.876 | 0.603 | 0.000 | 0.819 | 0.946 | 4.393 | 2766 | 49.950 | 18.736 | 14.744 | 43.889 | 1.421 | 0.927 | 0.965 | 0.792 | 0.909 | 0.266 | 0.297 | 44.138 | -40.590 | 33.070 | -20.971 |
| 2022_22_united_arab_emirates | 60 | 0.945 | 1.832 | 0.517 | 0.000 | 0.875 | 0.946 | 2.125 | 2079 | 57.312 | 12.691 | 9.692 | 57.312 | 1.811 | 0.945 | 0.965 | 0.665 | 0.830 | 0.236 | 0.293 | 31.068 | -1.065 | 17.565 | -7.789 |
| 2023_01_bahrain | 60 | 0.822 | 1.547 | 0.717 | 0.000 | 0.946 | 0.840 | 2.222 | 1762 | 85.524 | 33.421 | 31.144 | 85.524 | 2.329 | 0.831 | 0.972 | 0.817 | 0.960 | 0.476 | 0.571 | 31.898 | 23.858 | 28.689 | 23.032 |
| 2023_07_spain | 72 | 0.947 | 1.687 | 0.556 | 0.000 | 0.809 | 0.962 | 2.198 | 2473 | 59.806 | 14.589 | 14.332 | 59.806 | 1.704 | 0.918 | 0.977 | 0.752 | 0.915 | 0.242 | 0.452 | 28.997 | 17.765 | 17.338 | 7.860 |
| 2024_07_emilia_romagna | 66 | 0.910 | 1.307 | 0.652 | 0.000 | 0.909 | 0.922 | 2.080 | 2275 | 36.946 | 7.907 | 3.533 | 36.867 | 1.201 | 0.995 | 0.996 | 0.618 | 0.802 | 0.169 | 0.234 | 33.659 | -15.660 | 20.379 | -5.087 |
| pooled | 1225 | 0.886 | 2.932 | 0.609 | 0.000 | 0.865 | 0.908 | 3.196 | 41154 | 57.072 | 17.674 | 9.312 | 56.072 | 1.699 | 0.926 | 0.972 | 0.741 | 0.879 | 0.344 | 0.485 | 45.500 | 5.736 | 31.982 | 9.711 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1716.00 | 21.86 | 21.86 |
| 10.00 | 7842.00 | 24.91 | 22.62 |
| 20.00 | 8056.00 | 26.20 | 24.71 |
| 30.00 | 8369.00 | 44.36 | 43.10 |
| 40.00 | 7882.00 | 81.39 | 81.39 |
| 50.00 | 4783.00 | 111.27 | 111.26 |
| 60.00 | 2506.00 | 143.62 | 143.36 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 13205 | 106.37 | 105.96 |
| late | 10955 | 24.07 | 21.39 |
| mid | 16994 | 40.04 | 39.66 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12459 | 54.35 | 53.31 |
| real | 1155 | 53.56 | 52.45 |
| stay | 350 | 45.09 | 44.83 |
| two | 27190 | 58.62 | 57.64 |

B5 future blindness: {"status": "pass", "new_leak_differences": 0, "differences_with_known_leaks": 0, "known_leaks": [], "fixed_leaks": ["compound_list", "frame_filter", "gap_behind", "lap_window", "pit_loss", "race_length", "stint_limit"], "detail": ""}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | history prior (tyre_life_prior) | oldest age run | 52 | 19 | 7.518 | -1.129 | 0.673 | 0.885 |
| tyre_life | history prior (tyre_life_prior) | median stint-end age | 52 | 19 | 9.608 | 8.958 | 0.558 | 0.846 |
| tyre_life | history prior (tyre_life_prior) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.757 | 1.571 | 0.812 | 0.938 |
| tyre_life | judge cliff edge (as_of) | oldest age run | 53 | 19 | 7.344 | -1.139 | - | - |
| tyre_life | judge cliff edge (as_of) | median stint-end age | 53 | 19 | 9.395 | 8.757 | - | - |
| tyre_life | judge cliff edge (as_of) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.722 | 1.536 | - | - |
| tyre_life | belief cliff edge @k0 | oldest age run | 52 | 19 | 7.398 | -0.879 | 0.638 | 0.872 |
| tyre_life | belief cliff edge @k0 | median stint-end age | 52 | 19 | 9.857 | 9.207 | 0.553 | 0.872 |
| tyre_life | belief cliff edge @k0 | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.809 | 1.841 | 0.791 | 0.930 |
| tyre_life | as-of data (race to k0, practice) | oldest age run | 52 | 19 | 12.692 | -10.577 | - | - |
| tyre_life | as-of data (race to k0, practice) | median stint-end age | 52 | 19 | 8.221 | -0.490 | - | - |
| tyre_life | as-of data (race to k0, practice) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 9.923 | -7.769 | - | - |
| tyre_life | belief cliff edge @k10 | oldest age run | 52 | 19 | 7.398 | -0.879 | 0.638 | 0.872 |
| tyre_life | belief cliff edge @k10 | median stint-end age | 52 | 19 | 9.857 | 9.207 | 0.553 | 0.872 |
| tyre_life | belief cliff edge @k10 | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.809 | 1.841 | 0.791 | 0.930 |
| tyre_life | as-of data (race to k10, practice) | oldest age run | 52 | 19 | 12.481 | -10.365 | - | - |
| tyre_life | as-of data (race to k10, practice) | median stint-end age | 52 | 19 | 8.010 | -0.279 | - | - |
| tyre_life | as-of data (race to k10, practice) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 9.694 | -7.540 | - | - |
| tyre_life | belief cliff edge @k20 | oldest age run | 52 | 19 | 7.263 | -0.745 | 0.682 | 0.886 |
| tyre_life | belief cliff edge @k20 | median stint-end age | 52 | 19 | 9.953 | 9.342 | 0.523 | 0.864 |
| tyre_life | belief cliff edge @k20 | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.746 | 1.987 | 0.825 | 0.925 |
| tyre_life | as-of data (race to k20, practice) | oldest age run | 52 | 19 | 11.308 | -9.192 | - | - |
| tyre_life | as-of data (race to k20, practice) | median stint-end age | 52 | 19 | 7.260 | 0.894 | - | - |
| tyre_life | as-of data (race to k20, practice) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 8.506 | -6.269 | - | - |
| tyre_life | belief cliff edge @k(mid) | oldest age run | 52 | 19 | 6.375 | 0.144 | 0.676 | 0.865 |
| tyre_life | belief cliff edge @k(mid) | median stint-end age | 52 | 19 | 10.590 | 10.230 | 0.432 | 0.838 |
| tyre_life | belief cliff edge @k(mid) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.424 | 2.950 | 0.818 | 0.909 |
| tyre_life | as-of data (race to k(mid), practice) | oldest age run | 52 | 19 | 7.885 | -5.769 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | median stint-end age | 52 | 19 | 7.317 | 4.317 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.694 | -2.560 | - | - |
| wear_rate | belief @k0 | within-stint slope less fuel | 51 | 19 | 0.048 | 0.013 | 0.471 | 0.686 |
| wear_rate | belief @k10 | within-stint slope less fuel | 51 | 19 | 0.037 | 0.004 | 0.294 | 0.451 |
| wear_rate | belief @k20 | within-stint slope less fuel | 51 | 19 | 0.033 | 0.003 | 0.216 | 0.333 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 51 | 19 | 0.026 | -0.003 | 0.275 | 0.392 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 51 | 19 | 0.017 | -0.013 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 565 | 19 | 0.815 | 0.179 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 530 | 19 | 0.895 | 0.056 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 359 | 19 | 0.720 | 0.123 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 235 | 19 | 0.618 | 0.010 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.726 | -0.322 | - | - |
| compound_gap | belief @k0 | qualifying | 7 | 7 | 0.724 | 0.629 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.643 | -0.180 | - | - |
| compound_gap | belief @k10 | qualifying | 7 | 7 | 0.941 | 0.674 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.401 | 0.008 | - | - |
| compound_gap | belief @k20 | qualifying | 7 | 7 | 0.768 | 0.503 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.337 | -0.092 | - | - |
| compound_gap | belief @k(mid) | qualifying | 7 | 7 | 0.854 | 0.469 | - | - |
| fuel | prior (config) | race (judge full fit) | 19 | 19 | 0.019 | 0.016 | - | - |
| fuel | belief @k0 | race (judge full fit) | 19 | 19 | 0.008 | 0.002 | 0.632 | 0.947 |
| fuel | belief @k10 | race (judge full fit) | 19 | 19 | 0.008 | -0.001 | 0.316 | 0.421 |
| fuel | belief @k20 | race (judge full fit) | 19 | 19 | 0.008 | -0.003 | 0.158 | 0.421 |
| fuel | belief @k(mid) | race (judge full fit) | 19 | 19 | 0.006 | -0.001 | 0.421 | 0.526 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 11 | 11 | 2.457 | -0.959 | 0.545 | 0.636 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 11 | 11 | 2.841 | -2.021 | 0.545 | 0.727 |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 2.346 | -1.752 | - | - |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 3.317 | -2.342 | - | - |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.832 | -0.777 | 0.789 | 0.842 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.698 | -1.367 | 0.632 | 0.789 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.291 | 7.060 | 0.200 | 0.300 |
| pit_loss | belief charged @k0 | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.807 | -0.670 | - | - |
| pit_loss | belief charged @k0 | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.651 | -1.260 | - | - |
| pit_loss | belief charged @k0 | whole race, median SC stop | 10 | 10 | 13.303 | 7.120 | - | - |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.744 | -0.087 | 0.789 | 0.842 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.418 | -0.677 | 0.737 | 0.789 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.360 | 7.312 | 0.300 | 0.400 |
| pit_loss | belief charged @k10 | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.729 | 0.013 | - | - |
| pit_loss | belief charged @k10 | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.372 | -0.577 | - | - |
| pit_loss | belief charged @k10 | whole race, median SC stop | 10 | 10 | 13.374 | 7.368 | - | - |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.683 | -0.231 | 1.000 | 1.000 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.888 | -0.820 | 0.632 | 0.789 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.615 | 7.365 | 0.100 | 0.500 |
| pit_loss | belief charged @k20 | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.664 | -0.160 | - | - |
| pit_loss | belief charged @k20 | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.846 | -0.749 | - | - |
| pit_loss | belief charged @k20 | whole race, median SC stop | 10 | 10 | 12.615 | 7.401 | - | - |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.531 | -0.392 | 0.947 | 1.000 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.996 | -0.982 | 0.632 | 0.789 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.403 | 7.151 | 0.200 | 0.500 |
| pit_loss | belief charged @k(mid) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.507 | -0.327 | - | - |
| pit_loss | belief charged @k(mid) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.944 | -0.916 | - | - |
| pit_loss | belief charged @k(mid) | whole race, median SC stop | 10 | 10 | 12.402 | 7.185 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 19 | 19 | 2.022 | 1.339 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 19 | 19 | 0.180 | 0.082 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 601 | 19 | 0.594 | 0.115 | - | - |
| traffic_rejoin | judge rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 594 | 19 | 1.118 | 0.077 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 601 | 19 | 0.795 | 0.000 | - | - |
| traffic_rejoin | belief rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 594 | 19 | 1.943 | -0.630 | - | - |
| neutralisations | judge expected (recorded) | occurred | 76 | 19 | 0.000 | 0.000 | - | - |
| neutralisations | judge config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | judge config | observed (median VSC lap / green prediction) | 12 | 12 | 0.122 | 0.120 | - | - |
| neutralisations | belief expected (rate) | occurred | 76 | 19 | 1.348 | 0.443 | - | - |
| neutralisations | belief config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | belief config | observed (median VSC lap / green prediction) | 12 | 12 | 0.122 | 0.120 | - | - |
| neutralised_laps | judge replay (recorded lap times) | mean real lap time on the cars' neutralised laps | 22 | 15 | 2.081 | 0.738 | - | - |
| neutralised_laps | belief replay (factors) | mean real lap time on the cars' neutralised laps | 22 | 15 | 9.885 | 6.846 | - | - |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 6 | 6 | 0.869 | -0.641 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 6 | 6 | 2.654 | 0.059 | 0.167 | 0.833 |
| pit_loss | prior from earlier races (constant) | whole race, robust (pit_loss.whole_race: median green stop) | 2 | 2 | 1.281 | -0.187 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (constant) | whole race, replay mean (calibrate._pit_loss) | 2 | 2 | 2.045 | -2.045 | 1.000 | 1.000 |

Component status:

| component | status | races |
|---|---|---|
| compound_gap | ok | 19 |
| fresh_tyre_gain | ok | 19 |
| fuel | ok | 19 |
| lap1_and_warmup | ok | 19 |
| neutralisations | ok | 19 |
| neutralised_laps | no rows | 4 |
| neutralised_laps | ok | 15 |
| pit_loss | ok | 19 |
| traffic_rejoin | ok | 19 |
| tyre_life | ok | 19 |
| wear_rate | ok | 19 |

