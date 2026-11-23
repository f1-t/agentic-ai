# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_11_austria, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version 32671b6a2ad5-cd186815b161. World config 68caf788d1e0 (race default); belief config f75a461b7850. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "field_replay": false, "seed": 0, "max_stops": 2}. Run time 490 s.

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
| B2_rank_rho | 0.872 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 3.161 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.827 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.667 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.876 | same, Spearman, mean over k |
| A_abs_mae_s | 13.375 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -4.795 | same, mean error |
| A_stop_gain_bias_s | -0.191 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 55.224 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 54.066 | same, mean |
| B1_green_mae_s | 16.998 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.895 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.975 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.481 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 32.582 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |
| E_real_regret_s | 5.589 | expert consistency (secondary): judge time of the team's real remaining strategy less the judge's best panel plan, mean over snapshots; should fall where passing is hard, never chased to zero |
| E_real_over_20s | 0.056 | same, share of snapshots where the judge says the real strategy lost more than 20 s |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.650 | 0.961 | 3.737 | 12.838 | 0.650 | 0.961 | 3.375 | 12.657 | 0.721 | 4.813 | 39.383 | 36.394 | 0.540 | 0.866 | 17.538 | 23.911 |
| 2021_01_bahrain | 0.909 | 0.992 | 5.122 | 11.887 | 0.909 | 0.992 | 5.000 | 12.110 | 0.938 | 2.583 | 60.388 | 60.388 | 0.691 | 0.943 | 13.079 | 15.530 |
| 2021_04_spain | 0.714 | 0.974 | 4.901 | 12.988 | 0.714 | 0.972 | 5.102 | 13.145 | 0.794 | 4.635 | 63.251 | 63.219 | 0.786 | 0.972 | 10.623 | 21.534 |
| 2021_09_austrian | 0.750 | 0.958 | 5.516 | 12.286 | 0.750 | 0.970 | 5.720 | 12.291 | 0.812 | 3.415 | 59.338 | 59.330 | 0.500 | 0.912 | 19.718 | 25.863 |
| 2021_20_qatar | 0.462 | 0.973 | 6.811 | 13.731 | 0.462 | 0.973 | 6.668 | 14.150 | 0.834 | 4.051 | 59.450 | 59.449 | 0.754 | 0.940 | 23.191 | 48.538 |
| 2021_22_united_arab_emirates | 0.857 | 0.975 | 27.618 | 26.080 | 0.893 | 0.982 | 27.649 | 25.962 | 0.895 | 4.221 | 46.082 | 45.565 | 0.693 | 0.857 | 33.577 | 51.123 |
| 2022_01_bahrain | 0.761 | 0.796 | 9.529 | 19.074 | 0.761 | 0.821 | 9.598 | 19.067 | 0.922 | 2.009 | 64.018 | 63.883 | 0.709 | 0.734 | 47.981 | 105.360 |
| 2022_02_saudi_arabia | 0.692 | 0.960 | 6.116 | 10.246 | 0.692 | 0.943 | 6.575 | 10.672 | 0.881 | 0.824 | 46.355 | 44.261 | 0.615 | 0.849 | 16.779 | 31.284 |
| 2022_06_spain | 0.500 | 0.967 | 12.332 | 12.887 | 0.500 | 0.967 | 12.540 | 13.539 | 0.920 | 1.868 | 90.424 | 90.424 | 0.800 | 0.968 | 17.718 | 24.955 |
| 2022_08_azerbaijan | 0.364 | 0.938 | 9.094 | 12.229 | 0.364 | 0.944 | 9.014 | 12.280 | 0.814 | 4.933 | 49.718 | 48.766 | 0.491 | 0.815 | 23.465 | 20.418 |
| 2022_09_canada | 0.800 | 0.978 | 5.331 | 7.178 | 0.800 | 0.973 | 5.596 | 7.148 | 0.892 | 1.175 | 50.786 | 44.250 | 0.627 | 0.812 | 42.420 | 42.320 |
| 2022_11_austria | 1.000 | 0.964 | 6.431 | 11.652 | 1.000 | 0.964 | 5.845 | 11.334 | 0.821 | 2.235 | 49.074 | 49.074 | 0.467 | 0.750 | 15.792 | 19.583 |
| 2022_12_france | 0.727 | 0.979 | 5.045 | 19.828 | 0.773 | 0.982 | 5.008 | 20.309 | 0.926 | 2.228 | 40.835 | 40.786 | 0.736 | 0.884 | 12.289 | 28.010 |
| 2022_20_mexico | 0.714 | 0.967 | 8.809 | 12.580 | 0.786 | 0.972 | 8.682 | 12.338 | 0.891 | 6.341 | 47.049 | 46.513 | 0.386 | 0.821 | 28.165 | 33.605 |
| 2022_21_brazil | 0.854 | 0.971 | 3.380 | 7.890 | 0.875 | 0.973 | 3.556 | 8.044 | 0.931 | 4.070 | 49.170 | 43.014 | 0.725 | 0.875 | 29.925 | 33.039 |
| 2022_22_united_arab_emirates | 0.643 | 0.989 | 4.170 | 14.413 | 0.643 | 0.986 | 5.100 | 14.803 | 0.902 | 3.564 | 55.682 | 55.681 | 0.629 | 0.938 | 11.525 | 19.342 |
| 2023_01_bahrain | 0.583 | 0.978 | 5.805 | 13.520 | 0.500 | 0.975 | 5.807 | 13.714 | 0.836 | 2.334 | 73.766 | 73.766 | 0.767 | 0.964 | 19.733 | 29.641 |
| 2023_07_spain | 0.800 | 0.983 | 7.120 | 9.557 | 0.800 | 0.983 | 8.827 | 10.335 | 0.917 | 2.022 | 58.121 | 58.110 | 0.760 | 0.941 | 21.613 | 17.650 |
| 2024_07_emilia_romagna | 0.750 | 0.914 | 4.839 | 10.228 | 0.750 | 0.914 | 4.729 | 10.218 | 0.886 | 2.252 | 35.973 | 35.465 | 0.650 | 0.805 | 11.989 | 20.445 |
| pooled | 0.739 | 0.959 | 7.458 | 13.215 | 0.747 | 0.960 | 7.600 | 13.375 | 0.872 | 3.161 | 55.224 | 54.066 | 0.667 | 0.876 | 21.954 | 32.582 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.739 | 368.000 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.739 | 368.000 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.568 | 368.000 | 0.878 | 1.574 | 0.597 | 26.247 | 28.982 | -7.900 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.747 | 368.000 | 0.960 | 0.828 | 0.828 | 7.600 | 13.375 | -4.795 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.747 | 368.000 | 0.960 | 0.828 | 0.828 | 7.600 | 13.375 | -4.795 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.565 | 368.000 | 0.875 | 1.596 | 0.589 | 26.348 | 29.000 | -7.882 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.658 | 368.000 | 0.931 | 1.108 | 0.737 | 12.441 | 19.974 | -5.724 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.658 | 368.000 | 0.931 | 1.108 | 0.737 | 12.441 | 19.974 | -5.724 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.579 | 368.000 | 0.892 | 1.458 | 0.616 | 30.372 | 33.834 | -8.811 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.660 | 368.000 | 0.931 | 1.104 | 0.731 | 12.501 | 20.231 | -5.967 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.660 | 368.000 | 0.931 | 1.104 | 0.731 | 12.501 | 20.231 | -5.967 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.598 | 368.000 | 0.895 | 1.460 | 0.624 | 30.910 | 34.031 | -9.055 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.739 | 368.000 | 0.958 | 0.844 | 0.819 | 7.563 | 13.311 | -6.007 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.739 | 368.000 | 0.958 | 0.844 | 0.819 | 7.563 | 13.311 | -6.007 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.568 | 368.000 | 0.875 | 1.590 | 0.594 | 26.181 | 29.106 | -9.095 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.750 | 368.000 | 0.960 | 0.827 | 0.828 | 7.692 | 13.377 | -6.035 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.750 | 368.000 | 0.960 | 0.827 | 0.828 | 7.692 | 13.377 | -6.035 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.565 | 368.000 | 0.874 | 1.607 | 0.582 | 26.300 | 29.103 | -9.122 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.739 | 368.000 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.739 | 368.000 | 0.959 | 0.827 | 0.825 | 7.458 | 13.215 | -4.813 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.568 | 368.000 | 0.878 | 1.574 | 0.597 | 26.247 | 28.982 | -7.900 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.751 | 362.000 | 0.960 | 0.819 | 0.829 | 7.546 | 12.959 | -4.383 | 16.842 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.751 | 362.000 | 0.960 | 0.819 | 0.829 | 7.546 | 12.959 | -4.383 | 16.842 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.566 | 362.000 | 0.874 | 1.590 | 0.588 | 26.069 | 28.675 | -7.921 | 16.842 |
| 2 judge, substituted | events_rate | 0 | in-sample | insample | yes | 0.728 | 368.000 | 0.939 | 0.982 | 0.783 | 9.157 | 20.145 | 0.912 | 17.053 |
| 2 judge, substituted | events_rate | 0 | in-sample | none | yes | 0.728 | 368.000 | 0.939 | 0.982 | 0.783 | 9.157 | 20.145 | 0.912 | 17.053 |
| 2 judge, substituted | events_rate | 0 | in-sample | practice | yes | 0.562 | 368.000 | 0.867 | 1.654 | 0.582 | 26.984 | 32.198 | -2.176 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | insample | yes | 0.726 | 368.000 | 0.940 | 0.990 | 0.785 | 9.293 | 20.329 | 0.931 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | none | no | 0.726 | 368.000 | 0.940 | 0.990 | 0.785 | 9.293 | 20.329 | 0.931 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | practice | no | 0.560 | 368.000 | 0.861 | 1.692 | 0.562 | 27.092 | 32.297 | -2.156 | 17.053 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.669 | 251.000 | 0.926 | 1.166 | 0.714 | 17.127 | 32.008 | -1.777 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.669 | 251.000 | 0.926 | 1.166 | 0.714 | 17.127 | 32.008 | -1.777 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.586 | 251.000 | 0.888 | 1.467 | 0.602 | 35.614 | 43.941 | -4.629 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.665 | 251.000 | 0.925 | 1.167 | 0.710 | 16.964 | 32.129 | -1.859 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.665 | 251.000 | 0.925 | 1.167 | 0.710 | 16.964 | 32.129 | -1.859 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.602 | 251.000 | 0.895 | 1.443 | 0.607 | 36.319 | 44.095 | -4.711 | 17.071 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.546 | 368.000 | 0.773 | 2.270 | 0.496 | 28.370 | 79.943 | -15.119 | 17.053 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.639 | 368.000 | 0.868 | 1.640 | 0.594 | 27.535 | 50.489 | 15.495 | 17.053 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.658 | 368.000 | 0.886 | 1.415 | 0.667 | 22.842 | 32.845 | 8.612 | 17.053 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.715 | 368.000 | 0.918 | 1.110 | 0.743 | 17.046 | 25.092 | 6.410 | 17.053 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.780 | 368.000 | 0.935 | 0.912 | 0.824 | 13.975 | 21.013 | 4.715 | 17.053 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.514 | 368.000 | 0.695 | 2.877 | 0.383 | 25.728 | 81.797 | -12.031 | 17.053 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.543 | 368.000 | 0.773 | 2.450 | 0.431 | 22.471 | 52.573 | 19.686 | 17.053 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.609 | 368.000 | 0.819 | 1.932 | 0.544 | 17.834 | 37.580 | 9.618 | 17.053 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.698 | 368.000 | 0.888 | 1.428 | 0.677 | 16.575 | 29.985 | 7.349 | 17.053 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.755 | 368.000 | 0.914 | 1.129 | 0.747 | 13.668 | 24.420 | 5.326 | 17.053 |

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
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.52 |
| 2021_01_bahrain | 2 | 2 | 2:16 | 1.00 | 0.81 | 0.94 | 0.94 | 60 | 0.60 | 0.60 | 2.16 |
| 2021_04_spain | 2 | 2 | 1:9 2:10 | 0.53 | 0.53 | 1.00 | 1.00 | 72 | 0.36 | 0.36 | 1.35 |
| 2021_09_austrian | 1 | 1 | 1:19 | 1.00 | 0.58 | 0.26 | 0.26 | 78 | 0.10 | 0.10 | 0.68 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 0.00 | 0.00 | 60 | 0.23 | 0.23 | 0.67 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 0.00 | 0.00 | 60 | 0.02 | 0.02 | 0.62 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 2.82 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 0.93 | 0.93 | 54 | 0.33 | 0.33 | 0.56 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.44 | 0.44 | 72 | 0.08 | 0.08 | 3.71 |
| 2022_08_azerbaijan | 2 | 1 | 1:14 2:1 | 0.07 | 0.40 | 0.33 | 0.33 | 54 | 0.44 | 0.44 | 0.95 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 1.00 | 1.00 | 78 | 0.54 | 0.54 | 1.17 |
| 2022_11_austria | 2 | 2 | 2:17 | 1.00 | 0.82 | 0.76 | 0.76 | 55 | 0.45 | 0.45 | 2.16 |
| 2022_12_france | 1 | 1 | 1:13 2:2 | 0.87 | 0.67 | 0.13 | 0.13 | 54 | 0.11 | 0.11 | 0.79 |
| 2022_20_mexico | 1 | 2 | 2:18 | 0.00 | 0.17 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 2.15 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.06 | 0.06 | 78 | 0.00 | 0.00 | 1.90 |
| 2022_22_united_arab_emirates | 2 | 2 | 1:5 2:12 | 0.71 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.90 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 1.77 |
| 2023_07_spain | 2 | 1 | 1:14 2:6 | 0.30 | 0.30 | 0.20 | 0.20 | 72 | 0.01 | 0.01 | 1.06 |
| 2024_07_emilia_romagna | 1 | 1 | 1:19 | 1.00 | 0.84 | 0.95 | 0.95 | 66 | 0.41 | 0.41 | 1.15 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 368 | 368 | -0.082 | -0.028 | 4.982 | 6.824 | -1.327 | 0.075 | -0.991 | 4.177 | -0.040 | 0.195 | 0.195 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 368 | 368 | -0.087 | -0.030 | 4.902 | 6.877 | -1.581 | 0.112 | -1.047 | 3.977 | - | - | - |
| pit_loss_prior | in | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 368 | 368 | 0.000 | -0.001 | 0.104 | 0.197 | -1.354 | 0.000 | -0.619 | -0.032 | -0.076 | 0.070 | 0.070 |
| pit_loss_prior | lodo | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 368 | 368 | 0.003 | -0.001 | 0.092 | 0.106 | -1.404 | 0.000 | -0.650 | -0.031 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 368 | 368 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | -0.100 | -0.100 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 362 | 368 | 0.004 | -0.001 | -0.053 | -0.387 | 0.391 | 0.003 | 0.157 | -0.013 | - | - | - |
| events_rate | in | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 368 | 368 | -0.011 | -0.019 | 1.699 | 6.860 | 5.204 | 0.000 | 1.141 | 4.578 | -0.016 | 0.018 | 0.018 |
| events_rate | lodo | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 368 | 368 | -0.022 | -0.020 | 1.694 | 6.856 | 5.202 | 0.000 | 1.144 | 4.543 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 251 | 368 | -0.070 | -0.033 | 9.669 | 17.207 | 2.697 | 0.105 | 0.025 | 9.500 | 0.056 | 0.093 | 0.093 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 251 | 368 | -0.082 | -0.036 | 9.364 | 17.128 | 2.594 | 0.134 | -0.025 | 9.332 | - | - | - |

A6: hindsight ledger (judge and predictive):

| role | id | input | tag | substitute | source |
|---|---|---|---|---|---|
| judge | fit_window | degradation fit rows (fit_window full) | this race, whole | pre_race_fit | derived |
| judge | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, whole | - | derived |
| judge | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, whole | - | derived |
| judge | pit_loss | pit loss (Simulator.race_pit_loss_s, eval.score.world_pit_loss) | this race, whole | pit_loss_prior | derived |
| judge | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| judge | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
| judge | cliff_edge | wear-cliff support edge (world_edge history_only) | history | - | derived |
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
| predictive | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |
| predictive | field:belief_cliff_edge | config field belief_cliff_edge = "point" | unclassified | - | derived |
| predictive | field:fuel_anchor | config field fuel_anchor = {"prior": "anchored", "practice": "anchored", "full_fit": false} | unclassified | - | derived |

## B. Predictive versus judge

Plan panel per snapshot: the real remaining strategy, stay out, one stop at 4 laps spread from the soonest legal stop on each compound both model, two stops at 3 first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) over the plans legal under both configs' own rules. Snapshots every 5 laps for 6 sampled drivers per race. 20 draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.

| race_id | snapshots | rank_rho | regret_s | same_favourite | fav_b_illegal_w | pit_now_agree | rank_rho_green | regret_green_s | plans | b1_mae_s | b1_green_mae_s | b1_green_bias_s | b1_bias_s | b1_mae_per_lap_s | cov_68 | cov_90 | gap_cov_68 | gap_cov_90 | factor_cov_68 | factor_cov_90 | b4_mae_s | b4_bias_s | b4_mae_offset_s | b4_bias_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 54 | 0.721 | 4.813 | 0.519 | 0.000 | 0.830 | 0.774 | 4.093 | 1931 | 39.383 | 13.894 | 4.966 | 36.394 | 1.277 | 0.940 | 0.976 | 0.533 | 0.609 | 0.228 | 0.421 | 53.096 | 38.949 | 23.911 | 2.136 |
| 2021_01_bahrain | 60 | 0.938 | 2.583 | 0.533 | 0.000 | 0.862 | 0.944 | 3.033 | 2127 | 60.388 | 11.056 | 3.534 | 60.388 | 2.120 | 0.948 | 0.961 | 0.704 | 0.938 | 0.188 | 0.250 | 32.120 | -16.047 | 15.530 | -7.419 |
| 2021_04_spain | 72 | 0.794 | 4.635 | 0.375 | 0.000 | 0.775 | 0.832 | 4.050 | 2544 | 63.251 | 14.936 | 3.921 | 63.219 | 1.827 | 0.863 | 0.993 | 0.676 | 0.832 | 0.413 | 0.583 | 63.514 | -44.497 | 21.534 | -17.116 |
| 2021_09_austrian | 78 | 0.812 | 3.415 | 0.679 | 0.000 | 0.829 | 0.860 | 4.038 | 2707 | 59.338 | 22.912 | 16.440 | 59.330 | 1.415 | 0.872 | 0.930 | 0.709 | 0.869 | 0.333 | 0.540 | 49.224 | -17.327 | 25.863 | 2.222 |
| 2021_20_qatar | 60 | 0.834 | 4.051 | 0.733 | 0.000 | 0.931 | 0.873 | 4.793 | 2140 | 59.450 | 20.025 | 19.186 | 59.449 | 1.794 | 0.870 | 0.944 | 0.696 | 0.852 | 0.390 | 0.555 | 53.550 | 34.869 | 48.538 | 46.434 |
| 2021_22_united_arab_emirates | 60 | 0.895 | 4.221 | 0.583 | 0.000 | 0.678 | 0.919 | 5.265 | 2150 | 46.082 | 13.259 | 3.160 | 45.565 | 1.562 | 0.953 | 0.962 | 0.787 | 0.908 | 0.276 | 0.583 | 58.863 | 53.600 | 51.123 | 37.607 |
| 2022_01_bahrain | 60 | 0.922 | 2.009 | 0.583 | 0.000 | 0.741 | 0.949 | 1.626 | 2169 | 64.018 | 14.035 | 10.526 | 63.883 | 2.238 | 0.908 | 0.998 | 0.812 | 0.936 | 0.433 | 0.631 | 87.498 | 84.619 | 105.360 | 89.525 |
| 2022_02_saudi_arabia | 54 | 0.881 | 0.824 | 0.889 | 0.000 | 0.963 | 0.918 | 0.404 | 1779 | 46.355 | 14.813 | 2.847 | 44.261 | 1.879 | 0.954 | 0.996 | 0.777 | 0.873 | 0.553 | 0.696 | 40.246 | -24.644 | 31.284 | -22.326 |
| 2022_06_spain | 72 | 0.920 | 1.868 | 0.833 | 0.000 | 1.000 | 0.927 | 2.377 | 2654 | 90.424 | 38.402 | 34.826 | 90.424 | 2.380 | 0.904 | 0.956 | 0.778 | 0.950 | 0.522 | 0.633 | 68.704 | -43.781 | 24.955 | -4.459 |
| 2022_08_azerbaijan | 54 | 0.814 | 4.933 | 0.481 | 0.000 | 0.846 | 0.842 | 6.533 | 1858 | 49.718 | 12.344 | -6.959 | 48.766 | 2.049 | 1.000 | 1.000 | 0.737 | 0.852 | 0.404 | 0.528 | 29.920 | 9.192 | 20.418 | 6.441 |
| 2022_09_canada | 78 | 0.892 | 1.175 | 0.667 | 0.000 | 0.868 | 0.933 | 0.865 | 2758 | 50.786 | 11.592 | -0.871 | 44.250 | 1.285 | 0.944 | 0.999 | 0.920 | 0.975 | 0.460 | 0.566 | 30.602 | 11.997 | 42.320 | -3.488 |
| 2022_11_austria | 55 | 0.821 | 2.235 | 0.527 | 0.000 | 0.815 | 0.859 | 1.727 | 2005 | 49.074 | 8.653 | -1.787 | 49.074 | 1.257 | 0.986 | 1.000 | 0.837 | 0.933 | 0.483 | 0.701 | 34.316 | 4.828 | 19.583 | 1.083 |
| 2022_12_france | 54 | 0.926 | 2.228 | 0.407 | 0.000 | 0.755 | 0.918 | 4.172 | 1983 | 40.835 | 9.110 | 4.341 | 40.786 | 1.521 | 0.996 | 1.000 | 0.650 | 0.832 | 0.333 | 0.431 | 32.580 | 8.737 | 28.010 | 20.307 |
| 2022_20_mexico | 78 | 0.891 | 6.341 | 0.308 | 0.000 | 0.627 | 0.890 | 5.956 | 2723 | 47.049 | 26.063 | -7.292 | 46.513 | 1.309 | 0.924 | 0.958 | 0.391 | 0.483 | 0.214 | 0.278 | 54.405 | 26.049 | 33.605 | 19.812 |
| 2022_21_brazil | 78 | 0.931 | 4.070 | 0.667 | 0.000 | 0.875 | 0.942 | 5.237 | 2766 | 49.170 | 17.716 | 13.738 | 43.014 | 1.391 | 0.925 | 0.962 | 0.717 | 0.853 | 0.266 | 0.297 | 43.941 | -40.308 | 33.039 | -21.003 |
| 2022_22_united_arab_emirates | 60 | 0.902 | 3.564 | 0.367 | 0.000 | 0.857 | 0.908 | 3.970 | 2243 | 55.682 | 15.121 | 9.046 | 55.681 | 1.774 | 0.914 | 0.957 | 0.634 | 0.801 | 0.207 | 0.293 | 32.019 | 1.127 | 19.342 | -5.638 |
| 2023_01_bahrain | 60 | 0.836 | 2.334 | 0.483 | 0.000 | 0.857 | 0.850 | 2.757 | 2216 | 73.766 | 27.551 | 24.246 | 73.766 | 2.163 | 0.859 | 0.989 | 0.839 | 0.966 | 0.469 | 0.571 | 32.797 | 24.632 | 29.641 | 23.905 |
| 2023_07_spain | 72 | 0.917 | 2.022 | 0.486 | 0.000 | 0.735 | 0.937 | 2.551 | 2572 | 58.121 | 14.194 | 13.036 | 58.110 | 1.668 | 0.928 | 0.973 | 0.697 | 0.854 | 0.242 | 0.452 | 29.097 | 17.434 | 17.650 | 7.722 |
| 2024_07_emilia_romagna | 66 | 0.886 | 2.252 | 0.545 | 0.000 | 0.894 | 0.899 | 4.128 | 2351 | 35.973 | 8.590 | 2.563 | 35.465 | 1.182 | 0.992 | 0.995 | 0.579 | 0.764 | 0.143 | 0.216 | 33.655 | -15.626 | 20.445 | -5.115 |
| pooled | 1225 | 0.872 | 3.161 | 0.562 | 0.000 | 0.827 | 0.895 | 3.573 | 43676 | 55.224 | 16.998 | 8.269 | 54.066 | 1.678 | 0.928 | 0.975 | 0.709 | 0.847 | 0.343 | 0.481 | 45.866 | 5.960 | 32.582 | 9.935 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1867.00 | 21.45 | 21.42 |
| 10.00 | 8895.00 | 24.58 | 22.43 |
| 20.00 | 8833.00 | 25.78 | 23.49 |
| 30.00 | 8718.00 | 43.55 | 42.34 |
| 40.00 | 8020.00 | 80.84 | 80.84 |
| 50.00 | 4834.00 | 111.64 | 111.64 |
| 60.00 | 2509.00 | 142.69 | 142.42 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 13313 | 106.35 | 105.95 |
| late | 12334 | 23.92 | 20.87 |
| mid | 18029 | 38.89 | 38.47 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12732 | 53.76 | 52.58 |
| real | 1155 | 54.05 | 52.74 |
| stay | 350 | 44.66 | 44.18 |
| two | 29439 | 56.03 | 54.88 |

B5 future blindness: {"status": "pass", "new_leak_differences": 0, "differences_with_known_leaks": 0, "known_leaks": [], "fixed_leaks": ["compound_list", "frame_filter", "gap_behind", "lap_window", "pit_loss", "race_length", "stint_limit"], "detail": ""}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | history prior (tyre_life_prior) | oldest age run | 52 | 19 | 7.518 | -1.129 | 0.673 | 0.885 |
| tyre_life | history prior (tyre_life_prior) | median stint-end age | 52 | 19 | 9.608 | 8.958 | 0.558 | 0.846 |
| tyre_life | history prior (tyre_life_prior) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.757 | 1.571 | 0.812 | 0.938 |
| tyre_life | judge cliff edge (history_only) | oldest age run | 53 | 19 | 7.344 | -1.139 | - | - |
| tyre_life | judge cliff edge (history_only) | median stint-end age | 53 | 19 | 9.395 | 8.757 | - | - |
| tyre_life | judge cliff edge (history_only) | observed edge (tyre_history.observed_tyre_life) | 48 | 19 | 5.722 | 1.536 | - | - |
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
| wear_rate | belief @k10 | within-stint slope less fuel | 51 | 19 | 0.040 | 0.007 | 0.294 | 0.451 |
| wear_rate | belief @k20 | within-stint slope less fuel | 51 | 19 | 0.033 | 0.005 | 0.235 | 0.333 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 51 | 19 | 0.025 | -0.001 | 0.294 | 0.392 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 51 | 19 | 0.017 | -0.013 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 565 | 19 | 0.815 | 0.179 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 530 | 19 | 0.967 | 0.084 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 359 | 19 | 0.732 | 0.129 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 235 | 19 | 0.621 | 0.014 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.726 | -0.322 | - | - |
| compound_gap | belief @k0 | qualifying | 7 | 7 | 0.724 | 0.629 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.710 | -0.192 | - | - |
| compound_gap | belief @k10 | qualifying | 7 | 7 | 1.037 | 0.723 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.413 | -0.000 | - | - |
| compound_gap | belief @k20 | qualifying | 7 | 7 | 0.773 | 0.500 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 47 | 19 | 0.342 | -0.097 | - | - |
| compound_gap | belief @k(mid) | qualifying | 7 | 7 | 0.859 | 0.466 | - | - |
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
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.748 | -0.086 | 0.789 | 0.842 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.418 | -0.676 | 0.737 | 0.789 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.362 | 7.313 | 0.300 | 0.400 |
| pit_loss | belief charged @k10 | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 1.733 | 0.015 | - | - |
| pit_loss | belief charged @k10 | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 2.373 | -0.575 | - | - |
| pit_loss | belief charged @k10 | whole race, median SC stop | 10 | 10 | 13.375 | 7.368 | - | - |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.689 | -0.237 | 0.947 | 1.000 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.903 | -0.827 | 0.632 | 0.789 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.612 | 7.368 | 0.100 | 0.500 |
| pit_loss | belief charged @k20 | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.670 | -0.166 | - | - |
| pit_loss | belief charged @k20 | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.861 | -0.756 | - | - |
| pit_loss | belief charged @k20 | whole race, median SC stop | 10 | 10 | 12.611 | 7.404 | - | - |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.531 | -0.392 | 0.947 | 1.000 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.996 | -0.982 | 0.632 | 0.789 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.403 | 7.151 | 0.200 | 0.500 |
| pit_loss | belief charged @k(mid) | whole race, robust (pit_loss.whole_race: median green stop) | 19 | 19 | 0.508 | -0.327 | - | - |
| pit_loss | belief charged @k(mid) | whole race, replay mean (calibrate._pit_loss) | 19 | 19 | 1.945 | -0.917 | - | - |
| pit_loss | belief charged @k(mid) | whole race, median SC stop | 10 | 10 | 12.402 | 7.185 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 19 | 19 | 2.022 | 1.339 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 19 | 19 | 0.180 | 0.082 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 601 | 19 | 0.870 | 0.517 | - | - |
| traffic_rejoin | judge rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 594 | 19 | 1.268 | 0.449 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 601 | 19 | 0.792 | 0.003 | - | - |
| traffic_rejoin | belief rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 594 | 19 | 1.941 | -0.611 | - | - |
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

