# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_11_austria, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version df1ab3800c69-a4a7a2a12fe9. World config f916d3a10413 (race default); belief config 6ef149c3a23f. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "field_replay": false, "seed": 0, "max_stops": 2}. Run time 387 s.

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
| A_close_pair_acc | 0.753 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.957 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 13.186 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.882 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 3.360 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.834 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.667 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.876 | same, Spearman, mean over k |
| A_abs_mae_s | 18.926 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -4.205 | same, mean error |
| A_stop_gain_bias_s | -0.191 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 43.723 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 41.580 | same, mean |
| B1_green_mae_s | 16.998 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.895 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.988 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.481 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 32.582 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |
| E_real_regret_s | 5.523 | expert consistency (secondary): judge time of the team's real remaining strategy less the judge's best panel plan, mean over snapshots; should fall where passing is hard, never chased to zero |
| E_real_over_20s | 0.061 | same, share of snapshots where the judge says the real strategy lost more than 20 s |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.700 | 0.963 | 13.507 | 20.155 | 0.750 | 0.966 | 13.161 | 20.001 | 0.756 | 4.650 | 32.057 | 30.635 | 0.540 | 0.866 | 17.538 | 23.911 |
| 2021_01_bahrain | 0.909 | 0.992 | 4.520 | 9.929 | 0.909 | 0.992 | 4.526 | 10.017 | 0.945 | 2.566 | 47.430 | 47.430 | 0.691 | 0.943 | 13.079 | 15.530 |
| 2021_04_spain | 0.714 | 0.972 | 4.760 | 14.702 | 0.714 | 0.972 | 5.021 | 14.761 | 0.812 | 4.270 | 51.599 | 51.541 | 0.786 | 0.972 | 10.623 | 21.534 |
| 2021_09_austrian | 0.688 | 0.954 | 4.922 | 11.723 | 0.750 | 0.970 | 5.091 | 11.605 | 0.833 | 3.797 | 44.548 | 44.201 | 0.500 | 0.912 | 19.718 | 25.863 |
| 2021_20_qatar | 0.462 | 0.973 | 8.020 | 13.755 | 0.538 | 0.977 | 7.976 | 14.191 | 0.863 | 4.488 | 43.965 | 43.959 | 0.754 | 0.940 | 23.191 | 48.538 |
| 2021_22_united_arab_emirates | 0.786 | 0.964 | 22.921 | 49.173 | 0.786 | 0.964 | 22.776 | 49.056 | 0.895 | 5.030 | 34.078 | 29.152 | 0.693 | 0.857 | 33.577 | 51.123 |
| 2022_01_bahrain | 0.804 | 0.754 | 39.317 | 43.536 | 0.804 | 0.754 | 39.750 | 43.874 | 0.934 | 1.750 | 43.805 | 42.693 | 0.709 | 0.734 | 47.981 | 105.360 |
| 2022_02_saudi_arabia | 0.769 | 0.947 | 19.352 | 21.418 | 0.769 | 0.947 | 20.414 | 22.098 | 0.890 | 0.754 | 41.405 | 39.300 | 0.615 | 0.849 | 16.779 | 31.284 |
| 2022_06_spain | 0.500 | 0.967 | 12.332 | 12.887 | 0.500 | 0.967 | 12.540 | 13.539 | 0.924 | 2.051 | 77.082 | 77.082 | 0.800 | 0.968 | 17.718 | 24.955 |
| 2022_08_azerbaijan | 0.364 | 0.938 | 6.810 | 9.799 | 0.364 | 0.938 | 6.604 | 9.838 | 0.821 | 5.343 | 39.169 | 36.490 | 0.491 | 0.815 | 23.465 | 20.418 |
| 2022_09_canada | 0.733 | 0.978 | 31.333 | 32.092 | 0.733 | 0.978 | 32.178 | 32.729 | 0.900 | 1.283 | 44.397 | 34.069 | 0.627 | 0.812 | 42.420 | 42.320 |
| 2022_11_austria | 1.000 | 0.964 | 6.759 | 11.781 | 1.000 | 0.964 | 6.181 | 11.475 | 0.843 | 2.114 | 38.158 | 38.158 | 0.467 | 0.750 | 15.792 | 19.583 |
| 2022_12_france | 0.818 | 0.985 | 12.673 | 14.109 | 0.818 | 0.985 | 12.904 | 14.456 | 0.917 | 3.489 | 34.119 | 31.806 | 0.736 | 0.884 | 12.289 | 28.010 |
| 2022_20_mexico | 0.714 | 0.967 | 8.493 | 12.371 | 0.786 | 0.972 | 8.262 | 12.127 | 0.899 | 5.983 | 37.220 | 35.145 | 0.386 | 0.821 | 28.165 | 33.605 |
| 2022_21_brazil | 0.854 | 0.975 | 27.927 | 30.348 | 0.854 | 0.975 | 28.209 | 30.608 | 0.934 | 4.454 | 41.718 | 34.019 | 0.725 | 0.875 | 29.925 | 33.039 |
| 2022_22_united_arab_emirates | 0.643 | 0.989 | 4.170 | 14.413 | 0.643 | 0.986 | 5.100 | 14.803 | 0.909 | 3.911 | 41.881 | 41.051 | 0.629 | 0.938 | 11.525 | 19.342 |
| 2023_01_bahrain | 0.583 | 0.978 | 6.305 | 13.608 | 0.583 | 0.978 | 6.287 | 13.866 | 0.845 | 2.321 | 57.566 | 57.566 | 0.767 | 0.964 | 19.733 | 29.641 |
| 2023_07_spain | 0.800 | 0.983 | 7.120 | 9.557 | 0.800 | 0.983 | 8.827 | 10.335 | 0.916 | 2.684 | 44.809 | 44.752 | 0.760 | 0.941 | 21.613 | 17.650 |
| 2024_07_emilia_romagna | 0.750 | 0.914 | 4.839 | 10.228 | 0.750 | 0.914 | 4.729 | 10.218 | 0.891 | 2.546 | 26.167 | 23.608 | 0.650 | 0.805 | 11.989 | 20.445 |
| pooled | 0.742 | 0.956 | 12.952 | 18.715 | 0.753 | 0.957 | 13.186 | 18.926 | 0.882 | 3.360 | 43.723 | 41.580 | 0.667 | 0.876 | 21.954 | 32.582 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.742 | 368.000 | 0.956 | 0.836 | 0.820 | 12.952 | 18.715 | -4.225 | 17.053 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.519 | 368.000 | 0.705 | 2.819 | 0.400 | 22.075 | 35.597 | -1.204 | 17.053 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.546 | 368.000 | 0.775 | 2.304 | 0.482 | 26.318 | 30.237 | -4.292 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.753 | 368.000 | 0.957 | 0.824 | 0.822 | 13.186 | 18.926 | -4.205 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.514 | 368.000 | 0.632 | 3.140 | 0.367 | 23.479 | 37.430 | -1.113 | 17.053 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.530 | 368.000 | 0.759 | 2.370 | 0.480 | 26.872 | 31.127 | -4.201 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.663 | 368.000 | 0.933 | 1.102 | 0.737 | 13.914 | 22.036 | -8.910 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.500 | 368.000 | 0.657 | 3.001 | 0.392 | 23.596 | 64.887 | 24.182 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.557 | 368.000 | 0.752 | 2.372 | 0.503 | 26.601 | 63.390 | 21.095 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.671 | 368.000 | 0.935 | 1.058 | 0.749 | 13.933 | 22.331 | -9.265 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.511 | 368.000 | 0.616 | 3.117 | 0.381 | 24.449 | 65.753 | 25.024 | 17.053 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.543 | 368.000 | 0.749 | 2.443 | 0.469 | 27.226 | 63.926 | 21.936 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.745 | 368.000 | 0.955 | 0.841 | 0.820 | 13.111 | 19.112 | -5.208 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.514 | 368.000 | 0.703 | 2.855 | 0.385 | 22.027 | 35.920 | -2.188 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.543 | 368.000 | 0.772 | 2.316 | 0.479 | 26.193 | 30.567 | -5.276 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.750 | 368.000 | 0.956 | 0.836 | 0.822 | 13.342 | 19.260 | -5.242 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.511 | 368.000 | 0.639 | 3.120 | 0.365 | 23.397 | 37.620 | -2.150 | 17.053 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.535 | 368.000 | 0.761 | 2.360 | 0.480 | 26.708 | 31.346 | -5.238 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.742 | 368.000 | 0.956 | 0.836 | 0.820 | 12.952 | 18.715 | -4.225 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.519 | 368.000 | 0.705 | 2.819 | 0.400 | 22.075 | 35.597 | -1.204 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.546 | 368.000 | 0.775 | 2.304 | 0.482 | 26.318 | 30.237 | -4.292 | 17.053 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.757 | 362.000 | 0.956 | 0.821 | 0.827 | 13.075 | 18.412 | -3.958 | 16.842 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.511 | 362.000 | 0.623 | 3.141 | 0.366 | 23.476 | 37.306 | -0.594 | 16.842 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.528 | 362.000 | 0.752 | 2.383 | 0.480 | 26.672 | 30.912 | -4.132 | 16.842 |
| 2 judge, substituted | events_rate | 0 | in-sample | insample | yes | 0.726 | 368.000 | 0.936 | 1.006 | 0.781 | 15.272 | 31.344 | -2.988 | 17.053 |
| 2 judge, substituted | events_rate | 0 | in-sample | none | yes | 0.522 | 368.000 | 0.700 | 2.838 | 0.389 | 23.352 | 44.414 | 0.032 | 17.053 |
| 2 judge, substituted | events_rate | 0 | in-sample | practice | yes | 0.546 | 368.000 | 0.779 | 2.262 | 0.498 | 27.867 | 38.669 | -3.055 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | insample | yes | 0.726 | 368.000 | 0.937 | 1.000 | 0.781 | 15.410 | 31.500 | -2.938 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | none | no | 0.514 | 368.000 | 0.662 | 2.998 | 0.372 | 24.666 | 46.086 | 0.154 | 17.053 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | practice | no | 0.533 | 368.000 | 0.766 | 2.336 | 0.492 | 28.323 | 39.594 | -2.934 | 17.053 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.685 | 251.000 | 0.930 | 1.117 | 0.739 | 18.647 | 31.794 | -5.142 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.522 | 251.000 | 0.685 | 2.829 | 0.405 | 24.383 | 86.264 | 28.218 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.574 | 251.000 | 0.767 | 2.240 | 0.519 | 29.486 | 85.391 | 25.366 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.685 | 251.000 | 0.930 | 1.119 | 0.734 | 18.450 | 31.904 | -5.325 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.526 | 251.000 | 0.634 | 3.016 | 0.372 | 24.789 | 86.690 | 28.955 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.566 | 251.000 | 0.751 | 2.335 | 0.490 | 30.058 | 85.473 | 26.103 | 17.071 |
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
| 0 judge | 0.753 | 0.698 | 0.793 | 0.957 | 13.186 | 8.921 | 18.186 | 18.926 | 368 | 19 |
| 1 +stint pace | 0.696 | 0.610 | 0.761 | 0.957 | 11.936 | 7.421 | 17.109 | 18.366 | 368 | 19 |
| 2 +pit laps | 0.739 | 0.664 | 0.795 | 0.962 | 11.370 | 6.630 | 17.086 | 15.552 | 368 | 19 |
| 3 +neutral laps | 0.614 | 0.560 | 0.675 | 0.944 | 5.188 | 4.213 | 6.265 | 8.609 | 368 | 19 |
| 4 +lap 1 | 0.745 | 0.682 | 0.800 | 0.967 | 3.166 | 2.589 | 3.885 | 7.704 | 368 | 19 |
| alone: pit laps | 0.755 | 0.710 | 0.789 | 0.959 | 12.593 | 8.151 | 17.876 | 15.982 | 368 | 19 |
| alone: neutral laps | 0.614 | 0.559 | 0.679 | 0.902 | 8.800 | 7.076 | 10.987 | 13.454 | 368 | 19 |
| alone: lap 1 | 0.728 | 0.661 | 0.775 | 0.952 | 13.663 | 8.915 | 19.197 | 19.829 | 368 | 19 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 324.000 | 35.472 | -1.765 | 18.580 | -4.352 | 16.765 | -0.684 | 3.402 | 0.806 | 925 | 7.717 | -1.847 | 0.484 | 565.000 | 0.490 | -0.187 | -0.014 |
| pooled | lodo | 324.000 | 37.333 | -1.706 | 18.806 | -4.319 | 17.018 | -0.645 | 3.467 | 0.805 | 925 | 7.928 | -1.860 | 0.495 | 565.000 | 0.513 | -0.191 | -0.013 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 917 | 10.417 | -0.652 | 0.603 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 26.82 | -4.12 | 20.16 | -14.46 | 20.36 | -2.67 | 4.39 | -0.80 | 44 | 8.49 | -4.58 | 0.48 | 25.00 | 0.34 | 0.13 | -0.08 |
| 2019_10_united_kingdom | lodo | 17.00 | 28.66 | -4.11 | 20.00 | -14.81 | 20.26 | -2.64 | 4.60 | -0.80 | 44 | 8.54 | -4.74 | 0.48 | 25.00 | 0.38 | 0.12 | -0.07 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.52 | -5.19 | 0.53 | - | - | - | - |
| 2021_01_bahrain | in | 18.00 | 41.56 | -3.59 | 9.93 | -4.27 | 10.87 | -1.41 | 2.38 | -0.22 | 56 | 4.37 | -3.01 | 0.34 | 35.00 | 0.40 | -0.14 | 0.10 |
| 2021_01_bahrain | lodo | 18.00 | 43.27 | -3.49 | 10.02 | -4.47 | 10.88 | -1.40 | 2.41 | -0.23 | 56 | 4.47 | -3.07 | 0.34 | 35.00 | 0.42 | -0.14 | 0.10 |
| 2021_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 55 | 4.83 | -2.81 | 0.37 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 49.06 | -8.75 | 14.70 | -14.18 | 6.70 | -0.61 | 3.50 | -0.24 | 56 | 6.07 | -4.39 | 0.34 | 37.00 | 0.47 | -0.04 | -0.17 |
| 2021_04_spain | lodo | 19.00 | 51.50 | -8.68 | 14.76 | -14.14 | 6.46 | -0.58 | 3.54 | -0.24 | 56 | 6.15 | -4.40 | 0.34 | 37.00 | 0.48 | -0.04 | -0.17 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 6.84 | -4.08 | 0.38 | - | - | - | - |
| 2021_09_austrian | in | 19.00 | 36.92 | -8.71 | 11.72 | -10.24 | 9.98 | -3.34 | 2.87 | -0.75 | 46 | 6.52 | -5.98 | 0.27 | 26.00 | 0.44 | -0.25 | 0.16 |
| 2021_09_austrian | lodo | 19.00 | 38.62 | -8.76 | 11.61 | -10.02 | 9.83 | -3.30 | 2.85 | -0.75 | 46 | 6.64 | -5.92 | 0.27 | 26.00 | 0.45 | -0.24 | 0.16 |
| 2021_09_austrian | loso | - | - | - | - | - | - | - | - | - | 45 | 8.68 | -5.84 | 0.34 | - | - | - | - |
| 2021_20_qatar | in | 18.00 | 39.06 | -4.51 | 13.76 | -11.69 | 12.00 | -2.06 | 3.05 | 0.26 | 46 | 5.43 | -3.02 | 0.33 | 27.00 | 0.71 | -0.34 | 0.07 |
| 2021_20_qatar | lodo | 18.00 | 40.78 | -4.54 | 14.19 | -12.04 | 12.26 | -2.08 | 3.10 | 0.26 | 46 | 5.74 | -3.14 | 0.35 | 27.00 | 0.74 | -0.36 | 0.08 |
| 2021_20_qatar | loso | - | - | - | - | - | - | - | - | - | 46 | 7.71 | -2.80 | 0.44 | - | - | - | - |
| 2021_22_united_arab_emirates | in | 15.00 | 70.44 | 58.81 | 49.17 | 46.14 | 28.12 | 7.96 | 7.46 | 5.73 | 41 | 16.80 | 10.93 | 1.48 | 19.00 | 0.45 | -0.10 | 0.02 |
| 2021_22_united_arab_emirates | lodo | 15.00 | 71.70 | 59.21 | 49.06 | 46.15 | 28.36 | 7.94 | 7.44 | 5.73 | 41 | 17.03 | 10.94 | 1.50 | 19.00 | 0.48 | -0.10 | 0.01 |
| 2021_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 40 | 18.78 | 10.80 | 1.59 | - | - | - | - |
| 2022_01_bahrain | in | 19.00 | 19.03 | -6.79 | 43.54 | -2.54 | 44.21 | -1.16 | 2.73 | 1.50 | 75 | 14.92 | -0.43 | 1.33 | 54.00 | 0.50 | -0.24 | 0.13 |
| 2022_01_bahrain | lodo | 19.00 | 19.92 | -6.73 | 43.87 | -2.21 | 44.45 | -1.15 | 2.73 | 1.51 | 75 | 15.12 | -0.35 | 1.35 | 54.00 | 0.52 | -0.24 | 0.13 |
| 2022_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 75 | 16.36 | -0.10 | 1.42 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 31.77 | -9.87 | 21.42 | -19.57 | 18.17 | -5.15 | 8.45 | -0.26 | 29 | 10.58 | -8.56 | 0.44 | 15.00 | 0.31 | 0.02 | 0.13 |
| 2022_02_saudi_arabia | lodo | 14.00 | 33.92 | -9.65 | 22.10 | -19.74 | 19.04 | -5.26 | 8.49 | -0.26 | 29 | 10.92 | -8.58 | 0.46 | 15.00 | 0.35 | 0.01 | 0.13 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 12.25 | -8.62 | 0.51 | - | - | - | - |
| 2022_06_spain | in | 18.00 | 48.87 | -12.16 | 12.89 | -9.95 | 13.23 | 0.05 | 1.32 | - | 69 | 3.77 | -2.34 | 0.27 | 49.00 | 0.51 | -0.14 | 0.12 |
| 2022_06_spain | lodo | 18.00 | 53.29 | -14.26 | 13.54 | -10.51 | 14.17 | 0.15 | 1.35 | - | 69 | 4.23 | -2.56 | 0.29 | 49.00 | 0.53 | -0.14 | 0.15 |
| 2022_06_spain | loso | - | - | - | - | - | - | - | - | - | 67 | 21.00 | 13.11 | 0.94 | - | - | - | - |
| 2022_08_azerbaijan | in | 16.00 | 38.41 | -6.98 | 9.80 | -1.68 | 10.01 | 1.20 | 5.57 | 0.98 | 40 | 4.31 | -1.44 | 0.31 | 23.00 | 0.49 | -0.25 | 0.08 |
| 2022_08_azerbaijan | lodo | 16.00 | 39.61 | -7.12 | 9.84 | -1.28 | 10.00 | 1.17 | 5.66 | 0.99 | 40 | 4.40 | -1.27 | 0.31 | 23.00 | 0.51 | -0.27 | 0.07 |
| 2022_08_azerbaijan | loso | - | - | - | - | - | - | - | - | - | 40 | 5.74 | -1.20 | 0.38 | - | - | - | - |
| 2022_09_canada | in | 17.00 | 12.69 | 8.13 | 32.09 | 6.83 | 31.33 | -0.38 | 2.94 | 2.00 | 47 | 14.48 | 3.31 | 0.62 | 30.00 | 0.43 | -0.03 | 0.08 |
| 2022_09_canada | lodo | 17.00 | 14.16 | 8.20 | 32.73 | 6.66 | 32.18 | -0.39 | 3.01 | 2.00 | 47 | 14.69 | 3.26 | 0.63 | 30.00 | 0.46 | -0.04 | 0.09 |
| 2022_09_canada | loso | - | - | - | - | - | - | - | - | - | 47 | 15.00 | 3.30 | 0.65 | - | - | - | - |
| 2022_11_austria | in | 7.00 | 37.52 | 1.47 | 11.78 | -10.63 | 5.44 | -2.47 | 2.49 | -0.50 | 24 | 3.59 | -1.92 | 0.19 | 9.00 | 0.41 | -0.13 | 0.04 |
| 2022_11_austria | lodo | 7.00 | 38.75 | 2.43 | 11.48 | -10.79 | 5.76 | -2.50 | 2.53 | -0.51 | 24 | 3.92 | -1.94 | 0.21 | 9.00 | 0.44 | -0.10 | 0.02 |
| 2022_11_austria | loso | - | - | - | - | - | - | - | - | - | 23 | 5.81 | -1.57 | 0.30 | - | - | - | - |
| 2022_12_france | in | 16.00 | 23.71 | -9.60 | 14.11 | -8.85 | 14.45 | -1.47 | 4.99 | -0.01 | 36 | 7.13 | -2.47 | 0.32 | 20.00 | 0.32 | -0.11 | -0.05 |
| 2022_12_france | lodo | 16.00 | 25.05 | -8.72 | 14.46 | -8.50 | 14.74 | -1.09 | 5.24 | -0.02 | 36 | 7.18 | -2.52 | 0.32 | 20.00 | 0.36 | -0.12 | -0.05 |
| 2022_12_france | loso | - | - | - | - | - | - | - | - | - | 36 | 7.77 | -2.86 | 0.36 | - | - | - | - |
| 2022_20_mexico | in | 19.00 | 46.14 | 4.26 | 12.37 | -3.79 | 12.83 | 1.54 | 2.42 | -2.42 | 41 | 10.51 | -2.94 | 0.38 | 21.00 | 0.46 | -0.26 | -1.06 |
| 2022_20_mexico | lodo | 19.00 | 48.04 | 4.34 | 12.13 | -3.89 | 12.73 | 1.51 | 2.41 | -2.42 | 41 | 10.82 | -2.97 | 0.40 | 21.00 | 0.48 | -0.26 | -1.06 |
| 2022_20_mexico | loso | - | - | - | - | - | - | - | - | - | 41 | 14.00 | -1.55 | 0.49 | - | - | - | - |
| 2022_21_brazil | in | 17.00 | 18.26 | -9.30 | 30.35 | -3.71 | 30.73 | -1.14 | 2.39 | -0.30 | 59 | 10.53 | -2.67 | 0.61 | 39.00 | 0.39 | -0.18 | 0.07 |
| 2022_21_brazil | lodo | 17.00 | 20.00 | -9.57 | 30.61 | -3.79 | 30.98 | -1.18 | 2.43 | -0.30 | 59 | 10.69 | -2.67 | 0.61 | 39.00 | 0.40 | -0.18 | 0.07 |
| 2022_21_brazil | loso | - | - | - | - | - | - | - | - | - | 59 | 11.08 | -2.74 | 0.63 | - | - | - | - |
| 2022_22_united_arab_emirates | in | 19.00 | 39.47 | -9.88 | 14.41 | -12.29 | 14.50 | -3.95 | 4.43 | - | 49 | 4.30 | -3.22 | 0.23 | 30.00 | 0.43 | -0.15 | 0.13 |
| 2022_22_united_arab_emirates | lodo | 19.00 | 41.62 | -9.72 | 14.80 | -12.21 | 14.65 | -3.72 | 4.51 | - | 49 | 4.64 | -3.33 | 0.25 | 30.00 | 0.45 | -0.17 | 0.15 |
| 2022_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 49 | 6.10 | -3.48 | 0.31 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 31.92 | 4.37 | 13.61 | 0.52 | 13.45 | 0.42 | 2.80 | 2.43 | 62 | 4.97 | -0.98 | 0.41 | 42.00 | 0.78 | -0.26 | 0.12 |
| 2023_01_bahrain | lodo | 17.00 | 33.91 | 4.67 | 13.87 | 0.60 | 13.69 | 0.48 | 2.92 | 2.41 | 62 | 5.09 | -1.01 | 0.42 | 42.00 | 0.83 | -0.26 | 0.11 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.01 | -0.89 | 0.56 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 28.29 | -5.38 | 9.56 | -4.75 | 8.79 | -0.53 | 1.05 | - | 63 | 4.10 | -1.28 | 0.22 | 42.00 | 0.63 | -0.48 | 0.03 |
| 2023_07_spain | lodo | 20.00 | 30.00 | -4.64 | 10.34 | -3.97 | 9.62 | -0.51 | 1.14 | - | 63 | 4.47 | -1.04 | 0.23 | 42.00 | 0.66 | -0.49 | 0.01 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 6.15 | -1.60 | 0.32 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 36.42 | -0.29 | 10.23 | -1.18 | 10.26 | 0.36 | 1.44 | - | 42 | 5.47 | -1.66 | 0.23 | 22.00 | 0.44 | -0.34 | -0.42 |
| 2024_07_emilia_romagna | lodo | 19.00 | 38.35 | -0.02 | 10.22 | -0.96 | 10.35 | 0.48 | 1.55 | - | 42 | 5.59 | -1.63 | 0.23 | 22.00 | 0.43 | -0.34 | -0.43 |
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
| in | sc | 682 | -1.26 | 7.57 |
| in | vsc | 360 | 0.05 | 4.39 |
| lodo | sc | 682 | -1.26 | 7.59 |
| lodo | vsc | 360 | 0.05 | 4.40 |
| loso | sc | 682 | -1.23 | 7.56 |
| loso | vsc | 360 | 0.04 | 4.40 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.54 |
| 2021_01_bahrain | 2 | 2 | 2:16 | 1.00 | 0.81 | 1.00 | 1.00 | 60 | 0.57 | 0.57 | 2.20 |
| 2021_04_spain | 2 | 2 | 1:9 2:10 | 0.53 | 0.53 | 1.00 | 1.00 | 72 | 0.31 | 0.31 | 1.54 |
| 2021_09_austrian | 1 | 1 | 1:19 | 1.00 | 0.58 | 0.26 | 0.26 | 78 | 0.05 | 0.05 | 0.66 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 0.00 | 0.00 | 60 | 0.23 | 0.23 | 0.64 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 0.00 | 0.00 | 60 | 0.02 | 0.02 | 0.62 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 3.08 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 0.93 | 0.93 | 54 | 0.37 | 0.37 | 0.62 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.44 | 0.44 | 72 | 0.11 | 0.11 | 4.01 |
| 2022_08_azerbaijan | 2 | 2 | 1:7 2:8 | 0.53 | 0.47 | 0.80 | 0.80 | 54 | 0.31 | 0.31 | 0.73 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 1.00 | 1.00 | 78 | 0.54 | 0.54 | 1.22 |
| 2022_11_austria | 2 | 2 | 2:17 | 1.00 | 0.82 | 0.76 | 0.76 | 55 | 0.51 | 0.51 | 2.36 |
| 2022_12_france | 1 | 1 | 1:15 | 1.00 | 0.80 | 0.00 | 0.00 | 54 | 0.04 | 0.04 | 0.66 |
| 2022_20_mexico | 1 | 2 | 2:18 | 0.00 | 0.17 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 2.43 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.06 | 0.06 | 78 | 0.00 | 0.00 | 2.53 |
| 2022_22_united_arab_emirates | 2 | 1 | 1:16 2:1 | 0.06 | 0.35 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.78 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 1.85 |
| 2023_07_spain | 2 | 1 | 1:14 2:6 | 0.30 | 0.30 | 0.20 | 0.20 | 72 | 0.08 | 0.08 | 1.04 |
| 2024_07_emilia_romagna | 1 | 1 | 1:19 | 1.00 | 0.84 | 0.95 | 0.95 | 66 | 0.33 | 0.33 | 0.70 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 368 | 368 | -0.079 | -0.023 | 0.962 | 3.567 | -4.838 | 0.075 | -1.233 | 3.308 | -0.037 | 0.174 | 0.174 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 368 | 368 | -0.082 | -0.023 | 0.747 | 3.604 | -5.197 | 0.112 | -1.291 | 3.095 | - | - | - |
| pit_loss_prior | in | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 368 | 368 | 0.003 | -0.001 | 0.160 | 0.485 | -1.149 | 0.000 | -0.619 | 0.000 | -0.073 | 0.053 | 0.053 |
| pit_loss_prior | lodo | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 368 | 368 | -0.003 | -0.001 | 0.156 | 0.424 | -1.206 | 0.000 | -0.650 | 0.000 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 368 | 368 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 | -0.093 | -0.093 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 362 | 368 | 0.004 | -0.001 | -0.111 | -0.475 | 0.235 | 0.003 | 0.152 | -0.039 | - | - | - |
| events_rate | in | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 368 | 368 | -0.016 | -0.020 | 2.320 | 12.574 | 1.890 | 0.000 | 1.142 | 4.600 | -0.003 | -0.075 | -0.075 |
| events_rate | lodo | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 368 | 368 | -0.027 | -0.020 | 2.224 | 12.501 | 1.914 | 0.000 | 1.145 | 4.541 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 251 | 368 | -0.057 | -0.025 | 5.696 | 11.454 | -0.944 | 0.105 | -0.176 | 8.064 | 0.058 | 0.072 | 0.072 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 251 | 368 | -0.067 | -0.027 | 5.264 | 11.301 | -1.151 | 0.134 | -0.227 | 7.867 | - | - | - |

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
| 2019_10_united_kingdom | 54 | 0.756 | 4.650 | 0.537 | 0.000 | 0.811 | 0.774 | 4.093 | 1931 | 32.057 | 13.894 | 4.966 | 30.635 | 0.998 | 0.959 | 0.982 | 0.566 | 0.622 | 0.228 | 0.421 | 53.096 | 38.949 | 23.911 | 2.136 |
| 2021_01_bahrain | 60 | 0.945 | 2.566 | 0.567 | 0.000 | 0.879 | 0.944 | 3.033 | 2127 | 47.430 | 11.056 | 3.534 | 47.430 | 1.661 | 0.968 | 0.988 | 0.717 | 0.921 | 0.188 | 0.250 | 32.120 | -16.047 | 15.530 | -7.419 |
| 2021_04_spain | 72 | 0.812 | 4.270 | 0.389 | 0.000 | 0.775 | 0.832 | 4.050 | 2544 | 51.599 | 14.936 | 3.921 | 51.541 | 1.473 | 0.920 | 0.999 | 0.707 | 0.846 | 0.413 | 0.583 | 63.514 | -44.497 | 21.534 | -17.116 |
| 2021_09_austrian | 78 | 0.833 | 3.797 | 0.679 | 0.000 | 0.829 | 0.860 | 4.038 | 2707 | 44.548 | 22.912 | 16.440 | 44.201 | 0.987 | 0.906 | 0.945 | 0.741 | 0.893 | 0.333 | 0.540 | 49.224 | -17.327 | 25.863 | 2.222 |
| 2021_20_qatar | 60 | 0.863 | 4.488 | 0.733 | 0.000 | 0.914 | 0.873 | 4.793 | 2140 | 43.965 | 20.025 | 19.186 | 43.959 | 1.246 | 0.957 | 0.991 | 0.754 | 0.891 | 0.390 | 0.555 | 53.550 | 34.869 | 48.538 | 46.434 |
| 2021_22_united_arab_emirates | 60 | 0.895 | 5.030 | 0.600 | 0.000 | 0.712 | 0.919 | 5.265 | 2150 | 34.078 | 13.259 | 3.160 | 29.152 | 1.196 | 0.972 | 0.993 | 0.829 | 0.920 | 0.276 | 0.583 | 58.863 | 53.600 | 51.123 | 37.607 |
| 2022_01_bahrain | 60 | 0.934 | 1.750 | 0.567 | 0.000 | 0.778 | 0.949 | 1.626 | 2169 | 43.805 | 14.035 | 10.526 | 42.693 | 1.506 | 0.964 | 1.000 | 0.817 | 0.956 | 0.433 | 0.631 | 87.498 | 84.619 | 105.360 | 89.525 |
| 2022_02_saudi_arabia | 54 | 0.890 | 0.754 | 0.889 | 0.000 | 0.963 | 0.918 | 0.404 | 1779 | 41.405 | 14.813 | 2.847 | 39.300 | 1.632 | 0.945 | 0.995 | 0.781 | 0.876 | 0.553 | 0.696 | 40.246 | -24.644 | 31.284 | -22.326 |
| 2022_06_spain | 72 | 0.924 | 2.051 | 0.806 | 0.000 | 0.971 | 0.927 | 2.377 | 2654 | 77.082 | 38.402 | 34.826 | 77.082 | 1.977 | 0.953 | 0.979 | 0.795 | 0.953 | 0.522 | 0.633 | 68.704 | -43.781 | 24.955 | -4.459 |
| 2022_08_azerbaijan | 54 | 0.821 | 5.343 | 0.519 | 0.000 | 0.885 | 0.842 | 6.533 | 1858 | 39.169 | 12.344 | -6.959 | 36.490 | 1.603 | 1.000 | 1.000 | 0.755 | 0.853 | 0.404 | 0.528 | 29.920 | 9.192 | 20.418 | 6.441 |
| 2022_09_canada | 78 | 0.900 | 1.283 | 0.705 | 0.000 | 0.868 | 0.933 | 0.865 | 2758 | 44.397 | 11.592 | -0.871 | 34.069 | 1.144 | 0.950 | 1.000 | 0.962 | 0.991 | 0.460 | 0.566 | 30.602 | 11.997 | 42.320 | -3.488 |
| 2022_11_austria | 55 | 0.843 | 2.114 | 0.545 | 0.000 | 0.907 | 0.859 | 1.727 | 2005 | 38.158 | 8.653 | -1.787 | 38.158 | 0.994 | 0.996 | 1.000 | 0.848 | 0.934 | 0.483 | 0.701 | 34.316 | 4.828 | 19.583 | 1.083 |
| 2022_12_france | 54 | 0.917 | 3.489 | 0.389 | 0.000 | 0.717 | 0.918 | 4.172 | 1983 | 34.119 | 9.110 | 4.341 | 31.806 | 1.263 | 0.999 | 1.000 | 0.522 | 0.734 | 0.333 | 0.431 | 32.580 | 8.737 | 28.010 | 20.307 |
| 2022_20_mexico | 78 | 0.899 | 5.983 | 0.321 | 0.000 | 0.653 | 0.890 | 5.956 | 2723 | 37.220 | 26.063 | -7.292 | 35.145 | 1.020 | 0.930 | 0.964 | 0.415 | 0.530 | 0.214 | 0.278 | 54.405 | 26.049 | 33.605 | 19.812 |
| 2022_21_brazil | 78 | 0.934 | 4.454 | 0.641 | 0.000 | 0.847 | 0.942 | 5.237 | 2766 | 41.718 | 17.716 | 13.738 | 34.019 | 1.245 | 0.962 | 0.971 | 0.701 | 0.856 | 0.266 | 0.297 | 43.941 | -40.308 | 33.039 | -21.003 |
| 2022_22_united_arab_emirates | 60 | 0.909 | 3.911 | 0.383 | 0.000 | 0.839 | 0.908 | 3.970 | 2243 | 41.881 | 15.121 | 9.046 | 41.051 | 1.273 | 0.965 | 0.994 | 0.588 | 0.780 | 0.207 | 0.293 | 32.019 | 1.127 | 19.342 | -5.638 |
| 2023_01_bahrain | 60 | 0.845 | 2.321 | 0.583 | 0.000 | 0.911 | 0.850 | 2.757 | 2216 | 57.566 | 27.551 | 24.246 | 57.566 | 1.600 | 0.941 | 0.997 | 0.861 | 0.968 | 0.469 | 0.571 | 32.797 | 24.632 | 29.641 | 23.905 |
| 2023_07_spain | 72 | 0.916 | 2.684 | 0.444 | 0.000 | 0.721 | 0.937 | 2.551 | 2572 | 44.809 | 14.194 | 13.036 | 44.752 | 1.266 | 0.958 | 0.995 | 0.715 | 0.857 | 0.242 | 0.452 | 29.097 | 17.434 | 17.650 | 7.722 |
| 2024_07_emilia_romagna | 66 | 0.891 | 2.546 | 0.545 | 0.000 | 0.924 | 0.899 | 4.128 | 2351 | 26.167 | 8.590 | 2.563 | 23.608 | 0.864 | 0.994 | 0.997 | 0.544 | 0.745 | 0.143 | 0.216 | 33.655 | -15.626 | 20.445 | -5.115 |
| pooled | 1225 | 0.882 | 3.360 | 0.571 | 0.000 | 0.834 | 0.895 | 3.573 | 43676 | 43.723 | 16.998 | 8.269 | 41.580 | 1.307 | 0.958 | 0.988 | 0.718 | 0.850 | 0.343 | 0.481 | 45.866 | 5.960 | 32.582 | 9.935 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1867.00 | 18.04 | 17.97 |
| 10.00 | 8895.00 | 18.40 | 14.85 |
| 20.00 | 8833.00 | 18.89 | 13.51 |
| 30.00 | 8718.00 | 30.33 | 29.08 |
| 40.00 | 8020.00 | 64.45 | 64.40 |
| 50.00 | 4834.00 | 92.50 | 92.33 |
| 60.00 | 2509.00 | 126.33 | 125.46 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 13313 | 89.32 | 88.52 |
| late | 12334 | 18.81 | 13.94 |
| mid | 18029 | 27.09 | 25.82 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12732 | 43.28 | 41.20 |
| real | 1155 | 43.80 | 41.73 |
| stay | 350 | 34.26 | 33.25 |
| two | 29439 | 44.03 | 41.84 |

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
| pit_loss | prior from earlier races (constant) | whole race, robust (pit_loss.whole_race: median green stop) | 2 | 2 | 1.281 | -0.187 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (constant) | whole race, replay mean (calibrate._pit_loss) | 2 | 2 | 2.045 | -2.045 | 1.000 | 1.000 |
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
| neutralised_laps | judge replay (recorded lap times) | mean real lap time on the cars' neutralised laps | 22 | 15 | 1.172 | 0.583 | - | - |
| neutralised_laps | belief replay (factors) | mean real lap time on the cars' neutralised laps | 22 | 15 | 9.885 | 6.846 | - | - |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 11 | 11 | 2.457 | -0.959 | 0.545 | 0.636 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 11 | 11 | 2.841 | -2.021 | 0.545 | 0.727 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 6 | 6 | 0.869 | -0.641 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 6 | 6 | 2.654 | 0.059 | 0.167 | 0.833 |

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

