# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version 096753d89c0f-3cd6a5d8c2cf. World config ccb12649ff32 (race default); belief config 30ede0981519. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "field_replay": false, "seed": 0, "max_stops": 2}. Run time 308 s.

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
| A_close_pair_acc | 0.751 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.957 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 13.559 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.867 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 4.223 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.806 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.665 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.882 | same, Spearman, mean over k |
| A_abs_mae_s | 19.305 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -3.807 | same, mean error |
| A_stop_gain_bias_s | -0.191 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 50.440 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 45.703 | same, mean |
| B1_green_mae_s | 16.974 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.883 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.975 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.460 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 33.051 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.700 | 0.963 | 13.507 | 20.037 | 0.750 | 0.966 | 13.161 | 20.001 | 0.708 | 5.808 | 43.359 | 42.631 | 0.500 | 0.853 | 18.153 | 24.338 |
| 2021_01_bahrain | 0.909 | 0.992 | 4.520 | 9.929 | 0.909 | 0.992 | 4.526 | 10.017 | 0.942 | 2.397 | 51.934 | 51.319 | 0.691 | 0.943 | 13.106 | 15.555 |
| 2021_04_spain | 0.714 | 0.972 | 4.729 | 14.526 | 0.714 | 0.972 | 5.021 | 14.761 | 0.799 | 5.270 | 55.709 | 53.844 | 0.743 | 0.972 | 10.487 | 21.746 |
| 2021_09_austrian | 0.688 | 0.954 | 4.906 | 11.748 | 0.750 | 0.975 | 5.071 | 11.628 | 0.812 | 3.753 | 46.073 | 40.033 | 0.487 | 0.910 | 20.078 | 26.352 |
| 2021_20_qatar | 0.462 | 0.973 | 8.020 | 13.954 | 0.538 | 0.977 | 7.976 | 14.191 | 0.854 | 4.183 | 38.812 | 38.622 | 0.769 | 0.943 | 23.273 | 49.154 |
| 2021_22_united_arab_emirates | 0.786 | 0.964 | 22.921 | 49.109 | 0.786 | 0.964 | 22.776 | 49.056 | 0.886 | 5.664 | 37.540 | 27.393 | 0.700 | 0.861 | 33.590 | 51.680 |
| 2022_01_bahrain | 0.804 | 0.754 | 39.317 | 43.536 | 0.804 | 0.754 | 39.750 | 43.874 | 0.932 | 1.825 | 54.218 | 54.201 | 0.709 | 0.734 | 47.981 | 105.366 |
| 2022_02_saudi_arabia | 0.692 | 0.938 | 19.207 | 21.438 | 0.769 | 0.947 | 20.414 | 22.098 | 0.878 | 0.755 | 71.519 | 62.465 | 0.600 | 0.848 | 16.448 | 30.017 |
| 2022_06_spain | 0.500 | 0.967 | 12.331 | 12.828 | 0.500 | 0.967 | 12.540 | 13.482 | 0.912 | 2.085 | 80.983 | 78.714 | 0.850 | 0.971 | 17.830 | 24.978 |
| 2022_08_azerbaijan | 0.273 | 0.935 | 6.538 | 9.952 | 0.364 | 0.938 | 6.604 | 9.857 | 0.762 | 6.665 | 51.752 | 40.039 | 0.509 | 0.818 | 22.857 | 20.126 |
| 2022_09_canada | 0.733 | 0.978 | 31.333 | 32.092 | 0.733 | 0.978 | 32.178 | 32.729 | 0.898 | 1.410 | 46.136 | 41.973 | 0.627 | 0.802 | 42.747 | 42.749 |
| 2022_12_france | 0.818 | 0.985 | 12.673 | 14.109 | 0.818 | 0.985 | 12.904 | 14.456 | 0.909 | 4.324 | 58.966 | 51.817 | 0.736 | 0.880 | 12.556 | 29.902 |
| 2022_20_mexico | 0.714 | 0.967 | 9.581 | 13.245 | 0.786 | 0.972 | 8.262 | 12.127 | 0.894 | 5.760 | 47.433 | 36.441 | 0.400 | 0.824 | 26.143 | 35.276 |
| 2022_21_brazil | 0.854 | 0.975 | 27.947 | 30.366 | 0.854 | 0.975 | 28.194 | 30.594 | 0.862 | 7.415 | 50.010 | 46.494 | 0.725 | 0.875 | 29.925 | 33.038 |
| 2022_22_united_arab_emirates | 0.643 | 0.989 | 4.366 | 14.539 | 0.643 | 0.989 | 4.855 | 14.771 | 0.912 | 3.949 | 47.787 | 41.982 | 0.600 | 0.935 | 11.922 | 19.565 |
| 2023_01_bahrain | 0.583 | 0.978 | 6.305 | 13.506 | 0.583 | 0.978 | 6.287 | 13.454 | 0.879 | 2.311 | 46.357 | 45.260 | 0.750 | 0.962 | 19.764 | 29.750 |
| 2023_07_spain | 0.800 | 0.983 | 7.411 | 9.668 | 0.800 | 0.983 | 8.827 | 10.335 | 0.918 | 3.026 | 51.456 | 46.789 | 0.744 | 0.940 | 20.629 | 17.393 |
| 2024_07_emilia_romagna | 0.750 | 0.914 | 5.080 | 10.338 | 0.750 | 0.914 | 4.720 | 10.059 | 0.834 | 9.016 | 28.782 | 22.532 | 0.660 | 0.801 | 12.445 | 18.685 |
| pooled | 0.734 | 0.955 | 13.372 | 19.162 | 0.751 | 0.957 | 13.559 | 19.305 | 0.867 | 4.223 | 50.440 | 45.703 | 0.665 | 0.882 | 22.219 | 33.051 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.734 | 365.000 | 0.955 | 0.881 | 0.806 | 13.372 | 19.162 | -3.845 | 17.611 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.521 | 365.000 | 0.710 | 2.890 | 0.394 | 22.438 | 35.563 | -1.309 | 17.611 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.551 | 365.000 | 0.774 | 2.367 | 0.472 | 26.335 | 30.777 | -3.730 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.751 | 365.000 | 0.957 | 0.842 | 0.815 | 13.559 | 19.305 | -3.807 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.515 | 365.000 | 0.632 | 3.230 | 0.359 | 23.958 | 37.353 | -1.306 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.532 | 365.000 | 0.756 | 2.454 | 0.467 | 26.981 | 31.705 | -3.728 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.663 | 365.000 | 0.940 | 1.115 | 0.739 | 14.538 | 22.607 | -8.751 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.496 | 365.000 | 0.650 | 3.120 | 0.367 | 24.009 | 65.757 | 27.706 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.556 | 365.000 | 0.746 | 2.456 | 0.483 | 26.773 | 63.893 | 25.285 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.671 | 365.000 | 0.941 | 1.070 | 0.751 | 14.492 | 22.906 | -9.114 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.510 | 365.000 | 0.614 | 3.211 | 0.371 | 24.744 | 66.548 | 28.529 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.542 | 365.000 | 0.742 | 2.532 | 0.448 | 27.598 | 64.525 | 26.107 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.737 | 365.000 | 0.954 | 0.887 | 0.802 | 13.529 | 19.707 | -5.074 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.515 | 365.000 | 0.707 | 2.934 | 0.372 | 22.363 | 35.906 | -2.538 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.548 | 365.000 | 0.771 | 2.391 | 0.466 | 26.226 | 31.219 | -4.960 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.748 | 365.000 | 0.956 | 0.861 | 0.815 | 13.718 | 19.791 | -5.095 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.512 | 365.000 | 0.639 | 3.214 | 0.354 | 23.858 | 37.573 | -2.594 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.537 | 365.000 | 0.757 | 2.443 | 0.467 | 26.823 | 32.073 | -5.016 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.734 | 365.000 | 0.955 | 0.881 | 0.806 | 13.372 | 19.162 | -3.845 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.521 | 365.000 | 0.710 | 2.890 | 0.394 | 22.438 | 35.563 | -1.309 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.551 | 365.000 | 0.774 | 2.367 | 0.472 | 26.335 | 30.777 | -3.730 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.755 | 359.000 | 0.956 | 0.839 | 0.820 | 13.443 | 18.788 | -3.571 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.513 | 359.000 | 0.622 | 3.231 | 0.358 | 23.956 | 37.220 | -0.756 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.529 | 359.000 | 0.749 | 2.467 | 0.467 | 26.770 | 31.476 | -3.653 | 17.389 |
| 2 judge, substituted | events_rate | 0 | in-sample | insample | yes | 0.726 | 365.000 | 0.946 | 0.998 | 0.777 | 15.929 | 32.356 | -3.252 | 17.611 |
| 2 judge, substituted | events_rate | 0 | in-sample | none | yes | 0.521 | 365.000 | 0.708 | 2.907 | 0.387 | 23.676 | 44.412 | -0.716 | 17.611 |
| 2 judge, substituted | events_rate | 0 | in-sample | practice | yes | 0.551 | 365.000 | 0.780 | 2.323 | 0.489 | 28.105 | 39.833 | -3.137 | 17.611 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | insample | yes | 0.726 | 365.000 | 0.948 | 0.980 | 0.780 | 15.967 | 32.424 | -3.199 | 17.611 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | none | no | 0.515 | 365.000 | 0.669 | 3.058 | 0.372 | 25.086 | 45.989 | -0.698 | 17.611 |
| 2 judge, substituted | events_rate | 0 | leave-one-driver-out | practice | no | 0.534 | 365.000 | 0.763 | 2.418 | 0.480 | 28.638 | 40.754 | -3.120 | 17.611 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.685 | 251.000 | 0.930 | 1.117 | 0.739 | 18.647 | 31.794 | -5.142 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.522 | 251.000 | 0.685 | 2.829 | 0.405 | 24.383 | 86.264 | 28.218 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.574 | 251.000 | 0.767 | 2.240 | 0.519 | 29.486 | 85.391 | 25.366 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.685 | 251.000 | 0.930 | 1.119 | 0.734 | 18.450 | 31.904 | -5.325 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.526 | 251.000 | 0.634 | 3.016 | 0.372 | 24.789 | 86.690 | 28.955 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.566 | 251.000 | 0.751 | 2.335 | 0.490 | 30.058 | 85.473 | 26.103 | 17.071 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.542 | 365.000 | 0.767 | 2.373 | 0.480 | 28.947 | 83.173 | -13.928 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.644 | 365.000 | 0.877 | 1.661 | 0.588 | 27.573 | 52.295 | 17.167 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.649 | 365.000 | 0.893 | 1.484 | 0.645 | 22.963 | 33.697 | 10.517 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.701 | 365.000 | 0.926 | 1.159 | 0.722 | 17.430 | 25.577 | 7.128 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.789 | 365.000 | 0.946 | 0.887 | 0.822 | 14.180 | 21.386 | 4.604 | 17.611 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.521 | 365.000 | 0.695 | 2.984 | 0.353 | 25.920 | 84.017 | -11.506 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.545 | 365.000 | 0.776 | 2.531 | 0.413 | 23.131 | 53.918 | 21.402 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.611 | 365.000 | 0.827 | 1.992 | 0.509 | 18.335 | 38.203 | 11.375 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.693 | 365.000 | 0.891 | 1.526 | 0.630 | 16.932 | 30.140 | 7.655 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.753 | 365.000 | 0.924 | 1.124 | 0.738 | 13.776 | 24.246 | 4.998 | 17.611 |

## Noise floor (the judge handed the real race, step by step)

The judge's whole-race replay (version 1, leave-one-driver-out fit, whole-race pace offset: level 0 is the headline) with each level's real values in place of the model's, cumulative from 1 to 4, then each alone: stint_pace: each stint's own pace offset (mean clean-lap residual of that stint, five or more clean laps; else the car's), on every lap of the stint; pit_laps: in- and out-laps as recorded, no charged pit loss (the real stop durations and pit-lane driving); neutral_laps: SC and VSC laps as recorded; lap1: lap 1 as recorded (the start). What level 4 leaves is lap-to-lap noise around each stint's pace, incidents, errors and unmodelled effects. Pooled with race-cluster bootstrap 95% intervals:

| level | close_pair_acc | close_pair_lo | close_pair_hi | spearman | rel_gap_mae_s | rel_gap_lo | rel_gap_hi | abs_mae_s | close_pairs | races |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 judge | 0.751 | 0.694 | 0.789 | 0.957 | 13.559 | 9.187 | 18.320 | 19.305 | 365 | 18 |
| 1 +stint pace | 0.696 | 0.610 | 0.758 | 0.956 | 12.380 | 7.745 | 17.325 | 18.842 | 365 | 18 |
| 2 +pit laps | 0.740 | 0.667 | 0.793 | 0.962 | 11.752 | 6.685 | 17.078 | 16.039 | 365 | 18 |
| 3 +neutral laps | 0.614 | 0.559 | 0.674 | 0.943 | 5.197 | 4.191 | 6.289 | 8.740 | 365 | 18 |
| 4 +lap 1 | 0.745 | 0.683 | 0.803 | 0.967 | 3.161 | 2.571 | 3.880 | 7.820 | 365 | 18 |
| alone: pit laps | 0.756 | 0.712 | 0.790 | 0.960 | 12.913 | 8.348 | 17.976 | 16.405 | 365 | 18 |
| alone: neutral laps | 0.611 | 0.556 | 0.671 | 0.899 | 8.932 | 7.040 | 11.307 | 13.573 | 365 | 18 |
| alone: lap 1 | 0.726 | 0.662 | 0.772 | 0.956 | 14.141 | 9.208 | 19.309 | 20.190 | 365 | 18 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 317.000 | 35.511 | -1.797 | 18.797 | -4.192 | 17.072 | -0.633 | 3.414 | 0.818 | 901 | 7.820 | -1.837 | 0.491 | 556.000 | 0.492 | -0.184 | -0.019 |
| pooled | lodo | 317.000 | 37.297 | -1.793 | 18.933 | -4.144 | 17.232 | -0.588 | 3.489 | 0.818 | 901 | 8.031 | -1.849 | 0.502 | 556.000 | 0.513 | -0.191 | -0.016 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 894 | 10.540 | -0.621 | 0.611 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 26.86 | -4.16 | 20.04 | -14.34 | 20.22 | -2.67 | 4.38 | -0.79 | 44 | 8.47 | -4.54 | 0.48 | 25.00 | 0.34 | 0.13 | -0.08 |
| 2019_10_united_kingdom | lodo | 17.00 | 28.66 | -4.11 | 20.00 | -14.81 | 20.26 | -2.64 | 4.60 | -0.80 | 44 | 8.54 | -4.74 | 0.48 | 25.00 | 0.38 | 0.12 | -0.07 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.52 | -5.19 | 0.53 | - | - | - | - |
| 2021_01_bahrain | in | 18.00 | 41.56 | -3.59 | 9.93 | -4.27 | 10.87 | -1.41 | 2.38 | -0.22 | 56 | 4.37 | -3.01 | 0.34 | 35.00 | 0.40 | -0.14 | 0.10 |
| 2021_01_bahrain | lodo | 18.00 | 43.27 | -3.49 | 10.02 | -4.47 | 10.88 | -1.40 | 2.41 | -0.23 | 56 | 4.47 | -3.07 | 0.34 | 35.00 | 0.42 | -0.14 | 0.10 |
| 2021_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 55 | 4.83 | -2.81 | 0.37 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 49.26 | -8.54 | 14.53 | -14.00 | 6.72 | -0.58 | 3.49 | -0.24 | 56 | 6.03 | -4.35 | 0.34 | 37.00 | 0.47 | -0.04 | -0.18 |
| 2021_04_spain | lodo | 19.00 | 51.50 | -8.68 | 14.76 | -14.14 | 6.46 | -0.58 | 3.54 | -0.24 | 56 | 6.15 | -4.40 | 0.34 | 37.00 | 0.48 | -0.04 | -0.17 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 6.84 | -4.08 | 0.38 | - | - | - | - |
| 2021_09_austrian | in | 19.00 | 36.91 | -8.68 | 11.75 | -10.26 | 9.96 | -3.34 | 2.88 | -0.75 | 46 | 6.53 | -5.99 | 0.27 | 26.00 | 0.44 | -0.25 | 0.17 |
| 2021_09_austrian | lodo | 19.00 | 38.61 | -8.71 | 11.63 | -10.03 | 9.81 | -3.29 | 2.85 | -0.75 | 46 | 6.65 | -5.93 | 0.27 | 26.00 | 0.45 | -0.24 | 0.16 |
| 2021_09_austrian | loso | - | - | - | - | - | - | - | - | - | 45 | 8.68 | -5.85 | 0.34 | - | - | - | - |
| 2021_20_qatar | in | 18.00 | 39.09 | -4.85 | 13.95 | -12.02 | 12.05 | -2.09 | 3.03 | 0.26 | 46 | 5.46 | -3.13 | 0.33 | 27.00 | 0.71 | -0.34 | 0.10 |
| 2021_20_qatar | lodo | 18.00 | 40.78 | -4.54 | 14.19 | -12.04 | 12.26 | -2.08 | 3.10 | 0.26 | 46 | 5.74 | -3.14 | 0.35 | 27.00 | 0.74 | -0.36 | 0.08 |
| 2021_20_qatar | loso | - | - | - | - | - | - | - | - | - | 46 | 7.71 | -2.80 | 0.44 | - | - | - | - |
| 2021_22_united_arab_emirates | in | 15.00 | 70.38 | 58.75 | 49.11 | 46.07 | 28.19 | 7.95 | 7.45 | 5.73 | 41 | 16.81 | 10.91 | 1.48 | 19.00 | 0.45 | -0.10 | 0.02 |
| 2021_22_united_arab_emirates | lodo | 15.00 | 71.70 | 59.21 | 49.06 | 46.15 | 28.36 | 7.94 | 7.44 | 5.73 | 41 | 17.03 | 10.94 | 1.50 | 19.00 | 0.48 | -0.10 | 0.01 |
| 2021_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 40 | 18.78 | 10.80 | 1.59 | - | - | - | - |
| 2022_01_bahrain | in | 19.00 | 19.03 | -6.79 | 43.54 | -2.54 | 44.21 | -1.16 | 2.73 | 1.50 | 75 | 14.92 | -0.43 | 1.33 | 54.00 | 0.50 | -0.24 | 0.13 |
| 2022_01_bahrain | lodo | 19.00 | 19.92 | -6.73 | 43.87 | -2.21 | 44.45 | -1.15 | 2.73 | 1.51 | 75 | 15.12 | -0.35 | 1.35 | 54.00 | 0.52 | -0.24 | 0.13 |
| 2022_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 75 | 16.36 | -0.10 | 1.42 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 31.24 | -9.33 | 21.44 | -19.38 | 18.04 | -5.07 | 8.37 | -0.26 | 29 | 10.49 | -8.51 | 0.43 | 15.00 | 0.31 | 0.02 | 0.10 |
| 2022_02_saudi_arabia | lodo | 14.00 | 33.92 | -9.65 | 22.10 | -19.74 | 19.04 | -5.26 | 8.49 | -0.26 | 29 | 10.92 | -8.58 | 0.46 | 15.00 | 0.35 | 0.01 | 0.13 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 12.25 | -8.62 | 0.51 | - | - | - | - |
| 2022_06_spain | in | 18.00 | 48.85 | -12.14 | 12.83 | -9.89 | 13.18 | 0.05 | 1.32 | - | 69 | 3.75 | -2.32 | 0.27 | 49.00 | 0.50 | -0.14 | 0.11 |
| 2022_06_spain | lodo | 18.00 | 53.29 | -14.26 | 13.48 | -10.45 | 14.09 | 0.14 | 1.36 | - | 69 | 4.23 | -2.54 | 0.29 | 49.00 | 0.52 | -0.14 | 0.14 |
| 2022_06_spain | loso | - | - | - | - | - | - | - | - | - | 67 | 21.03 | 13.14 | 0.95 | - | - | - | - |
| 2022_08_azerbaijan | in | 16.00 | 38.72 | -6.74 | 9.95 | -1.41 | 10.13 | 1.18 | 5.57 | 0.98 | 40 | 4.31 | -1.32 | 0.31 | 23.00 | 0.50 | -0.23 | 0.05 |
| 2022_08_azerbaijan | lodo | 16.00 | 39.65 | -7.16 | 9.86 | -1.30 | 10.02 | 1.16 | 5.67 | 0.99 | 40 | 4.40 | -1.26 | 0.31 | 23.00 | 0.49 | -0.25 | 0.06 |
| 2022_08_azerbaijan | loso | - | - | - | - | - | - | - | - | - | 40 | 5.77 | -1.17 | 0.38 | - | - | - | - |
| 2022_09_canada | in | 17.00 | 12.69 | 8.13 | 32.09 | 6.83 | 31.33 | -0.38 | 2.94 | 2.00 | 47 | 14.48 | 3.31 | 0.62 | 30.00 | 0.43 | -0.03 | 0.08 |
| 2022_09_canada | lodo | 17.00 | 14.16 | 8.20 | 32.73 | 6.66 | 32.18 | -0.39 | 3.01 | 2.00 | 47 | 14.69 | 3.26 | 0.63 | 30.00 | 0.46 | -0.04 | 0.09 |
| 2022_09_canada | loso | - | - | - | - | - | - | - | - | - | 47 | 15.00 | 3.30 | 0.65 | - | - | - | - |
| 2022_12_france | in | 16.00 | 23.71 | -9.60 | 14.11 | -8.85 | 14.45 | -1.47 | 4.99 | -0.01 | 36 | 7.13 | -2.47 | 0.32 | 20.00 | 0.32 | -0.11 | -0.05 |
| 2022_12_france | lodo | 16.00 | 25.05 | -8.72 | 14.46 | -8.50 | 14.74 | -1.09 | 5.24 | -0.02 | 36 | 7.18 | -2.52 | 0.32 | 20.00 | 0.36 | -0.12 | -0.05 |
| 2022_12_france | loso | - | - | - | - | - | - | - | - | - | 36 | 7.77 | -2.86 | 0.36 | - | - | - | - |
| 2022_20_mexico | in | 19.00 | 47.21 | 4.22 | 13.24 | -4.02 | 13.76 | 1.49 | 2.35 | -2.42 | 41 | 10.35 | -3.02 | 0.38 | 21.00 | 0.46 | -0.26 | -1.05 |
| 2022_20_mexico | lodo | 19.00 | 48.04 | 4.34 | 12.13 | -3.89 | 12.73 | 1.51 | 2.41 | -2.42 | 41 | 10.82 | -2.97 | 0.40 | 21.00 | 0.48 | -0.26 | -1.06 |
| 2022_20_mexico | loso | - | - | - | - | - | - | - | - | - | 41 | 14.00 | -1.55 | 0.49 | - | - | - | - |
| 2022_21_brazil | in | 17.00 | 18.26 | -9.30 | 30.37 | -3.69 | 30.74 | -1.14 | 2.39 | -0.30 | 59 | 10.54 | -2.67 | 0.61 | 39.00 | 0.39 | -0.16 | 0.07 |
| 2022_21_brazil | lodo | 17.00 | 20.00 | -9.57 | 30.59 | -3.80 | 30.97 | -1.18 | 2.43 | -0.30 | 59 | 10.68 | -2.67 | 0.61 | 39.00 | 0.40 | -0.18 | 0.07 |
| 2022_21_brazil | loso | - | - | - | - | - | - | - | - | - | 59 | 11.09 | -2.74 | 0.63 | - | - | - | - |
| 2022_22_united_arab_emirates | in | 19.00 | 39.48 | -9.89 | 14.54 | -12.41 | 14.48 | -3.97 | 4.45 | - | 49 | 4.34 | -3.25 | 0.24 | 30.00 | 0.43 | -0.15 | 0.12 |
| 2022_22_united_arab_emirates | lodo | 19.00 | 41.62 | -9.76 | 14.77 | -12.22 | 14.82 | -3.74 | 4.51 | - | 49 | 4.63 | -3.32 | 0.25 | 30.00 | 0.45 | -0.16 | 0.14 |
| 2022_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 49 | 6.11 | -3.50 | 0.31 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 31.93 | 4.36 | 13.51 | 0.63 | 13.32 | 0.43 | 2.80 | 2.43 | 62 | 4.95 | -0.96 | 0.41 | 42.00 | 0.78 | -0.23 | 0.11 |
| 2023_01_bahrain | lodo | 17.00 | 33.91 | 4.67 | 13.45 | 1.01 | 13.16 | 0.52 | 2.91 | 2.41 | 62 | 5.05 | -0.92 | 0.41 | 42.00 | 0.83 | -0.26 | 0.10 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.03 | -0.84 | 0.56 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 28.42 | -5.25 | 9.67 | -4.62 | 8.89 | -0.53 | 1.05 | - | 63 | 4.14 | -1.25 | 0.22 | 42.00 | 0.63 | -0.48 | 0.03 |
| 2023_07_spain | lodo | 20.00 | 30.00 | -4.64 | 10.34 | -3.97 | 9.62 | -0.51 | 1.14 | - | 63 | 4.47 | -1.04 | 0.23 | 42.00 | 0.66 | -0.49 | 0.01 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 6.15 | -1.60 | 0.32 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 36.53 | -0.14 | 10.34 | -1.03 | 10.36 | 0.35 | 1.43 | - | 42 | 5.53 | -1.60 | 0.23 | 22.00 | 0.45 | -0.35 | -0.44 |
| 2024_07_emilia_romagna | lodo | 19.00 | 38.25 | 0.09 | 10.06 | -0.79 | 10.17 | 0.54 | 1.56 | - | 42 | 5.58 | -1.59 | 0.23 | 22.00 | 0.42 | -0.33 | -0.44 |
| 2024_07_emilia_romagna | loso | - | - | - | - | - | - | - | - | - | 42 | 7.48 | -1.56 | 0.29 | - | - | - | - |

Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:

| variant | compound | 1-5 | 11-15 | 16-20 | 21-25 | 26-30 | 31-35 | 36-40 | 41+ | 6-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| in | HARD | -0.04 | 0.05 | 0.04 | 0.02 | 0.02 | 0.02 | -0.07 | -1.42 | 0.04 |
| in | MEDIUM | 0.02 | 0.03 | 0.01 | -0.02 | 0.06 | -0.12 | -1.21 | -3.27 | 0.03 |
| in | SOFT | 0.01 | 0.12 | 0.04 | -0.01 | -0.67 | -2.57 | -4.08 | - | 0.03 |
| lodo | HARD | -0.04 | 0.06 | 0.05 | 0.02 | 0.03 | 0.03 | -0.05 | -1.45 | 0.04 |
| lodo | MEDIUM | 0.02 | 0.02 | 0.00 | -0.02 | 0.06 | -0.11 | -1.21 | -3.17 | 0.03 |
| lodo | SOFT | 0.00 | 0.12 | 0.04 | -0.03 | -0.65 | -2.57 | -4.14 | - | 0.02 |

Neutralised laps:

| variant | kind | n | mean_resid_s | mae_s |
|---|---|---|---|---|
| in | sc | 682 | -1.26 | 7.57 |
| in | vsc | 350 | 0.04 | 4.39 |
| lodo | sc | 682 | -1.26 | 7.59 |
| lodo | vsc | 350 | 0.04 | 4.40 |
| loso | sc | 682 | -1.23 | 7.56 |
| loso | vsc | 350 | 0.03 | 4.40 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.60 |
| 2021_01_bahrain | 2 | 2 | 2:16 | 1.00 | 0.81 | 1.00 | 1.00 | 60 | 0.57 | 0.57 | 2.17 |
| 2021_04_spain | 2 | 2 | 1:2 2:17 | 0.89 | 0.74 | 0.37 | 0.37 | 72 | 0.26 | 0.26 | 1.47 |
| 2021_09_austrian | 1 | 1 | 1:19 | 1.00 | 0.58 | 0.21 | 0.21 | 78 | 0.12 | 0.12 | 0.65 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 0.00 | 0.00 | 60 | 0.20 | 0.20 | 0.67 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 0.00 | 0.00 | 60 | 0.02 | 0.02 | 0.62 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 3.06 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 0.93 | 0.93 | 54 | 0.37 | 0.37 | 0.68 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.44 | 0.44 | 72 | 0.14 | 0.14 | 4.07 |
| 2022_08_azerbaijan | 2 | 2 | 1:4 2:11 | 0.73 | 0.67 | 1.00 | 1.00 | 54 | 0.44 | 0.44 | 0.84 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 1.00 | 1.00 | 78 | 0.44 | 0.44 | 1.20 |
| 2022_12_france | 1 | 1 | 1:15 | 1.00 | 0.80 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.67 |
| 2022_20_mexico | 1 | 2 | 2:18 | 0.00 | 0.17 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 2.48 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.06 | 0.06 | 78 | 0.00 | 0.00 | 2.57 |
| 2022_22_united_arab_emirates | 2 | 1 | 1:16 2:1 | 0.06 | 0.35 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.63 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 1.91 |
| 2023_07_spain | 2 | 1 | 1:14 2:6 | 0.30 | 0.30 | 0.10 | 0.10 | 72 | 0.14 | 0.14 | 0.98 |
| 2024_07_emilia_romagna | 1 | 2 | 1:3 2:16 | 0.16 | 0.26 | 1.00 | 1.00 | 66 | 0.62 | 0.62 | 1.70 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.071 | -0.015 | 1.166 | 3.579 | -4.940 | 0.079 | -1.293 | 3.381 | -0.000 | 0.223 | 0.223 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.079 | -0.016 | 0.933 | 3.707 | -5.318 | 0.119 | -1.357 | 3.165 | - | - | - |
| pit_loss_prior | in | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | 0.003 | -0.000 | 0.157 | 0.545 | -1.250 | 0.000 | -0.679 | 0.000 | -0.050 | 0.044 | 0.044 |
| pit_loss_prior | lodo | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | -0.003 | -0.001 | 0.159 | 0.489 | -1.310 | 0.000 | -0.711 | 0.000 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 365 | 365 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | -0.020 | -0.020 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 359 | 365 | 0.004 | -0.001 | -0.116 | -0.459 | 0.219 | 0.001 | 0.156 | -0.035 | - | - | - |
| events_rate | in | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 365 | 365 | -0.008 | -0.009 | 2.557 | 12.786 | 1.653 | 0.000 | 1.102 | 4.703 | 0.000 | 0.000 | 0.000 |
| events_rate | lodo | events | SC/VSC drawn at the rate model's rates (no recorded periods or lap times) |  | 365 | 365 | -0.025 | -0.009 | 2.407 | 12.715 | 1.667 | 0.000 | 1.102 | 4.637 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 251 | 365 | -0.049 | -0.024 | 5.276 | 11.237 | -1.104 | 0.102 | -0.227 | 7.961 | 0.077 | 0.093 | 0.093 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 251 | 365 | -0.065 | -0.027 | 4.891 | 11.175 | -1.326 | 0.133 | -0.284 | 7.764 | - | - | - |

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
| predictive | field:fuel_anchor | config field fuel_anchor = {"prior": "anchored", "practice": "anchored", "full_fit": false} | unclassified | - | derived |

## B. Predictive versus judge

Plan panel per snapshot: the real remaining strategy, stay out, one stop at 4 laps spread from the soonest legal stop on each compound both model, two stops at 3 first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) over the plans legal under both configs' own rules. Snapshots every 5 laps for 6 sampled drivers per race. 20 draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.

| race_id | snapshots | rank_rho | regret_s | same_favourite | fav_b_illegal_w | pit_now_agree | rank_rho_green | regret_green_s | plans | b1_mae_s | b1_green_mae_s | b1_green_bias_s | b1_bias_s | b1_mae_per_lap_s | cov_68 | cov_90 | gap_cov_68 | gap_cov_90 | factor_cov_68 | factor_cov_90 | b4_mae_s | b4_bias_s | b4_mae_offset_s | b4_bias_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 54 | 0.708 | 5.808 | 0.500 | 0.000 | 0.849 | 0.723 | 4.879 | 1931 | 43.359 | 14.764 | 3.352 | 42.631 | 1.344 | 0.972 | 0.991 | 0.559 | 0.626 | 0.228 | 0.421 | 53.678 | 39.653 | 24.338 | 2.794 |
| 2021_01_bahrain | 60 | 0.942 | 2.397 | 0.633 | 0.000 | 0.897 | 0.947 | 3.099 | 2127 | 51.934 | 10.723 | 3.045 | 51.319 | 1.838 | 0.977 | 0.995 | 0.771 | 0.941 | 0.188 | 0.250 | 31.842 | -15.744 | 15.555 | -7.378 |
| 2021_04_spain | 72 | 0.799 | 5.270 | 0.389 | 0.000 | 0.831 | 0.827 | 5.366 | 2544 | 55.709 | 14.472 | 2.566 | 53.844 | 1.639 | 0.936 | 0.969 | 0.717 | 0.864 | 0.413 | 0.583 | 63.759 | -44.267 | 21.746 | -16.735 |
| 2021_09_austrian | 78 | 0.812 | 3.753 | 0.692 | 0.000 | 0.842 | 0.838 | 4.064 | 2707 | 46.073 | 22.670 | 13.998 | 40.033 | 1.110 | 0.919 | 0.945 | 0.791 | 0.948 | 0.328 | 0.540 | 49.326 | -16.303 | 26.352 | 3.103 |
| 2021_20_qatar | 60 | 0.854 | 4.183 | 0.700 | 0.000 | 0.879 | 0.869 | 3.802 | 2140 | 38.812 | 19.164 | 18.219 | 38.622 | 1.022 | 0.985 | 1.000 | 0.744 | 0.899 | 0.390 | 0.555 | 53.531 | 35.895 | 49.154 | 47.096 |
| 2021_22_united_arab_emirates | 60 | 0.886 | 5.664 | 0.567 | 0.000 | 0.695 | 0.918 | 5.614 | 2150 | 37.540 | 13.053 | 2.821 | 27.393 | 1.315 | 0.923 | 0.986 | 0.873 | 0.923 | 0.276 | 0.583 | 59.573 | 54.624 | 51.680 | 38.408 |
| 2022_01_bahrain | 60 | 0.932 | 1.825 | 0.617 | 0.000 | 0.796 | 0.948 | 1.409 | 2169 | 54.218 | 13.555 | 10.485 | 54.201 | 2.939 | 0.932 | 0.939 | 0.807 | 0.935 | 0.433 | 0.631 | 87.549 | 84.669 | 105.366 | 89.532 |
| 2022_02_saudi_arabia | 54 | 0.878 | 0.755 | 0.907 | 0.000 | 0.963 | 0.911 | 0.457 | 1779 | 71.519 | 13.632 | 1.383 | 62.465 | 3.203 | 0.707 | 0.901 | 0.827 | 0.921 | 0.553 | 0.696 | 39.281 | -23.140 | 30.017 | -21.248 |
| 2022_06_spain | 72 | 0.912 | 2.085 | 0.778 | 0.000 | 0.957 | 0.921 | 2.377 | 2654 | 80.983 | 37.597 | 33.698 | 78.714 | 2.144 | 0.956 | 0.993 | 0.880 | 0.966 | 0.466 | 0.599 | 68.429 | -43.008 | 24.978 | -3.897 |
| 2022_08_azerbaijan | 54 | 0.762 | 6.665 | 0.444 | 0.000 | 0.788 | 0.781 | 7.683 | 1858 | 51.752 | 14.282 | -7.082 | 40.039 | 2.170 | 0.965 | 1.000 | 0.739 | 0.861 | 0.296 | 0.488 | 29.755 | 8.706 | 20.126 | 6.927 |
| 2022_09_canada | 78 | 0.898 | 1.410 | 0.718 | 0.000 | 0.809 | 0.933 | 0.948 | 2758 | 46.136 | 11.292 | -1.580 | 41.973 | 1.101 | 0.953 | 0.998 | 0.968 | 0.990 | 0.460 | 0.566 | 31.145 | 12.906 | 42.749 | -2.705 |
| 2022_12_france | 54 | 0.909 | 4.324 | 0.296 | 0.000 | 0.585 | 0.910 | 4.923 | 1983 | 58.966 | 8.110 | 3.754 | 51.817 | 2.070 | 0.847 | 0.920 | 0.584 | 0.730 | 0.333 | 0.431 | 33.918 | 11.001 | 29.902 | 22.496 |
| 2022_20_mexico | 78 | 0.894 | 5.760 | 0.269 | 0.000 | 0.640 | 0.886 | 5.876 | 2723 | 47.433 | 26.429 | -7.741 | 36.441 | 1.428 | 0.935 | 0.953 | 0.465 | 0.573 | 0.214 | 0.278 | 56.220 | 28.774 | 35.276 | 21.640 |
| 2022_21_brazil | 78 | 0.862 | 7.415 | 0.692 | 0.000 | 0.847 | 0.877 | 8.863 | 2766 | 50.010 | 18.673 | 10.584 | 46.494 | 1.416 | 0.931 | 0.969 | 0.719 | 0.879 | 0.231 | 0.253 | 43.928 | -40.295 | 33.038 | -21.002 |
| 2022_22_united_arab_emirates | 60 | 0.912 | 3.949 | 0.367 | 0.000 | 0.857 | 0.915 | 3.996 | 2243 | 47.787 | 14.263 | 8.660 | 41.982 | 1.462 | 0.945 | 0.989 | 0.608 | 0.780 | 0.207 | 0.293 | 32.053 | 2.171 | 19.565 | -4.744 |
| 2023_01_bahrain | 60 | 0.879 | 2.311 | 0.567 | 0.000 | 0.929 | 0.892 | 3.037 | 2216 | 46.357 | 22.323 | 19.662 | 45.260 | 1.278 | 0.946 | 1.000 | 0.865 | 0.947 | 0.369 | 0.457 | 32.775 | 24.920 | 29.750 | 24.069 |
| 2023_07_spain | 72 | 0.918 | 3.026 | 0.417 | 0.000 | 0.706 | 0.931 | 2.998 | 2572 | 51.456 | 13.849 | 12.512 | 46.789 | 1.500 | 0.986 | 1.000 | 0.770 | 0.926 | 0.242 | 0.452 | 29.084 | 18.050 | 17.393 | 8.053 |
| 2024_07_emilia_romagna | 66 | 0.834 | 9.016 | 0.348 | 0.000 | 0.667 | 0.843 | 10.678 | 2351 | 28.782 | 8.864 | 2.314 | 22.532 | 0.978 | 0.985 | 0.995 | 0.609 | 0.772 | 0.143 | 0.247 | 34.043 | -16.487 | 18.685 | -5.835 |
| pooled | 1170 | 0.867 | 4.223 | 0.551 | 0.000 | 0.806 | 0.883 | 4.483 | 41671 | 50.440 | 16.974 | 7.579 | 45.703 | 1.625 | 0.936 | 0.975 | 0.743 | 0.865 | 0.320 | 0.460 | 46.355 | 6.703 | 33.051 | 10.737 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1837.00 | 24.74 | 24.61 |
| 10.00 | 8653.00 | 29.12 | 22.95 |
| 20.00 | 8588.00 | 16.12 | 6.56 |
| 30.00 | 8312.00 | 41.24 | 35.12 |
| 40.00 | 7630.00 | 68.27 | 68.06 |
| 50.00 | 4468.00 | 118.80 | 118.76 |
| 60.00 | 2183.00 | 124.39 | 120.23 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 12621 | 98.06 | 95.76 |
| late | 11940 | 27.13 | 20.12 |
| mid | 17110 | 31.58 | 26.63 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12150 | 49.45 | 44.79 |
| real | 1116 | 51.49 | 47.41 |
| stay | 319 | 37.34 | 32.43 |
| two | 28086 | 50.97 | 46.18 |

B5 future blindness: {"status": "pass", "new_leak_differences": 0, "differences_with_known_leaks": 0, "known_leaks": [], "fixed_leaks": ["compound_list", "frame_filter", "gap_behind", "lap_window", "pit_loss", "race_length", "stint_limit"], "detail": ""}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | history prior (tyre_life_prior) | oldest age run | 50 | 18 | 7.542 | -1.450 | 0.660 | 0.880 |
| tyre_life | history prior (tyre_life_prior) | median stint-end age | 50 | 18 | 9.216 | 8.540 | 0.580 | 0.860 |
| tyre_life | history prior (tyre_life_prior) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.513 | 1.146 | 0.826 | 0.935 |
| tyre_life | judge cliff edge (history_only) | oldest age run | 50 | 18 | 7.509 | -1.483 | - | - |
| tyre_life | judge cliff edge (history_only) | median stint-end age | 50 | 18 | 9.183 | 8.507 | - | - |
| tyre_life | judge cliff edge (history_only) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.477 | 1.110 | - | - |
| tyre_life | belief cliff edge @k0 | oldest age run | 50 | 18 | 7.317 | -1.291 | 0.630 | 0.870 |
| tyre_life | belief cliff edge @k0 | median stint-end age | 50 | 18 | 9.375 | 8.699 | 0.565 | 0.870 |
| tyre_life | belief cliff edge @k0 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.458 | 1.318 | 0.810 | 0.929 |
| tyre_life | as-of data (race to k0, practice) | oldest age run | 50 | 18 | 12.460 | -10.500 | - | - |
| tyre_life | as-of data (race to k0, practice) | median stint-end age | 50 | 18 | 7.630 | -0.510 | - | - |
| tyre_life | as-of data (race to k0, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 9.574 | -7.757 | - | - |
| tyre_life | belief cliff edge @k10 | oldest age run | 50 | 18 | 7.317 | -1.291 | 0.630 | 0.870 |
| tyre_life | belief cliff edge @k10 | median stint-end age | 50 | 18 | 9.375 | 8.699 | 0.565 | 0.870 |
| tyre_life | belief cliff edge @k10 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.458 | 1.318 | 0.810 | 0.929 |
| tyre_life | as-of data (race to k10, practice) | oldest age run | 50 | 18 | 12.400 | -10.440 | - | - |
| tyre_life | as-of data (race to k10, practice) | median stint-end age | 50 | 18 | 7.570 | -0.450 | - | - |
| tyre_life | as-of data (race to k10, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 9.509 | -7.691 | - | - |
| tyre_life | belief cliff edge @k20 | oldest age run | 50 | 18 | 7.177 | -1.151 | 0.674 | 0.884 |
| tyre_life | belief cliff edge @k20 | median stint-end age | 50 | 18 | 9.475 | 8.839 | 0.535 | 0.860 |
| tyre_life | belief cliff edge @k20 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.393 | 1.470 | 0.846 | 0.923 |
| tyre_life | as-of data (race to k20, practice) | oldest age run | 50 | 18 | 11.380 | -9.420 | - | - |
| tyre_life | as-of data (race to k20, practice) | median stint-end age | 50 | 18 | 6.990 | 0.570 | - | - |
| tyre_life | as-of data (race to k20, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 8.487 | -6.583 | - | - |
| tyre_life | belief cliff edge @k(mid) | oldest age run | 50 | 18 | 6.253 | -0.227 | 0.667 | 0.861 |
| tyre_life | belief cliff edge @k(mid) | median stint-end age | 50 | 18 | 10.137 | 9.763 | 0.444 | 0.833 |
| tyre_life | belief cliff edge @k(mid) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.057 | 2.475 | 0.844 | 0.906 |
| tyre_life | as-of data (race to k(mid), practice) | oldest age run | 50 | 18 | 7.980 | -6.020 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | median stint-end age | 50 | 18 | 7.090 | 3.970 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.726 | -2.887 | - | - |
| wear_rate | belief @k0 | within-stint slope less fuel | 49 | 18 | 0.049 | 0.015 | 0.449 | 0.673 |
| wear_rate | belief @k10 | within-stint slope less fuel | 49 | 18 | 0.041 | 0.007 | 0.306 | 0.449 |
| wear_rate | belief @k20 | within-stint slope less fuel | 49 | 18 | 0.034 | 0.006 | 0.245 | 0.347 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 49 | 18 | 0.026 | -0.000 | 0.286 | 0.388 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 49 | 18 | 0.011 | -0.006 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 556 | 18 | 0.819 | 0.191 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 521 | 18 | 0.976 | 0.082 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 351 | 18 | 0.736 | 0.135 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 233 | 18 | 0.624 | 0.016 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.820 | -0.321 | - | - |
| compound_gap | belief @k0 | qualifying | 7 | 7 | 0.724 | 0.629 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.724 | -0.202 | - | - |
| compound_gap | belief @k10 | qualifying | 7 | 7 | 1.037 | 0.723 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.450 | -0.005 | - | - |
| compound_gap | belief @k20 | qualifying | 7 | 7 | 0.773 | 0.500 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.384 | -0.100 | - | - |
| compound_gap | belief @k(mid) | qualifying | 7 | 7 | 0.859 | 0.466 | - | - |
| fuel | prior (config) | race (judge full fit) | 18 | 18 | 0.020 | 0.017 | - | - |
| fuel | belief @k0 | race (judge full fit) | 18 | 18 | 0.008 | 0.002 | 0.611 | 0.944 |
| fuel | belief @k10 | race (judge full fit) | 18 | 18 | 0.009 | -0.001 | 0.333 | 0.444 |
| fuel | belief @k20 | race (judge full fit) | 18 | 18 | 0.008 | -0.003 | 0.167 | 0.389 |
| fuel | belief @k(mid) | race (judge full fit) | 18 | 18 | 0.006 | -0.001 | 0.389 | 0.500 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 10 | 10 | 2.545 | -1.213 | 0.500 | 0.600 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 10 | 10 | 3.100 | -2.248 | 0.500 | 0.700 |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 2.450 | -1.875 | - | - |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 3.454 | -2.424 | - | - |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.846 | -0.908 | 0.778 | 0.833 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.834 | -1.457 | 0.611 | 0.778 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.291 | 7.060 | 0.200 | 0.300 |
| pit_loss | belief charged @k0 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.814 | -0.800 | - | - |
| pit_loss | belief charged @k0 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.779 | -1.349 | - | - |
| pit_loss | belief charged @k0 | whole race, median SC stop | 10 | 10 | 13.303 | 7.120 | - | - |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.738 | -0.198 | 0.778 | 0.833 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.520 | -0.746 | 0.722 | 0.778 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.362 | 7.313 | 0.300 | 0.400 |
| pit_loss | belief charged @k10 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.718 | -0.096 | - | - |
| pit_loss | belief charged @k10 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.467 | -0.645 | - | - |
| pit_loss | belief charged @k10 | whole race, median SC stop | 10 | 10 | 13.375 | 7.368 | - | - |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.710 | -0.268 | 0.944 | 1.000 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.952 | -0.816 | 0.611 | 0.778 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.612 | 7.368 | 0.100 | 0.500 |
| pit_loss | belief charged @k20 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.686 | -0.196 | - | - |
| pit_loss | belief charged @k20 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.911 | -0.745 | - | - |
| pit_loss | belief charged @k20 | whole race, median SC stop | 10 | 10 | 12.611 | 7.404 | - | - |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.548 | -0.428 | 0.944 | 1.000 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.047 | -0.977 | 0.611 | 0.778 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.403 | 7.151 | 0.200 | 0.500 |
| pit_loss | belief charged @k(mid) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.520 | -0.363 | - | - |
| pit_loss | belief charged @k(mid) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.996 | -0.911 | - | - |
| pit_loss | belief charged @k(mid) | whole race, median SC stop | 10 | 10 | 12.402 | 7.185 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 18 | 18 | 2.082 | 1.362 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 18 | 18 | 0.183 | 0.090 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.613 | 0.161 | - | - |
| traffic_rejoin | judge rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 577 | 18 | 1.132 | 0.109 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.803 | 0.015 | - | - |
| traffic_rejoin | belief rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 577 | 18 | 2.088 | -0.785 | - | - |
| neutralisations | judge expected (recorded) | occurred | 72 | 18 | 0.000 | 0.000 | - | - |
| neutralisations | judge config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | judge config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralisations | belief expected (rate) | occurred | 72 | 18 | 1.355 | 0.405 | - | - |
| neutralisations | belief config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | belief config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralised_laps | judge replay (recorded lap times) | mean real lap time on the cars' neutralised laps | 21 | 14 | 1.206 | 0.632 | - | - |
| neutralised_laps | belief replay (factors) | mean real lap time on the cars' neutralised laps | 21 | 14 | 10.121 | 6.938 | - | - |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 6 | 6 | 0.869 | -0.641 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 6 | 6 | 2.654 | 0.059 | 0.167 | 0.833 |
| pit_loss | prior from earlier races (constant) | whole race, robust (pit_loss.whole_race: median green stop) | 2 | 2 | 1.281 | -0.187 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (constant) | whole race, replay mean (calibrate._pit_loss) | 2 | 2 | 2.045 | -2.045 | 1.000 | 1.000 |

Component status:

| component | status | races |
|---|---|---|
| compound_gap | ok | 18 |
| fresh_tyre_gain | ok | 18 |
| fuel | ok | 18 |
| lap1_and_warmup | ok | 18 |
| neutralisations | ok | 18 |
| neutralised_laps | no rows | 4 |
| neutralised_laps | ok | 14 |
| pit_loss | ok | 18 |
| traffic_rejoin | ok | 18 |
| tyre_life | ok | 18 |
| wear_rate | ok | 18 |


Failed and left out: 2022_11_austria (ValueError: OCO's set at lap 45 has no recorded compound)
