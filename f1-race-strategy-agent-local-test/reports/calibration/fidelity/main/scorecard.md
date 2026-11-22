# Simulator fidelity scorecard: main

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version f7415672be70-134c5546de2d. World config ba21b60fc268 (race default); belief config 9e7580bde6f9. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "seed": 0, "max_stops": 2}. Run time 178 s.

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
| A_close_pair_acc | 0.704 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.947 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 15.577 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.719 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 3.195 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.830 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.635 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.877 | same, Spearman, mean over k |
| A_abs_mae_s | 32.240 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -4.134 | same, mean error |
| A_stop_gain_bias_s | -0.250 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 94.389 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 94.281 | same, mean |
| B1_green_mae_s | 21.262 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.722 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.482 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.514 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 34.988 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.650 | 0.958 | 12.878 | 21.159 | 0.700 | 0.961 | 13.116 | 21.403 | 0.853 | 1.431 | 103.831 | 103.831 | 0.470 | 0.856 | 18.574 | 32.315 |
| 2021_01_bahrain | 0.818 | 0.990 | 4.411 | 17.005 | 1.000 | 0.994 | 4.693 | 17.091 | 0.536 | 3.182 | 109.641 | 109.641 | 0.582 | 0.937 | 10.983 | 13.379 |
| 2021_04_spain | 0.714 | 0.981 | 4.775 | 17.251 | 0.643 | 0.977 | 4.894 | 17.715 | 0.824 | 0.680 | 97.282 | 97.282 | 0.586 | 0.936 | 11.243 | 22.318 |
| 2021_09_austrian | 0.750 | 0.972 | 5.059 | 42.868 | 0.750 | 0.970 | 5.170 | 42.497 | 0.698 | 1.191 | 83.082 | 83.082 | 0.512 | 0.895 | 18.989 | 22.754 |
| 2021_20_qatar | 0.308 | 0.961 | 8.196 | 24.906 | 0.385 | 0.963 | 8.114 | 24.947 | 0.590 | 5.998 | 103.997 | 103.997 | 0.738 | 0.938 | 22.381 | 64.022 |
| 2021_22_united_arab_emirates | 0.786 | 0.964 | 27.687 | 31.980 | 0.750 | 0.961 | 26.839 | 31.439 | 0.791 | 1.030 | 94.153 | 94.153 | 0.657 | 0.859 | 31.255 | 50.820 |
| 2022_01_bahrain | 0.804 | 0.754 | 43.839 | 84.119 | 0.804 | 0.754 | 44.614 | 83.690 | 0.720 | 3.682 | 114.593 | 114.593 | 0.700 | 0.718 | 45.076 | 104.029 |
| 2022_02_saudi_arabia | 0.615 | 0.930 | 13.939 | 77.821 | 0.615 | 0.930 | 15.005 | 78.363 | 0.872 | 0.112 | 109.813 | 109.813 | 0.538 | 0.822 | 16.939 | 24.188 |
| 2022_06_spain | 0.375 | 0.955 | 13.960 | 13.761 | 0.500 | 0.957 | 14.174 | 14.433 | 0.713 | 3.557 | 98.011 | 98.011 | 0.700 | 0.969 | 16.598 | 23.386 |
| 2022_08_azerbaijan | 0.273 | 0.879 | 16.759 | 32.912 | 0.364 | 0.871 | 17.014 | 32.850 | 0.608 | 5.710 | 101.777 | 101.777 | 0.527 | 0.818 | 23.041 | 29.912 |
| 2022_09_canada | 0.700 | 0.946 | 44.082 | 44.481 | 0.667 | 0.951 | 44.270 | 44.672 | 0.816 | 1.924 | 95.556 | 95.556 | 0.580 | 0.813 | 38.748 | 38.237 |
| 2022_12_france | 0.864 | 0.988 | 13.396 | 22.354 | 0.864 | 0.988 | 13.537 | 23.188 | 0.849 | 2.932 | 88.688 | 88.688 | 0.727 | 0.867 | 11.715 | 37.749 |
| 2022_20_mexico | 0.286 | 0.870 | 11.965 | 20.321 | 0.357 | 0.854 | 12.635 | 20.152 | 0.605 | 6.489 | 79.501 | 79.501 | 0.457 | 0.825 | 25.826 | 27.333 |
| 2022_21_brazil | 0.833 | 0.978 | 32.816 | 71.031 | 0.833 | 0.975 | 31.906 | 70.030 | 0.897 | 0.759 | 94.727 | 93.105 | 0.721 | 0.897 | 29.195 | 31.877 |
| 2022_22_united_arab_emirates | 0.643 | 0.988 | 4.163 | 14.229 | 0.643 | 0.988 | 4.075 | 14.065 | 0.501 | 9.002 | 80.314 | 80.314 | 0.586 | 0.928 | 12.619 | 25.441 |
| 2023_01_bahrain | 0.583 | 0.978 | 9.037 | 23.126 | 0.583 | 0.978 | 8.675 | 22.873 | 0.464 | 3.106 | 97.841 | 97.841 | 0.667 | 0.959 | 17.851 | 43.616 |
| 2023_07_spain | 0.640 | 0.974 | 3.729 | 8.214 | 0.760 | 0.982 | 5.543 | 8.985 | 0.717 | 3.079 | 90.449 | 90.449 | 0.744 | 0.948 | 16.569 | 18.156 |
| 2024_07_emilia_romagna | 0.700 | 0.988 | 6.147 | 11.780 | 0.650 | 0.984 | 6.108 | 11.924 | 0.838 | 4.446 | 68.718 | 68.718 | 0.590 | 0.794 | 12.965 | 21.007 |
| pooled | 0.688 | 0.947 | 15.380 | 32.184 | 0.704 | 0.947 | 15.577 | 32.240 | 0.719 | 3.195 | 94.389 | 94.281 | 0.635 | 0.877 | 21.143 | 34.988 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.947 | 0.966 | 0.783 | 15.380 | 32.184 | -4.151 | 17.611 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.545 | 365.000 | 0.727 | 2.791 | 0.388 | 23.294 | 44.295 | -5.057 | 17.611 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.534 | 365.000 | 0.775 | 2.300 | 0.491 | 27.941 | 40.754 | -7.479 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.704 | 365.000 | 0.947 | 0.954 | 0.760 | 15.577 | 32.240 | -4.134 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.499 | 365.000 | 0.651 | 3.198 | 0.326 | 25.291 | 46.586 | -5.165 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.532 | 365.000 | 0.754 | 2.414 | 0.488 | 28.717 | 41.994 | -7.587 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.660 | 365.000 | 0.933 | 1.162 | 0.729 | 17.357 | 32.731 | -4.472 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.510 | 365.000 | 0.695 | 2.869 | 0.414 | 24.986 | 78.936 | 31.775 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.559 | 365.000 | 0.762 | 2.388 | 0.492 | 28.714 | 76.879 | 29.354 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.668 | 365.000 | 0.933 | 1.161 | 0.728 | 17.284 | 32.978 | -4.798 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.512 | 365.000 | 0.660 | 2.969 | 0.397 | 25.608 | 79.673 | 32.699 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.540 | 365.000 | 0.749 | 2.486 | 0.469 | 29.499 | 77.584 | 30.278 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.949 | 0.953 | 0.773 | 15.484 | 31.534 | -5.531 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.551 | 365.000 | 0.728 | 2.779 | 0.391 | 23.222 | 43.881 | -6.437 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.532 | 365.000 | 0.775 | 2.330 | 0.492 | 27.849 | 40.373 | -8.859 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.707 | 365.000 | 0.947 | 0.949 | 0.763 | 15.650 | 31.506 | -5.532 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.518 | 365.000 | 0.656 | 3.167 | 0.336 | 25.184 | 46.070 | -6.563 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.529 | 365.000 | 0.757 | 2.399 | 0.482 | 28.551 | 41.593 | -8.985 | 17.611 |
| 2 judge, substituted | r8_off | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.947 | 0.966 | 0.783 | 15.380 | 32.184 | -4.151 | 17.611 |
| 2 judge, substituted | r8_off | 0 | in-sample | none | yes | 0.545 | 365.000 | 0.727 | 2.791 | 0.388 | 23.294 | 44.295 | -5.057 | 17.611 |
| 2 judge, substituted | r8_off | 0 | in-sample | practice | yes | 0.534 | 365.000 | 0.775 | 2.300 | 0.491 | 27.941 | 40.754 | -7.479 | 17.611 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | insample | yes | 0.704 | 365.000 | 0.947 | 0.954 | 0.760 | 15.577 | 32.240 | -4.134 | 17.611 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | none | no | 0.499 | 365.000 | 0.651 | 3.198 | 0.326 | 25.291 | 46.586 | -5.165 | 17.611 |
| 2 judge, substituted | r8_off | 0 | leave-one-driver-out | practice | no | 0.532 | 365.000 | 0.754 | 2.414 | 0.488 | 28.717 | 41.994 | -7.587 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.947 | 0.966 | 0.783 | 15.380 | 32.184 | -4.151 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.545 | 365.000 | 0.727 | 2.791 | 0.388 | 23.294 | 44.295 | -5.057 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.534 | 365.000 | 0.775 | 2.300 | 0.491 | 27.941 | 40.754 | -7.479 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.705 | 359.000 | 0.944 | 0.953 | 0.756 | 15.477 | 32.162 | -3.922 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.493 | 359.000 | 0.641 | 3.197 | 0.322 | 25.302 | 46.437 | -4.630 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.526 | 359.000 | 0.746 | 2.433 | 0.485 | 28.473 | 42.170 | -7.528 | 17.389 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.685 | 251.000 | 0.933 | 1.083 | 0.756 | 18.672 | 32.130 | -5.602 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.518 | 251.000 | 0.685 | 2.819 | 0.405 | 24.399 | 86.633 | 27.490 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.570 | 251.000 | 0.772 | 2.199 | 0.523 | 29.542 | 85.752 | 24.637 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.681 | 251.000 | 0.931 | 1.110 | 0.742 | 18.488 | 32.224 | -5.798 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.522 | 251.000 | 0.637 | 3.000 | 0.376 | 24.793 | 87.056 | 28.227 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.558 | 251.000 | 0.753 | 2.304 | 0.501 | 30.147 | 85.841 | 25.375 | 17.071 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.545 | 365.000 | 0.764 | 2.418 | 0.470 | 28.690 | 86.027 | 42.553 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.608 | 365.000 | 0.880 | 1.689 | 0.581 | 25.481 | 53.835 | 39.415 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.589 | 365.000 | 0.887 | 1.581 | 0.631 | 20.656 | 39.286 | 23.911 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.685 | 365.000 | 0.916 | 1.277 | 0.692 | 16.919 | 25.987 | 9.502 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.748 | 365.000 | 0.936 | 1.055 | 0.776 | 13.966 | 21.550 | 3.553 | 17.611 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.474 | 365.000 | 0.702 | 2.912 | 0.376 | 24.355 | 87.662 | 44.975 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.540 | 365.000 | 0.790 | 2.385 | 0.446 | 21.751 | 57.623 | 40.869 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.551 | 365.000 | 0.805 | 2.118 | 0.508 | 19.132 | 41.539 | 23.102 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.674 | 365.000 | 0.883 | 1.579 | 0.635 | 17.286 | 29.174 | 8.870 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.734 | 365.000 | 0.924 | 1.188 | 0.726 | 14.204 | 24.492 | 3.137 | 17.611 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 317.000 | 44.158 | -4.548 | 31.411 | -3.527 | 19.004 | 0.442 | 3.890 | 0.903 | 901 | 12.576 | -2.127 | 0.822 | 556.000 | 0.534 | -0.246 | 0.124 |
| pooled | lodo | 317.000 | 46.479 | -4.674 | 31.467 | -3.500 | 19.205 | 0.461 | 3.963 | 0.909 | 901 | 12.734 | -2.129 | 0.829 | 556.000 | 0.555 | -0.250 | 0.128 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 894 | 14.152 | -2.313 | 0.897 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 30.75 | -15.28 | 21.16 | -16.37 | 19.98 | -2.44 | 5.30 | -1.57 | 44 | 8.53 | -5.51 | 0.47 | 25.00 | 0.35 | -0.05 | 0.10 |
| 2019_10_united_kingdom | lodo | 17.00 | 35.58 | -15.78 | 21.40 | -16.73 | 20.24 | -2.34 | 5.56 | -1.55 | 44 | 8.86 | -5.71 | 0.48 | 25.00 | 0.37 | -0.05 | 0.13 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.75 | -6.48 | 0.51 | - | - | - | - |
| 2021_01_bahrain | in | 18.00 | 40.38 | 5.76 | 17.01 | 7.59 | 13.11 | -1.06 | 1.39 | 3.14 | 56 | 8.11 | 1.29 | 0.85 | 35.00 | 0.46 | -0.27 | 0.10 |
| 2021_01_bahrain | lodo | 18.00 | 43.26 | 5.65 | 17.09 | 7.24 | 13.35 | -1.07 | 1.40 | 3.12 | 56 | 8.29 | 1.18 | 0.86 | 35.00 | 0.48 | -0.27 | 0.10 |
| 2021_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 55 | 9.40 | 1.20 | 0.93 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 52.84 | -20.15 | 17.25 | -16.29 | 6.98 | -1.35 | 2.37 | -0.13 | 56 | 7.14 | -4.67 | 0.39 | 37.00 | 0.54 | -0.31 | 0.19 |
| 2021_04_spain | lodo | 19.00 | 56.42 | -20.61 | 17.72 | -16.90 | 6.99 | -1.37 | 2.40 | -0.13 | 56 | 7.38 | -4.87 | 0.40 | 37.00 | 0.55 | -0.30 | 0.20 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 8.79 | -4.62 | 0.46 | - | - | - | - |
| 2021_09_austrian | in | 19.00 | 50.24 | -42.75 | 42.87 | -42.87 | 9.81 | -4.00 | 3.30 | -12.65 | 46 | 15.75 | -15.42 | 0.68 | 26.00 | 0.52 | -0.37 | 0.20 |
| 2021_09_austrian | lodo | 19.00 | 51.68 | -42.80 | 42.50 | -42.50 | 10.03 | -3.93 | 3.27 | -12.64 | 46 | 15.71 | -15.31 | 0.68 | 26.00 | 0.54 | -0.36 | 0.19 |
| 2021_09_austrian | loso | - | - | - | - | - | - | - | - | - | 45 | 17.72 | -15.53 | 0.75 | - | - | - | - |
| 2021_20_qatar | in | 18.00 | 48.22 | 22.41 | 24.91 | 20.09 | 15.16 | -1.41 | 2.50 | 12.04 | 46 | 12.75 | 8.97 | 0.80 | 27.00 | 0.67 | -0.09 | 0.07 |
| 2021_20_qatar | lodo | 18.00 | 51.17 | 22.37 | 24.95 | 20.13 | 15.38 | -1.40 | 2.56 | 12.05 | 46 | 12.93 | 8.97 | 0.81 | 27.00 | 0.69 | -0.10 | 0.06 |
| 2021_20_qatar | loso | - | - | - | - | - | - | - | - | - | 46 | 15.20 | 8.23 | 0.91 | - | - | - | - |
| 2021_22_united_arab_emirates | in | 15.00 | 40.05 | 18.01 | 31.98 | 13.50 | 29.28 | 7.00 | 6.32 | 0.93 | 41 | 19.25 | -0.45 | 2.17 | 19.00 | 0.50 | 0.01 | 0.03 |
| 2021_22_united_arab_emirates | lodo | 15.00 | 43.06 | 19.10 | 31.44 | 13.67 | 28.70 | 7.01 | 6.33 | 0.93 | 41 | 19.28 | -0.39 | 2.18 | 19.00 | 0.60 | -0.00 | 0.03 |
| 2021_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 40 | 21.64 | -0.91 | 2.32 | - | - | - | - |
| 2022_01_bahrain | in | 19.00 | 64.14 | 64.14 | 84.12 | 70.54 | 48.80 | 3.29 | 4.37 | 13.50 | 75 | 21.02 | 14.72 | 2.00 | 54.00 | 0.54 | -0.36 | 0.15 |
| 2022_01_bahrain | lodo | 19.00 | 63.90 | 63.90 | 83.69 | 70.15 | 49.44 | 3.27 | 4.34 | 13.49 | 75 | 21.15 | 14.64 | 2.01 | 54.00 | 0.57 | -0.36 | 0.16 |
| 2022_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 75 | 22.20 | 14.82 | 2.07 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 74.90 | -74.90 | 77.82 | -77.82 | 13.11 | -2.98 | 8.48 | -8.93 | 29 | 41.22 | -37.86 | 1.53 | 15.00 | 0.41 | 0.23 | 0.15 |
| 2022_02_saudi_arabia | lodo | 14.00 | 74.61 | -74.61 | 78.36 | -78.36 | 13.96 | -3.07 | 8.48 | -8.94 | 29 | 41.47 | -38.08 | 1.55 | 15.00 | 0.45 | 0.27 | 0.16 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 41.39 | -37.71 | 1.56 | - | - | - | - |
| 2022_06_spain | in | 18.00 | 47.98 | -11.45 | 13.76 | -9.16 | 14.35 | 0.04 | 1.31 | - | 69 | 4.00 | -2.18 | 0.27 | 49.00 | 0.56 | -0.29 | 0.16 |
| 2022_06_spain | lodo | 18.00 | 53.18 | -13.57 | 14.43 | -9.37 | 15.21 | 0.16 | 1.37 | - | 69 | 4.50 | -2.32 | 0.29 | 49.00 | 0.57 | -0.30 | 0.19 |
| 2022_06_spain | loso | - | - | - | - | - | - | - | - | - | 67 | 5.93 | -3.00 | 0.37 | - | - | - | - |
| 2022_08_azerbaijan | in | 16.00 | 43.57 | 30.88 | 32.91 | 32.91 | 14.83 | 6.77 | 10.79 | 10.38 | 40 | 12.44 | 9.16 | 0.72 | 23.00 | 0.55 | 0.18 | 0.05 |
| 2022_08_azerbaijan | lodo | 16.00 | 45.79 | 31.31 | 32.85 | 32.85 | 15.32 | 6.74 | 10.94 | 10.40 | 40 | 12.68 | 9.15 | 0.74 | 23.00 | 0.58 | 0.19 | 0.04 |
| 2022_08_azerbaijan | loso | - | - | - | - | - | - | - | - | - | 40 | 13.36 | 9.36 | 0.78 | - | - | - | - |
| 2022_09_canada | in | 17.00 | 16.95 | 4.56 | 44.48 | 6.78 | 44.08 | 3.54 | 5.58 | 1.27 | 47 | 17.39 | 0.79 | 0.79 | 30.00 | 0.44 | 0.02 | 0.12 |
| 2022_09_canada | lodo | 17.00 | 17.45 | 4.49 | 44.67 | 6.83 | 44.27 | 3.55 | 5.66 | 1.28 | 47 | 17.41 | 0.80 | 0.79 | 30.00 | 0.45 | 0.01 | 0.13 |
| 2022_09_canada | loso | - | - | - | - | - | - | - | - | - | 47 | 17.91 | 0.89 | 0.81 | - | - | - | - |
| 2022_12_france | in | 16.00 | 22.20 | 11.79 | 22.35 | 11.81 | 17.15 | 1.49 | 6.67 | 5.75 | 36 | 11.57 | 5.14 | 0.54 | 20.00 | 0.45 | -0.33 | 0.11 |
| 2022_12_france | lodo | 16.00 | 23.96 | 12.49 | 23.19 | 12.88 | 17.43 | 1.91 | 7.02 | 5.77 | 36 | 11.87 | 5.38 | 0.56 | 20.00 | 0.50 | -0.36 | 0.10 |
| 2022_12_france | loso | - | - | - | - | - | - | - | - | - | 36 | 11.57 | 5.39 | 0.56 | - | - | - | - |
| 2022_20_mexico | in | 19.00 | 52.64 | 5.04 | 20.32 | 5.09 | 19.67 | 0.17 | 1.40 | 11.54 | 41 | 11.54 | 1.86 | 0.41 | 21.00 | 0.46 | -0.35 | 0.13 |
| 2022_20_mexico | lodo | 19.00 | 56.14 | 4.95 | 20.15 | 5.47 | 19.41 | 0.03 | 1.46 | 11.57 | 41 | 11.81 | 2.11 | 0.41 | 21.00 | 0.48 | -0.36 | 0.12 |
| 2022_20_mexico | loso | - | - | - | - | - | - | - | - | - | 41 | 15.96 | 1.63 | 0.55 | - | - | - | - |
| 2022_21_brazil | in | 17.00 | 74.61 | -74.61 | 71.03 | -69.30 | 34.53 | -1.11 | 2.46 | -5.94 | 59 | 28.23 | -22.19 | 1.98 | 39.00 | 0.50 | -0.24 | 0.11 |
| 2022_21_brazil | lodo | 17.00 | 75.52 | -75.52 | 70.03 | -69.27 | 33.74 | -1.14 | 2.53 | -5.92 | 59 | 27.90 | -22.17 | 1.96 | 39.00 | 0.45 | -0.26 | 0.12 |
| 2022_21_brazil | loso | - | - | - | - | - | - | - | - | - | 59 | 28.48 | -22.44 | 1.99 | - | - | - | - |
| 2022_22_united_arab_emirates | in | 19.00 | 38.77 | -12.11 | 14.23 | -12.30 | 14.25 | -3.99 | 4.48 | - | 49 | 4.75 | -3.18 | 0.24 | 30.00 | 0.36 | -0.06 | 0.08 |
| 2022_22_united_arab_emirates | lodo | 19.00 | 41.31 | -12.19 | 14.07 | -12.17 | 14.39 | -3.84 | 4.52 | - | 49 | 4.86 | -3.22 | 0.24 | 30.00 | 0.40 | -0.07 | 0.10 |
| 2022_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 49 | 7.16 | -4.27 | 0.33 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 31.20 | 16.30 | 23.13 | 16.02 | 14.39 | 2.56 | 4.28 | 10.64 | 62 | 5.58 | 1.65 | 0.41 | 42.00 | 0.81 | -0.51 | 0.20 |
| 2023_01_bahrain | lodo | 17.00 | 33.14 | 16.21 | 22.87 | 15.35 | 13.95 | 2.53 | 4.35 | 10.59 | 62 | 5.53 | 1.49 | 0.42 | 42.00 | 0.87 | -0.52 | 0.22 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.10 | 1.88 | 0.53 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 24.95 | -8.03 | 8.21 | -6.37 | 7.11 | -0.51 | 1.04 | - | 63 | 3.66 | -1.81 | 0.21 | 42.00 | 0.72 | -0.62 | 0.18 |
| 2023_07_spain | lodo | 20.00 | 27.14 | -7.38 | 8.98 | -5.23 | 8.05 | -0.50 | 1.09 | - | 63 | 4.00 | -1.44 | 0.22 | 42.00 | 0.74 | -0.64 | 0.16 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 5.33 | -1.59 | 0.29 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 42.91 | -10.62 | 11.78 | -8.58 | 8.89 | 0.38 | 1.25 | - | 42 | 6.54 | -5.05 | 0.25 | 22.00 | 0.40 | -0.22 | -0.05 |
| 2024_07_emilia_romagna | lodo | 19.00 | 45.24 | -10.97 | 11.92 | -8.45 | 9.14 | 0.25 | 1.39 | - | 42 | 6.71 | -4.91 | 0.26 | 22.00 | 0.38 | -0.20 | -0.07 |
| 2024_07_emilia_romagna | loso | - | - | - | - | - | - | - | - | - | 42 | 8.29 | -5.02 | 0.32 | - | - | - | - |

Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:

| variant | compound | 1-5 | 11-15 | 16-20 | 21-25 | 26-30 | 31-35 | 36-40 | 41+ | 6-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| in | HARD | -0.04 | 0.07 | 0.05 | 0.03 | 0.03 | 0.03 | 0.05 | -0.05 | 0.04 |
| in | MEDIUM | 0.02 | -0.00 | -0.02 | -0.04 | 0.06 | 0.17 | 0.36 | 0.45 | 0.02 |
| in | SOFT | -0.17 | 0.08 | 0.06 | 0.08 | 0.24 | 0.09 | -0.11 | - | -0.09 |
| lodo | HARD | -0.04 | 0.07 | 0.06 | 0.03 | 0.04 | 0.04 | 0.01 | -0.13 | 0.04 |
| lodo | MEDIUM | 0.02 | -0.01 | -0.03 | -0.04 | 0.08 | 0.21 | 0.47 | 0.47 | 0.01 |
| lodo | SOFT | -0.18 | 0.08 | 0.06 | 0.07 | 0.23 | 0.10 | -0.12 | - | -0.10 |

Neutralised laps:

| variant | kind | n | mean_resid_s | mae_s |
|---|---|---|---|---|
| in | sc | 682 | 3.65 | 14.54 |
| in | vsc | 350 | -9.78 | 12.00 |
| lodo | sc | 682 | 3.64 | 14.53 |
| lodo | vsc | 350 | -9.78 | 12.00 |
| loso | sc | 682 | 3.68 | 14.53 |
| loso | vsc | 350 | -9.78 | 11.99 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 1.00 | 0.00 | 54 | 0.37 | 0.00 | 0.61 |
| 2021_01_bahrain | 2 | 2 | 1:4 2:12 | 0.75 | 0.69 | 0.69 | 0.00 | 60 | 0.12 | 0.00 | 1.22 |
| 2021_04_spain | 2 | 2 | 1:9 2:10 | 0.53 | 0.53 | 1.00 | 0.53 | 72 | 0.50 | 0.50 | 1.83 |
| 2021_09_austrian | 1 | 1 | 1:18 2:1 | 0.95 | 0.63 | 1.00 | 0.00 | 78 | 0.45 | 0.00 | 1.13 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 0.56 | 0.00 | 60 | 0.13 | 0.00 | 0.35 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 1.00 | 0.00 | 60 | 0.52 | 0.00 | 2.69 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.28 | 0.00 | 60 | 0.07 | 0.00 | 2.83 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 1.00 | 0.00 | 54 | 0.44 | 0.00 | 0.32 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.00 | 0.00 | 72 | 0.00 | 0.00 | 2.87 |
| 2022_08_azerbaijan | 2 | 1 | 1:15 | 0.00 | 0.47 | 0.27 | 0.00 | 54 | 0.24 | 0.00 | 0.30 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 1.00 | 0.00 | 78 | 0.54 | 0.00 | 1.75 |
| 2022_12_france | 1 | 1 | 1:15 | 1.00 | 0.80 | 0.80 | 0.80 | 54 | 0.28 | 0.28 | 0.74 |
| 2022_20_mexico | 1 | 1 | 1:18 | 1.00 | 0.83 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 0.23 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 78 | 0.17 | 0.00 | 1.48 |
| 2022_22_united_arab_emirates | 2 | 2 | 2:17 | 1.00 | 0.59 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.37 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.15 | 0.00 | 1.24 |
| 2023_07_spain | 2 | 1 | 1:20 | 0.00 | 0.00 | 1.00 | 0.00 | 72 | 0.36 | 0.00 | 1.24 |
| 2024_07_emilia_romagna | 1 | 1 | 1:17 2:2 | 0.89 | 0.89 | 0.11 | 0.00 | 66 | 0.00 | 0.00 | 0.19 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.027 | -0.014 | 1.978 | 0.650 | -0.565 | 0.129 | -0.917 | 2.935 | 0.009 | -0.080 | 0.038 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.036 | -0.014 | 1.708 | 0.832 | -0.891 | 0.157 | -0.967 | 2.783 | - | - | - |
| pit_loss_prior | in | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | 0.000 | 0.001 | 0.104 | -0.645 | -1.392 | 0.000 | -0.756 | 0.000 | -0.044 | -0.014 | 0.000 |
| pit_loss_prior | lodo | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | 0.003 | 0.001 | 0.073 | -0.729 | -1.408 | 0.000 | -0.764 | 0.000 | - | - | - |
| r8_off | in | stint_limit | no stint limit (A4 only: replays do not apply R8) |  | 365 | 365 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.118 | 0.123 | 0.587 |
| r8_off | lodo | stint_limit | no stint limit (A4 only: replays do not apply R8) |  | 365 | 365 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 365 | 365 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | -0.065 | -0.074 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 359 | 365 | 0.001 | -0.002 | -0.100 | -0.128 | 0.258 | 0.001 | 0.108 | 0.005 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 251 | 365 | -0.002 | -0.014 | 3.292 | -1.049 | -2.224 | 0.164 | -1.349 | 3.301 | -0.062 | -0.047 | 0.418 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 251 | 365 | -0.023 | -0.016 | 2.912 | -1.045 | -2.437 | 0.193 | -1.381 | 3.154 | - | - | - |

A6: hindsight ledger (judge and predictive):

| role | id | input | tag | substitute | source |
|---|---|---|---|---|---|
| judge | fit_window | degradation fit rows (fit_window full) | this race, whole | pre_race_fit | derived |
| judge | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, whole | - | derived |
| judge | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, whole | - | derived |
| judge | pit_loss | pit loss (Simulator.race_pit_loss_s, eval.score.world_pit_loss) | this race, whole | pit_loss_prior | derived |
| judge | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| judge | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
| judge | stint_limit | R8 stint limit (rank-1 oldest age per compound + 3) | this race, whole | r8_off | derived |
| judge | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| judge | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| judge | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| judge | events | neutralisations (events none) | a priori | - | derived |
| judge | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| judge | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| judge | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |
| predictive | fit_window | degradation fit rows (fit_window to_snapshot) | this race, to snapshot | - | derived |
| predictive | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, to snapshot | - | derived |
| predictive | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, to snapshot | - | derived |
| predictive | pit_loss | pit loss (pit_loss.source asof) | this race, to snapshot | - | derived |
| predictive | pit_loss_prior | pit loss prior table (configs/sim/pit_loss_prior.json, earlier races only) | history | - | derived |
| predictive | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| predictive | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
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
| 2019_10_united_kingdom | 54 | 0.853 | 1.431 | 0.870 | 0.000 | 0.887 | 0.865 | 1.521 | 1931 | 103.831 | 30.347 | 30.347 | 103.831 | 3.457 | 0.068 | 0.270 | 0.617 | 0.674 | 0.349 | 0.603 | 70.093 | 63.451 | 32.315 | 22.291 |
| 2021_01_bahrain | 60 | 0.536 | 3.182 | 0.567 | 0.000 | 0.875 | 0.549 | 3.437 | 2127 | 109.641 | 26.077 | 21.142 | 109.641 | 3.375 | 0.511 | 0.745 | 0.594 | 0.787 | 0.379 | 0.443 | 60.045 | 37.568 | 13.379 | -7.849 |
| 2021_04_spain | 72 | 0.824 | 0.680 | 0.694 | 0.000 | 0.845 | 0.847 | 0.709 | 2544 | 97.282 | 17.039 | 12.554 | 97.282 | 2.717 | 0.454 | 0.544 | 0.754 | 0.856 | 0.359 | 0.512 | 37.520 | 9.912 | 22.318 | 8.762 |
| 2021_09_austrian | 78 | 0.698 | 1.191 | 0.897 | 0.000 | 0.960 | 0.708 | 1.101 | 2707 | 83.082 | 19.428 | 10.056 | 83.082 | 2.090 | 0.543 | 0.651 | 0.566 | 0.822 | 0.407 | 0.529 | 34.180 | 7.208 | 22.754 | -1.825 |
| 2021_20_qatar | 60 | 0.590 | 5.998 | 0.483 | 0.000 | 0.862 | 0.586 | 5.569 | 2140 | 103.997 | 31.521 | 31.240 | 103.997 | 3.181 | 0.121 | 0.212 | 0.365 | 0.568 | 0.207 | 0.336 | 73.127 | 68.084 | 64.022 | 63.267 |
| 2021_22_united_arab_emirates | 60 | 0.791 | 1.030 | 0.700 | 0.000 | 0.789 | 0.777 | 1.035 | 2150 | 94.153 | 23.265 | 23.045 | 94.153 | 2.934 | 0.104 | 0.342 | 0.588 | 0.810 | 0.507 | 0.667 | 59.470 | 50.001 | 50.820 | 34.798 |
| 2022_01_bahrain | 60 | 0.720 | 3.682 | 0.500 | 0.000 | 0.820 | 0.745 | 3.557 | 2169 | 114.593 | 23.463 | 19.197 | 114.593 | 4.751 | 0.232 | 0.390 | 0.445 | 0.638 | 0.521 | 0.721 | 89.286 | 86.033 | 104.029 | 90.801 |
| 2022_02_saudi_arabia | 54 | 0.872 | 0.112 | 0.926 | 0.000 | 0.917 | 0.887 | 0.030 | 1779 | 109.813 | 29.885 | 29.816 | 109.813 | 4.350 | 0.115 | 0.216 | 0.788 | 0.906 | 0.577 | 0.690 | 35.138 | 6.728 | 24.188 | 7.468 |
| 2022_06_spain | 72 | 0.713 | 3.557 | 0.431 | 0.000 | 0.853 | 0.704 | 3.256 | 2654 | 98.011 | 20.555 | 17.480 | 98.011 | 2.852 | 0.258 | 0.495 | 0.658 | 0.795 | 0.538 | 0.716 | 57.628 | -26.804 | 23.386 | 13.236 |
| 2022_08_azerbaijan | 54 | 0.608 | 5.710 | 0.426 | 0.000 | 0.745 | 0.628 | 5.372 | 1858 | 101.777 | 20.384 | 8.852 | 101.777 | 3.876 | 0.124 | 0.334 | 0.575 | 0.706 | 0.450 | 0.596 | 44.805 | 39.016 | 29.912 | 24.122 |
| 2022_09_canada | 78 | 0.816 | 1.924 | 0.705 | 0.000 | 0.779 | 0.827 | 1.733 | 2758 | 95.556 | 14.870 | 3.634 | 95.556 | 2.585 | 0.384 | 0.628 | 0.713 | 0.881 | 0.414 | 0.608 | 34.083 | 13.872 | 38.237 | -3.431 |
| 2022_12_france | 54 | 0.849 | 2.932 | 0.630 | 0.000 | 0.868 | 0.830 | 2.606 | 1983 | 88.688 | 15.474 | 14.866 | 88.688 | 3.008 | 0.357 | 0.447 | 0.539 | 0.694 | 0.415 | 0.519 | 35.675 | 27.213 | 37.749 | 33.947 |
| 2022_20_mexico | 78 | 0.605 | 6.489 | 0.436 | 0.000 | 0.797 | 0.596 | 7.475 | 2723 | 79.501 | 24.206 | -9.035 | 79.501 | 2.323 | 0.707 | 0.828 | 0.338 | 0.535 | 0.350 | 0.496 | 49.094 | 16.668 | 27.333 | 9.422 |
| 2022_21_brazil | 78 | 0.897 | 0.759 | 0.821 | 0.000 | 0.901 | 0.902 | 0.918 | 2766 | 94.727 | 25.019 | 6.484 | 93.105 | 2.781 | 0.146 | 0.298 | 0.170 | 0.418 | 0.066 | 0.136 | 36.760 | -29.201 | 31.877 | -13.123 |
| 2022_22_united_arab_emirates | 60 | 0.501 | 9.002 | 0.233 | 0.000 | 0.818 | 0.499 | 9.585 | 2243 | 80.314 | 18.103 | 14.444 | 80.314 | 2.439 | 0.436 | 0.519 | 0.308 | 0.469 | 0.357 | 0.500 | 40.154 | 18.172 | 25.441 | 7.719 |
| 2023_01_bahrain | 60 | 0.464 | 3.106 | 0.383 | 0.000 | 0.768 | 0.444 | 3.370 | 2216 | 97.841 | 24.016 | 17.159 | 97.841 | 3.010 | 0.404 | 0.524 | 0.503 | 0.712 | 0.343 | 0.531 | 45.750 | 40.550 | 43.616 | 39.955 |
| 2023_07_spain | 72 | 0.717 | 3.079 | 0.472 | 0.000 | 0.894 | 0.727 | 3.345 | 2572 | 90.449 | 15.468 | 14.728 | 90.449 | 2.597 | 0.301 | 0.435 | 0.471 | 0.657 | 0.278 | 0.375 | 26.758 | 19.656 | 18.156 | 15.097 |
| 2024_07_emilia_romagna | 66 | 0.838 | 4.446 | 0.364 | 0.000 | 0.561 | 0.834 | 4.676 | 2351 | 68.718 | 9.203 | 5.328 | 68.718 | 2.009 | 0.444 | 0.569 | 0.256 | 0.414 | 0.292 | 0.390 | 36.198 | -12.696 | 21.007 | 2.441 |
| pooled | 1170 | 0.719 | 3.195 | 0.588 | 0.000 | 0.830 | 0.722 | 3.264 | 41671 | 94.389 | 21.262 | 14.166 | 94.281 | 2.952 | 0.330 | 0.482 | 0.509 | 0.684 | 0.372 | 0.514 | 48.130 | 23.988 | 34.988 | 19.713 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1837.00 | 24.92 | 24.92 |
| 10.00 | 8653.00 | 42.01 | 42.01 |
| 20.00 | 8588.00 | 62.52 | 62.52 |
| 30.00 | 8312.00 | 97.37 | 97.37 |
| 40.00 | 7630.00 | 131.18 | 131.14 |
| 50.00 | 4468.00 | 170.81 | 170.34 |
| 60.00 | 2183.00 | 189.50 | 188.54 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 12621 | 162.80 | 162.47 |
| late | 11940 | 43.35 | 43.35 |
| mid | 17110 | 79.54 | 79.52 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12150 | 89.72 | 89.40 |
| real | 1116 | 98.17 | 98.17 |
| stay | 319 | 95.03 | 95.03 |
| two | 28086 | 96.25 | 96.23 |

B5 future blindness: {"status": "pass", "new_leak_differences": 0, "differences_with_known_leaks": 8, "known_leaks": ["stint_limit"], "fixed_leaks": ["compound_list", "frame_filter", "gap_behind", "lap_window", "pit_loss", "race_length"], "detail": ""}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | history prior (tyre_life_prior) | oldest age run | 50 | 18 | 7.553 | -1.268 | 0.600 | 0.880 |
| tyre_life | history prior (tyre_life_prior) | median stint-end age | 50 | 18 | 9.267 | 8.722 | 0.520 | 0.800 |
| tyre_life | history prior (tyre_life_prior) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.540 | 1.302 | 0.783 | 0.913 |
| tyre_life | belief support edge @k0 | oldest age run | 50 | 18 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k0 | median stint-end age | 50 | 18 | 9.990 | 9.990 | - | - |
| tyre_life | belief support edge @k0 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 3.917 | 3.917 | - | - |
| tyre_life | as-of data (race to k0, practice) | oldest age run | 50 | 18 | 12.460 | -10.500 | - | - |
| tyre_life | as-of data (race to k0, practice) | median stint-end age | 50 | 18 | 7.630 | -0.510 | - | - |
| tyre_life | as-of data (race to k0, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 9.574 | -7.757 | - | - |
| tyre_life | belief support edge @k10 | oldest age run | 50 | 18 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k10 | median stint-end age | 50 | 18 | 9.990 | 9.990 | - | - |
| tyre_life | belief support edge @k10 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 3.917 | 3.917 | - | - |
| tyre_life | as-of data (race to k10, practice) | oldest age run | 50 | 18 | 12.400 | -10.440 | - | - |
| tyre_life | as-of data (race to k10, practice) | median stint-end age | 50 | 18 | 7.570 | -0.450 | - | - |
| tyre_life | as-of data (race to k10, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 9.509 | -7.691 | - | - |
| tyre_life | belief support edge @k20 | oldest age run | 50 | 18 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k20 | median stint-end age | 50 | 18 | 9.990 | 9.990 | - | - |
| tyre_life | belief support edge @k20 | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 3.917 | 3.917 | - | - |
| tyre_life | as-of data (race to k20, practice) | oldest age run | 50 | 18 | 11.380 | -9.420 | - | - |
| tyre_life | as-of data (race to k20, practice) | median stint-end age | 50 | 18 | 6.990 | 0.570 | - | - |
| tyre_life | as-of data (race to k20, practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 8.487 | -6.583 | - | - |
| tyre_life | belief support edge @k(mid) | oldest age run | 50 | 18 | 0.000 | 0.000 | - | - |
| tyre_life | belief support edge @k(mid) | median stint-end age | 50 | 18 | 9.990 | 9.990 | - | - |
| tyre_life | belief support edge @k(mid) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 3.917 | 3.917 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | oldest age run | 50 | 18 | 7.980 | -6.020 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | median stint-end age | 50 | 18 | 7.090 | 3.970 | - | - |
| tyre_life | as-of data (race to k(mid), practice) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.726 | -2.887 | - | - |
| wear_rate | belief @k0 | within-stint slope less fuel | 49 | 18 | 0.047 | 0.005 | 0.449 | 0.673 |
| wear_rate | belief @k10 | within-stint slope less fuel | 49 | 18 | 0.046 | -0.012 | 0.408 | 0.653 |
| wear_rate | belief @k20 | within-stint slope less fuel | 49 | 18 | 0.041 | -0.008 | 0.347 | 0.469 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 49 | 18 | 0.033 | -0.013 | 0.265 | 0.469 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 49 | 18 | 0.016 | -0.012 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 556 | 18 | 0.819 | -0.012 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 521 | 18 | 1.066 | -0.249 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 351 | 18 | 0.879 | 0.034 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 233 | 18 | 0.662 | -0.039 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.914 | -0.498 | - | - |
| compound_gap | belief @k0 | qualifying | 7 | 7 | 0.706 | 0.605 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.739 | -0.452 | - | - |
| compound_gap | belief @k10 | qualifying | 7 | 7 | 0.720 | 0.524 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.558 | -0.118 | - | - |
| compound_gap | belief @k20 | qualifying | 7 | 7 | 0.709 | 0.473 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.420 | -0.144 | - | - |
| compound_gap | belief @k(mid) | qualifying | 7 | 7 | 0.839 | 0.581 | - | - |
| fuel | prior (config) | race (judge full fit) | 18 | 18 | 0.020 | 0.017 | - | - |
| fuel | belief @k0 | race (judge full fit) | 18 | 18 | 0.020 | 0.017 | 0.278 | 0.444 |
| fuel | belief @k10 | race (judge full fit) | 18 | 18 | 0.019 | 0.015 | 0.222 | 0.333 |
| fuel | belief @k20 | race (judge full fit) | 18 | 18 | 0.013 | 0.009 | 0.278 | 0.389 |
| fuel | belief @k(mid) | race (judge full fit) | 18 | 18 | 0.011 | 0.005 | 0.222 | 0.444 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 10 | 10 | 2.637 | -1.291 | 0.500 | 0.600 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 10 | 10 | 3.115 | -2.291 | 0.500 | 0.700 |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 2.494 | -1.952 | - | - |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 3.454 | -2.424 | - | - |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.941 | -1.017 | 0.778 | 0.833 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.834 | -1.489 | 0.667 | 0.778 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.283 | 7.037 | 0.200 | 0.300 |
| pit_loss | belief charged @k0 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.900 | -0.910 | - | - |
| pit_loss | belief charged @k0 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.779 | -1.382 | - | - |
| pit_loss | belief charged @k0 | whole race, median SC stop | 10 | 10 | 13.295 | 7.097 | - | - |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.746 | -0.577 | 0.778 | 0.833 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.619 | -1.049 | 0.667 | 0.778 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.359 | 7.280 | 0.300 | 0.400 |
| pit_loss | belief charged @k10 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.705 | -0.475 | - | - |
| pit_loss | belief charged @k10 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.556 | -0.947 | - | - |
| pit_loss | belief charged @k10 | whole race, median SC stop | 10 | 10 | 13.373 | 7.335 | - | - |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.826 | -0.421 | 0.944 | 1.000 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.959 | -0.893 | 0.611 | 0.722 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.614 | 7.330 | 0.100 | 0.500 |
| pit_loss | belief charged @k20 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.803 | -0.350 | - | - |
| pit_loss | belief charged @k20 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.922 | -0.822 | - | - |
| pit_loss | belief charged @k20 | whole race, median SC stop | 10 | 10 | 12.614 | 7.366 | - | - |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.648 | -0.417 | 1.000 | 1.000 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.068 | -0.889 | 0.611 | 0.778 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.407 | 7.228 | 0.200 | 0.500 |
| pit_loss | belief charged @k(mid) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.639 | -0.351 | - | - |
| pit_loss | belief charged @k(mid) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.024 | -0.823 | - | - |
| pit_loss | belief charged @k(mid) | whole race, median SC stop | 10 | 10 | 12.406 | 7.262 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 18 | 18 | 2.167 | 1.360 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 18 | 18 | 0.238 | 0.093 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.598 | 0.125 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.630 | 0.089 | - | - |
| neutralisations | judge expected (none) | occurred | 72 | 18 | 1.417 | -1.417 | - | - |
| neutralisations | judge config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | judge config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralisations | belief expected (rate) | occurred | 72 | 18 | 1.355 | 0.405 | - | - |
| neutralisations | belief config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | belief config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralised_laps | judge replay (factors) | mean real lap time on the cars' neutralised laps | 21 | 14 | 9.314 | 3.642 | - | - |
| neutralised_laps | belief replay (factors) | mean real lap time on the cars' neutralised laps | 21 | 14 | 9.940 | 5.193 | - | - |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 6 | 6 | 1.044 | -0.781 | 0.833 | 1.000 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 6 | 6 | 2.627 | 0.031 | 0.167 | 0.833 |
| pit_loss | prior from earlier races (constant) | whole race, robust (pit_loss.whole_race: median green stop) | 2 | 2 | 1.147 | -0.357 | 1.000 | 1.000 |
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


Failed and left out: 2022_11_austria (ValueError: cannot convert float NaN to integer)
