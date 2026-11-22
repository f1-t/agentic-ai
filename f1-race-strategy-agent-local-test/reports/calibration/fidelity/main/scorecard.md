# Simulator fidelity scorecard: main

Computed on main f7be251 (the wear cliff, R8 retired, traffic off by default), before #102 (as-of tables, belief.yaml) and #100 merged.

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Code version 3ed6e2af6d0b-64df2bb5099f. World config af6c10d29af6 (race default); belief config 99d87e6a4aef. Options: {"b_drivers": 6, "b_every": 5, "samples": 20, "n_one": 4, "n_two": 3, "lodo": true, "loso": true, "grid_references": true, "hindsight_variants": true, "components": true, "field_replay": false, "seed": 0, "max_stops": 2}. Run time 207 s.

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
| A_close_pair_acc | 0.682 | judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order |
| A_spearman | 0.945 | same, Spearman of simulated against real finishing order |
| A_rel_gap_mae_s | 15.548 | same, mean /time error - the race's mean error/ over cars on the lead lap (relative gap error) |
| B2_rank_rho | 0.810 | Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots |
| B2_regret_s | 6.198 | judge time of the predictive favourite less the judge favourite, mean over snapshots |
| B2_pit_now_agree | 0.769 | share of snapshots where predictive and judge agree on pit-now against stay |
| P_close_pair_acc | 0.644 | predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k |
| P_spearman | 0.879 | same, Spearman, mean over k |
| A_abs_mae_s | 32.231 | judge LODO pace-adjusted whole-race time, mean /error/ |
| A_abs_bias_s | -4.896 | same, mean error |
| A_stop_gain_bias_s | -0.188 | judge LODO predicted less real gain at real stops |
| B1_mae_s | 94.210 | predictive less judge remaining time over the panel, mean /error/ |
| B1_bias_s | 93.952 | same, mean |
| B1_green_mae_s | 23.511 | B1 with no SC/VSC on either side (the physics difference without the belief's event model) |
| B2_rank_rho_green | 0.819 | B2 rank agreement with no SC/VSC on either side |
| B3_cov_90 | 0.481 | share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90) |
| B3_factor_cov_90 | 0.520 | share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90) |
| B4_mae_offset_s | 36.669 | predictive replay of the real remaining strategy against reality, pace-offset error (G) |

## The three distances (per race and pooled)

JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace measure, pace-offset error).

| race_id | JR_in_close_pair_acc | JR_in_spearman | JR_in_rel_gap_mae_s | JR_in_abs_mae_s | JR_lodo_close_pair_acc | JR_lodo_spearman | JR_lodo_rel_gap_mae_s | JR_lodo_abs_mae_s | PJ_rank_rho | PJ_regret_s | PJ_mae_s | PJ_bias_s | PR_close_pair_acc | PR_spearman | PR_rel_gap_mae_s | PR_g_mae_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 0.600 | 0.951 | 12.680 | 22.101 | 0.700 | 0.961 | 12.845 | 21.391 | 0.811 | 4.226 | 99.796 | 99.796 | 0.520 | 0.868 | 17.655 | 33.599 |
| 2021_01_bahrain | 0.909 | 0.992 | 4.471 | 13.594 | 0.909 | 0.992 | 4.409 | 13.716 | 0.681 | 6.270 | 107.304 | 107.304 | 0.582 | 0.937 | 11.010 | 13.382 |
| 2021_04_spain | 0.643 | 0.974 | 4.959 | 18.742 | 0.643 | 0.974 | 5.032 | 18.390 | 0.897 | 2.901 | 90.607 | 90.605 | 0.543 | 0.937 | 10.893 | 23.998 |
| 2021_09_austrian | 0.688 | 0.960 | 4.934 | 42.751 | 0.688 | 0.972 | 5.187 | 42.836 | 0.858 | 3.976 | 75.801 | 75.801 | 0.512 | 0.895 | 19.224 | 23.055 |
| 2021_20_qatar | 0.308 | 0.961 | 9.152 | 24.917 | 0.308 | 0.961 | 9.080 | 24.863 | 0.808 | 5.141 | 107.636 | 107.636 | 0.738 | 0.930 | 22.045 | 65.726 |
| 2021_22_united_arab_emirates | 0.750 | 0.961 | 28.505 | 32.824 | 0.786 | 0.946 | 28.927 | 32.596 | 0.879 | 7.822 | 89.637 | 89.637 | 0.657 | 0.858 | 31.246 | 51.516 |
| 2022_01_bahrain | 0.804 | 0.754 | 45.402 | 82.178 | 0.804 | 0.754 | 45.462 | 82.337 | 0.837 | 5.964 | 111.294 | 111.294 | 0.700 | 0.718 | 45.076 | 104.035 |
| 2022_02_saudi_arabia | 0.615 | 0.934 | 13.772 | 79.271 | 0.615 | 0.930 | 14.277 | 79.402 | 0.764 | 2.468 | 104.116 | 104.116 | 0.600 | 0.851 | 15.456 | 25.192 |
| 2022_06_spain | 0.500 | 0.967 | 12.707 | 12.573 | 0.500 | 0.959 | 12.804 | 13.636 | 0.826 | 6.072 | 88.037 | 87.561 | 0.700 | 0.969 | 16.723 | 23.602 |
| 2022_08_azerbaijan | 0.364 | 0.882 | 16.879 | 34.647 | 0.364 | 0.882 | 16.547 | 34.762 | 0.675 | 9.425 | 105.898 | 105.898 | 0.509 | 0.819 | 24.722 | 31.608 |
| 2022_09_canada | 0.667 | 0.944 | 44.072 | 44.450 | 0.667 | 0.944 | 44.103 | 44.437 | 0.768 | 3.526 | 90.348 | 87.472 | 0.587 | 0.825 | 40.097 | 39.629 |
| 2022_12_france | 0.818 | 0.985 | 14.134 | 22.536 | 0.818 | 0.985 | 14.452 | 23.018 | 0.904 | 4.518 | 95.304 | 95.304 | 0.745 | 0.877 | 11.926 | 40.660 |
| 2022_20_mexico | 0.357 | 0.875 | 11.390 | 21.621 | 0.357 | 0.865 | 11.388 | 21.605 | 0.876 | 4.688 | 88.453 | 88.453 | 0.571 | 0.834 | 26.827 | 35.420 |
| 2022_21_brazil | 0.833 | 0.971 | 32.265 | 70.050 | 0.854 | 0.966 | 31.724 | 69.929 | 0.723 | 11.853 | 106.611 | 106.132 | 0.721 | 0.895 | 29.200 | 31.890 |
| 2022_22_united_arab_emirates | 0.571 | 0.986 | 4.688 | 15.136 | 0.571 | 0.986 | 4.786 | 15.208 | 0.868 | 6.715 | 81.864 | 81.864 | 0.600 | 0.929 | 13.040 | 26.171 |
| 2023_01_bahrain | 0.583 | 0.978 | 8.428 | 21.836 | 0.500 | 0.975 | 8.191 | 21.395 | 0.769 | 6.430 | 95.785 | 95.675 | 0.667 | 0.959 | 17.884 | 43.769 |
| 2023_07_spain | 0.720 | 0.974 | 3.931 | 8.435 | 0.640 | 0.968 | 5.781 | 9.368 | 0.796 | 6.125 | 93.192 | 93.192 | 0.744 | 0.950 | 19.371 | 19.944 |
| 2024_07_emilia_romagna | 0.750 | 0.988 | 5.266 | 11.495 | 0.600 | 0.982 | 4.870 | 11.273 | 0.812 | 13.255 | 76.407 | 76.407 | 0.590 | 0.774 | 13.939 | 26.453 |
| pooled | 0.688 | 0.946 | 15.424 | 32.176 | 0.682 | 0.945 | 15.548 | 32.231 | 0.810 | 6.198 | 94.210 | 93.952 | 0.644 | 0.879 | 21.463 | 36.669 |

## O. Order and time, three versions (pooled)

fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of simulated times; it becomes a traffic measure when the traffic branch lands.

| version | label | k | fit | pace | in_sample | close_pair_acc | close_pairs | spearman | mae_pos | within_1 | rel_gap_mae_s | abs_mae_s | bias_s | n_cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 judge | as configured | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.946 | 0.978 | 0.767 | 15.424 | 32.176 | -4.961 | 17.611 |
| 1 judge | as configured | 0 | in-sample | none | yes | 0.490 | 365.000 | 0.697 | 2.936 | 0.368 | 23.783 | 44.882 | -3.072 | 17.611 |
| 1 judge | as configured | 0 | in-sample | practice | yes | 0.521 | 365.000 | 0.768 | 2.354 | 0.492 | 28.631 | 40.687 | -5.493 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | insample | yes | 0.682 | 365.000 | 0.945 | 1.024 | 0.742 | 15.548 | 32.231 | -4.896 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | none | no | 0.485 | 365.000 | 0.655 | 3.162 | 0.343 | 25.036 | 46.499 | -3.009 | 17.611 |
| 1 judge | as configured | 0 | leave-one-driver-out | practice | no | 0.518 | 365.000 | 0.753 | 2.440 | 0.483 | 28.992 | 41.567 | -5.431 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | insample | yes | 0.668 | 365.000 | 0.933 | 1.179 | 0.720 | 17.447 | 32.719 | -4.090 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | none | yes | 0.518 | 365.000 | 0.695 | 2.880 | 0.405 | 25.101 | 78.771 | 32.399 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | in-sample | practice | yes | 0.564 | 365.000 | 0.757 | 2.405 | 0.495 | 28.721 | 76.722 | 29.977 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | insample | yes | 0.671 | 365.000 | 0.931 | 1.178 | 0.725 | 17.366 | 32.962 | -4.405 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | none | no | 0.515 | 365.000 | 0.661 | 2.982 | 0.394 | 25.733 | 79.510 | 33.323 | 17.611 |
| 2 judge, substituted | pre_race_fit | 0 | leave-one-driver-out | practice | no | 0.545 | 365.000 | 0.747 | 2.497 | 0.466 | 29.393 | 77.427 | 30.901 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | insample | yes | 0.696 | 365.000 | 0.947 | 0.961 | 0.766 | 15.555 | 31.586 | -6.218 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | none | yes | 0.493 | 365.000 | 0.703 | 2.906 | 0.382 | 23.691 | 44.559 | -4.329 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | in-sample | practice | yes | 0.523 | 365.000 | 0.766 | 2.372 | 0.496 | 28.545 | 40.383 | -6.750 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | insample | yes | 0.679 | 365.000 | 0.944 | 1.011 | 0.745 | 15.658 | 31.578 | -6.201 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | none | no | 0.493 | 365.000 | 0.653 | 3.159 | 0.340 | 24.941 | 46.041 | -4.315 | 17.611 |
| 2 judge, substituted | pit_loss_prior | 0 | leave-one-driver-out | practice | no | 0.518 | 365.000 | 0.753 | 2.439 | 0.480 | 28.882 | 41.188 | -6.737 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | in-sample | insample | yes | 0.707 | 365.000 | 0.945 | 1.013 | 0.767 | 15.922 | 32.336 | -3.273 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | in-sample | none | yes | 0.523 | 365.000 | 0.711 | 2.879 | 0.389 | 23.573 | 44.396 | -0.729 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | in-sample | practice | yes | 0.548 | 365.000 | 0.777 | 2.322 | 0.498 | 28.073 | 39.815 | -3.150 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | leave-one-driver-out | insample | yes | 0.707 | 365.000 | 0.944 | 1.028 | 0.762 | 15.950 | 32.423 | -3.243 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | leave-one-driver-out | none | no | 0.512 | 365.000 | 0.670 | 3.044 | 0.371 | 24.980 | 45.949 | -0.723 | 17.611 |
| 2 judge, substituted | cliff_history_only | 0 | leave-one-driver-out | practice | no | 0.523 | 365.000 | 0.759 | 2.421 | 0.483 | 28.590 | 40.753 | -3.145 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | insample | yes | 0.688 | 365.000 | 0.946 | 0.978 | 0.767 | 15.424 | 32.176 | -4.961 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | none | yes | 0.490 | 365.000 | 0.697 | 2.936 | 0.368 | 23.783 | 44.882 | -3.072 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | in-sample | practice | yes | 0.521 | 365.000 | 0.768 | 2.354 | 0.492 | 28.631 | 40.687 | -5.493 | 17.611 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | insample | yes | 0.682 | 359.000 | 0.942 | 1.025 | 0.739 | 15.444 | 32.143 | -4.703 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | none | no | 0.479 | 359.000 | 0.643 | 3.175 | 0.339 | 25.049 | 46.372 | -2.492 | 17.389 |
| 2 judge, substituted | proxy_off | 0 | leave-one-driver-out | practice | no | 0.515 | 359.000 | 0.746 | 2.453 | 0.483 | 28.752 | 41.767 | -5.389 | 17.389 |
| 2 judge, substituted | all_substituted | 0 | in-sample | insample | yes | 0.685 | 251.000 | 0.930 | 1.117 | 0.739 | 18.644 | 31.800 | -5.162 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | none | yes | 0.522 | 251.000 | 0.685 | 2.829 | 0.405 | 24.383 | 86.255 | 28.198 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | in-sample | practice | yes | 0.574 | 251.000 | 0.767 | 2.240 | 0.519 | 29.487 | 85.389 | 25.345 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | insample | yes | 0.685 | 251.000 | 0.930 | 1.119 | 0.734 | 18.447 | 31.910 | -5.345 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | none | no | 0.526 | 251.000 | 0.636 | 3.016 | 0.372 | 24.790 | 86.681 | 28.935 | 17.071 |
| 2 judge, substituted | all_substituted | 0 | leave-one-driver-out | practice | no | 0.566 | 251.000 | 0.751 | 2.335 | 0.490 | 30.059 | 85.473 | 26.083 | 17.071 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | asof | no | 0.540 | 365.000 | 0.762 | 2.423 | 0.470 | 28.559 | 87.398 | 45.989 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | asof | no | 0.625 | 365.000 | 0.883 | 1.637 | 0.613 | 26.236 | 56.330 | 42.983 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | asof | no | 0.614 | 365.000 | 0.896 | 1.492 | 0.644 | 21.079 | 41.640 | 27.321 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | asof | no | 0.690 | 365.000 | 0.919 | 1.248 | 0.704 | 17.078 | 27.054 | 11.384 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | asof | no | 0.753 | 365.000 | 0.936 | 1.036 | 0.789 | 14.363 | 22.091 | 4.466 | 17.611 |
| 3 predictive | predictive | 0 | as-of (to_snapshot) | none | no | 0.504 | 365.000 | 0.703 | 2.927 | 0.369 | 25.038 | 89.207 | 48.411 | 17.611 |
| 3 predictive | predictive | 10 | as-of (to_snapshot) | none | no | 0.545 | 365.000 | 0.803 | 2.354 | 0.426 | 22.356 | 59.690 | 44.437 | 17.611 |
| 3 predictive | predictive | 20 | as-of (to_snapshot) | none | no | 0.567 | 365.000 | 0.802 | 2.162 | 0.484 | 19.270 | 43.138 | 26.512 | 17.611 |
| 3 predictive | predictive | 30 | as-of (to_snapshot) | none | no | 0.668 | 365.000 | 0.888 | 1.582 | 0.616 | 17.537 | 29.854 | 10.751 | 17.611 |
| 3 predictive | predictive | 40 | as-of (to_snapshot) | none | no | 0.734 | 365.000 | 0.922 | 1.206 | 0.724 | 14.534 | 24.947 | 4.050 | 17.611 |

## A. Judge versus reality

A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). loso rows are the held-out stint only.

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | in | 317.000 | 44.801 | -2.609 | 31.373 | -4.342 | 19.126 | 0.356 | 3.858 | 0.846 | 901 | 12.331 | -2.342 | 0.802 | 556.000 | 0.474 | -0.184 | 0.106 |
| pooled | lodo | 317.000 | 46.446 | -2.552 | 31.436 | -4.270 | 19.347 | 0.395 | 3.921 | 0.851 | 901 | 12.486 | -2.342 | 0.810 | 556.000 | 0.495 | -0.188 | 0.108 |
| pooled | loso | - | - | - | - | - | - | - | - | - | 894 | 14.784 | -1.133 | 0.911 | - | - | - | - |

Per race:

| race_id | variant | race_n | race_raw_mae_s | race_raw_bias_s | race_pace_mae_s | race_pace_bias_s | race_rel_mae_s | pit_err_per_stop_s | pit_mae_per_stop_s | neutral_err_per_lap_s | stint_n | stint_mae_s | stint_bias_s | stint_mae_per_lap_s | stop_gain_n | stop_gain_mae_s | stop_gain_bias_s | late_stint_resid_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | in | 17.00 | 28.37 | -8.89 | 22.10 | -18.17 | 19.64 | -2.82 | 5.23 | -1.74 | 44 | 8.54 | -5.97 | 0.45 | 25.00 | 0.31 | 0.05 | 0.04 |
| 2019_10_united_kingdom | lodo | 17.00 | 30.70 | -9.02 | 21.39 | -17.61 | 20.07 | -2.78 | 5.40 | -1.71 | 44 | 8.69 | -5.78 | 0.47 | 25.00 | 0.34 | 0.05 | 0.06 |
| 2019_10_united_kingdom | loso | - | - | - | - | - | - | - | - | - | 44 | 9.80 | -6.56 | 0.51 | - | - | - | - |
| 2021_01_bahrain | in | 18.00 | 42.01 | 6.03 | 13.59 | 5.39 | 11.05 | -1.09 | 1.45 | 2.75 | 56 | 6.18 | 0.62 | 0.69 | 35.00 | 0.41 | -0.15 | 0.10 |
| 2021_01_bahrain | lodo | 18.00 | 43.92 | 6.15 | 13.72 | 5.17 | 11.26 | -1.08 | 1.47 | 2.74 | 56 | 6.22 | 0.54 | 0.70 | 35.00 | 0.42 | -0.15 | 0.10 |
| 2021_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 55 | 6.31 | 0.89 | 0.71 | - | - | - | - |
| 2021_04_spain | in | 19.00 | 51.23 | -15.23 | 18.74 | -17.46 | 7.06 | -1.48 | 2.28 | -0.23 | 56 | 6.71 | -4.96 | 0.37 | 37.00 | 0.42 | -0.14 | 0.06 |
| 2021_04_spain | lodo | 19.00 | 53.89 | -15.12 | 18.39 | -17.19 | 6.97 | -1.46 | 2.28 | -0.22 | 56 | 6.63 | -4.88 | 0.37 | 37.00 | 0.43 | -0.14 | 0.06 |
| 2021_04_spain | loso | - | - | - | - | - | - | - | - | - | 56 | 7.23 | -4.86 | 0.39 | - | - | - | - |
| 2021_09_austrian | in | 19.00 | 49.60 | -41.63 | 42.75 | -42.75 | 10.02 | -3.92 | 3.29 | -12.66 | 46 | 15.72 | -15.41 | 0.68 | 26.00 | 0.37 | -0.20 | 0.15 |
| 2021_09_austrian | lodo | 19.00 | 50.81 | -41.60 | 42.84 | -42.84 | 9.96 | -3.87 | 3.30 | -12.66 | 46 | 15.82 | -15.48 | 0.69 | 26.00 | 0.38 | -0.19 | 0.14 |
| 2021_09_austrian | loso | - | - | - | - | - | - | - | - | - | 45 | 17.42 | -15.46 | 0.74 | - | - | - | - |
| 2021_20_qatar | in | 18.00 | 50.93 | 23.58 | 24.92 | 17.42 | 17.90 | -1.39 | 2.60 | 11.91 | 46 | 12.69 | 7.96 | 0.77 | 27.00 | 0.68 | -0.35 | 0.21 |
| 2021_20_qatar | lodo | 18.00 | 52.76 | 23.40 | 24.86 | 17.18 | 18.23 | -1.47 | 2.66 | 11.90 | 46 | 12.81 | 7.92 | 0.77 | 27.00 | 0.71 | -0.38 | 0.21 |
| 2021_20_qatar | loso | - | - | - | - | - | - | - | - | - | 46 | 14.69 | 8.31 | 0.86 | - | - | - | - |
| 2021_22_united_arab_emirates | in | 15.00 | 43.83 | 24.41 | 32.82 | 12.35 | 30.35 | 6.60 | 5.94 | 0.88 | 41 | 19.62 | -0.58 | 2.17 | 19.00 | 0.47 | -0.12 | 0.03 |
| 2021_22_united_arab_emirates | lodo | 15.00 | 45.32 | 24.91 | 32.60 | 12.34 | 30.13 | 6.59 | 5.93 | 0.89 | 41 | 19.95 | -0.58 | 2.18 | 19.00 | 0.51 | -0.13 | 0.03 |
| 2021_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 40 | 22.01 | -1.01 | 2.31 | - | - | - | - |
| 2022_01_bahrain | in | 19.00 | 64.51 | 64.51 | 82.18 | 68.75 | 49.86 | 3.16 | 4.25 | 13.46 | 75 | 20.96 | 14.41 | 1.99 | 54.00 | 0.50 | -0.24 | 0.13 |
| 2022_01_bahrain | lodo | 19.00 | 64.51 | 64.51 | 82.34 | 69.04 | 50.03 | 3.17 | 4.25 | 13.47 | 75 | 21.09 | 14.48 | 2.00 | 54.00 | 0.52 | -0.24 | 0.13 |
| 2022_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 75 | 22.29 | 14.75 | 2.08 | - | - | - | - |
| 2022_02_saudi_arabia | in | 14.00 | 69.72 | -69.72 | 79.27 | -79.27 | 12.97 | -3.43 | 8.55 | -9.00 | 29 | 41.81 | -38.28 | 1.54 | 15.00 | 0.29 | 0.01 | 0.18 |
| 2022_02_saudi_arabia | lodo | 14.00 | 70.17 | -69.31 | 79.40 | -79.40 | 13.31 | -3.48 | 8.53 | -8.99 | 29 | 42.10 | -38.31 | 1.56 | 15.00 | 0.34 | 0.02 | 0.18 |
| 2022_02_saudi_arabia | loso | - | - | - | - | - | - | - | - | - | 29 | 42.45 | -38.40 | 1.59 | - | - | - | - |
| 2022_06_spain | in | 18.00 | 48.77 | -12.01 | 12.57 | -9.57 | 13.03 | 0.13 | 1.32 | - | 69 | 3.67 | -2.31 | 0.26 | 49.00 | 0.51 | -0.18 | 0.10 |
| 2022_06_spain | lodo | 18.00 | 53.35 | -14.50 | 13.64 | -10.49 | 14.56 | 0.13 | 1.35 | - | 69 | 4.29 | -2.55 | 0.29 | 49.00 | 0.52 | -0.17 | 0.14 |
| 2022_06_spain | loso | - | - | - | - | - | - | - | - | - | 67 | 21.35 | 13.15 | 0.96 | - | - | - | - |
| 2022_08_azerbaijan | in | 16.00 | 43.21 | 28.43 | 34.65 | 34.65 | 14.92 | 7.06 | 10.81 | 10.44 | 40 | 12.39 | 9.57 | 0.72 | 23.00 | 0.50 | -0.24 | 0.17 |
| 2022_08_azerbaijan | lodo | 16.00 | 44.97 | 28.25 | 34.76 | 34.76 | 14.80 | 7.02 | 10.90 | 10.45 | 40 | 12.32 | 9.64 | 0.72 | 23.00 | 0.50 | -0.25 | 0.16 |
| 2022_08_azerbaijan | loso | - | - | - | - | - | - | - | - | - | 40 | 12.47 | 9.56 | 0.74 | - | - | - | - |
| 2022_09_canada | in | 17.00 | 17.40 | 7.09 | 44.45 | 6.43 | 44.07 | 3.28 | 5.49 | 1.24 | 47 | 17.38 | 0.82 | 0.79 | 30.00 | 0.42 | -0.05 | 0.13 |
| 2022_09_canada | lodo | 17.00 | 17.48 | 7.01 | 44.44 | 5.69 | 44.10 | 3.25 | 5.52 | 1.24 | 47 | 17.27 | 0.56 | 0.79 | 30.00 | 0.45 | -0.05 | 0.15 |
| 2022_09_canada | loso | - | - | - | - | - | - | - | - | - | 47 | 17.83 | 0.64 | 0.81 | - | - | - | - |
| 2022_12_france | in | 16.00 | 23.24 | 11.75 | 22.54 | 11.72 | 17.52 | 1.60 | 6.75 | 5.81 | 36 | 10.99 | 5.01 | 0.50 | 20.00 | 0.31 | -0.01 | 0.03 |
| 2022_12_france | lodo | 16.00 | 24.40 | 12.84 | 23.02 | 12.20 | 17.80 | 1.94 | 7.04 | 5.85 | 36 | 11.15 | 5.04 | 0.51 | 20.00 | 0.35 | -0.02 | 0.01 |
| 2022_12_france | loso | - | - | - | - | - | - | - | - | - | 36 | 11.02 | 4.65 | 0.53 | - | - | - | - |
| 2022_20_mexico | in | 19.00 | 57.22 | 8.91 | 21.62 | 5.36 | 20.86 | 0.13 | 1.39 | 11.63 | 41 | 11.42 | 2.00 | 0.39 | 21.00 | 0.47 | -0.26 | 0.09 |
| 2022_20_mexico | lodo | 19.00 | 58.96 | 9.08 | 21.60 | 5.12 | 20.88 | 0.13 | 1.43 | 11.61 | 41 | 11.58 | 1.90 | 0.40 | 21.00 | 0.48 | -0.25 | 0.10 |
| 2022_20_mexico | loso | - | - | - | - | - | - | - | - | - | 41 | 13.85 | 2.77 | 0.47 | - | - | - | - |
| 2022_21_brazil | in | 17.00 | 74.97 | -74.97 | 70.05 | -69.10 | 33.85 | -1.00 | 2.49 | -5.94 | 59 | 27.79 | -22.22 | 1.95 | 39.00 | 0.39 | -0.16 | 0.09 |
| 2022_21_brazil | lodo | 17.00 | 74.99 | -74.99 | 69.93 | -68.90 | 33.39 | -0.97 | 2.54 | -5.93 | 59 | 27.74 | -22.18 | 1.95 | 39.00 | 0.40 | -0.17 | 0.09 |
| 2022_21_brazil | loso | - | - | - | - | - | - | - | - | - | 59 | 28.11 | -22.29 | 1.97 | - | - | - | - |
| 2022_22_united_arab_emirates | in | 19.00 | 39.77 | -10.26 | 15.14 | -12.79 | 14.86 | -3.99 | 4.45 | - | 49 | 4.50 | -3.39 | 0.24 | 30.00 | 0.44 | -0.19 | 0.14 |
| 2022_22_united_arab_emirates | lodo | 19.00 | 41.95 | -10.16 | 15.21 | -12.50 | 15.30 | -3.77 | 4.52 | - | 49 | 4.73 | -3.41 | 0.25 | 30.00 | 0.46 | -0.20 | 0.15 |
| 2022_22_united_arab_emirates | loso | - | - | - | - | - | - | - | - | - | 49 | 6.19 | -3.57 | 0.31 | - | - | - | - |
| 2023_01_bahrain | in | 17.00 | 33.30 | 17.90 | 21.84 | 14.17 | 14.05 | 2.31 | 4.20 | 10.54 | 62 | 5.66 | 1.39 | 0.42 | 42.00 | 0.78 | -0.23 | 0.11 |
| 2023_01_bahrain | lodo | 17.00 | 35.04 | 18.25 | 21.40 | 14.55 | 13.60 | 2.39 | 4.30 | 10.52 | 62 | 5.62 | 1.43 | 0.41 | 42.00 | 0.83 | -0.25 | 0.10 |
| 2023_01_bahrain | loso | - | - | - | - | - | - | - | - | - | 61 | 7.38 | 1.54 | 0.55 | - | - | - | - |
| 2023_07_spain | in | 20.00 | 25.55 | -7.05 | 8.44 | -7.57 | 6.84 | -0.58 | 1.10 | - | 63 | 3.44 | -2.14 | 0.19 | 42.00 | 0.56 | -0.37 | 0.12 |
| 2023_07_spain | lodo | 20.00 | 27.52 | -5.93 | 9.37 | -6.68 | 7.91 | -0.55 | 1.16 | - | 63 | 3.97 | -1.87 | 0.21 | 42.00 | 0.59 | -0.38 | 0.11 |
| 2023_07_spain | loso | - | - | - | - | - | - | - | - | - | 62 | 4.97 | -1.72 | 0.26 | - | - | - | - |
| 2024_07_emilia_romagna | in | 19.00 | 44.24 | -8.14 | 11.50 | -8.86 | 8.77 | 0.05 | 1.10 | - | 42 | 6.18 | -4.95 | 0.24 | 22.00 | 0.35 | -0.23 | 0.01 |
| 2024_07_emilia_romagna | lodo | 19.00 | 46.25 | -7.92 | 11.27 | -8.56 | 9.07 | 0.23 | 1.26 | - | 42 | 6.37 | -4.91 | 0.24 | 22.00 | 0.33 | -0.22 | -0.00 |
| 2024_07_emilia_romagna | loso | - | - | - | - | - | - | - | - | - | 42 | 8.05 | -5.03 | 0.31 | - | - | - | - |

Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:

| variant | compound | 1-5 | 11-15 | 16-20 | 21-25 | 26-30 | 31-35 | 36-40 | 41+ | 6-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| in | HARD | -0.03 | 0.05 | 0.04 | 0.00 | 0.01 | 0.00 | 0.04 | 0.12 | 0.04 |
| in | MEDIUM | -0.01 | 0.02 | 0.01 | 0.01 | 0.09 | -0.00 | 0.05 | 0.15 | 0.01 |
| in | SOFT | -0.00 | 0.11 | 0.02 | 0.02 | 0.10 | 0.06 | -0.07 | - | 0.02 |
| lodo | HARD | -0.03 | 0.05 | 0.04 | 0.00 | 0.02 | 0.03 | 0.03 | 0.03 | 0.04 |
| lodo | MEDIUM | -0.01 | 0.02 | 0.01 | 0.01 | 0.08 | -0.01 | 0.12 | 0.17 | 0.01 |
| lodo | SOFT | -0.01 | 0.11 | 0.02 | 0.02 | 0.11 | 0.01 | -0.32 | - | 0.01 |

Neutralised laps:

| variant | kind | n | mean_resid_s | mae_s |
|---|---|---|---|---|
| in | sc | 682 | 3.70 | 14.56 |
| in | vsc | 350 | -9.71 | 11.93 |
| lodo | sc | 682 | 3.70 | 14.55 |
| lodo | vsc | 350 | -9.71 | 11.93 |
| loso | sc | 682 | 3.72 | 14.55 |
| loso | vsc | 350 | -9.72 | 11.93 |

A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus the R8 margin):

| race_id | field_mode | ref_mode | ref_stop_counts | match_field_mode | match_own_stops | lap0_beyond_raced | lap0_beyond_raced_margin | grid_refs | grid_beyond_raced | grid_beyond_raced_margin | panel_gap_to_ref_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 1 | 1 | 1:17 | 1.00 | 0.47 | 1.00 | 1.00 | 54 | 0.30 | 0.30 | 0.44 |
| 2021_01_bahrain | 2 | 2 | 2:16 | 1.00 | 0.81 | 1.00 | 1.00 | 60 | 0.60 | 0.60 | 1.75 |
| 2021_04_spain | 2 | 1 | 1:19 | 0.00 | 0.11 | 1.00 | 1.00 | 72 | 0.38 | 0.38 | 0.65 |
| 2021_09_austrian | 1 | 1 | 1:19 | 1.00 | 0.58 | 0.21 | 0.21 | 78 | 0.03 | 0.03 | 0.30 |
| 2021_20_qatar | 1 | 1 | 1:18 | 1.00 | 0.50 | 1.00 | 1.00 | 60 | 0.52 | 0.52 | 0.56 |
| 2021_22_united_arab_emirates | 2 | 1 | 1:14 | 0.00 | 0.43 | 1.00 | 1.00 | 60 | 0.40 | 0.40 | 0.45 |
| 2022_01_bahrain | 3 | 2 | 2:18 | 0.00 | 0.06 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 2.43 |
| 2022_02_saudi_arabia | 1 | 1 | 1:14 | 1.00 | 0.93 | 0.00 | 0.00 | 54 | 0.00 | 0.00 | 0.18 |
| 2022_06_spain | 3 | 2 | 2:18 | 0.00 | 0.22 | 0.00 | 0.00 | 72 | 0.00 | 0.00 | 3.37 |
| 2022_08_azerbaijan | 2 | 2 | 1:4 2:11 | 0.73 | 0.67 | 1.00 | 1.00 | 54 | 0.33 | 0.33 | 0.20 |
| 2022_09_canada | 2 | 1 | 1:17 | 0.00 | 0.24 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 0.90 |
| 2022_12_france | 1 | 1 | 1:15 | 1.00 | 0.80 | 0.80 | 0.80 | 54 | 0.28 | 0.28 | 0.34 |
| 2022_20_mexico | 1 | 1 | 1:18 | 1.00 | 0.83 | 0.00 | 0.00 | 78 | 0.00 | 0.00 | 0.48 |
| 2022_21_brazil | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 78 | 0.01 | 0.01 | 0.96 |
| 2022_22_united_arab_emirates | 2 | 1 | 1:16 2:1 | 0.06 | 0.35 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 0.30 |
| 2023_01_bahrain | 2 | 2 | 2:17 | 1.00 | 0.53 | 0.00 | 0.00 | 60 | 0.00 | 0.00 | 1.34 |
| 2023_07_spain | 2 | 1 | 1:20 | 0.00 | 0.00 | 0.00 | 0.00 | 72 | 0.06 | 0.06 | 0.31 |
| 2024_07_emilia_romagna | 1 | 2 | 2:19 | 0.00 | 0.11 | 1.00 | 1.00 | 66 | 0.67 | 0.67 | 1.01 |

A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as configured (order: pace insample; positive d_close_pair_acc is better):

| substitute | variant | replaces | with | races_failed | close_pairs | close_pairs_base | d_close_pair_acc | d_spearman | d_rel_gap_mae_s | d_race_pace_mae_s | d_race_pace_bias_s | d_stop_gain_bias_s | d_pit_err_per_stop_s | d_stint_mae_s | d_match_field_mode | d_lap0_beyond_raced | d_lap0_beyond_raced_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_race_fit | in | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.019 | -0.014 | 2.023 | 0.687 | 0.660 | 0.068 | -0.803 | 3.137 | 0.075 | -0.011 | -0.011 |
| pre_race_fit | lodo | fit_window | practice long runs plus the pre_race base delta (no race laps) |  | 365 | 365 | -0.011 | -0.013 | 1.818 | 0.856 | 0.299 | 0.095 | -0.873 | 2.991 | - | - | - |
| pit_loss_prior | in | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | 0.008 | 0.001 | 0.131 | -0.574 | -1.275 | 0.000 | -0.692 | 0.000 | 0.000 | 0.000 | 0.000 |
| pit_loss_prior | lodo | pit_loss | the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else across circuits, else 22 s) |  | 365 | 365 | -0.003 | -0.000 | 0.110 | -0.639 | -1.325 | 0.000 | -0.719 | 0.000 | - | - | - |
| cliff_history_only | in | cliff_edge | the wear-cliff edge from the historical prior only, nothing from this race |  | 365 | 365 | 0.019 | -0.002 | 0.498 | 0.187 | 1.780 | 0.000 | 0.126 | 0.150 | -0.053 | -0.139 | -0.139 |
| cliff_history_only | lodo | cliff_edge | the wear-cliff edge from the historical prior only, nothing from this race |  | 365 | 365 | 0.025 | -0.001 | 0.402 | 0.213 | 1.749 | 0.000 | 0.125 | 0.158 | - | - | - |
| proxy_off | in | proxy_raced | no compound proxy: only raced compounds are priced |  | 365 | 365 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.056 | -0.093 | -0.093 |
| proxy_off | lodo | proxy_raced | no compound proxy: only raced compounds are priced |  | 359 | 365 | 0.000 | -0.002 | -0.104 | -0.138 | 0.230 | 0.001 | 0.100 | 0.003 | - | - | - |
| all_substituted | in | every whole-race input above | every substitute together |  | 251 | 365 | -0.002 | -0.016 | 3.220 | -1.335 | -0.973 | 0.102 | -1.226 | 3.450 | 0.049 | -0.123 | -0.123 |
| all_substituted | lodo | every whole-race input above | every substitute together |  | 251 | 365 | 0.003 | -0.015 | 2.899 | -1.324 | -1.218 | 0.131 | -1.277 | 3.310 | - | - | - |

A6: hindsight ledger (judge and predictive):

| role | id | input | tag | substitute | source |
|---|---|---|---|---|---|
| judge | fit_window | degradation fit rows (fit_window full) | this race, whole | pre_race_fit | derived |
| judge | clean_flag | lap filter (clean flag, pace residual, driver offset) | this race, whole | - | derived |
| judge | ref_lap_s | SC lap time base ref_lap_s (median clean lap of the fit rows) | this race, whole | - | derived |
| judge | pit_loss | pit loss (Simulator.race_pit_loss_s, eval.score.world_pit_loss) | this race, whole | pit_loss_prior | derived |
| judge | race_length | race length (scheduled distance: session.total_laps in the scenario snapshots, calibration.snapshot_as_of) | regulation | - | derived |
| judge | compound_list | compound list (RaceParams.compounds: nominated) | regulation | - | derived |
| judge | cliff_edge | wear-cliff support edge (world_edge history) | this race, whole | cliff_history_only | derived |
| judge | cliff_prior | tyre-life history prior (tyre_history) | history | - | derived |
| judge | cliff_shape | cliff cost min(0.05 o^2, 4.0) s/lap past the edge | a priori | - | derived |
| judge | proxy_raced | compound proxy: which compounds count as raced | this race, whole | proxy_off | derived |
| judge | proxy_steps | compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json) | history | - | derived |
| judge | proxy_weekend | compound proxy tiers 2-3 (practice long runs, qualifying) | weekend (pre-race) | - | derived |
| judge | events | neutralisations (events none) | a priori | - | derived |
| judge | event_effects | SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing (EventEffects) | history | - | derived |
| judge | practice | practice long runs and qualifying (this weekend) | weekend (pre-race) | - | derived |
| judge | driver_effects | driver effects (race) in the degradation fit | this race, whole | - | derived |
| judge | hinge | wear hinge at age 25 (knots default) | a priori | - | derived |
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
| predictive | calibrator | calibrator shrinkage | a priori | - | derived |
| predictive | wear_lower_bound | wear bounded below at 0.0 | a priori | - | derived |
| predictive | draws | parameter draws (posterior) and lap noise (residual) | this race, to snapshot | - | derived |
| predictive | min_stint_two_compounds | R1-R3: minimum stint, two dry compounds | regulation | - | derived |

## B. Predictive versus judge

Plan panel per snapshot: the real remaining strategy, stay out, one stop at 4 laps spread from the soonest legal stop on each compound both model, two stops at 3 first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) over the plans legal under both configs' own rules. Snapshots every 5 laps for 6 sampled drivers per race. 20 draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.

| race_id | snapshots | rank_rho | regret_s | same_favourite | fav_b_illegal_w | pit_now_agree | rank_rho_green | regret_green_s | plans | b1_mae_s | b1_green_mae_s | b1_green_bias_s | b1_bias_s | b1_mae_per_lap_s | cov_68 | cov_90 | gap_cov_68 | gap_cov_90 | factor_cov_68 | factor_cov_90 | b4_mae_s | b4_bias_s | b4_mae_offset_s | b4_bias_offset_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | 54 | 0.811 | 4.226 | 0.481 | 0.000 | 0.792 | 0.834 | 3.373 | 1931 | 99.796 | 27.290 | 27.278 | 99.796 | 3.328 | 0.141 | 0.324 | 0.582 | 0.667 | 0.368 | 0.577 | 71.255 | 65.812 | 33.599 | 24.770 |
| 2021_01_bahrain | 60 | 0.681 | 6.270 | 0.367 | 0.000 | 0.828 | 0.707 | 6.183 | 2127 | 107.304 | 26.994 | 19.975 | 107.304 | 3.427 | 0.283 | 0.513 | 0.417 | 0.629 | 0.188 | 0.310 | 60.034 | 37.613 | 13.382 | -7.808 |
| 2021_04_spain | 72 | 0.897 | 2.901 | 0.319 | 0.000 | 0.620 | 0.918 | 3.137 | 2544 | 90.607 | 21.516 | 7.708 | 90.605 | 2.570 | 0.513 | 0.581 | 0.678 | 0.788 | 0.417 | 0.583 | 38.837 | 14.568 | 23.998 | 12.967 |
| 2021_09_austrian | 78 | 0.858 | 3.976 | 0.744 | 0.000 | 0.895 | 0.866 | 3.759 | 2707 | 75.801 | 22.011 | 4.488 | 75.801 | 1.952 | 0.618 | 0.734 | 0.640 | 0.826 | 0.359 | 0.520 | 34.661 | 8.132 | 23.055 | -0.912 |
| 2021_20_qatar | 60 | 0.808 | 5.141 | 0.683 | 0.000 | 0.931 | 0.812 | 6.709 | 2140 | 107.636 | 36.802 | 36.024 | 107.636 | 3.281 | 0.134 | 0.223 | 0.413 | 0.619 | 0.424 | 0.552 | 74.690 | 70.403 | 65.726 | 65.017 |
| 2021_22_united_arab_emirates | 60 | 0.879 | 7.822 | 0.467 | 0.000 | 0.678 | 0.880 | 7.499 | 2150 | 89.637 | 21.199 | 19.634 | 89.637 | 2.746 | 0.367 | 0.486 | 0.515 | 0.708 | 0.493 | 0.764 | 60.439 | 51.162 | 51.516 | 35.765 |
| 2022_01_bahrain | 60 | 0.837 | 5.964 | 0.333 | 0.000 | 0.852 | 0.856 | 5.711 | 2169 | 111.294 | 26.226 | 17.020 | 111.294 | 4.702 | 0.230 | 0.412 | 0.400 | 0.579 | 0.269 | 0.548 | 89.293 | 86.041 | 104.035 | 90.807 |
| 2022_02_saudi_arabia | 54 | 0.764 | 2.468 | 0.556 | 0.000 | 0.741 | 0.792 | 2.188 | 1779 | 104.116 | 26.270 | 25.064 | 104.116 | 4.135 | 0.188 | 0.271 | 0.318 | 0.609 | 0.582 | 0.685 | 35.054 | 9.834 | 25.192 | 10.491 |
| 2022_06_spain | 72 | 0.826 | 6.072 | 0.486 | 0.000 | 0.884 | 0.823 | 4.648 | 2654 | 88.037 | 18.165 | 8.582 | 87.561 | 2.618 | 0.367 | 0.578 | 0.600 | 0.770 | 0.288 | 0.506 | 57.365 | -26.157 | 23.602 | 13.833 |
| 2022_08_azerbaijan | 54 | 0.675 | 9.425 | 0.352 | 0.000 | 0.731 | 0.687 | 9.055 | 1858 | 105.898 | 21.186 | 13.932 | 105.898 | 3.970 | 0.194 | 0.347 | 0.362 | 0.547 | 0.412 | 0.698 | 46.450 | 41.166 | 31.608 | 26.093 |
| 2022_09_canada | 78 | 0.768 | 3.526 | 0.577 | 0.000 | 0.735 | 0.775 | 3.411 | 2758 | 90.348 | 22.515 | -2.812 | 87.472 | 2.489 | 0.449 | 0.653 | 0.690 | 0.826 | 0.617 | 0.777 | 35.531 | 16.004 | 39.629 | -1.280 |
| 2022_12_france | 54 | 0.904 | 4.518 | 0.352 | 0.000 | 0.811 | 0.906 | 4.952 | 1983 | 95.304 | 22.592 | 22.405 | 95.304 | 3.198 | 0.295 | 0.422 | 0.677 | 0.800 | 0.585 | 0.646 | 38.526 | 30.351 | 40.660 | 37.175 |
| 2022_20_mexico | 78 | 0.876 | 4.688 | 0.282 | 0.000 | 0.587 | 0.877 | 4.688 | 2723 | 88.453 | 20.420 | 1.785 | 88.453 | 2.473 | 0.599 | 0.755 | 0.499 | 0.648 | 0.260 | 0.364 | 55.883 | 31.375 | 35.420 | 22.342 |
| 2022_21_brazil | 78 | 0.723 | 11.853 | 0.731 | 0.000 | 0.778 | 0.720 | 11.785 | 2766 | 106.611 | 31.817 | 21.693 | 106.132 | 3.087 | 0.080 | 0.133 | 0.443 | 0.669 | 0.168 | 0.253 | 36.769 | -29.159 | 31.890 | -13.082 |
| 2022_22_united_arab_emirates | 60 | 0.868 | 6.715 | 0.200 | 0.000 | 0.732 | 0.871 | 7.153 | 2243 | 81.864 | 20.404 | 17.007 | 81.864 | 2.468 | 0.461 | 0.530 | 0.447 | 0.604 | 0.171 | 0.343 | 40.978 | 19.425 | 26.171 | 8.924 |
| 2023_01_bahrain | 60 | 0.769 | 6.430 | 0.467 | 0.000 | 0.875 | 0.777 | 5.776 | 2216 | 95.785 | 22.066 | 16.237 | 95.675 | 2.998 | 0.351 | 0.528 | 0.657 | 0.851 | 0.252 | 0.412 | 45.914 | 40.756 | 43.769 | 40.158 |
| 2023_07_spain | 72 | 0.796 | 6.125 | 0.278 | 0.000 | 0.794 | 0.810 | 6.294 | 2572 | 93.192 | 20.081 | 19.092 | 93.192 | 2.674 | 0.271 | 0.449 | 0.536 | 0.698 | 0.222 | 0.431 | 28.768 | 21.903 | 19.944 | 17.161 |
| 2024_07_emilia_romagna | 66 | 0.812 | 13.255 | 0.273 | 0.000 | 0.636 | 0.812 | 13.519 | 2351 | 76.407 | 18.271 | 14.448 | 76.407 | 2.169 | 0.420 | 0.511 | 0.340 | 0.492 | 0.357 | 0.513 | 38.163 | -6.249 | 26.453 | 8.636 |
| pooled | 1170 | 0.810 | 6.198 | 0.447 | 0.000 | 0.769 | 0.819 | 6.110 | 41671 | 94.210 | 23.511 | 15.241 | 93.952 | 2.951 | 0.342 | 0.481 | 0.520 | 0.692 | 0.353 | 0.520 | 49.430 | 26.692 | 36.669 | 22.242 |

B1 by horizon and by race phase:

| horizon_bucket | n | mae_s | bias_s |
|---|---|---|---|
| 0.00 | 1837.00 | 24.88 | 24.88 |
| 10.00 | 8653.00 | 41.99 | 41.99 |
| 20.00 | 8588.00 | 62.74 | 62.74 |
| 30.00 | 8312.00 | 98.32 | 98.32 |
| 40.00 | 7630.00 | 130.98 | 130.96 |
| 50.00 | 4468.00 | 169.58 | 168.21 |
| 60.00 | 2183.00 | 184.90 | 182.82 |

| phase | n | mae_s | bias_s |
|---|---|---|---|
| early | 12621 | 161.57 | 160.71 |
| late | 11940 | 43.31 | 43.31 |
| mid | 17110 | 80.05 | 80.04 |

B1 by plan kind:

| kind | n | mae_s | bias_s |
|---|---|---|---|
| one | 12150 | 88.55 | 88.13 |
| real | 1116 | 98.27 | 98.27 |
| stay | 319 | 90.09 | 90.09 |
| two | 28086 | 96.55 | 96.34 |

B5 future blindness: {"status": "pass", "new_leak_differences": 0, "differences_with_known_leaks": 0, "known_leaks": [], "fixed_leaks": ["compound_list", "frame_filter", "gap_behind", "lap_window", "pit_loss", "race_length", "stint_limit"], "detail": ""}

## D. Component targets (pooled over races)

Each component's prediction from allowed information against the whole-race measurement of the same quantity. err = pred - target; coverage only where the predictor states an sd.

| component | source | target_kind | n | races | mae | bias | cov_68 | cov_90 |
|---|---|---|---|---|---|---|---|---|
| tyre_life | history prior (tyre_life_prior) | oldest age run | 50 | 18 | 7.542 | -1.450 | 0.660 | 0.880 |
| tyre_life | history prior (tyre_life_prior) | median stint-end age | 50 | 18 | 9.216 | 8.540 | 0.580 | 0.860 |
| tyre_life | history prior (tyre_life_prior) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.513 | 1.146 | 0.826 | 0.935 |
| tyre_life | judge cliff edge (history) | oldest age run | 50 | 18 | 3.013 | 3.013 | - | - |
| tyre_life | judge cliff edge (history) | median stint-end age | 50 | 18 | 13.003 | 13.003 | - | - |
| tyre_life | judge cliff edge (history) | observed edge (tyre_history.observed_tyre_life) | 46 | 18 | 5.997 | 5.997 | - | - |
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
| wear_rate | belief @k0 | within-stint slope less fuel | 49 | 18 | 0.046 | 0.005 | 0.449 | 0.673 |
| wear_rate | belief @k10 | within-stint slope less fuel | 49 | 18 | 0.046 | -0.012 | 0.408 | 0.653 |
| wear_rate | belief @k20 | within-stint slope less fuel | 49 | 18 | 0.042 | -0.009 | 0.347 | 0.469 |
| wear_rate | belief @k(mid) | within-stint slope less fuel | 49 | 18 | 0.034 | -0.013 | 0.347 | 0.490 |
| wear_rate | judge (full fit) | within-stint slope less fuel | 49 | 18 | 0.012 | -0.008 | - | - |
| fresh_tyre_gain | belief @k0 | real gain at the stop | 556 | 18 | 0.819 | -0.012 | - | - |
| fresh_tyre_gain | belief @k10 | real gain at the stop | 521 | 18 | 1.066 | -0.249 | - | - |
| fresh_tyre_gain | belief @k20 | real gain at the stop | 351 | 18 | 0.879 | 0.034 | - | - |
| fresh_tyre_gain | belief @k(mid) | real gain at the stop | 233 | 18 | 0.662 | -0.039 | - | - |
| compound_gap | belief @k0 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.803 | -0.310 | - | - |
| compound_gap | belief @k0 | qualifying | 7 | 7 | 0.706 | 0.605 | - | - |
| compound_gap | belief @k10 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.640 | -0.263 | - | - |
| compound_gap | belief @k10 | qualifying | 7 | 7 | 0.720 | 0.524 | - | - |
| compound_gap | belief @k20 | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.513 | 0.071 | - | - |
| compound_gap | belief @k20 | qualifying | 7 | 7 | 0.709 | 0.473 | - | - |
| compound_gap | belief @k(mid) | race (judge full fit, fresh tyre, mid-race) | 46 | 18 | 0.476 | 0.044 | - | - |
| compound_gap | belief @k(mid) | qualifying | 7 | 7 | 0.839 | 0.581 | - | - |
| fuel | prior (config) | race (judge full fit) | 18 | 18 | 0.020 | 0.017 | - | - |
| fuel | belief @k0 | race (judge full fit) | 18 | 18 | 0.020 | 0.017 | 0.222 | 0.444 |
| fuel | belief @k10 | race (judge full fit) | 18 | 18 | 0.019 | 0.015 | 0.222 | 0.444 |
| fuel | belief @k20 | race (judge full fit) | 18 | 18 | 0.013 | 0.009 | 0.222 | 0.389 |
| fuel | belief @k(mid) | race (judge full fit) | 18 | 18 | 0.011 | 0.005 | 0.167 | 0.278 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 10 | 10 | 2.628 | -1.236 | 0.500 | 0.600 |
| pit_loss | prior from earlier races (cross_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 10 | 10 | 3.115 | -2.291 | 0.500 | 0.700 |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 2.483 | -1.864 | - | - |
| pit_loss | constant (calibrate.DEFAULT_PIT_LOSS_S) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 3.454 | -2.424 | - | - |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.909 | -0.929 | 0.778 | 0.833 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.834 | -1.489 | 0.667 | 0.778 |
| pit_loss | as-of estimate @k0 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.283 | 7.037 | 0.200 | 0.300 |
| pit_loss | belief charged @k0 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.879 | -0.822 | - | - |
| pit_loss | belief charged @k0 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.779 | -1.382 | - | - |
| pit_loss | belief charged @k0 | whole race, median SC stop | 10 | 10 | 13.295 | 7.097 | - | - |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.714 | -0.489 | 0.833 | 0.833 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.619 | -1.049 | 0.667 | 0.778 |
| pit_loss | as-of estimate @k10 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 13.359 | 7.280 | 0.300 | 0.400 |
| pit_loss | belief charged @k10 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 1.685 | -0.387 | - | - |
| pit_loss | belief charged @k10 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.556 | -0.947 | - | - |
| pit_loss | belief charged @k10 | whole race, median SC stop | 10 | 10 | 13.373 | 7.335 | - | - |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.789 | -0.333 | 0.889 | 1.000 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.959 | -0.893 | 0.611 | 0.722 |
| pit_loss | as-of estimate @k20 (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.614 | 7.330 | 0.100 | 0.500 |
| pit_loss | belief charged @k20 | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.778 | -0.262 | - | - |
| pit_loss | belief charged @k20 | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 1.922 | -0.822 | - | - |
| pit_loss | belief charged @k20 | whole race, median SC stop | 10 | 10 | 12.614 | 7.366 | - | - |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.655 | -0.329 | 0.944 | 1.000 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.068 | -0.889 | 0.611 | 0.778 |
| pit_loss | as-of estimate @k(mid) (pit_loss.estimate) | whole race, median SC stop | 10 | 10 | 12.407 | 7.228 | 0.200 | 0.500 |
| pit_loss | belief charged @k(mid) | whole race, robust (pit_loss.whole_race: median green stop) | 18 | 18 | 0.655 | -0.263 | - | - |
| pit_loss | belief charged @k(mid) | whole race, replay mean (calibrate._pit_loss) | 18 | 18 | 2.024 | -0.823 | - | - |
| pit_loss | belief charged @k(mid) | whole race, median SC stop | 10 | 10 | 12.406 | 7.262 | - | - |
| lap1_and_warmup | prior (config) | observed (median lap-1 residual + prior) | 18 | 18 | 2.083 | 1.356 | - | - |
| lap1_and_warmup | model (no warm-up term: 0) | median residual of the first lap after the out-lap | 18 | 18 | 0.174 | 0.096 | - | - |
| traffic_rejoin | judge rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.592 | 0.127 | - | - |
| traffic_rejoin | judge rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 577 | 18 | 1.111 | 0.092 | - | - |
| traffic_rejoin | belief rejoin position (horizon 2) | real position after the out-lap | 584 | 18 | 0.630 | 0.089 | - | - |
| traffic_rejoin | belief rejoin position (horizon 5, traffic_accuracy) | real position three laps later | 577 | 18 | 1.338 | -0.180 | - | - |
| neutralisations | judge expected (none) | occurred | 72 | 18 | 1.417 | -1.417 | - | - |
| neutralisations | judge config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | judge config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralisations | belief expected (rate) | occurred | 72 | 18 | 1.355 | 0.405 | - | - |
| neutralisations | belief config | observed (mean SC lap / ref lap) | 10 | 10 | 0.082 | -0.056 | - | - |
| neutralisations | belief config | observed (median VSC lap / green prediction) | 11 | 11 | 0.129 | 0.127 | - | - |
| neutralised_laps | judge replay (factors) | mean real lap time on the cars' neutralised laps | 21 | 14 | 9.300 | 3.628 | - | - |
| neutralised_laps | belief replay (factors) | mean real lap time on the cars' neutralised laps | 21 | 14 | 9.945 | 5.199 | - | - |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, robust (pit_loss.whole_race: median green stop) | 6 | 6 | 0.916 | -0.671 | 1.000 | 1.000 |
| pit_loss | prior from earlier races (same_circuit_asof) | whole race, replay mean (calibrate._pit_loss) | 6 | 6 | 2.627 | 0.031 | 0.167 | 0.833 |
| pit_loss | prior from earlier races (constant) | whole race, robust (pit_loss.whole_race: median green stop) | 2 | 2 | 1.291 | -0.171 | 1.000 | 1.000 |
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
