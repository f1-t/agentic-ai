# Simulator fidelity ablation

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

Races: 2021_04_spain, 2023_07_spain, 2023_01_bahrain, 2019_10_united_kingdom, 2022_02_saudi_arabia, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (driver_effects_hinge, wear_lower_bound_judge). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over 6 races, so intervals are wide). Run time 15 s.

Factors:

- `driver_effects`: one pace offset per driver in every degradation fit (configs/sim/arms/driver_effects.yaml) {"both": {"degradation": {"params": {"driver_effects": "race"}}}}
- `hinge`: linear_fuel_hinge, wear steepens past the default knot {"both": {"degradation": {"name": "linear_fuel_hinge"}}}
- `driver_effects_hinge`: both of the above {"both": {"degradation": {"name": "linear_fuel_hinge", "params": {"driver_effects": "race"}}}}
- `wear_lower_bound_judge`: wear bounded below at 0 in the judge too (the belief already has it) {"world": {"degradation": {"params": {"wear_lower_bound": 0.0}}}}
- `no_wear_lower_bound_belief`: the belief without its wear bound {"belief": {"degradation": {"params": {"wear_lower_bound": null}}}}
- `compound_proxy_off`: no proxy pricing of unraced compounds, judge and belief {"both": {"compound_proxy": {"enabled": false}}}
- `stint_limit_off`: R8 stint limit off, judge and belief {"both": {"stint_limit_margin_laps": null}}
- `learned_ridge`: the belief's calibrator is the learned ridge (SPEC 0024), trained per race on the other races of the run {"belief": {"calibrator": {"name": "ridge"}}}

Measures (primary first; direction: + better for acc/rho/agree, - better for errors):

- `A_close_pair_acc`: judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order
- `A_spearman`: same, Spearman of simulated against real finishing order
- `A_rel_gap_mae_s`: same, mean |time error - the race's mean error| over cars on the lead lap (relative gap error)
- `B2_rank_rho`: Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots
- `B2_regret_s`: judge time of the predictive favourite less the judge favourite, mean over snapshots
- `B2_pit_now_agree`: share of snapshots where predictive and judge agree on pit-now against stay
- `P_close_pair_acc`: predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k
- `P_spearman`: same, Spearman, mean over k
- `A_abs_mae_s`: judge LODO pace-adjusted whole-race time, mean |error|
- `A_abs_bias_s`: same, mean error
- `A_stop_gain_bias_s`: judge LODO predicted less real gain at real stops
- `B1_mae_s`: predictive less judge remaining time over the panel, mean |error|
- `B1_bias_s`: same, mean
- `B1_green_mae_s`: B1 with no SC/VSC on either side (the physics difference without the belief's event model)
- `B2_rank_rho_green`: B2 rank agreement with no SC/VSC on either side
- `B3_cov_90`: share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90)
- `B3_factor_cov_90`: share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90)
- `B4_mae_offset_s`: predictive replay of the real remaining strategy against reality, pace-offset error (G)

| factor | A_close_pair_acc | A_spearman | A_rel_gap_mae_s | B2_rank_rho | B2_regret_s | B2_pit_now_agree | P_close_pair_acc | P_spearman | A_abs_mae_s | A_abs_bias_s | A_stop_gain_bias_s | B1_mae_s | B1_bias_s | B1_green_mae_s | B2_rank_rho_green | B3_cov_90 | B3_factor_cov_90 | B4_mae_offset_s | ledger_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.644 | 0.967 | 8.883 | 0.748 | 2.002 | 0.846 | 0.602 | 0.888 | 27.056 | -16.967 | -0.336 | 95.350 | 95.341 | 21.419 | 0.766 | 0.302 | 0.533 | 27.821 |  |
| driver_effects | +0.048 [+0.012, +0.077] | +0.003 [-0.000, +0.007] | -0.453 [-0.703, -0.186] | -0.032 [-0.121, +0.063] | +0.237 [-1.230, +1.734] | +0.006 [-0.066, +0.077] | +0.062 [+0.034, +0.087] | +0.014 [-0.002, +0.033] | -0.172 [-0.631, +0.283] | -0.277 [-0.472, +0.022] | +0.135 [-0.029, +0.230] | -3.011 [-5.009, -1.135] | -3.008 [-5.009, -1.127] | -2.933 [-4.668, -1.611] | -0.027 [-0.111, +0.072] | -0.038 [-0.107, +0.017] | -0.172 [-0.264, -0.092] | -2.992 [-5.337, -0.403] | +j:driver_effects(this race, whole) +p:driver_effects(this race, to snapshot) |
| hinge | +0.000 [-0.065, +0.052] | -0.001 [-0.005, +0.003] | -0.007 [-0.168, +0.242] | -0.042 [-0.084, +0.005] | +0.080 [-0.265, +0.385] | -0.023 [-0.056, +0.013] | -0.002 [-0.007, +0.000] | +0.003 [-0.000, +0.009] | -0.032 [-0.454, +0.399] | +0.128 [-0.350, +0.590] | -0.010 [-0.107, +0.085] | -1.360 [-3.369, +0.458] | -1.361 [-3.371, +0.458] | -0.183 [-1.402, +0.872] | -0.033 [-0.067, +0.011] | +0.030 [+0.012, +0.049] | +0.011 [-0.021, +0.045] | -0.085 [-0.263, +0.062] | +j:hinge(a priori) +p:hinge(a priori) |
| driver_effects_hinge | -0.019 [-0.074, +0.058] | -0.001 [-0.007, +0.006] | -0.405 [-0.760, -0.037] | -0.050 [-0.139, +0.060] | +0.619 [-1.079, +2.389] | -0.006 [-0.051, +0.042] | +0.062 [+0.034, +0.087] | +0.015 [-0.003, +0.037] | +0.047 [-0.510, +0.544] | -0.558 [-0.961, -0.211] | +0.145 [-0.004, +0.230] | -3.163 [-7.472, +0.715] | -3.162 [-7.479, +0.717] | -1.148 [-4.109, +1.004] | -0.054 [-0.153, +0.064] | -0.016 [-0.094, +0.044] | -0.057 [-0.172, +0.031] | -3.157 [-5.626, -0.503] | +j:driver_effects(this race, whole) +j:hinge(a priori) +p:driver_effects(this race, to snapshot) +p:hinge(a priori) |
| wear_lower_bound_judge | +0.019 [+0.000, +0.056] | +0.002 [+0.000, +0.005] | +0.046 [-0.002, +0.141] | -0.000 [-0.001, +0.000] | +0.000 [+0.000, +0.001] | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.006] | +0.000 [-0.002, +0.004] | +0.049 [+0.000, +0.127] | +0.029 [-0.030, +0.117] | +0.003 [+0.000, +0.009] | -0.124 [-0.368, +0.000] | -0.124 [-0.368, +0.000] | -0.123 [-0.364, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.001] | +0.002 [+0.000, +0.007] | +0.000 [+0.000, +0.000] | +j:wear_lower_bound(a priori) |
| no_wear_lower_bound_belief | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.024 [-0.096, +0.035] | -0.121 [-0.671, +0.266] | -0.026 [-0.070, +0.006] | +0.012 [-0.020, +0.052] | -0.008 [-0.022, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -6.553 [-13.796, -0.858] | -6.554 [-13.799, -0.858] | -2.366 [-6.723, +0.423] | -0.032 [-0.113, +0.038] | +0.092 [+0.009, +0.212] | -0.104 [-0.251, -0.008] | +1.015 [-0.329, +2.839] | -p:wear_lower_bound |
| compound_proxy_off | +0.000 [+0.000, +0.000] | -0.001 [-0.002, +0.000] | +0.000 [+0.000, +0.000] | +0.033 [+0.001, +0.068] | +0.054 [-0.145, +0.255] | -0.020 [-0.051, +0.000] | +0.001 [-0.004, +0.012] | +0.000 [-0.001, +0.004] | -0.331 [-0.994, +0.000] | +0.792 [+0.000, +2.375] | +0.002 [+0.000, +0.006] | -1.593 [-5.475, +1.255] | -1.614 [-5.475, +1.206] | -0.455 [-2.988, +1.425] | +0.038 [+0.003, +0.075] | -0.012 [-0.047, +0.037] | -0.074 [-0.133, -0.018] | +0.224 [+0.001, +0.583] | -j:proxy_raced -j:proxy_steps -j:proxy_weekend -p:proxy_raced -p:proxy_steps -p:proxy_weekend |
| stint_limit_off | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [-0.113, +0.111] | +5.752 [+1.462, +10.286] | -0.069 [-0.118, -0.021] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.010 [-0.140, +0.112] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -j:stint_limit -p:stint_limit |
| learned_ridge | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.098 [-0.177, -0.047] | +2.015 [+0.686, +4.473] | -0.103 [-0.254, -0.013] | -0.027 [-0.053, +0.008] | -0.014 [-0.020, -0.008] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -3.125 [-11.383, +4.913] | -3.225 [-11.591, +4.913] | +3.409 [-2.608, +9.217] | -0.095 [-0.173, -0.045] | +0.216 [+0.101, +0.343] | +0.095 [-0.050, +0.227] | +7.420 [-0.565, +16.523] | +p:calibrator(history) |
| all_on | -0.019 [-0.074, +0.058] | -0.002 [-0.007, +0.005] | -0.441 [-0.824, -0.058] | -0.049 [-0.139, +0.061] | +0.621 [-1.079, +2.389] | -0.006 [-0.051, +0.042] | +0.063 [+0.039, +0.087] | +0.015 [-0.003, +0.038] | +0.014 [-0.500, +0.453] | -0.525 [-0.901, -0.237] | +0.148 [+0.009, +0.230] | -3.182 [-7.351, +0.565] | -3.182 [-7.353, +0.571] | -1.167 [-4.055, +0.869] | -0.054 [-0.153, +0.065] | -0.016 [-0.094, +0.044] | -0.054 [-0.172, +0.033] | -3.157 [-5.626, -0.503] | +j:driver_effects(this race, whole) +j:hinge(a priori) +j:wear_lower_bound(a priori) +p:driver_effects(this race, to snapshot) +p:hinge(a priori) |

