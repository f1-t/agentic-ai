# Simulator fidelity ablation

STALE: computed on main 42abdb5, before the wear cliff, the retired R8 and traffic. The f7be251 ablation replaces it.

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

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (driver_effects, wear_lower_bound_judge, world_events_hybrid). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over the 18 races every configuration ran). Run time 623 s.

Factors:

- `world_events_hybrid`: the judge with SC/VSC (configs/sim/world_hybrid.yaml); whole-race replays run it as `recorded` (every period and its recorded lap times), the plan panel as `hybrid` (the period in progress as recorded, later ones drawn) {"world": {"events": {"name": "hybrid", "params": {"lap_times": "recorded"}}}}
- `driver_effects`: one pace offset per driver in every degradation fit (configs/sim/arms/driver_effects.yaml) {"both": {"degradation": {"params": {"driver_effects": "race"}}}}
- `hinge`: linear_fuel_hinge, wear steepens past the default knot {"both": {"degradation": {"name": "linear_fuel_hinge"}}}
- `driver_effects_hinge`: both of the above {"both": {"degradation": {"name": "linear_fuel_hinge", "params": {"driver_effects": "race"}}}}
- `wear_lower_bound_judge`: wear bounded below at 0 in the judge too (the belief already has it) {"world": {"degradation": {"params": {"wear_lower_bound": 0.0}}}}
- `no_wear_lower_bound_belief`: the belief without its wear bound {"belief": {"degradation": {"params": {"wear_lower_bound": null}}}}
- `compound_proxy_off`: no proxy pricing of unraced compounds, judge and belief {"both": {"compound_proxy": {"enabled": false}}}
- `belief_pit_loss_prior_only`: the belief prices stops at the snapshot's pre-race prior, not the as-of estimate updated with this race's stops {"belief": {"pit_loss": {"source": "snapshot"}}}
- `learned_trained`: the belief with the trained learned calibrator (configs/sim/belief_learned.yaml, committed artifacts) {"belief_file": "configs/sim/belief_learned.yaml"}
- `stint_limit_off`: R8 stint limit off, judge and belief. To be replaced by the edge-source switches below when fix/tyre-life-no-hindsight merges {"both": {"stint_limit_margin_laps": null}}
- `tyre_edge_history`: support edge from history only (tyre_history.tyre_life_prior) {"both": {"tyre_life": {"edge_source": "history"}}}
- `tyre_edge_history_race`: support edge from history extended by this race's laps seen so far {"both": {"tyre_life": {"edge_source": "history_race"}}}
- `tyre_edge_race`: support edge from this race (the R8 behaviour) {"both": {"tyre_life": {"edge_source": "race"}}}

Skipped (switch not on this branch):

- `tyre_edge_history`: switch not on this branch (ValidationError: 1 validation error for SimConfig)
- `tyre_edge_history_race`: switch not on this branch (ValidationError: 1 validation error for SimConfig)
- `tyre_edge_race`: switch not on this branch (ValidationError: 1 validation error for SimConfig)

Failed and left out (configuration, race, error):

- base 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_events_hybrid 2022_11_austria: ValueError: cannot convert float NaN to integer
- driver_effects 2022_11_austria: ValueError: cannot convert float NaN to integer
- hinge 2022_11_austria: ValueError: cannot convert float NaN to integer
- driver_effects_hinge 2022_11_austria: ValueError: cannot convert float NaN to integer
- wear_lower_bound_judge 2022_11_austria: ValueError: cannot convert float NaN to integer
- no_wear_lower_bound_belief 2022_11_austria: ValueError: cannot convert float NaN to integer
- compound_proxy_off 2022_11_austria: ValueError: cannot convert float NaN to integer
- belief_pit_loss_prior_only 2022_11_austria: ValueError: cannot convert float NaN to integer
- learned_trained 2022_11_austria: ValueError: cannot convert float NaN to integer
- stint_limit_off 2022_11_austria: ValueError: cannot convert float NaN to integer
- all_on 2022_11_austria: ValueError: cannot convert float NaN to integer

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
| base | 0.704 | 0.947 | 15.577 | 0.719 | 3.195 | 0.830 | 0.635 | 0.877 | 32.240 | -4.134 | -0.250 | 94.389 | 94.281 | 21.262 | 0.722 | 0.482 | 0.514 | 34.988 |  |
| world_events_hybrid | +0.022 [+0.004, +0.046] | +0.011 [+0.002, +0.025] | -2.427 [-4.437, -0.555] | +0.000 [-0.004, +0.005] | -0.042 [-0.144, +0.050] | -0.008 [-0.021, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -12.500 [-20.780, -4.607] | -0.501 [-14.804, +14.112] | +0.000 [+0.000, +0.000] | -58.741 [-62.364, -55.351] | -65.618 [-68.462, -63.049] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.463 [+0.384, +0.539] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:events(this race, whole) |
| driver_effects | +0.003 [-0.019, +0.024] | +0.002 [+0.000, +0.004] | -0.058 [-0.428, +0.314] | +0.001 [-0.050, +0.046] | -0.593 [-1.559, +0.407] | +0.003 [-0.042, +0.046] | +0.027 [+0.016, +0.041] | +0.010 [+0.004, +0.016] | -0.055 [-0.645, +0.491] | -0.667 [-1.147, -0.169] | +0.060 [-0.030, +0.131] | -0.501 [-2.052, +1.073] | -0.418 [-2.019, +1.191] | -1.421 [-2.607, -0.279] | +0.012 [-0.043, +0.060] | -0.038 [-0.084, +0.005] | -0.119 [-0.180, -0.059] | -1.830 [-3.012, -0.701] | +j:driver_effects(this race, whole) +p:driver_effects(this race, to snapshot) |
| hinge | -0.016 [-0.036, -0.002] | -0.000 [-0.003, +0.003] | +0.039 [-0.148, +0.257] | -0.020 [-0.048, +0.003] | +0.365 [+0.153, +0.617] | -0.025 [-0.067, +0.007] | +0.001 [-0.001, +0.004] | +0.001 [-0.001, +0.004] | -0.091 [-0.301, +0.106] | +0.122 [-0.104, +0.362] | -0.008 [-0.045, +0.029] | -2.153 [-3.769, -0.611] | -2.245 [-3.907, -0.658] | +0.445 [-0.739, +1.529] | -0.021 [-0.048, +0.003] | +0.034 [+0.020, +0.049] | +0.004 [-0.041, +0.049] | -0.122 [-0.279, +0.007] | +j:hinge(a priori) +p:hinge(a priori) |
| driver_effects_hinge | -0.022 [-0.047, -0.003] | -0.002 [-0.005, +0.001] | -0.029 [-0.433, +0.372] | -0.005 [-0.060, +0.047] | -0.191 [-1.400, +1.087] | +0.005 [-0.034, +0.048] | +0.029 [+0.017, +0.045] | +0.011 [+0.004, +0.019] | -0.009 [-0.573, +0.543] | -0.762 [-1.213, -0.299] | +0.062 [-0.030, +0.137] | -1.184 [-3.941, +1.587] | -1.203 [-4.138, +1.762] | -0.088 [-2.003, +1.695] | +0.004 [-0.056, +0.059] | -0.025 [-0.076, +0.023] | -0.086 [-0.169, -0.008] | -1.939 [-3.151, -0.758] | +j:driver_effects(this race, whole) +j:hinge(a priori) +p:driver_effects(this race, to snapshot) +p:hinge(a priori) |
| wear_lower_bound_judge | -0.008 [-0.020, +0.000] | -0.000 [-0.001, +0.000] | +0.026 [-0.012, +0.077] | +0.002 [-0.000, +0.005] | -0.015 [-0.039, -0.000] | +0.003 [+0.000, +0.009] | -0.002 [-0.004, +0.000] | -0.001 [-0.002, +0.000] | -0.034 [-0.096, +0.008] | -0.063 [-0.182, +0.017] | +0.007 [+0.001, +0.018] | -0.061 [-0.154, -0.000] | -0.061 [-0.154, -0.000] | -0.065 [-0.155, -0.006] | +0.001 [-0.001, +0.005] | -0.001 [-0.005, +0.001] | +0.009 [+0.000, +0.028] | +0.000 [+0.000, +0.000] | +j:wear_lower_bound(a priori) |
| no_wear_lower_bound_belief | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.031 [-0.099, +0.019] | +0.527 [-0.532, +2.124] | -0.005 [-0.026, +0.017] | +0.012 [-0.004, +0.036] | -0.001 [-0.009, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -6.113 [-9.946, -3.185] | -6.342 [-10.166, -3.342] | -0.779 [-2.993, +1.384] | -0.031 [-0.101, +0.021] | +0.037 [-0.006, +0.091] | -0.035 [-0.088, +0.003] | -0.687 [-2.390, +0.742] | -p:wear_lower_bound |
| compound_proxy_off | +0.001 [-0.005, +0.008] | -0.002 [-0.005, -0.000] | -0.100 [-0.299, +0.000] | -0.004 [-0.062, +0.039] | +0.276 [-0.611, +1.606] | -0.004 [-0.019, +0.008] | +0.008 [-0.004, +0.020] | +0.005 [-0.001, +0.012] | -0.078 [-0.688, +0.529] | +0.212 [-0.566, +1.049] | +0.001 [-0.003, +0.003] | -2.543 [-5.233, -0.060] | -2.564 [-5.249, -0.076] | -0.275 [-1.858, +1.237] | -0.000 [-0.056, +0.041] | +0.001 [-0.018, +0.022] | -0.072 [-0.110, -0.040] | -0.867 [-2.653, +0.645] | -j:proxy_raced -j:proxy_steps -j:proxy_weekend -p:proxy_raced -p:proxy_steps -p:proxy_weekend |
| belief_pit_loss_prior_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.002 [-0.008, +0.003] | -0.024 [-0.169, +0.125] | -0.007 [-0.024, +0.008] | -0.003 [-0.007, +0.000] | +0.000 [-0.002, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.039 [-1.226, +1.262] | +0.033 [-1.233, +1.252] | +0.341 [-0.207, +0.900] | +0.001 [-0.006, +0.007] | -0.037 [-0.081, +0.002] | +0.000 [+0.000, +0.000] | +0.527 [+0.103, +1.015] | +p:pit_loss(history) |
| learned_trained | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.020 [-0.040, -0.003] | +0.649 [+0.369, +1.001] | -0.036 [-0.064, -0.015] | +0.000 [-0.008, +0.007] | -0.000 [-0.004, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -14.955 [-19.855, -10.627] | -15.452 [-20.960, -10.866] | -0.137 [-3.848, +3.116] | -0.018 [-0.038, -0.001] | +0.183 [+0.131, +0.245] | +0.004 [-0.039, +0.049] | +1.354 [-3.260, +6.462] | +p:calibrator(history) |
| stint_limit_off | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.036 [-0.039, +0.105] | +4.831 [+2.053, +8.044] | -0.030 [-0.071, +0.013] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.036 [-0.045, +0.107] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -j:stint_limit -p:stint_limit |
| all_on | +0.025 [+0.003, +0.049] | +0.013 [+0.003, +0.028] | -2.438 [-4.384, -0.706] | +0.002 [-0.043, +0.045] | -0.651 [-1.581, +0.285] | +0.000 [-0.041, +0.042] | +0.025 [+0.013, +0.040] | +0.009 [+0.002, +0.016] | -12.592 [-20.908, -4.765] | -1.174 [-15.374, +13.471] | +0.062 [-0.027, +0.131] | -60.322 [-64.394, -56.526] | -65.953 [-69.070, -63.033] | -1.460 [-2.648, -0.325] | +0.012 [-0.042, +0.060] | +0.466 [+0.387, +0.543] | -0.115 [-0.178, -0.051] | -1.830 [-3.012, -0.701] | +j:driver_effects(this race, whole) +j:events(this race, whole) +j:wear_lower_bound(a priori) +p:driver_effects(this race, to snapshot) |

