# Simulator fidelity ablation

Computed on main f7be251 (the wear cliff, R8 retired, traffic off by default), 18 calibration races; 2022_11_austria did not load then.

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

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (traffic_both, world_events_hybrid). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over the 18 races every configuration ran). Run time 2822 s.

Factors:

- `world_traffic`: the judge with ghost-car traffic (configs/sim/world.yaml, score.WORLD_TRAFFIC_CONFIG; recorded ghosts, own pace, earned passes, auto dirty air) {"world_file": "configs/sim/world.yaml"}
- `belief_traffic`: the belief with the projected field (configs/sim/belief_traffic.yaml, score.BELIEF_TRAFFIC_CONFIG; visible rows only) {"belief_file": "configs/sim/belief_traffic.yaml"}
- `traffic_both`: traffic in the judge and the belief {"world_file": "configs/sim/world.yaml", "belief_file": "configs/sim/belief_traffic.yaml"}
- `world_events_hybrid`: the judge with SC/VSC (configs/sim/world_hybrid.yaml); whole-race replays run it as `recorded` (every period and its recorded lap times), the plan panel as `hybrid` (the period in progress as recorded, later ones drawn) {"world": {"events": {"name": "hybrid", "params": {"lap_times": "recorded"}}}}
- `world_no_driver_effects`: the judge's wear fit, hinge only (linear_fuel_hinge without driver effects) {"world": {"degradation": {"params": {"driver_effects": "off"}}}}
- `world_no_hinge`: the judge's wear fit, driver effects only (linear_fuel with driver effects) {"world": {"degradation": {"name": "linear_fuel"}}}
- `world_linear`: the judge's wear fit, neither (linear_fuel, no driver effects; the world before WORLD_DEGRADATION) {"world": {"degradation": {"name": "linear_fuel", "params": {"driver_effects": "off"}}}}
- `belief_driver_effects`: driver effects in the belief's fits too {"belief": {"degradation": {"params": {"driver_effects": "race"}}}}
- `belief_hinge`: the belief on linear_fuel_hinge {"belief": {"degradation": {"name": "linear_fuel_hinge"}}}
- `wear_lower_bound_judge`: wear bounded below at 0 in the judge too (the belief already has it) {"world": {"degradation": {"params": {"wear_lower_bound": 0.0}}}}
- `no_wear_lower_bound_belief`: the belief without its wear bound {"belief": {"degradation": {"params": {"wear_lower_bound": null}}}}
- `compound_proxy_off`: no proxy pricing of unraced compounds, judge and belief {"both": {"compound_proxy": {"enabled": false}}}
- `belief_pit_loss_prior_only`: the belief prices stops at the snapshot's pre-race prior, not the as-of estimate updated with this race's stops {"belief": {"pit_loss": {"source": "snapshot"}}}
- `learned_trained`: the belief with the trained learned calibrator (configs/sim/belief_learned.yaml, committed artifacts) {"belief_file": "configs/sim/belief_learned.yaml"}
- `cliff_edge_history_only`: the judge's edge from the historical prior only, nothing from this race (no hindsight) {"world": {"wear_cliff": {"world_edge": "history_only"}}}
- `cliff_edge_race`: the judge's edge from this race alone (the oldest age raced, the earlier design) {"world": {"wear_cliff": {"world_edge": "race"}}}
- `wear_cliff_off`: no wear cliff in either simulator (and no R8), linear wear past every edge {"both": {"wear_cliff": {"enabled": false}}}
- `r8_legacy`: the retired R8 hindsight limit on and the cliff off in both (configs/sim/legacy/*_r8.yaml) {"both": {"stint_limit_margin_laps": 3, "wear_cliff": {"enabled": false}}}

Failed and left out (configuration, race, error):

- base 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_traffic 2022_11_austria: ValueError: cannot convert float NaN to integer
- belief_traffic 2022_11_austria: ValueError: cannot convert float NaN to integer
- traffic_both 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_events_hybrid 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_no_driver_effects 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_no_hinge 2022_11_austria: ValueError: cannot convert float NaN to integer
- world_linear 2022_11_austria: ValueError: cannot convert float NaN to integer
- belief_driver_effects 2022_11_austria: ValueError: cannot convert float NaN to integer
- belief_hinge 2022_11_austria: ValueError: cannot convert float NaN to integer
- wear_lower_bound_judge 2022_11_austria: ValueError: cannot convert float NaN to integer
- no_wear_lower_bound_belief 2022_11_austria: ValueError: cannot convert float NaN to integer
- compound_proxy_off 2022_11_austria: ValueError: cannot convert float NaN to integer
- belief_pit_loss_prior_only 2022_11_austria: ValueError: cannot convert float NaN to integer
- learned_trained 2022_11_austria: ValueError: cannot convert float NaN to integer
- cliff_edge_history_only 2022_11_austria: ValueError: cannot convert float NaN to integer
- cliff_edge_race 2022_11_austria: ValueError: cannot convert float NaN to integer
- wear_cliff_off 2022_11_austria: ValueError: cannot convert float NaN to integer
- r8_legacy 2022_11_austria: ValueError: cannot convert float NaN to integer
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
| base | 0.682 | 0.945 | 15.548 | 0.810 | 6.198 | 0.769 | 0.644 | 0.879 | 32.231 | -4.896 | -0.188 | 94.210 | 93.952 | 23.511 | 0.819 | 0.481 | 0.520 | 36.669 |  |
| world_traffic | -0.008 [-0.035, +0.024] | -0.001 [-0.007, +0.004] | +0.058 [-0.341, +0.491] | -0.010 [-0.017, -0.005] | +0.455 [+0.191, +0.756] | -0.007 [-0.024, +0.010] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.151 [-0.623, +0.288] | +0.859 [+0.536, +1.214] | +0.000 [+0.000, +0.000] | -3.841 [-4.577, -3.161] | -3.937 [-4.774, -3.192] | -0.966 [-1.769, -0.111] | -0.012 [-0.019, -0.007] | +0.044 [+0.036, +0.052] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:traffic(this race, whole) |
| belief_traffic | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.000 [-0.001, +0.001] | -0.015 [-0.276, +0.274] | +0.016 [+0.004, +0.028] | -0.007 [-0.017, +0.005] | -0.001 [-0.005, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +1.109 [+0.957, +1.258] | +1.118 [+0.960, +1.269] | +0.597 [+0.381, +0.804] | +0.000 [-0.001, +0.002] | -0.017 [-0.023, -0.013] | +0.000 [+0.000, +0.000] | -0.129 [-0.544, +0.227] | +p:traffic(this race, to snapshot) |
| traffic_both | -0.008 [-0.035, +0.024] | -0.001 [-0.007, +0.004] | +0.058 [-0.341, +0.491] | -0.009 [-0.015, -0.004] | +0.416 [+0.098, +0.769] | +0.007 [-0.013, +0.027] | -0.007 [-0.017, +0.005] | -0.001 [-0.005, +0.002] | -0.151 [-0.623, +0.288] | +0.859 [+0.536, +1.214] | +0.000 [+0.000, +0.000] | -2.733 [-3.509, -2.034] | -2.819 [-3.692, -2.054] | -0.576 [-1.214, +0.092] | -0.011 [-0.017, -0.005] | +0.030 [+0.023, +0.037] | +0.000 [+0.000, +0.000] | -0.129 [-0.544, +0.227] | +j:traffic(this race, whole) +p:traffic(this race, to snapshot) |
| world_events_hybrid | +0.016 [+0.000, +0.038] | +0.011 [+0.002, +0.025] | -2.384 [-4.382, -0.520] | -0.004 [-0.007, -0.001] | -0.361 [-0.542, -0.206] | -0.007 [-0.023, +0.008] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -12.485 [-20.793, -4.685] | -0.491 [-14.769, +14.084] | +0.000 [+0.000, +0.000] | -57.373 [-60.861, -53.916] | -64.457 [-67.283, -61.908] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.463 [+0.389, +0.540] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:events(this race, whole) |
| world_no_driver_effects | +0.005 [-0.016, +0.032] | +0.002 [-0.001, +0.005] | +0.068 [-0.278, +0.384] | +0.033 [+0.009, +0.060] | -0.210 [-1.882, +1.393] | +0.006 [-0.015, +0.030] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.083 [-0.736, +0.600] | +0.884 [+0.302, +1.415] | -0.071 [-0.149, +0.026] | -0.088 [-2.182, +1.949] | -0.090 [-2.359, +2.106] | -0.153 [-1.392, +1.323] | +0.032 [+0.008, +0.058] | +0.013 [-0.023, +0.051] | -0.013 [-0.068, +0.045] | +0.000 [+0.000, +0.000] | -j:driver_effects |
| world_no_hinge | +0.025 [-0.003, +0.060] | +0.004 [+0.002, +0.007] | -0.029 [-0.182, +0.123] | +0.010 [-0.001, +0.021] | -0.480 [-0.869, -0.138] | +0.002 [-0.013, +0.017] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.047 [-0.183, +0.080] | +0.095 [-0.083, +0.284] | -0.002 [-0.028, +0.024] | +0.418 [-1.118, +1.937] | +0.639 [-1.028, +2.397] | -1.271 [-2.572, -0.020] | +0.010 [-0.001, +0.021] | -0.002 [-0.010, +0.007] | -0.024 [-0.060, +0.007] | +0.000 [+0.000, +0.000] | -j:hinge |
| world_linear | +0.022 [+0.003, +0.047] | +0.002 [-0.001, +0.005] | +0.029 [-0.372, +0.433] | +0.042 [+0.015, +0.070] | -0.399 [-2.140, +1.295] | +0.016 [-0.014, +0.048] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.009 [-0.543, +0.573] | +0.762 [+0.299, +1.213] | -0.062 [-0.137, +0.030] | +1.735 [-1.035, +4.350] | +1.868 [-1.094, +4.689] | -0.346 [-2.080, +1.473] | +0.041 [+0.014, +0.067] | +0.002 [-0.035, +0.043] | -0.007 [-0.066, +0.058] | +0.000 [+0.000, +0.000] | -j:driver_effects -j:hinge |
| belief_driver_effects | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.031 [+0.007, +0.053] | -1.014 [-1.781, -0.321] | +0.013 [-0.016, +0.044] | +0.021 [+0.009, +0.035] | +0.006 [+0.001, +0.011] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.787 [-0.034, +1.594] | +0.811 [-0.022, +1.619] | -0.725 [-1.791, +0.305] | +0.033 [+0.008, +0.055] | -0.032 [-0.062, -0.001] | -0.105 [-0.155, -0.055] | -2.020 [-3.316, -0.793] | +p:driver_effects(this race, to snapshot) |
| belief_hinge | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.008 [-0.000, +0.019] | +0.010 [-0.093, +0.102] | -0.004 [-0.011, +0.003] | +0.003 [-0.001, +0.010] | +0.001 [-0.000, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.267 [-0.592, +0.023] | -0.287 [-0.641, +0.017] | +0.048 [-0.158, +0.330] | +0.009 [+0.000, +0.020] | +0.013 [+0.005, +0.023] | +0.009 [-0.002, +0.022] | -0.186 [-0.399, -0.022] | +p:hinge(a priori) |
| wear_lower_bound_judge | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.033 [-0.075, -0.002] | +0.002 [-0.001, +0.006] | -0.209 [-0.587, -0.009] | +0.004 [+0.000, +0.014] | -0.002 [-0.004, +0.000] | -0.000 [-0.001, +0.000] | -0.014 [-0.071, +0.043] | -0.020 [-0.076, +0.035] | +0.003 [+0.001, +0.008] | +0.106 [-0.047, +0.277] | +0.106 [-0.047, +0.277] | +0.011 [-0.118, +0.159] | +0.002 [-0.001, +0.006] | -0.001 [-0.003, +0.000] | +0.003 [-0.003, +0.010] | +0.000 [+0.000, +0.000] | +j:wear_lower_bound(a priori) |
| no_wear_lower_bound_belief | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.058 [-0.109, -0.016] | +6.933 [+2.808, +11.519] | +0.003 [-0.032, +0.036] | +0.015 [-0.001, +0.039] | -0.000 [-0.007, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.879 [-9.650, -3.030] | -6.342 [-10.166, -3.342] | -0.771 [-2.940, +1.412] | -0.060 [-0.112, -0.017] | +0.035 [-0.003, +0.087] | -0.052 [-0.108, -0.011] | -1.140 [-2.827, +0.095] | -p:wear_lower_bound |
| compound_proxy_off | +0.000 [-0.005, +0.007] | -0.002 [-0.005, -0.000] | -0.104 [-0.311, +0.000] | -0.011 [-0.034, +0.007] | +0.786 [-0.448, +2.996] | -0.002 [-0.013, +0.008] | +0.008 [-0.005, +0.021] | +0.005 [-0.001, +0.012] | -0.088 [-0.700, +0.499] | +0.193 [-0.580, +1.016] | +0.001 [-0.000, +0.003] | -1.837 [-4.469, +0.382] | -1.800 [-4.431, +0.440] | -0.317 [-1.686, +1.071] | -0.010 [-0.034, +0.008] | -0.004 [-0.024, +0.018] | -0.067 [-0.104, -0.035] | -0.854 [-2.635, +0.656] | -j:proxy_raced -j:proxy_steps -j:proxy_weekend -p:proxy_raced -p:proxy_steps -p:proxy_weekend |
| belief_pit_loss_prior_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.002 [-0.005, +0.001] | -0.106 [-0.329, +0.124] | +0.004 [-0.019, +0.024] | -0.003 [-0.008, +0.001] | +0.000 [-0.002, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.017 [-1.233, +1.208] | +0.033 [-1.233, +1.252] | +0.142 [-0.382, +0.682] | -0.001 [-0.004, +0.002] | -0.036 [-0.078, -0.001] | +0.000 [+0.000, +0.000] | +0.546 [+0.121, +1.043] | +p:pit_loss(history) |
| learned_trained | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [-0.005, +0.013] | +0.759 [+0.180, +1.568] | -0.043 [-0.071, -0.018] | -0.008 [-0.018, +0.000] | -0.002 [-0.006, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -14.695 [-19.327, -10.517] | -15.452 [-20.960, -10.866] | -0.123 [-3.273, +2.921] | +0.001 [-0.007, +0.010] | +0.185 [+0.137, +0.238] | +0.002 [-0.043, +0.049] | +0.045 [-4.624, +5.098] | +p:calibrator(history) |
| cliff_edge_history_only | +0.025 [+0.000, +0.060] | -0.001 [-0.011, +0.009] | +0.402 [-0.116, +1.107] | +0.007 [-0.005, +0.020] | +0.540 [-0.304, +1.565] | +0.003 [-0.016, +0.024] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.192 [-0.553, +1.106] | +1.653 [+0.451, +3.403] | +0.000 [+0.000, +0.000] | -4.404 [-6.323, -2.793] | -4.440 [-6.360, -2.801] | -1.559 [-2.856, -0.181] | +0.009 [-0.003, +0.022] | +0.045 [+0.025, +0.066] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:cliff_edge(history) |
| cliff_edge_race | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.006 [-0.017, +0.007] | +1.443 [-0.455, +3.501] | +0.000 [-0.011, +0.010] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -1.863 [-2.802, -0.950] | -2.235 [-3.501, -1.092] | -0.284 [-0.939, +0.292] | -0.005 [-0.016, +0.008] | +0.019 [+0.008, +0.031] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |  |
| wear_cliff_off | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.126 [-0.189, -0.071] | +5.435 [+1.108, +10.616] | -0.010 [-0.046, +0.028] | -0.009 [-0.024, +0.000] | -0.003 [-0.007, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -1.653 [-3.248, -0.292] | -1.539 [-3.164, -0.177] | -1.622 [-2.506, -0.667] | -0.131 [-0.198, -0.074] | +0.007 [-0.007, +0.024] | +0.000 [+0.000, +0.000] | -1.681 [-2.789, -0.808] | -j:cliff_edge -j:cliff_prior -j:cliff_shape -p:cliff_edge -p:cliff_prior -p:cliff_shape |
| r8_legacy | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.140 [-0.191, -0.091] | -2.396 [-4.048, -0.930] | +0.027 [-0.009, +0.063] | -0.009 [-0.024, +0.000] | -0.003 [-0.007, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -1.653 [-3.248, -0.292] | -1.539 [-3.164, -0.177] | -1.622 [-2.506, -0.667] | -0.142 [-0.194, -0.091] | +0.007 [-0.007, +0.024] | +0.000 [+0.000, +0.000] | -1.681 [-2.789, -0.808] | +j:stint_limit(this race, whole) +p:stint_limit(this race, whole) -j:cliff_edge -j:cliff_prior -j:cliff_shape -p:cliff_edge -p:cliff_prior -p:cliff_shape |
| all_on | -0.008 [-0.042, +0.031] | +0.005 [-0.008, +0.022] | -2.390 [-4.461, -0.531] | -0.012 [-0.019, -0.006] | -0.073 [-0.387, +0.304] | +0.004 [-0.018, +0.025] | -0.007 [-0.017, +0.005] | -0.001 [-0.005, +0.002] | -12.682 [-20.950, -5.141] | +0.375 [-13.934, +15.059] | +0.000 [+0.000, +0.000] | -58.652 [-62.220, -55.238] | -66.989 [-70.027, -64.219] | -0.576 [-1.214, +0.092] | -0.011 [-0.017, -0.005] | +0.468 [+0.395, +0.545] | +0.000 [+0.000, +0.000] | -0.129 [-0.544, +0.227] | +j:events(this race, whole) +j:traffic(this race, whole) +p:traffic(this race, to snapshot) |

