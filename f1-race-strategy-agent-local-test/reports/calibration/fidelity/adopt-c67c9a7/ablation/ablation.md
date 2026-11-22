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

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (traffic_both). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over the 18 races every configuration ran). Run time 1066 s.

Factors:

- `world_traffic`: the judge with ghost-car traffic (configs/sim/world.yaml, score.WORLD_TRAFFIC_CONFIG; recorded ghosts, own pace, earned passes, auto dirty air) {"world_file": "configs/sim/world.yaml"}
- `belief_traffic`: the belief with the projected field (configs/sim/belief_traffic.yaml, score.BELIEF_TRAFFIC_CONFIG; visible rows only) {"belief_file": "configs/sim/belief_traffic.yaml"}
- `traffic_both`: traffic in the judge and the belief {"world_file": "configs/sim/world.yaml", "belief_file": "configs/sim/belief_traffic.yaml"}
- `world_events_off`: the judge without SC/VSC (the world before the adoption drew none; the base's hybrid events, configs/sim/world_hybrid.yaml, replay whole races as `recorded` and run the plan panel as `hybrid`) {"world": {"events": {"name": "none"}}}
- `world_linear`: the judge's wear fit, neither (linear_fuel, no driver effects; the world before WORLD_DEGRADATION) {"world": {"degradation": {"name": "linear_fuel", "params": {"driver_effects": "off"}}}}
- `world_hinge`: the judge's wear fit with the hinge too (linear_fuel_hinge with driver effects, configs/sim/world_de_hinge.yaml) {"world": {"degradation": {"name": "linear_fuel_hinge"}}}
- `belief_no_driver_effects`: the belief without driver effects in its fits (configs/sim/legacy/belief_pre_scorecard.yaml) {"belief": {"degradation": {"params": {"driver_effects": "off"}}}}
- `belief_hinge`: the belief on linear_fuel_hinge {"belief": {"degradation": {"name": "linear_fuel_hinge"}}}
- `wear_lower_bound_judge`: wear bounded below at 0 in the judge too (the belief already has it) {"world": {"degradation": {"params": {"wear_lower_bound": 0.0}}}}
- `no_wear_lower_bound_belief`: the belief without its wear bound {"belief": {"degradation": {"params": {"wear_lower_bound": null}}}}
- `compound_proxy_off`: no proxy pricing of unraced compounds, judge and belief {"both": {"compound_proxy": {"enabled": false}}}
- `belief_pit_loss_prior_only`: the belief prices stops at the snapshot's pre-race prior, not the as-of estimate updated with this race's stops {"belief": {"pit_loss": {"source": "snapshot"}}}
- `learned_trained`: the belief with the trained learned calibrator (configs/sim/belief_learned.yaml, committed artifacts) {"belief_file": "configs/sim/belief_learned.yaml"}
- `cliff_edge_history`: the judge's edge from the historical prior extended by this race's tyre ages (the world before the adoption) {"world": {"wear_cliff": {"world_edge": "history"}}}
- `cliff_edge_race`: the judge's edge from this race alone (the oldest age raced, the earlier design) {"world": {"wear_cliff": {"world_edge": "race"}}}
- `wear_cliff_off`: no wear cliff in either simulator (and no R8), linear wear past every edge {"both": {"wear_cliff": {"enabled": false}}}
- `r8_legacy`: the retired R8 hindsight limit on and the cliff off in both (configs/sim/legacy/*_r8.yaml) {"both": {"stint_limit_margin_laps": 3, "wear_cliff": {"enabled": false}}}

Failed and left out (configuration, race, error):

- base 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- traffic_both 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- world_events_off 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- world_linear 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- world_hinge 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- belief_no_driver_effects 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound
- cliff_edge_history 2022_11_austria: ValueError: OCO's set at lap 45 has no recorded compound

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
| base | 0.751 | 0.957 | 13.559 | 0.867 | 4.223 | 0.806 | 0.665 | 0.882 | 19.305 | -3.807 | -0.191 | 50.440 | 45.703 | 16.974 | 0.883 | 0.975 | 0.460 | 33.051 |  |
| traffic_both | -0.041 [-0.084, +0.003] | -0.004 [-0.010, +0.001] | -0.215 [-0.691, +0.204] | -0.008 [-0.014, -0.003] | +0.070 [-0.132, +0.273] | +0.006 [-0.012, +0.025] | -0.002 [-0.014, +0.009] | +0.001 [-0.001, +0.003] | -0.098 [-0.641, +0.414] | +0.752 [+0.007, +1.467] | +0.000 [+0.000, +0.000] | -1.606 [-2.074, -1.155] | -2.605 [-3.427, -1.849] | +0.450 [-0.223, +1.246] | -0.011 [-0.018, -0.004] | +0.003 [+0.001, +0.005] | +0.000 [+0.000, +0.000] | -0.241 [-0.590, +0.077] | +j:traffic(this race, whole) +p:traffic(this race, to snapshot) |
| world_events_off | -0.025 [-0.050, -0.007] | -0.009 [-0.019, -0.002] | +2.407 [+0.435, +4.513] | -0.001 [-0.005, +0.003] | +0.397 [+0.121, +0.696] | +0.002 [-0.010, +0.013] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +13.119 [+5.269, +21.373] | +0.608 [-13.943, +14.848] | +0.000 [+0.000, +0.000] | +59.527 [+55.932, +63.287] | +64.265 [+61.671, +67.079] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.254 [-0.311, -0.203] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:events(a priori) |
| world_linear | -0.014 [-0.036, +0.002] | -0.002 [-0.005, +0.001] | -0.205 [-0.689, +0.225] | -0.030 [-0.074, +0.005] | +0.813 [-0.529, +1.944] | +0.011 [-0.024, +0.048] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.101 [-0.355, +0.506] | +0.571 [+0.113, +1.033] | -0.060 [-0.130, +0.030] | +1.395 [+0.311, +2.507] | +1.116 [-0.497, +2.625] | +1.515 [+0.567, +2.507] | -0.032 [-0.073, +0.001] | -0.001 [-0.006, +0.003] | -0.084 [-0.142, -0.025] | +0.000 [+0.000, +0.000] | -j:driver_effects |
| world_hinge | -0.033 [-0.065, -0.009] | -0.005 [-0.008, -0.002] | -0.004 [-0.155, +0.151] | -0.002 [-0.010, +0.005] | +0.170 [+0.014, +0.331] | -0.006 [-0.018, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.032 [-0.149, +0.191] | -0.036 [-0.342, +0.265] | +0.002 [-0.024, +0.028] | -0.120 [-1.387, +1.165] | -0.605 [-2.236, +0.879] | +1.465 [+0.265, +2.707] | -0.003 [-0.011, +0.004] | -0.002 [-0.006, +0.001] | +0.026 [-0.018, +0.071] | +0.000 [+0.000, +0.000] | +j:hinge(a priori) |
| belief_no_driver_effects | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.030 [-0.051, -0.008] | +1.045 [+0.308, +1.819] | -0.021 [-0.067, +0.024] | -0.031 [-0.050, -0.016] | -0.008 [-0.015, -0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.919 [-1.017, +2.404] | +0.435 [-1.599, +2.123] | +1.556 [-0.192, +3.278] | -0.029 [-0.050, -0.008] | +0.005 [+0.002, +0.008] | +0.084 [+0.040, +0.129] | +0.711 [-0.686, +2.174] | -p:driver_effects |
| cliff_edge_history | -0.022 [-0.068, +0.006] | +0.003 [-0.004, +0.012] | -0.404 [-1.021, +0.073] | -0.003 [-0.012, +0.007] | -0.439 [-1.051, +0.051] | -0.029 [-0.051, -0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.323 [-0.182, +1.012] | -1.460 [-3.141, -0.309] | +0.000 [+0.000, +0.000] | +3.659 [+2.277, +5.235] | +4.186 [+2.614, +6.014] | +1.635 [+0.487, +2.737] | -0.002 [-0.012, +0.008] | -0.006 [-0.009, -0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:cliff_edge(this race, whole) |

