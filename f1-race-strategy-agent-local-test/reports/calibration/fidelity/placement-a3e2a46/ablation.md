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

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_11_austria, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (traffic_both). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over the 19 races every configuration ran). Run time 1142 s.

Factors:

- `world_traffic`: the judge with ghost-car traffic (configs/sim/world.yaml, score.WORLD_TRAFFIC_CONFIG; recorded ghosts, own pace, earned passes, auto dirty air) {"world_file": "configs/sim/world.yaml"}
- `belief_traffic`: the belief with the projected field (configs/sim/belief_traffic.yaml, score.BELIEF_TRAFFIC_CONFIG; visible rows only) {"belief_file": "configs/sim/belief_traffic.yaml"}
- `traffic_both`: traffic in the judge and the belief {"world_file": "configs/sim/world.yaml", "belief_file": "configs/sim/belief_traffic.yaml"}
- `world_difficulty`: the judge's traffic with the circuit's passing difficulty (configs/sim/world_difficulty.yaml; the circuit's races, this one included) {"world_file": "configs/sim/world_difficulty.yaml", "requires": ["src.sim.overtaking"]}
- `world_difficulty_race`: the judge's traffic with this race's own passing rate (shrunk toward the pooled one) {"world": {"traffic": {"name": "ghost", "params": {"align_neutral": true, "own_pace": true, "pass_mode": "earned", "dirty_air_auto": true, "difficulty": "race"}}}, "requires": ["src.sim.overtaking"]}
- `belief_difficulty`: the belief's projected traffic with the as-of passing difficulty (configs/sim/belief_difficulty.yaml; earlier races only) {"belief_file": "configs/sim/belief_difficulty.yaml", "requires": ["src.sim.overtaking"]}
- `difficulty_both`: the judge's circuit difficulty and the belief's as-of difficulty {"world_file": "configs/sim/world_difficulty.yaml", "belief_file": "configs/sim/belief_difficulty.yaml", "requires": ["src.sim.overtaking"]}
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
- `wear_guard_off`: the judge's thin wear fits as fitted, no guard (configs/sim/legacy/world_pre_v3.yaml) {"world": {"wear_guard": {"enabled": false, "scale_sd_by_drivers": false}}}
- `wear_guard_unscaled`: the guard without scaling the thin fit's standard error by the number of drivers {"world": {"wear_guard": {"scale_sd_by_drivers": false}}}
- `belief_cliff_edge_draw`: the belief draws each sample's wear-cliff edge, max(seen so far, prior + sd z) (configs/sim/legacy/belief_pre_v3.yaml) {"belief": {"belief_cliff_edge": "draw"}}
- `wear_cliff_off`: no wear cliff in either simulator (and no R8), linear wear past every edge {"both": {"wear_cliff": {"enabled": false}}}
- `r8_legacy`: the retired R8 hindsight limit on and the cliff off in both (configs/sim/legacy/*_r8.yaml) {"both": {"stint_limit_margin_laps": 3, "wear_cliff": {"enabled": false}}}
- `pit_stops_replay`: the judge's whole-race replays charge each real stop its own lane time (less the race's median) {"world": {"pit_stops": {"replay": "actual"}}}
- `pit_stops_team`: the judge prices every stop at the pit loss plus the car's team's typical stop (whole race) {"world": {"pit_stops": {"team": true}}}
- `pit_stops_both`: actual stops in replays and the team's typical stop for every plan, judge; the belief's as-of team stop {"world": {"pit_stops": {"replay": "actual", "team": true}}, "belief": {"pit_stops": {"team": true}}}
- `sc_queue`: the judge places each simulated car in the SC queue among the other cars' recorded race (mode single) {"world": {"sc_queue": {"enabled": true, "mode": "single"}}}
- `sc_queue_field`: the judge's whole-race replays queue every replayed car together (mode field; Section A only) {"world": {"sc_queue": {"enabled": true, "mode": "field"}}}
- `sc_queue_both`: the single-car queue in the judge and in the belief (the field visible at the snapshot) {"world": {"sc_queue": {"enabled": true, "mode": "single"}}, "belief": {"sc_queue": {"enabled": true, "mode": "single"}}}
- `sc_queue_held`: the single-car queue in the judge, cars lapped by more than 1.25 laps at the deployment held {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}}
- `sc_queue_held_field`: the whole-field queue (Section A only) with the near-lapped rule {"world": {"sc_queue": {"enabled": true, "mode": "field", "held_behind_laps": 1.25}}}
- `sc_queue_held_both`: the single-car queue with the near-lapped rule in the judge and in the belief (the field visible at the snapshot) {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}, "belief": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}}
- `sc_queue_placed`: the single-car queue with the near-lapped rule, the car placed by its recorded place at the deployment {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25, "placement": "record"}}}
- `sc_queue_placed_both`: the placed queue in the judge, the near-lapped rule in the belief (which reads no record) {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25, "placement": "record"}}, "belief": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}}
- `sc_queue_traffic`: the single-car queue with the earned-pass ghost traffic in the judge (the order holds in a train unless a car has the pace) {"world": {"sc_queue": {"enabled": true, "mode": "single"}, "traffic": {"name": "ghost", "params": {"align_neutral": true, "own_pace": true, "pass_mode": "earned", "dirty_air_auto": true}}}}
- `sc_alignment_off`: SC/VSC lengths memoryless geometric again (belief) and the judge's later periods at EventModel's provisional rates (configs/sim/legacy/*_pre_sc_alignment.yaml) {"belief": {"events": {"name": "rate", "params": {"as_of": true, "duration": "geometric"}}}, "world": {"events": {"name": "hybrid", "params": {"lap_times": "recorded", "duration": "geometric", "pooled": false}}}}

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
- `E_real_regret_s`: expert consistency (secondary): judge time of the team's real remaining strategy less the judge's best panel plan, mean over snapshots; should fall where passing is hard, never chased to zero
- `E_real_over_20s`: same, share of snapshots where the judge says the real strategy lost more than 20 s

| factor | A_close_pair_acc | A_spearman | A_rel_gap_mae_s | B2_rank_rho | B2_regret_s | B2_pit_now_agree | P_close_pair_acc | P_spearman | A_abs_mae_s | A_abs_bias_s | A_stop_gain_bias_s | B1_mae_s | B1_bias_s | B1_green_mae_s | B2_rank_rho_green | B3_cov_90 | B3_factor_cov_90 | B4_mae_offset_s | E_real_regret_s | E_real_over_20s | ledger_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.753 | 0.957 | 13.186 | 0.882 | 3.360 | 0.834 | 0.667 | 0.876 | 18.926 | -4.205 | -0.191 | 43.723 | 41.580 | 16.998 | 0.895 | 0.988 | 0.481 | 32.582 | 5.523 | 0.061 |  |
| sc_queue | -0.062 [-0.115, -0.008] | -0.075 [-0.192, -0.000] | -3.821 [-7.828, -0.873] | -0.011 [-0.017, -0.006] | -0.332 [-0.533, -0.134] | -0.007 [-0.023, +0.008] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.023 [-9.222, -1.600] | -4.336 [-12.213, +1.403] | +0.000 [+0.000, +0.000] | +18.090 [+15.160, +20.817] | +19.062 [+16.331, +21.594] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.034 [-0.050, -0.020] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.014 [-0.218, +0.229] | -0.009 [-0.018, -0.001] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_held | -0.043 [-0.081, -0.010] | -0.004 [-0.010, +0.000] | -5.697 [-10.906, -1.501] | -0.010 [-0.015, -0.006] | -0.197 [-0.383, -0.028] | -0.009 [-0.023, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.691 [-10.639, -1.342] | -0.683 [-8.721, +5.340] | +0.000 [+0.000, +0.000] | +11.665 [+10.175, +13.189] | +12.332 [+10.807, +13.728] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.013 [-0.020, -0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.095 [-0.123, +0.344] | -0.003 [-0.017, +0.010] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_held_field | -0.019 [-0.036, -0.003] | -0.001 [-0.004, +0.001] | -5.718 [-10.922, -1.469] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -4.383 [-7.496, -1.619] | -1.962 [-11.406, +5.478] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_held_both | -0.043 [-0.081, -0.010] | -0.004 [-0.010, +0.000] | -5.697 [-10.906, -1.501] | +0.008 [-0.002, +0.018] | +0.559 [-0.036, +1.157] | -0.117 [-0.175, -0.065] | -0.007 [-0.015, +0.005] | -0.002 [-0.007, +0.002] | -5.691 [-10.639, -1.342] | -0.683 [-8.721, +5.340] | +0.000 [+0.000, +0.000] | -5.974 [-8.228, -3.607] | -15.260 [-19.537, -11.570] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.022 [-0.036, -0.010] | +0.000 [+0.000, +0.000] | -0.246 [-8.954, +6.650] | +0.095 [-0.123, +0.344] | -0.003 [-0.017, +0.010] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) +p:sc_queue(this race, to snapshot) +p:sc_queue_spacing(history) |
| sc_queue_placed | -0.005 [-0.033, +0.021] | +0.003 [-0.002, +0.012] | -5.587 [-10.723, -1.467] | -0.010 [-0.015, -0.006] | -0.199 [-0.382, -0.030] | -0.008 [-0.022, +0.006] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.552 [-10.334, -1.311] | -0.590 [-8.522, +5.331] | +0.000 [+0.000, +0.000] | +11.501 [+9.892, +13.123] | +12.486 [+10.925, +13.998] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.013 [-0.019, -0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.066 [-0.148, +0.308] | -0.005 [-0.021, +0.009] | +j:sc_queue(this race, whole) +j:sc_queue_placement(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_placed_both | -0.005 [-0.033, +0.021] | +0.003 [-0.002, +0.012] | -5.587 [-10.723, -1.467] | +0.008 [-0.002, +0.018] | +0.554 [-0.040, +1.156] | -0.116 [-0.175, -0.063] | -0.007 [-0.015, +0.005] | -0.002 [-0.007, +0.002] | -5.552 [-10.334, -1.311] | -0.590 [-8.522, +5.331] | +0.000 [+0.000, +0.000] | -6.298 [-8.416, -4.119] | -15.106 [-19.303, -11.488] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.021 [-0.034, -0.009] | +0.000 [+0.000, +0.000] | -0.246 [-8.954, +6.650] | +0.066 [-0.148, +0.308] | -0.005 [-0.021, +0.009] | +j:sc_queue(this race, whole) +j:sc_queue_placement(this race, whole) +j:sc_queue_spacing(history) +p:sc_queue(this race, to snapshot) +p:sc_queue_spacing(history) |

