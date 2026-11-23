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

Races: 2019_10_united_kingdom, 2021_01_bahrain, 2021_04_spain, 2021_09_austrian, 2021_20_qatar, 2021_22_united_arab_emirates, 2022_01_bahrain, 2022_02_saudi_arabia, 2022_06_spain, 2022_08_azerbaijan, 2022_09_canada, 2022_11_austria, 2022_12_france, 2022_20_mexico, 2022_21_brazil, 2022_22_united_arab_emirates, 2023_01_bahrain, 2023_07_spain, 2024_07_emilia_romagna. Rows: the base, each factor switched alone, and all_on (traffic_both). Cells: change against the base, pooled, with a race-cluster bootstrap 95% interval (2000 draws over the 19 races every configuration ran). Run time 1507 s.

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
- `pit_stops_replay`: the judge's whole-race replays charge each real stop its own lane time (less the race's median) {"world": {"pit_stops": {"replay": "actual"}}}
- `pit_stops_team`: the judge prices every stop at the pit loss plus the car's team's typical stop (whole race) {"world": {"pit_stops": {"team": true}}}
- `pit_stops_both`: actual stops in replays and the team's typical stop for every plan, judge; the belief's as-of team stop {"world": {"pit_stops": {"replay": "actual", "team": true}}, "belief": {"pit_stops": {"team": true}}}
- `sc_queue`: the judge places each simulated car in the SC queue among the other cars' recorded race (mode single) {"world": {"sc_queue": {"enabled": true, "mode": "single"}}}
- `sc_queue_field`: the judge's whole-race replays queue every replayed car together (mode field; Section A only) {"world": {"sc_queue": {"enabled": true, "mode": "field"}}}
- `sc_queue_both`: the single-car queue in the judge and in the belief (the field visible at the snapshot) {"world": {"sc_queue": {"enabled": true, "mode": "single"}}, "belief": {"sc_queue": {"enabled": true, "mode": "single"}}}
- `sc_queue_held`: the single-car queue in the judge, cars lapped by more than 1.25 laps at the deployment held {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}}
- `sc_queue_held_field`: the whole-field queue (Section A only) with the near-lapped rule {"world": {"sc_queue": {"enabled": true, "mode": "field", "held_behind_laps": 1.25}}}
- `sc_queue_held_both`: the single-car queue with the near-lapped rule in the judge and in the belief (the field visible at the snapshot) {"world": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}, "belief": {"sc_queue": {"enabled": true, "mode": "single", "held_behind_laps": 1.25}}}
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

| factor | A_close_pair_acc | A_spearman | A_rel_gap_mae_s | B2_rank_rho | B2_regret_s | B2_pit_now_agree | P_close_pair_acc | P_spearman | A_abs_mae_s | A_abs_bias_s | A_stop_gain_bias_s | B1_mae_s | B1_bias_s | B1_green_mae_s | B2_rank_rho_green | B3_cov_90 | B3_factor_cov_90 | B4_mae_offset_s | ledger_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.753 | 0.958 | 13.171 | 0.868 | 4.110 | 0.816 | 0.664 | 0.875 | 18.893 | -4.175 | -0.189 | 42.941 | 40.657 | 16.611 | 0.882 | 0.991 | 0.470 | 32.756 |  |
| sc_queue | -0.062 [-0.115, -0.008] | -0.076 [-0.193, -0.000] | -3.820 [-7.827, -0.872] | -0.009 [-0.015, -0.003] | -0.337 [-0.574, -0.123] | -0.013 [-0.031, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.021 [-9.221, -1.600] | -4.336 [-12.213, +1.403] | +0.000 [+0.000, +0.000] | +18.033 [+15.088, +20.802] | +19.072 [+16.348, +21.616] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.023 [-0.033, -0.014] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_field | -0.027 [-0.044, -0.010] | -0.003 [-0.006, -0.001] | -5.467 [-10.323, -1.379] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -4.817 [-8.518, -1.708] | -2.773 [-12.122, +4.497] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_both | -0.062 [-0.115, -0.008] | -0.076 [-0.193, -0.000] | -3.820 [-7.827, -0.872] | +0.012 [+0.001, +0.023] | +0.533 [-0.189, +1.189] | -0.101 [-0.153, -0.049] | -0.008 [-0.022, +0.005] | -0.003 [-0.010, +0.002] | -5.021 [-9.221, -1.600] | -4.336 [-12.213, +1.403] | +0.000 [+0.000, +0.000] | +4.709 [-4.266, +18.366] | -26.007 [-48.843, -9.108] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.038 [-0.059, -0.018] | +0.000 [+0.000, +0.000] | +7.958 [-8.444, +32.985] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) +p:sc_queue(this race, to snapshot) +p:sc_queue_spacing(history) |
| sc_queue_held | -0.043 [-0.081, -0.010] | -0.004 [-0.011, +0.000] | -5.697 [-10.906, -1.501] | -0.008 [-0.013, -0.003] | -0.164 [-0.400, +0.064] | -0.009 [-0.025, +0.006] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -5.690 [-10.638, -1.342] | -0.683 [-8.722, +5.339] | +0.000 [+0.000, +0.000] | +11.624 [+10.120, +13.146] | +12.337 [+10.808, +13.732] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.008 [-0.013, -0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_held_field | -0.019 [-0.036, -0.003] | -0.001 [-0.004, +0.001] | -5.718 [-10.922, -1.469] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -4.383 [-7.495, -1.619] | -1.962 [-11.406, +5.478] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) |
| sc_queue_held_both | -0.043 [-0.081, -0.010] | -0.004 [-0.011, +0.000] | -5.697 [-10.906, -1.501] | +0.009 [-0.001, +0.020] | +0.691 [-0.053, +1.419] | -0.093 [-0.143, -0.043] | -0.009 [-0.023, +0.004] | -0.003 [-0.009, +0.002] | -5.690 [-10.638, -1.342] | -0.683 [-8.722, +5.339] | +0.000 [+0.000, +0.000] | -5.779 [-8.025, -3.345] | -15.255 [-19.536, -11.570] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.019 [-0.034, -0.007] | +0.000 [+0.000, +0.000] | -0.337 [-9.049, +6.496] | +j:sc_queue(this race, whole) +j:sc_queue_spacing(history) +p:sc_queue(this race, to snapshot) +p:sc_queue_spacing(history) |

