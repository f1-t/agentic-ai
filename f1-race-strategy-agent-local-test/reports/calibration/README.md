# Calibration reports

A committed report is regenerated when any number in it would change at the current code and config. A drift in `config_hash` or `sim_code_version` alone does not trigger a regeneration: a schema change such as a new `SimConfig` field, or a code change that leaves the report's numbers as they were, moves those stamps and nothing else. The two columns record the config and code the report was generated with, not the current ones.

`cross_race/` is the cross-race calibration report (SPEC 0016, `python -m src.sim.calibrate cross run` then `cross report`, src/sim/cross_race.py). Its `summary.md`, `factors.csv` and the small csv and json tables beside them are committed. The per-race and per-fold parquets under `races/` and `h/` are left out (see its .gitignore); `cross run` rebuilds them, resumably.
