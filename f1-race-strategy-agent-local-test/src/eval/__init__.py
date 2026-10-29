"""Evaluation harness: versioned scenario sets, regret scoring, result tables.

    python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json
    python -m src.eval score --scenarios data/scenarios/v1.json --policies baselines --out reports/eval/v1.csv
    python -m src.eval report reports/eval/v1.csv

See docs/eval.md.
"""
from .policies import BASELINE_POLICIES, Decision, agent_policy, recorded_policy, resolve  # noqa: F401
from .scenarios import Scenario, ScenarioSet, build_set, sample_race  # noqa: F401
from .score import score_scenario, score_set, summarize, to_markdown  # noqa: F401
