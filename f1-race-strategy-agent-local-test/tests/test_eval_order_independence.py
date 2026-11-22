"""A scenario's numbers do not depend on what was scored before it (docs/eval.md "Common random
numbers"): every draw is a function of (seed, sample index, stream) and every per-simulator cache
is keyed by what it reads (engine._race_key: the race, its length and compounds), so the same
scenario and plan give bit-identical time, regret and reference alone, after other scenarios, in
reversed order, in a second process, and under any --jobs. The default world (hybrid SC/VSC, the
history-only tyre-life edge) and the default belief (the agent's tools and the continuation).

The cause this pins: the world's wear-cliff edges were cached under one key for the whole race
but built from the first snapshot's RaceParams; the legacy builder (snapshot_at) gives each driver
their own total_laps, so a scenario scored after another driver's saw that driver's edges
(2021 Spain: fixed_one_stop 1.4 s, the reference 0.01-0.1 s). Committed sets only, no held-out race."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.eval import rerun
from src.eval.scenarios import ScenarioSet

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "scenarios" / "v1.json"
DEV6 = ROOT / "data" / "scenarios" / "dev6.json"
pytestmark = pytest.mark.skipif(not V1.exists() or not DEV6.exists(), reason="committed scenario sets not present")
POLICIES = ["belief_optimizer", "fixed_one_stop", "run_to_flag"]
COLS = ["status", "plan", "time_s", "opt_time_s", "opt_plan", "regret_s"]
# 2021 Spain, four drivers whose snapshots carry different total_laps (snapshot_at)
PICK = (10, 60, 120, 180)


def _items():
    full = ScenarioSet.load(V1)
    return [full.items[i] for i in PICK]


def _score(items, order):
    """score_set over items[order] on fresh simulators (default world and belief)."""
    from src.eval.policies import resolve
    from src.eval.score import race_simulators, score_set
    from src.sim import SimConfig
    from src.sim.compute import load_profile
    prof = load_profile("local")
    sim_for = race_simulators(prof, SimConfig.from_file(rerun.belief_path("default")), None)
    s = ScenarioSet(version="t_order", items=items)
    rows = score_set(s, sim_for, resolve(POLICIES), n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops,
                     items=[items[i] for i in order], warn=False)
    return rows.set_index(["scenario_id", "policy"])[COLS].sort_index()


def test_the_default_world_has_lengths_that_differ_within_the_race():
    lens = {sc.snapshot.race.total_laps for sc in _items()}
    assert len(lens) > 1   # otherwise the tests below could not see the old cache collision


def test_scores_do_not_depend_on_the_scenarios_scored_before():
    items = _items()
    fwd = _score(items, [0, 1, 2, 3])
    rev = _score(items, [3, 2, 1, 0])
    pd.testing.assert_frame_equal(fwd, rev, check_exact=True)
    for i in (1, 3):   # alone, on a fresh pair of simulators
        alone = _score(items, [i])
        pd.testing.assert_frame_equal(fwd.loc[alone.index], alone, check_exact=True)
    # a second process
    code = ("import sys, json; sys.path.insert(0, 'tests'); from test_eval_order_independence import _items, _score; "
            "r = _score(_items(), [2]); print('JSON' + r.reset_index().to_json(orient='records', double_precision=15))")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    other = pd.DataFrame(json.loads(next(x for x in out.stdout.splitlines() if x.startswith("JSON"))[4:]))
    other = other.set_index(["scenario_id", "policy"])[COLS].sort_index()
    mine = fwd.loc[other.index]
    for c in ("time_s", "opt_time_s", "regret_s"):
        np.testing.assert_allclose(other[c].astype(float), mine[c].astype(float), rtol=0, atol=1e-9)
    assert list(other["plan"]) == list(mine["plan"])


def _fresh_tools():
    from src.sim import agent_tools
    agent_tools._SIMS.clear()
    agent_tools._RACES.clear()
    return agent_tools


def _tool_outputs(snap):
    from src.eval import decisions
    from src.sim.compute import load_profile
    tools = _tools()
    rest = snap.remaining_laps
    plans = [[[snap.compound, min(12, rest - 10)], ["HARD", rest - min(12, rest - 10)]],
             {"action": "stay_out", "laps": 3}]
    cmp = tools.compare_plans(snap, plans)
    ev = tools.evaluate_action(snap, {"action": "pit", "compound": "HARD"})
    prof = load_profile("local")
    belief = tools.belief_simulator(snap)
    cont = decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": 3}, belief,
                             prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples)
    opt = decisions.belief_optimum(snap, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples)
    return json.dumps({"compare": cmp, "evaluate": ev, "continuation": cont.as_pairs(), "optimum": opt.as_pairs()},
                      sort_keys=True, default=str)


def _tools():
    from src.sim import agent_tools
    return agent_tools


def test_the_belief_tools_and_continuations_do_not_depend_on_history():
    items = _items()
    a, b = items[1].snapshot, items[0].snapshot   # different total_laps
    _fresh_tools()
    alone = _tool_outputs(a)
    _fresh_tools()
    _tool_outputs(b)
    after = _tool_outputs(a)   # the tools' cached belief simulator was built and used on b first
    assert alone == after


def test_references_do_not_depend_on_build_order(tmp_path):
    from src.eval.reference import build_references
    from src.eval.score import race_simulators
    from src.sim import SimConfig
    from src.sim.compute import load_profile
    prof = load_profile("local")
    items = _items()
    out = {}
    for name, order in (("fwd", [0, 1, 2, 3]), ("rev", [3, 2, 1, 0])):
        s = ScenarioSet(version=f"t_ref_{name}", items=[items[i] for i in order])
        sim_for = race_simulators(prof, SimConfig.from_file(rerun.belief_path("default")), None)
        written = build_references(s, sim_for, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops,
                                   root=tmp_path / name)
        out[name] = {r.scenario_id: r for _, refs in written for r in refs}
    for sid, r in out["fwd"].items():
        q = out["rev"][sid]
        assert r.plan.as_pairs() == q.plan.as_pairs() and r.mean_s == q.mean_s
        assert np.array_equal(r.times, q.times) and np.array_equal(r.plan_gap_s, q.plan_gap_s)


def _args(d, **kw):
    base = dict(set="dev20", sets=str(rerun.SETS_FILE), out_dir=str(d), world_config="default",
                belief_config="default", rule="as_of", final=False, jobs=1, profile="local", harness=False)
    return SimpleNamespace(**{**base, **kw})


def test_the_references_and_baselines_stages_do_not_depend_on_jobs_or_order(tmp_path):
    """Two races of the committed dev6 set, two scenarios each with different total_laps: the stages
    with --jobs 1 and --jobs 3 and on the set in reversed order give the same rows."""
    full = ScenarioSet.load(DEV6)
    items = []
    for race in ("2021_04_spain", "2022_02_saudi_arabia"):
        seen = {}
        for sc in full.items:
            if sc.race_id == race and sc.snapshot.race.total_laps not in seen:
                seen[sc.snapshot.race.total_laps] = sc
        items += list(seen.values())[:2]
    assert len(items) == 4
    runs = {}
    for name, its, jobs in (("one", items, 1), ("three", items, 3), ("reversed", items[::-1], 1)):
        d = tmp_path / name
        d.mkdir()
        ScenarioSet(version="t_jobs", items=its).save(d / "scenarios.json")
        rerun.stage_references(_args(d, jobs=jobs))
        rerun.stage_baselines(_args(d, jobs=jobs))
        rows = pd.read_csv(d / "baselines.csv")
        runs[name] = rows.drop(columns=[c for c in ("latency_s",) if c in rows]).set_index(
            ["scenario_id", "policy"]).sort_index()
    pd.testing.assert_frame_equal(runs["one"], runs["three"], check_exact=True)
    # the reversed set is another set (its digest names the order); every number is the same
    pd.testing.assert_frame_equal(runs["one"].drop(columns="scenario_digest"),
                                  runs["reversed"].drop(columns="scenario_digest"), check_exact=True)
