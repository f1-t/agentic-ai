"""The rerun's agent stage with --jobs (src/eval/agent_pass.py), a scripted backend standing in for
the model (tests/agent_pass_scripted.py): N=1 and N=3 give the same rows and trajectories, the same
rows as the sequential score_set before it, in the set's order; resume skips finished scenarios; a
failing scenario is one error row; a transient error is retried and counted; an auth error stops the
pass; the timing sample and its paired table. Six scenarios of the committed Spain set (data/scenarios/v1.json)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.eval import agent_pass, rerun
from src.eval.scenarios import ScenarioSet

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "scenarios" / "v1.json"
TIMING = ["latency_s"]
# the world here is the pre-scorecard one: the adopted default world's results depend on the order the
# scenarios are scored in within one process (found by this test, a simulator matter), which would
# make the N=1 / N=3 comparison test the simulator rather than the pass
WORLD = str(ROOT / "configs" / "sim" / "legacy" / "world_pre_scorecard.yaml")
pytestmark = pytest.mark.skipif(not V1.exists(), reason="committed scenario set not present")


def args(d, **kw):
    base = dict(set="dev20", sets=str(rerun.SETS_FILE), out_dir=str(d), world_config=WORLD,
                belief_config="default", rule="as_of", final=False, jobs=1, profile="local", harness=False,
                resume=False, force=False, rerun_errors=False, timing_sample=0, timing_seed=0, only=None,
                progress_every=2, backend="agent_pass_scripted:make", backoff=(0.0, 0.0))
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A six-scenario set (two drivers, laps across the race, in a shuffled order so the merge order
    is the set's, not a sorted one) and its references."""
    d = tmp_path_factory.mktemp("set")
    full = ScenarioSet.load(V1)
    by_driver = {}
    for sc in full.items:
        by_driver.setdefault(sc.driver, []).append(sc)
    drivers = sorted(by_driver, key=lambda x: -len(by_driver[x]))[:2]
    picks = []
    for drv in drivers:
        xs = sorted(by_driver[drv], key=lambda s: s.lap)
        picks += [xs[len(xs) // 6], xs[len(xs) // 2], xs[5 * len(xs) // 6]]
    items = [picks[i] for i in (3, 0, 5, 1, 4, 2)]
    s = ScenarioSet(version="t_agent", items=items)
    s.save(d / "scenarios.json")
    rerun.stage_references(args(d))
    return d, [sc.scenario_id for sc in items]


def fresh(built, tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    src, _ = built
    (d / "scenarios.json").write_bytes((src / "scenarios.json").read_bytes())
    (d / "references").symlink_to(src / "references")
    return d


def content(d):
    rows = pd.read_csv(d / "agent.csv")
    return rows.drop(columns=[c for c in TIMING if c in rows])


def trajectories(d, sub="trajectories"):
    """name -> events, the per-run fields (timestamps, episode ids) dropped."""
    out = {}
    for p in sorted((d / sub).glob("*.jsonl")):
        out[p.name] = [{k: v for k, v in json.loads(x).items() if k not in ("timestamp", "episode_id")}
                       for x in p.read_text().splitlines()]
    return out


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AGENT_BELIEF_CONFIG", "unset")   # restored after the stage sets it
    for v in ("AGENT_PASS_TEST_FAIL", "AGENT_PASS_TEST_FLAKY", "AGENT_PASS_TEST_AUTH", "AGENT_PASS_TEST_CRASH"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.slow
def test_jobs_1_and_3_give_the_same_rows_in_the_set_order_and_match_the_sequential_pass(built, tmp_path):
    _, ids = built
    one, three = fresh(built, tmp_path, "one"), fresh(built, tmp_path, "three")
    rerun.stage_agent(args(one, jobs=1))
    rerun.stage_agent(args(three, jobs=3))
    a, b = content(one), content(three)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["scenario_id"]) == ids                       # the set's order, not a sorted one
    assert set(a["status"]) <= {"ok", "invalid"} and (a["api_retries"] == 0).all()
    assert trajectories(one) == trajectories(three) and len(trajectories(one)) == len(ids)
    assert len(list((three / "parts" / "agent").glob("rows_w*.jsonl"))) >= 2   # more than one worker ran
    run = json.loads((three / "agent_run.json").read_text())
    assert run["agent_jobs"] == 3 and run["passes"][-1]["jobs"] == 3 and run["agent_wall_s"] > 0
    assert "concurrent" in run["latency_measured_under"] and "isolated" in run["headline"]

    # the sequential score_set the stage ran before: the same rows, the new api_retries column apart
    from src.eval.reference import ReferenceStore
    from src.eval.score import race_simulators, score_set
    from src.sim import SimConfig
    from src.sim.compute import load_profile
    import agent_pass_scripted
    prof = load_profile("local")
    old = score_set(ScenarioSet.load(one / "scenarios.json"),
                    race_simulators(prof, SimConfig.from_file(rerun.belief_path("default")), SimConfig.from_file(WORLD)),
                    {"agent": _per_scenario_policy(tmp_path / "old", prof, agent_pass_scripted)},
                    n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops,
                    references=ReferenceStore(one / "references", require=True))
    old.to_csv(tmp_path / "old.csv", index=False)
    old = pd.read_csv(tmp_path / "old.csv").drop(columns=TIMING)
    pd.testing.assert_frame_equal(a.drop(columns=["api_retries"]), old)


def _per_scenario_policy(log_dir, prof, mod):
    """An AgentPolicy that tells the scripted backend which scenario it is on, as Runner does."""
    from src.eval.policies import AgentPolicy

    class Policy(AgentPolicy):
        def __call__(self, scenario, belief, max_stops):
            self.backend.on_scenario(scenario.scenario_id)
            return super().__call__(scenario, belief, max_stops)
    return Policy(set_name=".", profile=prof, log_dir=log_dir, backend=mod.make())


@pytest.mark.slow
def test_a_failing_scenario_is_one_error_row_and_the_rest_complete(built, tmp_path, monkeypatch):
    _, ids = built
    monkeypatch.setenv("AGENT_PASS_TEST_FAIL", ids[2])
    d = fresh(built, tmp_path, "fail")
    rerun.stage_agent(args(d, jobs=3))
    rows = content(d)
    assert list(rows["scenario_id"]) == ids
    err = rows[rows["status"] == "error"]
    assert list(err["scenario_id"]) == [ids[2]] and "injected API failure" in err["reason"].iloc[0]
    assert (rows.drop(err.index)["status"] != "error").all()


@pytest.mark.slow
def test_a_worker_crash_is_an_error_row_and_a_new_worker_goes_on(built, tmp_path, monkeypatch):
    _, ids = built
    monkeypatch.setenv("AGENT_PASS_TEST_CRASH", ids[1])
    d = fresh(built, tmp_path, "crash")
    rerun.stage_agent(args(d, jobs=2))
    rows = content(d)
    assert list(rows["scenario_id"]) == ids
    err = rows[rows["status"] == "error"]
    assert list(err["scenario_id"]) == [ids[1]] and "worker crashed" in err["reason"].iloc[0]
    assert json.loads((d / "agent_run.json").read_text())["passes"][-1]["crashed"] == 1


def test_a_transient_error_is_retried_and_counted(built, tmp_path, monkeypatch):
    _, ids = built
    monkeypatch.setenv("AGENT_PASS_TEST_FLAKY", ids[0])
    d = fresh(built, tmp_path, "flaky")
    rerun.stage_agent(args(d, jobs=1))
    rows = content(d).set_index("scenario_id")
    assert rows.loc[ids[0], "api_retries"] == 1 and rows.loc[ids[0], "status"] != "error"
    assert (rows.drop(ids[0])["api_retries"] == 0).all()
    assert json.loads((d / "agent_run.json").read_text())["passes"][-1]["retry_event_counts"] == {"503": 1}


@pytest.mark.slow
def test_an_auth_error_stops_the_pass_and_resume_finishes_it_without_rerunning(built, tmp_path, monkeypatch):
    _, ids = built
    d = fresh(built, tmp_path, "auth")
    monkeypatch.setenv("AGENT_PASS_TEST_AUTH", ids[3])
    with pytest.raises(SystemExit, match="authentication failed"):
        rerun.stage_agent(args(d, jobs=1))
    assert not (d / "agent.csv").exists()
    done = agent_pass.read_records(d / "parts" / "agent")
    assert set(done) == set(ids[:3])                 # the scenario that failed auth has no record
    with pytest.raises(SystemExit, match="--resume"):   # a started pass is not overwritten
        rerun.stage_agent(args(d, jobs=1))
    before = {p.name: p.stat().st_mtime_ns for p in (d / "trajectories").glob("*.jsonl")}
    monkeypatch.delenv("AGENT_PASS_TEST_AUTH")
    rerun.stage_agent(args(d, jobs=2, resume=True))
    after = {p.name: p.stat().st_mtime_ns for p in (d / "trajectories").glob("*.jsonl")}
    assert all(after[n] == t for n, t in before.items())   # finished scenarios were not rerun
    assert len(after) == len(ids) and list(content(d)["scenario_id"]) == ids
    run = json.loads((d / "agent_run.json").read_text())
    assert [p["skipped"] for p in run["passes"]] == [0, 3] and "stopped" not in run
    with pytest.raises(SystemExit, match="finished"):
        rerun.stage_agent(args(d, jobs=2, resume=True))
    with pytest.raises(SystemExit, match="--force"):
        rerun.stage_agent(args(d, jobs=2))


@pytest.mark.slow
def test_the_timing_sample_runs_first_apart_and_the_tables_pair_it(built, tmp_path):
    _, ids = built
    d = fresh(built, tmp_path, "timing")
    rerun.stage_agent(args(d, jobs=3, timing_sample=3))
    ts = pd.read_csv(d / "timing_sample.csv")
    assert len(ts) == 3 and set(ts["scenario_id"]) < set(ids)
    assert len(list((d / "timing_sample" / "trajectories").glob("*.jsonl"))) == 3
    assert len(content(d)) == len(ids)                          # the pass reran the sampled ones too
    run = json.loads((d / "agent_run.json").read_text())
    assert [p["label"] for p in run["passes"]] == ["timing sample", "agent"] and run["passes"][0]["jobs"] == 1
    assert run["timing_sample"]["k"] == 3 and run["timing_sample_wall_s"] > 0
    runs, paired, pairs = rerun.agent_timing(d)
    assert paired.loc[0, "pairs"] == 3 and paired.loc[0, "same_decision"] == 3   # the script is deterministic
    assert list(runs["episodes"]) == [3, 3, len(ids)]


def test_the_timing_sample_spreads_over_races_and_phases():
    items = []
    for race in ("r1", "r2", "r3"):
        for lap in range(0, 60, 3):
            items.append(SimpleNamespace(scenario_id=f"{race}:D:{lap}", race_id=race, lap=lap,
                                         snapshot=SimpleNamespace(race=SimpleNamespace(total_laps=60))))
    pick = agent_pass.timing_sample(items, 9, seed=0)
    assert len(pick) == 9 and {p.race_id for p in pick} == {"r1", "r2", "r3"}
    assert sorted(agent_pass.phase_of(p) for p in pick) == ["early"] * 3 + ["late"] * 3 + ["mid"] * 3
    assert [p.scenario_id for p in pick] == [p.scenario_id for p in agent_pass.timing_sample(items, 9, seed=0)]
    assert pick == [x for x in items if x in pick]               # in the set's order


def test_retry_classification_and_redaction(monkeypatch):
    from agent_pass_scripted import StatusError
    assert agent_pass.classify(StatusError(429)) == "transient" and agent_pass.classify(StatusError(502)) == "transient"
    assert agent_pass.classify(TimeoutError()) == "transient" and agent_pass.classify(StatusError(401)) == "auth"
    assert agent_pass.classify(ValueError("x")) is None and agent_pass.classify(StatusError(400)) is None
    monkeypatch.setenv("VOYAGER_API_KEY", "secret-value-123")
    assert "secret-value-123" not in agent_pass.redact("bad key secret-value-123 and sk-abcdefghijkl")
    assert "abcdefghijkl" not in agent_pass.redact("sk-abcdefghijkl")

    class Inner:
        name = "test"

        def __init__(self):
            self.n = 0

        def run_turn(self, **kw):
            self.n += 1
            raise StatusError(503)
    b = agent_pass.RetryingBackend(Inner(), backoff=(0, 0))
    with pytest.raises(StatusError):
        b.run_turn()
    assert b.inner.n == agent_pass.TRIES and b.retries == agent_pass.TRIES - 1 and b.name == "test"
