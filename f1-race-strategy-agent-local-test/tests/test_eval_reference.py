"""SPEC 0019: the reference cache, paired standard errors for regret, and within_mc. Offline,
on the synthetic race of test_eval."""
import json

import numpy as np
import pandas as pd
import pytest
from test_eval import HAS_CACHE, RACE, SC_LAPS, SCENARIOS_V1, race_frame

from src.eval import Decision, ScenarioSet, resolve, sample_race, score_scenario, summarize
from src.eval.__main__ import main
from src.eval.reference import (MISS_NOTICE, NO_STOP, REFERENCE_FORMAT, Reference, ReferenceMissing,
                                ReferenceStore, build_references, compute_reference, load_references, near_ties,
                                optimal_set, paired_se, recorded_events, reference_inputs, reference_key,
                                set_agreement, stale_references)
from src.eval.score import _first_pit_error, score_set, summarize_near_ties, to_markdown
from src.sim import SimConfig, Simulator
from src.sim.optimizer_bridge import optimize_batch


@pytest.fixture(scope="module")
def frame():
    return race_frame()


@pytest.fixture(scope="module")
def world(frame):
    """Drawn SC/VSC events, so the samples differ and the standard errors are not 0."""
    return Simulator(frame, SimConfig(degradation={"name": "linear"},
                                      events={"name": "rate", "params": {"p_sc": 0.05, "p_vsc": 0.05}}))


@pytest.fixture(scope="module")
def flat(frame):
    return Simulator(frame, SimConfig(degradation={"name": "linear"}))


@pytest.fixture(scope="module")
def scenario_set(frame):
    s = ScenarioSet(version="tref", items=sample_race(frame, RACE)[0][:4])
    s.digest = s.compute_digest()
    return s


def _fixed(plan):
    return lambda scenario, sim, max_stops: Decision(plan)


def _drop_latency(rows):
    return rows.drop(columns=["latency_s"]).reset_index(drop=True)


# --- key ---------------------------------------------------------------------------
KEY_ARGS = dict(set_digest="d" * 64, world_config_hash="abc123abc123", sim_code_version="0123456789ab",
                seed=0, n_samples=50, max_stops=2, hindsight=False)


def test_reference_key_is_stable_and_moves_with_every_input():
    key = reference_key(**KEY_ARGS)
    assert key == reference_key(**dict(reversed(list(KEY_ARGS.items()))))
    assert key == "3324c78ed123"   # pinned: a change here orphans every cached reference
    changed = {"set_digest": "e" * 64, "world_config_hash": "abc123abc124", "sim_code_version": "0123456789ac",
               "seed": 1, "n_samples": 51, "max_stops": 1, "hindsight": True}
    for name, value in changed.items():
        assert reference_key(**{**KEY_ARGS, name: value}) != key, name


def test_reference_inputs_resolve_the_worlds_n_samples(scenario_set, world):
    inputs = reference_inputs(scenario_set, world)
    assert inputs["n_samples"] == world.config.n_samples and inputs["hindsight"] is False
    assert inputs["set_digest"] == scenario_set.digest
    assert set(inputs) == set(KEY_ARGS)


# --- optimize_batch per-draw times (opt-in) -------------------------------------------
def test_optimize_batch_times_are_opt_in(scenario_set, world):
    snap = scenario_set.items[0].snapshot
    default = optimize_batch(snap, world, n_samples=8)
    assert "times" not in default.diagnostics and "runner_up" not in default.diagnostics
    on = optimize_batch(snap, world, n_samples=8, return_times=True)
    assert on.plan == default.plan and on.evaluation == default.evaluation
    assert np.array_equal(on.diagnostics["times"], np.asarray(default.evaluation.seed_times_s))
    second = on.diagnostics["runner_up"]
    assert second["plan"] != on.plan and second["mean_time_s"] >= on.evaluation.mean_time_s
    assert second["times"].shape == (8,)


# --- the cache -----------------------------------------------------------------------
def test_cached_and_uncached_rows_are_identical(scenario_set, world, tmp_path, capsys):
    (path, refs), = build_references(scenario_set, lambda race_id, role: world, n_samples=12, root=tmp_path)
    assert path.parent == tmp_path / "tref" and path.with_suffix(".json").exists()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert path.stem == meta["key"] == reference_key(**{k: meta[k] for k in KEY_ARGS})
    loaded = load_references(path)
    for ref in refs:
        got = loaded[ref.scenario_id]
        assert got.times.dtype == np.float32 and np.array_equal(got.times, ref.times)
        assert got.plan == ref.plan and got.mean_s == ref.mean_s and got.runner_up_se_s == ref.runner_up_se_s
        assert got.cvar_90_s >= got.mean_s and got.pit_now_opt == ref.pit_now_opt

    pols = {**resolve(["baselines"]), "probe": _fixed([["SOFT", 20], ["MEDIUM", 20]])}
    sims = lambda race_id, role: world   # noqa: E731
    capsys.readouterr()
    uncached = score_set(scenario_set, sims, pols, n_samples=12)
    cached = score_set(scenario_set, sims, pols, n_samples=12, references=ReferenceStore(tmp_path, require=True))
    assert MISS_NOTICE not in capsys.readouterr().out
    pd.testing.assert_frame_equal(_drop_latency(uncached), _drop_latency(cached))
    assert (cached.loc[cached.status == "ok", "regret_se_s"] > 0).any()


def test_a_miss_computes_and_says_so_once(scenario_set, world, tmp_path, capsys):
    pols = {"probe": _fixed([["SOFT", 20], ["MEDIUM", 20]])}
    store = ReferenceStore(tmp_path)
    rows = score_set(scenario_set, lambda race_id, role: world, pols, n_samples=4, references=store)
    assert capsys.readouterr().out.count(MISS_NOTICE) == 1 and store.missed
    assert len(rows) == len(scenario_set.items)
    with pytest.raises(ReferenceMissing):
        score_set(scenario_set, lambda race_id, role: world, pols, n_samples=4,
                  references=ReferenceStore(tmp_path, require=True))


def test_other_inputs_miss_the_cache(scenario_set, world, tmp_path):
    build_references(scenario_set, lambda race_id, role: world, n_samples=4, root=tmp_path)
    store = ReferenceStore(tmp_path, require=True)
    scenario = scenario_set.items[0]
    assert store.get(scenario_set, scenario, world, n_samples=4) is not None
    for kw in ({"n_samples": 5}, {"n_samples": 4, "seed": 1}, {"n_samples": 4, "max_stops": 1}):
        with pytest.raises(ReferenceMissing):
            store.get(scenario_set, scenario, world, **kw)


def test_check_finds_a_reference_from_another_code_version(scenario_set, world, tmp_path, capsys):
    (path, _), = build_references(scenario_set, lambda race_id, role: world, n_samples=4, root=tmp_path)
    assert stale_references(tmp_path) == []
    old = tmp_path / "tref" / "000000000000.json"
    meta = json.loads(path.with_suffix(".json").read_text())
    old.write_text(json.dumps({**meta, "sim_code_version": "ffffffffffff", "key": "000000000000"}))
    stale = stale_references(tmp_path)
    assert [p.name for p, _ in stale] == ["000000000000.parquet"]
    assert main(["reference", "--check", "--reference-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "000000000000.parquet" in out and "ffffffffffff" in out and path.name not in out


def test_hindsight_fixes_the_events_to_the_record(scenario_set, world):
    scenario = scenario_set.items[0]
    events = recorded_events(world, scenario.snapshot)
    assert set(events.sc_laps) == SC_LAPS and events.vsc_laps == []
    ref = compute_reference(scenario, world, n_samples=6, hindsight=True)
    assert np.ptp(ref.times) == 0   # every draw runs the recorded SC: no spread left
    assert reference_key(**reference_inputs(scenario_set, world, hindsight=True)) != \
        reference_key(**reference_inputs(scenario_set, world))


# --- regret_se_s and within_mc -----------------------------------------------------------
class KnownWorld:
    """The synthetic world, with each scored plan's per-draw times replaced by the reference's
    plus known differences, so regret and its standard error are known in closed form."""

    def __init__(self, real, base, diff):
        self.real, self.config, self.base, self.diff = real, real.config, base, diff

    def simulate_batch(self, snapshot, plans, *, seed=0, n_samples=None, **kw):
        out = self.real.simulate_batch(snapshot, plans, seed=seed, n_samples=n_samples, **kw)
        out.times = (self.base.astype(float) + self.diff)[None, :].repeat(len(plans), axis=0)
        return out


@pytest.mark.parametrize("diff", [[1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, -1.0, 1.5], [0.5, 0.5, 0.5, 0.5]])
def test_regret_se_on_a_world_with_known_differences(scenario_set, flat, diff):
    scenario = scenario_set.items[0]
    real = compute_reference(scenario, flat, n_samples=4)
    base = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)
    ref = Reference(scenario.scenario_id, plan=real.plan, times=base, mean_s=float(base.mean()),
                    pit_now_opt=real.pit_now_opt)
    diff = np.array(diff)
    world = KnownWorld(flat, base, diff)
    (row,) = score_scenario(scenario, world, {"p": _fixed([["SOFT", 20], ["MEDIUM", 20]])}, belief=flat,
                            n_samples=4, reference=ref)
    se = np.std(diff, ddof=1) / 2
    assert row["status"] == "ok"
    assert row["regret_s"] == pytest.approx(diff.mean())
    assert row["regret_se_s"] == pytest.approx(se, abs=1e-12)
    assert row["within_mc"] == (abs(diff.mean()) <= 2 * se)
    assert paired_se(base + diff, base) == pytest.approx(se, abs=1e-12)


def test_within_mc_at_one_draw_is_the_tie_rule(scenario_set, world):
    snap_opt = compute_reference(scenario_set.items[0], world, n_samples=1)
    pols = {**resolve(["baselines"]), "opt": _fixed(snap_opt.plan.as_pairs())}
    rows = score_set(scenario_set, lambda race_id, role: world, pols, n_samples=1)
    ok = rows[rows.status == "ok"]
    assert (ok["regret_se_s"] == 0).all()
    assert (ok["within_mc"] == (ok["regret_s"] == 0)).all()
    assert ok["within_mc"].any() and not ok["within_mc"].all()
    table = summarize(rows).set_index(["race_id", "policy"])
    assert table.loc[("ALL", "opt"), "share_within_mc"] >= 0.25
    for pol, g in ok.groupby("policy"):
        assert table.loc[("ALL", pol), "share_within_mc"] == pytest.approx(g["within_mc"].mean())


def test_reports_without_within_mc_still_summarize():
    rows = pd.DataFrame([{"race_id": "r", "policy": "p", "status": "ok", "regret_s": 1.0, "regret_pct": 0.1,
                          "stop_agree": True, "first_pit_error": 0}])
    assert np.isnan(summarize(rows)["share_within_mc"]).all()


# --- the jobs runner reads the cache -------------------------------------------------------
@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_eval_jobs_runner_reads_the_cache_and_carries_require_reference(tmp_path):
    from src.eval.score import install_job_runner, race_simulators
    from src.sim import jobs
    from src.sim.compute import load_profile

    profile = load_profile("local")
    small = ScenarioSet(version="tjobs", items=ScenarioSet.load(SCENARIOS_V1).items[:3])
    path = small.save(tmp_path / "tjobs.json")
    refs, empty = tmp_path / "refs", tmp_path / "empty"
    build_references(small, race_simulators(profile), n_samples=profile.sim.n_samples,
                     max_stops=profile.sim.max_stops, root=refs)
    try:
        install_job_runner(path, ["fixed_one_stop"], reference_dir=refs, require_reference=True)
        spec = jobs._EVAL_SPEC[0]
        assert spec["require_reference"] is True and spec["reference_dir"] == str(refs.resolve())
        assert jobs.run_task(jobs.JOBS["eval_chunk"], "chunk_0000", profile, tmp_path / "a") == jobs.DONE
        got = pd.read_parquet(tmp_path / "a" / "eval" / "chunk_0000.parquet")
        want = score_set(small, race_simulators(profile), resolve(["fixed_one_stop"]),
                         n_samples=profile.sim.n_samples, max_stops=profile.sim.max_stops, warn=False)
        assert len(got) == 3 and "regret_se_s" in got
        pd.testing.assert_frame_equal(_drop_latency(got), _drop_latency(want), check_dtype=False)

        jobs.set_eval_runner(None)
        jobs.install_eval_runner({**spec, "reference_dir": str(empty)})   # as a run-all worker does
        with pytest.raises(ReferenceMissing):
            jobs.run_task(jobs.JOBS["eval_chunk"], "chunk_0000", profile, tmp_path / "b")
    finally:
        jobs.set_eval_runner(None)


# --- near ties (SPEC 0019 D-measure, information only) ----------------------------------
NEW_COLUMNS = ["opt_set_size", "best_pit_now_s", "best_stay_s", "pit_now_gap_s", "pit_now_gap_se_s",
               "pit_now_decisive", "stop_agree_set", "first_pit_error_set"]


def test_optimize_batch_returns_every_searched_plan_in_tie_break_order(scenario_set, world):
    snap = scenario_set.items[0].snapshot
    on = optimize_batch(snap, world, n_samples=8, return_times=True)
    searched = on.diagnostics["searched"]
    k = on.candidates_evaluated
    assert len(searched["plans"]) == k and searched["times"].shape == (k, 8) and searched["mean_time_s"].shape == (k,)
    assert searched["plans"][0] == on.plan and searched["plans"][1] == on.diagnostics["runner_up"]["plan"]
    assert np.array_equal(searched["times"][0], on.diagnostics["times"])
    assert np.array_equal(searched["times"][1], on.diagnostics["runner_up"]["times"])
    assert (np.diff(searched["mean_time_s"]) >= 0).all()
    assert "searched" not in optimize_batch(snap, world, n_samples=8).diagnostics


def test_searched_plans_round_trip_and_match_a_direct_computation(scenario_set, world, tmp_path):
    (path, refs), = build_references(scenario_set, lambda race_id, role: world, n_samples=12, root=tmp_path)
    assert json.loads(path.with_suffix(".json").read_text())["format"] == REFERENCE_FORMAT == 2
    loaded = load_references(path)
    pitted = []
    for scenario, ref in zip(scenario_set.items, refs):
        got = loaded[ref.scenario_id]
        for name, dtype in (("plan_gap_s", np.float32), ("plan_se_s", np.float32), ("plan_num_stops", np.int8),
                            ("plan_first_pit_lap", np.int16)):
            assert getattr(got, name).dtype == dtype and np.array_equal(getattr(got, name), getattr(ref, name))
        assert got.pit_now_gap_se_s == ref.pit_now_gap_se_s or np.isnan(got.pit_now_gap_se_s) == np.isnan(
            ref.pit_now_gap_se_s) == True   # noqa: E712

        snap = scenario.snapshot
        opt = optimize_batch(snap, world, n_samples=12, return_times=True).diagnostics["searched"]
        assert got.plan_gap_s[0] == 0 and got.plan_se_s[0] == 0
        assert got.plan_se_s[1] == pytest.approx(ref.runner_up_se_s, rel=1e-6)
        se = np.array([paired_se(t, opt["times"][0]) for t in opt["times"]])
        gap = opt["mean_time_s"] - opt["mean_time_s"][0]
        firsts = [(p.pit_laps(snap) or [NO_STOP])[0] for p in opt["plans"]]
        now = np.array(firsts) == snap.laps_completed + 1
        keep = gap <= np.maximum(10.0, 5 * se)   # the storage bound, plus the best of each kind
        keep[[int(np.argmax(m)) for m in (now, ~now) if m.any()]] = True
        assert keep.sum() < len(keep)   # the bound drops plans here
        assert np.allclose(got.plan_se_s, se[keep], rtol=1e-6, atol=1e-6)
        assert np.allclose(got.plan_gap_s, gap[keep], rtol=1e-6, atol=1e-6)
        assert got.plan_first_pit_lap.tolist() == np.array(firsts)[keep].tolist()

        ties = near_ties(got, snap)
        assert ties["opt_set_size"] == len(optimal_set(got)) >= 1 and optimal_set(got)[0] == 0
        assert ties["opt_set_size"] == int((gap <= 2 * se + 1e-6).sum())
        assert (~now).any()
        if not now.any():   # pitting now would leave a stint under the minimum
            assert np.isnan(ties["best_pit_now_s"]) and np.isnan(ties["pit_now_gap_se_s"])
            assert ties["pit_now_decisive"] is None and np.isfinite(ties["best_stay_s"])
            continue
        pitted.append(scenario.scenario_id)
        i_pit, i_stay = int(np.argmax(now)), int(np.argmax(~now))
        assert ties["best_pit_now_s"] == pytest.approx(opt["mean_time_s"][i_pit], abs=1e-3)
        assert ties["best_stay_s"] == pytest.approx(opt["mean_time_s"][i_stay], abs=1e-3)
        assert ties["pit_now_gap_s"] == pytest.approx(gap[i_pit] - gap[i_stay], abs=1e-3)
        assert ties["pit_now_gap_se_s"] == pytest.approx(paired_se(opt["times"][i_pit], opt["times"][i_stay]))
        assert ties["pit_now_decisive"] == (abs(ties["pit_now_gap_s"]) > 2 * ties["pit_now_gap_se_s"])
    assert pitted   # both branches are covered
    assert any(len(optimal_set(r)) > 1 for r in loaded.values())   # drawn events: some near ties


def test_set_agreement_takes_the_closest_member_of_the_optimal_set():
    ref = Reference("s", plan_gap_s=np.array([0.0, 0.1, 0.5, 3.0], dtype=np.float32),
                    plan_se_s=np.array([0.0, 0.1, 0.2, 0.1], dtype=np.float32),
                    plan_num_stops=np.array([1, 2, 0, 1], dtype=np.int8),
                    plan_first_pit_lap=np.array([30, 20, NO_STOP, 25], dtype=np.int16))
    assert optimal_set(ref).tolist() == [0, 1]   # 0.5 > 2 * 0.2 and 3.0 > 2 * 0.1 are out
    assert set_agreement(ref, 2, [22, 40], _first_pit_error) == (True, 2)
    assert set_agreement(ref, 1, [27], _first_pit_error) == (True, 3)
    assert set_agreement(ref, 0, [], _first_pit_error) == (False, None)
    assert set_agreement(Reference("s"), 1, [27], _first_pit_error) == (None, None)


def test_without_drawn_events_the_optimal_set_holds_only_exact_ties(scenario_set, flat):
    """No spread, so every standard error is 0 and only a plan with the reference's very mean
    joins it. The synthetic race has no fuel effect, so swapping two later stints is such a
    tie (scenario AAA:05); the Spain world below has none."""
    for scenario in scenario_set.items:
        ref = compute_reference(scenario, flat, n_samples=6)
        members = optimal_set(ref)
        assert (ref.plan_se_s == 0).all() and (ref.plan_gap_s[members] == 0).all()
        ties = near_ties(ref, scenario.snapshot)
        if ties["pit_now_decisive"] is not None:
            assert ties["pit_now_decisive"] and ties["pit_now_gap_se_s"] == 0


def _deterministic_world_rows(scenario_set, sims, pols, n_samples):
    rows = score_set(scenario_set, sims, pols, n_samples=n_samples)
    assert set(NEW_COLUMNS) <= set(rows.columns)
    return rows, rows.drop(columns=NEW_COLUMNS)


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_the_default_world_is_decisive_with_one_plan_sets_and_the_same_tables():
    """The deterministic default world on a slice of v1: every optimal set is the reference
    alone, every pit-now call is decisive, the set-based agreement is the plain one, and the
    summary table is the one without the near-tie columns."""
    from src.eval.score import race_simulators
    from src.sim.compute import load_profile
    profile = load_profile("local")
    v1 = ScenarioSet.load(SCENARIOS_V1)
    part = ScenarioSet(version="tv1", items=v1.items[::12])
    part.digest = part.compute_digest()
    rows, old = _deterministic_world_rows(part, race_simulators(profile), resolve(["baselines"]),
                                          profile.sim.n_samples)
    assert (rows["opt_set_size"] == 1).all()
    decisive = rows["pit_now_decisive"].dropna()
    assert len(decisive) > len(rows) / 2 and decisive.astype(bool).all()
    ok = rows[rows.status == "ok"]
    assert len(ok) and (ok["stop_agree_set"] == ok["stop_agree"]).all()
    assert ok["first_pit_error_set"].fillna(-1).tolist() == ok["first_pit_error"].fillna(-1).tolist()
    pd.testing.assert_frame_equal(summarize(rows), summarize(old))
    ties = summarize_near_ties(rows).set_index("policy")
    assert (ties["share_opt_set_gt1"] == 0).all() and (ties["share_pit_now_decisive"] == 1).all()


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
@pytest.mark.parametrize("events", [None, {"name": "rate"}])
def test_the_storage_bound_leaves_the_optimal_set_and_the_best_of_each_kind_unchanged(events):
    """On a slice of v1, in the default world and with drawn events: the stored plans
    (gap <= max(10 s, 5 SE), plus the best pit-now and stay-out plans) give the same optimal
    set and near-tie columns as every searched plan, and set_agreement agrees for any plan."""
    from src.eval.score import race_simulators
    from src.sim.compute import load_profile
    profile = load_profile("local")
    config = None if events is None else SimConfig.model_validate(
        {"fit_window": "full", "traffic": {"name": "none"}, "events": events})
    sims = race_simulators(profile, None, config)
    dropped = 0
    for scenario in ScenarioSet.load(SCENARIOS_V1).items[::24]:
        world = sims(scenario.race_id, "world")
        kw = dict(n_samples=20, max_stops=profile.sim.max_stops)
        bounded, full = compute_reference(scenario, world, **kw), compute_reference(scenario, world, keep_all=True,
                                                                                      **kw)
        dropped += len(full.plan_gap_s) - len(bounded.plan_gap_s)
        a, b = optimal_set(bounded), optimal_set(full)
        for name in ("plan_gap_s", "plan_se_s", "plan_num_stops", "plan_first_pit_lap"):
            assert np.array_equal(getattr(bounded, name)[a], getattr(full, name)[b]), name
        snap = scenario.snapshot
        assert pd.Series(near_ties(bounded, snap)).equals(pd.Series(near_ties(full, snap)))
        for stops, pits in ((0, []), (1, [snap.laps_completed + 5]), (2, [snap.laps_completed + 1, 60])):
            assert set_agreement(bounded, stops, pits, _first_pit_error) == set_agreement(full, stops, pits,
                                                                                           _first_pit_error)
    assert dropped > 0


def test_the_near_tie_columns_leave_the_existing_tables_unchanged(scenario_set, flat, tmp_path, capsys):
    pols = {**resolve(["baselines"]), "probe": _fixed([["SOFT", 20], ["MEDIUM", 20]])}
    rows, old = _deterministic_world_rows(scenario_set, lambda race_id, role: flat, pols, 6)
    pd.testing.assert_frame_equal(summarize(rows), summarize(old))
    assert to_markdown(summarize(rows)) == to_markdown(summarize(old))
    new_csv, old_csv = tmp_path / "new.csv", tmp_path / "old.csv"
    rows.to_csv(new_csv, index=False)
    old.to_csv(old_csv, index=False)
    capsys.readouterr()
    assert main(["report", str(old_csv)]) == 0
    before = capsys.readouterr().out
    assert "Near-tie view" not in before
    assert main(["report", str(new_csv)]) == 0
    after = capsys.readouterr().out
    assert after.startswith(before) and "Near-tie view" in after
    assert to_markdown(summarize_near_ties(pd.read_csv(new_csv))) == to_markdown(summarize_near_ties(rows))
    assert after == before + "\nNear-tie view (information only; optimal set: plans within 2 SE of the " \
                             "reference):\n" + to_markdown(summarize_near_ties(rows)) + "\n"


def test_a_format_1_cache_still_loads_and_is_flagged_stale(scenario_set, world, tmp_path, capsys):
    import pyarrow.parquet as pq
    (path, _), = build_references(scenario_set, lambda race_id, role: world, n_samples=4, root=tmp_path)
    pq.write_table(pq.read_table(path).drop(["plan_gap_s", "plan_se_s", "plan_num_stops", "plan_first_pit_lap",
                                             "pit_now_gap_se_s"]), path)
    meta = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(json.dumps({k: v for k, v in meta.items() if k != "format"}))
    assert [p for p, _ in stale_references(tmp_path)] == [path]
    assert main(["reference", "--check", "--reference-dir", str(tmp_path)]) == 0
    assert "format 1" in capsys.readouterr().out
    assert all(r.plan_gap_s is None for r in load_references(path).values())
    rows = score_set(scenario_set, lambda race_id, role: world, {"probe": _fixed([["SOFT", 20], ["MEDIUM", 20]])},
                     n_samples=4, references=ReferenceStore(tmp_path, require=True))
    assert rows["opt_set_size"].isna().all() and rows["stop_agree_set"].isna().all()
    assert summarize_near_ties(rows) is None
