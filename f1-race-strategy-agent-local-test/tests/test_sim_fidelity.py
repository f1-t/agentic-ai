"""The fidelity scorecard (src/sim/fidelity.py) on synthetic races where the truth is known,
its held-out refusal and its hindsight ledger. Real races are `slow`."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sim import cross_race, fidelity
from src.sim.config import SimConfig
from src.sim.engine import Simulator

SYN = "2099_01_synthetic"


@pytest.fixture(scope="module")
def noiseless():
    """A synthetic race drawn from the simulator's own model (linear wear, linear fuel, a pace per
    driver, the pit loss on the in and out laps, the lap-1 extra) with no lap noise and no SC."""
    keep = cross_race.SYN_NOISE_S
    cross_race.SYN_NOISE_S = 0.0
    try:
        return cross_race.synthetic_race(SYN, sc=False)
    finally:
        cross_race.SYN_NOISE_S = keep


def _spec():
    """Judge and belief on the generating model: linear wear and fuel with a pace per driver
    (driver_effects), full window."""
    cfg = SimConfig(degradation={"name": "linear_fuel", "params": {"driver_effects": "race"}}).model_dump()
    # the judge charges the generating model's pit loss (the whole-race estimate is a median of
    # stops that carry each driver's pace, so it is near, not at, the truth)
    return fidelity.Spec(label="t", world=cfg, belief=copy.deepcopy(cfg),
                         world_race={"pit_loss_s": cross_race.SYN_PIT_LOSS_S})


FAST = fidelity.Options(b_drivers=3, b_every=10, samples=1, n_one=3, n_two=2, loso=False, grid_references=False,
                        hindsight_variants=False, components=False)


# --- refusal ------------------------------------------------------------------------------------
@pytest.mark.parametrize("race", cross_race.held_out_races())
def test_held_out_races_are_refused_before_loading(race, monkeypatch, tmp_path):
    def no_load(*a, **k):
        raise AssertionError("a held-out race was loaded")
    monkeypatch.setattr(fidelity, "load", no_load)
    monkeypatch.setattr(cross_race, "load_race", no_load)
    with pytest.raises(ValueError, match="held out|evaluation races"):
        fidelity.main(["run", "--races", "2021_04_spain", race, "--out", str(tmp_path), "--cache", "none"])
    with pytest.raises(ValueError):
        fidelity.guard([race])


def test_never_list_is_refused_without_eval_set(monkeypatch, tmp_path):
    monkeypatch.setattr(cross_race, "EVAL_SET", tmp_path / "missing.json")
    monkeypatch.setattr(cross_race.guard_held_out, "__defaults__", (tmp_path / "missing.json", False))
    with pytest.raises(ValueError, match="evaluation races"):
        fidelity.guard(["2024_11_austria"])


# --- ledger -------------------------------------------------------------------------------------
def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_ledger_lists_whole_race_inputs():
    judge = _by_id(fidelity.ledger(SimConfig(fit_window="full", stint_limit_margin_laps=3), "judge"))
    assert judge["fit_window"]["tag"] == "this race, whole"
    assert judge["fit_window"]["substitute"] == "pre_race_fit"
    assert judge["pit_loss"]["tag"] == "this race, whole"
    assert judge["stint_limit"]["tag"] == "this race, whole"
    belief = _by_id(fidelity.ledger(SimConfig(fit_window="to_snapshot"), "predictive"))
    assert belief["fit_window"]["tag"] == "this race, to snapshot"
    off = _by_id(fidelity.ledger(SimConfig(stint_limit_margin_laps=None, compound_proxy={"enabled": False}), "judge"))
    assert "stint_limit" not in off and "proxy_raced" not in off


def test_ledger_traffic_declared_and_unclassified(tmp_path):
    cfg = SimConfig(traffic={"name": "ghost"}, min_samples=7)
    rows = _by_id(fidelity.ledger(cfg, "judge"))
    assert rows["traffic"]["tag"] == "this race, whole"
    assert rows["field:min_samples"]["tag"] == "unclassified"
    path = tmp_path / "w.yaml"
    path.write_text("fit_window: full\nx_ledger:\n  - {id: pit_loss, input: pit loss from history, tag: history}\n")
    spec = fidelity.make_spec(world=path)
    led = fidelity.ledger_frame(spec)
    row = led[(led["role"] == "judge") & (led["id"] == "pit_loss")].iloc[0]
    assert row["tag"] == "history" and row["source"] == "declared"
    with pytest.raises(ValueError):
        fidelity.ledger(SimConfig(), "judge", [{"id": "x", "input": "y", "tag": "not a tag"}])


def test_substitutes_follow_the_ledger():
    spec = fidelity.Spec(label="t", world=SimConfig().model_dump(), belief=SimConfig().model_dump())
    subs = fidelity.substitutes_for(spec)
    assert {"pre_race_fit", "pit_loss_prior", "cliff_history_only", "proxy_off"} <= set(subs)
    v = fidelity.substituted(spec, ["pit_loss_prior", "pre_race_fit"])
    assert v.world_race["pit_loss_s"] == "prior"
    assert v.world_config().fit_window == "pre_race"
    assert "pit_loss_prior" not in fidelity.substitutes_for(v)


# --- A: a judge fitted on a race drawn from its own model ----------------------------------------
def test_judge_on_its_own_model_is_near_exact(noiseless):
    a = fidelity.section_a(noiseless, _spec(), FAST)
    d = a["a_drivers"]
    assert set(d["variant"]) == {"in", "lodo"}
    for v in ("in", "lodo"):
        part = d[d["variant"] == v]
        assert len(part) == 16
        assert part["err_pace_s"].abs().max() < 1e-6, part[["driver", "err_pace_s"]]
    assert a["a_stop_gain"]["err_s"].abs().max() < 1e-6
    o = fidelity.order_rows_from_a(d.assign(race_id=SYN), noiseless, None, "1 judge", "t").assign(race_id=SYN)
    m = fidelity.order_measures(o[(o["fit"] == "leave-one-driver-out") & (o["pace"] == "insample")])
    assert m["spearman"] > 0.99 and m["mae_pos"] < 0.2
    assert m["close_pairs"] == 0 or m["close_pair_acc"] > 0.95


def test_leave_one_stint_out_reports_the_held_out_stint(noiseless):
    a = fidelity.section_a(noiseless, _spec(), fidelity.Options(lodo=False, grid_references=False), variants=("in", "loso"))
    s = a["a_stints"]
    loso = s[s["variant"] == "loso"]
    assert len(loso) == len(s[s["variant"] == "in"])
    assert loso["err_s"].abs().max() < 1e-6


# --- B: a deliberately wrong predictive simulator -----------------------------------------------
def test_wrong_belief_shows_bias_and_regret(noiseless):
    world = Simulator(noiseless.frame, SimConfig(), noiseless.practice)
    belief = Simulator(noiseless.frame, SimConfig(), noiseless.practice)
    wrong = copy.copy(belief.degradation)
    # 2 s a lap slower, and wear five times too steep: it wants more stops than the judge
    wrong.coefs = {c: (b + 2.0, 5 * w) for c, (b, w) in belief.degradation.coefs.items()}
    belief.degradation = wrong
    b = fidelity.section_b(noiseless, _spec(), FAST, belief=belief, world=world)
    rows, snaps = b["b_rows"], b["b_snaps"]
    valid = rows[rows["valid"]]
    assert (valid["err_s"] > 0).all()
    assert (valid["err_s"] / valid["horizon"]).min() > 2.0   # the base error plus the extra wear
    assert (snaps["fav_b_stops"] >= snaps["fav_w_stops"]).all()
    assert snaps["regret_s"].mean() > 0
    assert (snaps["regret_s"] >= -1e-9).all()


def test_right_belief_shows_no_bias_and_no_regret(noiseless):
    world = Simulator(noiseless.frame, SimConfig(), noiseless.practice)
    b = fidelity.section_b(noiseless, _spec(), FAST, belief=world, world=world)
    assert b["b_rows"]["err_s"].abs().max() < 1e-6
    assert b["b_snaps"]["regret_s"].abs().max() < 1e-6
    assert (b["b_snaps"]["rho"] > 0.999).all()


# --- the panel, the order measures, the registry -------------------------------------------------
def test_plan_panel(noiseless):
    from src.sim import calibration
    snap = calibration.snapshot_at(noiseless.frame, noiseless.race, "D00", 10)
    real = calibration.remaining_plan(noiseless.frame, "D00", 10)
    panel = fidelity.plan_panel(snap, ["HARD", "MEDIUM", "SOFT"], real, 4, 3)
    names = [n for n, _, _ in panel]
    assert names[0] == "real" and "stay" in names
    assert any(p.stints[0].laps == 1 for _, _, p in panel)   # pit now
    assert all(p.total_laps == snap.remaining_laps for _, _, p in panel)
    assert len({tuple((s.compound, s.laps) for s in p.stints) for _, _, p in panel}) == len(panel)


def test_order_measures_close_pairs():
    t = pd.DataFrame({"laps": [50] * 4, "real_s": [100.0, 105.0, 130.0, 200.0], "real_pos": [1, 2, 3, 4],
                      "pred_total_s": [106.0, 104.0, 131.0, 150.0]})
    m = fidelity.order_measures(t)
    assert m["close_pairs"] == 1 and m["close_pair_acc"] == 0.0   # the 5 s pair is swapped
    assert m["mae_pos"] == pytest.approx(0.5)
    assert m["rel_gap_mae_s"] == pytest.approx(np.mean(np.abs(np.array([6, -1, 1, -50]) - (-11.0))))


def test_component_registry_reports_missing_code(noiseless, monkeypatch):
    def absent(ctx):
        raise fidelity.NotPresent("needs branch feat/example")
    comps = {"fuel": fidelity.COMPONENTS["fuel"],
             "example": fidelity.Component("example", absent, lambda ctx: [])}
    monkeypatch.setattr(fidelity, "COMPONENTS", comps)
    d = fidelity.section_d(noiseless, _spec(), FAST)
    status = d["d_status"].set_index("component")["status"]
    assert status["example"] == "not present" and status["fuel"] == "ok"
    fuel = d["d_rows"][d["d_rows"]["component"] == "fuel"]
    judge_fit = fuel[fuel["source"].str.startswith("belief")]
    assert judge_fit["err"].abs().max() < 1e-6   # a full-window belief fits the true fuel slope exactly


def test_cache_reuses_sections(noiseless, monkeypatch, tmp_path):
    monkeypatch.setattr(fidelity, "load", lambda race_id, profile=None: noiseless)
    calls = []
    real_a = fidelity.section_a

    def counted(*a, **k):
        calls.append(1)
        return real_a(*a, **k)
    monkeypatch.setattr(fidelity, "section_a", counted)
    opts = fidelity.Options(**{**FAST.__dict__, "lodo": False})
    fidelity.race_tables(SYN, _spec(), opts, tmp_path)
    fidelity.race_tables(SYN, _spec(), opts, tmp_path)
    belief_only = _spec().with_factor("b", belief={"fit_window": "pre_race"})
    fidelity.race_tables(SYN, belief_only, opts, tmp_path)
    assert len(calls) == 1   # the judge's section A is shared by the belief-only change


def test_bootstrap_delta_of_identical_scorecards_is_zero(noiseless, monkeypatch):
    monkeypatch.setattr(fidelity, "load", lambda race_id, profile=None: noiseless)
    t = fidelity.race_tables(SYN, _spec(), FAST, None)
    per = {"r1": t, "r2": t}
    d = fidelity.bootstrap_delta(per, per, n_boot=50)
    assert all(v[0] == 0 or np.isnan(v[0]) for v in d.values())


# --- real races ---------------------------------------------------------------------------------
@pytest.mark.slow
def test_scorecard_on_spain(tmp_path):
    spec = fidelity.make_spec()
    opts = fidelity.Options(b_drivers=2, b_every=15, samples=4, loso=False)
    res = fidelity.run_specs([spec], ["2021_04_spain"], opts, tmp_path / "cache", n_jobs=1, log=lambda *a: None)
    per = res["main"]
    out = fidelity.write_scorecard("main", per, spec, opts, tmp_path / "main")
    assert (out / "scorecard.md").exists() and (out / "order.csv").exists()
    h = fidelity.headline(per)
    assert 0 <= h["A_close_pair_acc"] <= 1 and np.isfinite(h["B2_regret_s"])
    led = pd.read_csv(out / "ledger.csv")
    assert ((led["role"] == "judge") & (led["tag"] == "this race, whole")).any()


def test_hybrid_judge_replays_recorded_and_offers_the_rate_substitute():
    cfg = SimConfig(events={"name": "hybrid", "params": {"lap_times": "recorded"}})
    assert fidelity._replay_config(cfg).events.name == "recorded"
    assert fidelity.replay_events(fidelity._replay_config(cfg)) == "recorded"
    rows = _by_id(fidelity.ledger(cfg, "judge"))
    assert rows["events"]["tag"] == "this race, whole" and rows["events"]["substitute"] == "events_rate"
    belief = _by_id(fidelity.ledger(SimConfig.from_file(fidelity.BELIEF_FILE), "predictive"))
    assert belief["clean_flag"]["tag"] == "this race, to snapshot"
    assert belief["race_length"]["tag"] == "regulation" and belief["compound_list"]["tag"] == "regulation"


def test_absent_switches_are_skipped():
    ab = {"factors": {"ok": {"both": {"degradation": {"name": "linear_fuel_hinge"}}},
                      "later": {"both": {"not_a_field_yet": {"x": 1}}},
                      "module": {"requires": ["src.sim.no_such_module"], "both": {}}},
          "all_on": ["ok", "later"]}
    skipped = []
    specs = fidelity.ablation_specs(_spec(), ab, skipped)
    assert [s.label for s in specs] == ["base", "ok", "all_on"]
    assert {x["factor"] for x in skipped} == {"later", "module"}


def test_recorded_judge_replays_its_own_model_exactly(noiseless):
    cfg = SimConfig(degradation={"name": "linear_fuel", "params": {"driver_effects": "race"}},
                    events={"name": "recorded", "params": {"lap_times": "factor"}}).model_dump()
    spec = fidelity.Spec(label="t", world=cfg, belief=copy.deepcopy(cfg),
                         world_race={"pit_loss_s": cross_race.SYN_PIT_LOSS_S})
    a = fidelity.section_a(noiseless, spec, FAST)
    assert a["a_drivers"]["err_pace_s"].abs().max() < 1e-6


def _blas_threads(_):
    import os
    return os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")


def test_workers_run_one_blas_thread():
    assert fidelity.DEFAULT_JOBS <= 6
    with fidelity.ProcessPoolExecutor(1) as ex:   # the pool run_specs uses: spawned, one BLAS thread
        assert list(ex.map(_blas_threads, [0])) == [("1", "1")]


def test_ledger_follows_the_wear_cliff_edge():
    def edge(world_edge):
        return _by_id(fidelity.ledger(SimConfig(wear_cliff={"world_edge": world_edge}), "judge"))["cliff_edge"]
    assert edge("history")["tag"] == "this race, whole" and edge("history")["substitute"] == "cliff_history_only"
    assert edge("race")["tag"] == "this race, whole"
    assert edge("history_only")["tag"] == "history" and edge("history_only")["substitute"] is None
    belief = _by_id(fidelity.ledger(SimConfig.from_file(fidelity.BELIEF_FILE), "predictive"))
    assert belief["cliff_edge"]["tag"] == "this race, to snapshot"
    assert "stint_limit" not in belief   # R8 retired


def test_only_runs_the_named_factors():
    ab = {"factors": {"a": {"both": {"degradation": {"name": "linear_fuel_hinge"}}},
                      "b": {"world": {"wear_cliff": {"world_edge": "race"}}}}, "all_on": ["a", "b"]}
    assert [s.label for s in fidelity.ablation_specs(_spec(), ab, only=["b"])] == ["base", "b"]
    assert [s.label for s in fidelity.ablation_specs(_spec(), ab, only=["b", "all_on"])] == ["base", "b", "all_on"]
    with pytest.raises(ValueError, match="unknown"):
        fidelity.ablation_specs(_spec(), ab, only=["c"])
