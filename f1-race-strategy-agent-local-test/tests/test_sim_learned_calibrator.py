"""The learned calibrator on real races (src/sim/learned_calibrator.py): the held-out guard on
training, the trained artifacts and their loading by race id, the per-factor hybrid, and the
adoption and default rules. Synthetic races only, offline; WANDB_MODE is disabled (conftest.py)."""
import json

import pandas as pd
import pytest

from src.contracts import split_factor
from src.sim import SimConfig, Simulator, calibration, calibrators, cross_race, learned_calibrator as lc
from src.sim.calibrators import CALIBRATORS
from src.sim.compute import load_profile
from src.sim.config import ComponentSpec

SYN = [f"2099_{i:02d}_synthetic" for i in range(1, 7)]
HYBRID = {"targets": "degradation", "factors": ["wear_s_per_lap"]}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts") / "ridge_hybrid"
    res = lc.train_artifacts(SYN, HYBRID, load_profile(), str(out), arm="wear_nonneg",
                             eval_set=tmp_path_factory.mktemp("e") / "none.json")
    return out, res


# --- training refuses held-out races --------------------------------------------------------------
def test_training_refuses_a_held_out_race_before_loading_anything(tmp_path, monkeypatch):
    eval_set = tmp_path / "eval_set.json"
    eval_set.write_text(json.dumps({"races": [SYN[5]]}))
    loaded = []
    monkeypatch.setattr(cross_race, "load_race", lambda r, p: loaded.append(r))
    monkeypatch.setattr(cross_race.jobs, "load_race", lambda r, p: loaded.append(r))
    with pytest.raises(ValueError, match="held out"):
        lc.train_artifacts(SYN, HYBRID, load_profile(), str(tmp_path / "a"), eval_set=eval_set)
    with pytest.raises(ValueError, match="held out"):
        lc.train(SYN, n_jobs=1, work=tmp_path / "w", report_dir=tmp_path / "r", artifacts=str(tmp_path / "a"),
                 eval_set=eval_set)
    assert loaded == [] and not (tmp_path / "a").exists() and not (tmp_path / "w").exists()


def test_the_committed_held_out_races_are_refused():
    held = cross_race.held_out_races()
    assert held, "data/eval_set.json lists the held-out races"
    with pytest.raises(ValueError, match="held out"):
        lc.refuse_held_out([*lc.CALIBRATION_RACES, held[0]])
    assert not set(held) & set(lc.CALIBRATION_RACES)


# --- artifacts and the trained calibrator -----------------------------------------------------------
def test_artifacts_hold_one_leave_one_out_model_per_race_and_one_on_all(artifacts):
    out, res = artifacts
    m = json.loads((out / "manifest.json").read_text())
    assert m["races"] == SYN and m["command"] == lc.COMMAND and m["calibrator"]["params"] == HYBRID
    assert m["sources"]["wear_s_per_lap"] == "ridge" and m["sources"]["base_delta_s"] == "shrinkage"
    assert {"commit", "sim_code_version", "temporal_caveat"} <= set(m)
    for r in SYN:
        loo = calibrators.Calibrator.load(out / "loo" / f"{r}.json")
        assert loo.train_races == sorted(set(SYN) - {r})
        assert set(loo.models) and all(split_factor(f)[0] == "wear_s_per_lap" for f in loo.models)
        assert {f for f, why in loo.fallback.items() if why == "not adopted"} >= {"fuel_s_per_lap"}
    assert calibrators.Calibrator.load(out / "all.json").train_races == sorted(SYN)
    assert res["sha256"] == calibrators.artifact_sha256(out)


def test_the_trained_calibrator_picks_the_model_not_fitted_on_the_race(artifacts):
    out, res = artifacts
    cal = CALIBRATORS.get("trained")(artifacts=str(out), sha256=res["sha256"])
    assert cal.artifact_for(SYN[2]) == out / "loo" / f"{SYN[2]}.json"
    assert cal.artifact_for("2098_03_elsewhere") == out / "all.json"    # a race never trained on
    assert SYN[2] not in cal.model_for(SYN[2]).train_races
    with pytest.raises(ValueError, match="race id"):
        cal.artifact_for("spain")                                       # not canonical: refused
    with pytest.raises(ValueError, match="sha256"):
        CALIBRATORS.get("trained")(artifacts=str(out), sha256="000000000000").manifest()


def test_the_trained_calibrator_reports_per_factor_sources_in_the_model_report(artifacts):
    out, res = artifacts
    r = cross_race.synthetic_race(SYN[1])
    cfg = r.config.model_copy(update={"fit_window": "to_snapshot",
                                      "calibrator": ComponentSpec(name="trained", params={"artifacts": str(out),
                                                                                           "sha256": res["sha256"],
                                                                                           "force_not_as_of": True})})
    sim = Simulator(r.frame, cfg, r.practice)
    snap = calibration.snapshot_at(r.frame, r.race, "D00", 15)
    o = sim.fitted_for(snap).calibration.output
    assert o.source == "trained" and o.diagnostics["artifact"] == f"loo/{SYN[1]}.json"
    src = o.diagnostics["sources"]
    assert {v for f, v in src.items() if split_factor(f)[0] == "wear_s_per_lap"} == {"ridge"}
    assert {v for f, v in src.items() if split_factor(f)[0] != "wear_s_per_lap"} == {"shrinkage"}
    assert sim.model_report(snap).prior_sources == src
    # shrinkage says nothing about sources: the field stays None
    plain = Simulator(r.frame, r.config.model_copy(update={"fit_window": "to_snapshot"}), r.practice)
    assert plain.model_report(snap).prior_sources is None


def test_ridge_factors_fits_only_the_adopted_kinds():
    with pytest.raises(ValueError, match="factors"):
        CALIBRATORS.get("ridge")(factors=["tyre_magic"])
    cal = CALIBRATORS.get("ridge")(targets="all", factors=["pit_loss_s"])
    assert cal._targeted("pit_loss_s") and not cal._targeted("wear_s_per_lap[SOFT]")


# --- the rules --------------------------------------------------------------------------------------
def _row(cal, window, kind, rmse, c68, c90):
    return {"calibrator": cal, "window": window, "kind": kind, "rmse": rmse, "coverage_68": c68, "coverage_90": c90}


def test_adoption_needs_lower_rmse_and_no_worse_coverage_in_both_windows():
    rows = []
    for w in lc.WINDOWS:
        rows += [_row("shrinkage", w, "wear_s_per_lap", 0.05, 0.5, 0.8), _row("ridge", w, "wear_s_per_lap", 0.04, 0.6, 0.85),
                 _row("shrinkage", w, "fuel_s_per_lap", 0.02, 0.6, 0.9), _row("ridge", w, "fuel_s_per_lap", 0.01, 0.4, 0.9),
                 _row("shrinkage", w, "pit_loss_s", 1.0, 0.0, 0.0), _row("ridge_all", w, "pit_loss_s", 0.9, 0.7, 0.9)]
    # base: lower rmse in pre_race only
    rows += [_row("shrinkage", "pre_race", "base_delta_s", 1.0, 0.7, 0.9), _row("ridge", "pre_race", "base_delta_s", 0.9, 0.7, 0.9),
             _row("shrinkage", "to_snapshot", "base_delta_s", 1.0, 0.7, 0.9),
             _row("ridge", "to_snapshot", "base_delta_s", 1.1, 0.7, 0.9)]
    a = lc.adopt(pd.DataFrame(rows)).set_index("kind")["adopted"].to_dict()
    assert a == {"wear_s_per_lap": True, "fuel_s_per_lap": False, "pit_loss_s": True, "base_delta_s": False}


def test_the_default_rule_takes_the_first_candidate_that_passes():
    def b(cal, g, sg, c68, c90):
        return {"window": "to_snapshot", "calibrator": cal, "g_mae_s": g, "sg_mae_s": sg, "g_coverage_68": c68,
                "g_coverage_90": c90}
    t = pd.DataFrame([b("shrinkage", 30, 1.0, 0.5, 0.8), b("hybrid", 31, 0.9, 0.6, 0.85), b("ridge", 29, 1.0, 0.55, 0.8)])
    d = lc.decide(t)
    assert d["default"] == "ridge" and not d["checks"]["hybrid"]["g_mae_lower"]
    t.loc[t["calibrator"] == "ridge", "sg_mae_s"] = 1.01
    assert lc.decide(t)["default"] is None


def test_the_committed_artifacts_never_saw_a_held_out_race_or_the_race_they_calibrate():
    """Reads the committed JSON only; a held-out race id is used as a string, never loaded."""
    out = lc.ROOT / lc.ARTIFACTS
    m = json.loads((out / "manifest.json").read_text())
    assert m["races"] == list(lc.CALIBRATION_RACES) and m["command"] == lc.COMMAND
    held = set(cross_race.held_out_races())
    cal = CALIBRATORS.get("trained")(artifacts=lc.ARTIFACTS)
    for r in m["races"]:
        model = calibrators.Calibrator.load(out / "loo" / f"{r}.json")
        assert model.train_races == sorted(set(m["races"]) - {r}) and not held & set(model.train_races)
        assert cal.artifact_for(r) == out / "loo" / f"{r}.json"
    assert calibrators.Calibrator.load(out / "all.json").train_races == sorted(m["races"])
    for r in held:
        assert cal.artifact_for(r) == out / "all.json"
    # the hybrid arm, the learned belief option and the artifacts agree
    arm = cross_race.arm_config(lc.HYBRID_ARM)
    assert arm.calibrator.name == "ridge"
    if m["decision"]["default"] != "ridge":
        assert dict(arm.calibrator.params) == m["calibrator"]["params"]
    learned = SimConfig.from_file(lc.BELIEF_LEARNED)
    assert learned.calibrator.name == "trained"
    assert learned.calibrator.params == {"artifacts": lc.ARTIFACTS, "sha256": calibrators.artifact_sha256(out)}
    belief = SimConfig.from_file(lc.BELIEF_CONFIG)
    assert learned.model_copy(update={"calibrator": belief.calibrator}) == belief


def test_the_learned_belief_on_2021_spain_uses_the_model_fitted_without_it():
    from src.sim import jobs
    data = jobs.load_race("2021_04_spain", load_profile())
    cfg = SimConfig.from_file(lc.BELIEF_LEARNED)
    driver = data.frame.loc[data.frame["lap"] == 21, "driver"].iloc[0]
    snap = calibration.snapshot_at(data.frame, data.race, driver, 20)
    # the artifacts were fitted on races after 2021 Spain: refused unless forced (docs/eval.md
    # "Cross-race tables, as of the race")
    with pytest.raises(ValueError, match="not as of the race"):
        Simulator(data.frame, cfg, data.practice).fitted_for(snap)
    cfg = cfg.model_copy(update={"calibrator": cfg.calibrator.model_copy(
        update={"params": {**cfg.calibrator.params, "force_not_as_of": True}})})
    sim = Simulator(data.frame, cfg, data.practice)
    out = sim.fitted_for(snap).calibration.output
    assert out.source == "trained" and out.diagnostics["artifact"] == "loo/2021_04_spain.json"
    manifest = json.loads((lc.ROOT / lc.ARTIFACTS / "manifest.json").read_text())
    sources = sim.model_report(snap).prior_sources
    for f, src in sources.items():   # a kind not adopted is shrinkage's (an adopted one may fall back per compound)
        if manifest["sources"][split_factor(f)[0]] == "shrinkage":
            assert src == "shrinkage", f
    assert "ridge" in sources.values() or not manifest["adopted"]


def test_the_calibrator_arms_are_frozen_files():
    for arm in cross_race.CALIBRATOR_ARMS:
        assert (cross_race.ARMS_DIR / f"{arm}.yaml").is_file(), arm
        cfg = cross_race.arm_config(arm)
        assert cfg.degradation.params["wear_lower_bound"] == 0.0
    assert set(lc.ARMS.values()) <= {*cross_race.CALIBRATOR_ARMS, "wear_nonneg"}
