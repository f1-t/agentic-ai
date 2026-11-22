"""The learned (ridge) calibrator on the real calibration races: evaluated leave-one-race-out
against shrinkage, adopted per factor, and committed as trained artifacts.

    python -m src.sim.learned_calibrator train [--races ...] [--jobs 4] [--resume]

One command does all of it (docs/eval.md, "The learned calibrator"):

  1. Leave-one-race-out folds (cross_race.fold_job, the cross-race runner's own measure) on the
     calibration races only, for the arms ARMS: none, shrinkage (the belief's calibrator), ridge
     (degradation targets) and ridge_all (event factors too), every one the belief simulator as
     configs/sim/belief.yaml stands (arm base_config: wear bound, posterior draws, residual lap
     noise, traffic off; G replays the race's real SC and VSC periods). Each fold trains on the other races only, its inner alpha search
     included (calibrators.RidgeCalibrator); the windows are the belief's, pre_race and
     to_snapshot. Fold outputs go under --work (gitignored), wiped first unless --resume.
  2. The per-factor table (factor_table): per factor kind (pooled over compounds, as the engine
     takes one prior sd per kind) and window, the rmse of the effective factor against the
     race's hindsight value and its 68 and 90 percent interval coverage.
  3. The adoption rule (adopt, fixed before any result was seen): ridge is adopted for a factor
     kind when, in BOTH belief windows, its rmse is below shrinkage's AND its coverage is no
     worse at both levels (|coverage - nominal| no larger than shrinkage's at 68 and at 90
     percent). Degradation kinds are read from ridge, event kinds from ridge_all (shrinkage has
     no event prior: the config or race registry value at sd 0). Every other kind stays on
     shrinkage. The result is written to configs/sim/arms/belief_hybrid.yaml (ridge with
     `factors`, the adopted kinds) and run through the same folds.
  4. The belief measures (belief_table) for none, shrinkage, ridge, ridge_all and the hybrid:
     the replay of the teams' real remaining plans from each calibration point (G: error and
     signed bias of the pace-corrected remaining time at horizons 10-49, race-weighted, with
     race-bootstrap CIs and paired differences against shrinkage, per race and pooled), the
     stop-gain error and signed bias at the real pit stops, and G's 68 and 90 percent interval
     coverage. The default rule (decide, fixed before any result): the first of the hybrid and
     plain ridge whose to_snapshot (belief) G mae is below shrinkage's, whose stop-gain mae is
     no worse and whose G coverage is no worse at 68 and 90 percent (|coverage - nominal|) is
     adopted as the belief's default; otherwise shrinkage stays.
  5. Trained artifacts of the hybrid (train_artifacts): loo/<race_id>.json fitted on every
     calibration race but that one (used when that race is evaluated as development data) and
     all.json fitted on every calibration race (used for any other race: the held-out races,
     which never enter training), with manifest.json (the command, the races, the commit, the
     params, the per-factor sources, the decision). calibrators.TrainedCalibrator loads the
     right one by race id and never trains. configs/sim/belief_learned.yaml is written: the
     belief with that calibrator (artifacts and sha256), an option via --belief-config. The
     command never edits configs/sim/belief.yaml; when the rule adopts, copy its calibrator block.
  6. Report: reports/calibration/learned/{adoption.csv, factors.csv, factors_by_compound.csv,
     belief.csv, belief_by_race.csv, g_by_horizon.csv, fold_fits.csv, artifact_fits.csv, summary.md}.

A race data/eval_set.json holds out is refused before anything about it is loaded
(cross_race.guard_held_out, never allowed here), and again by train_artifacts.

Temporal caveat (docs/eval.md): a leave-one-out model learns from races both before and after
the race it calibrates; nothing of that race is used, but it is not strictly as of the race.
--as-of is a stub that says what a strictly as-of variant would need.
"""
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..contracts import split_factor
from . import calibration, calibrators, cross_race, jobs
from .compound_steps import CALIBRATION_RACES
from .compute import load_profile, training_run
from .race_ids import canonical_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports" / "calibration" / "learned"
WORK_DIR = REPORT_DIR / "work"               # fold parquets, gitignored
ARTIFACTS = "models/calibrators/ridge_hybrid"  # relative to ROOT; TrainedCalibrator's `artifacts`
HYBRID_ARM = "belief_hybrid"
HYBRID_FILE = cross_race.ARMS_DIR / f"{HYBRID_ARM}.yaml"
BELIEF_CONFIG = ROOT / "configs" / "sim" / "belief.yaml"
BELIEF_LEARNED = ROOT / "configs" / "sim" / "belief_learned.yaml"   # written: the belief with the trained calibrator
COMMAND = "python -m src.sim.learned_calibrator train"
# calibrator -> cross-race arm; every arm is the belief simulator (base_config configs/sim/belief.yaml)
ARMS = {"none": "belief_none", "shrinkage": "belief_shrinkage", "ridge": "belief_ridge", "ridge_all": "belief_ridge_all"}
WINDOWS = ("pre_race", "to_snapshot")
BELIEF_WINDOW = cross_race.BELIEF_WINDOW
G_HORIZONS = cross_race.G_HORIZONS
LEVELS = {"68": 1.0, "90": 1.6448536269514722}   # interval half-width in sds per nominal level
DEGRADATION_KINDS = ("base_delta_s", "wear_s_per_lap", "fuel_s_per_lap", "hinge_s_per_lap")
EVENT_KINDS = tuple(calibrators.EVENT_BOUNDS)
DEFAULT_CANDIDATES = ("hybrid", "ridge")   # tried in this order by the default rule
N_BOOT = cross_race.N_BOOT


def profile_for(name=None):
    """The compute profile with G's windows the belief's two (the world, which never calls a calibrator, left out)."""
    p = load_profile(name)
    return p.model_copy(update={"g": p.g.model_copy(update={"windows": list(WINDOWS)})})


def refuse_held_out(races, eval_set=cross_race.EVAL_SET) -> list[str]:
    """Canonical ids of `races`; a race `eval_set` holds out is refused, before anything is loaded."""
    cross_race.guard_held_out(races, eval_set, allow=False)
    return [canonical_race_id(r) for r in races]


# --- folds -----------------------------------------------------------------------------------
def run_folds(arms, races, profile, work, n_jobs) -> list[tuple[str, str]]:
    items = cross_race._fold_items(list(arms), list(races), profile, work, stop_check=False)
    return cross_race._run_items(items, n_jobs)


def fold_outputs(arms: dict, races, profile, work) -> dict[str, dict]:
    """{calibrator: cross_race.arm_outputs}; a fold error or a missing fold is fatal."""
    out = {}
    for name, arm in arms.items():
        d = cross_race.arm_outputs(work, arm, races, profile)
        if len(d["fold_error"]):
            raise RuntimeError(f"{arm}: folds failed: {d['fold_error'].to_dict('records')}")
        missing = sorted(set(races) - set(d["g"]["race_id"]))
        if missing:
            raise RuntimeError(f"{arm}: no fold output for {missing}")
        out[name] = d
    return out


# --- per-factor table and the adoption rule -----------------------------------------------------
def _factor_rows(f: pd.DataFrame) -> pd.DataFrame:
    f = f[f["window"].isin(WINDOWS)].copy()
    f["kind"] = [split_factor(x)[0] for x in f["factor"]]
    for lvl, z in LEVELS.items():
        f[f"covered_{lvl}"] = (f["err"].abs() <= z * f["sd"]).astype(float)
    return f


def _summary(g: pd.DataFrame) -> dict:
    return {"n": int(len(g)), "n_races": int(g["race_id"].nunique()), "rmse": float(np.sqrt((g["err"] ** 2).mean())),
            "bias": float(g["err"].mean()), "mean_sd": float(g["sd"].mean()),
            **{f"coverage_{lvl}": float(g[f"covered_{lvl}"].mean()) for lvl in LEVELS}}


def factor_table(data: dict, by="kind") -> pd.DataFrame:
    """Per calibrator, window and factor kind (by="kind", pooled over compounds) or factor name
    (by="factor"): n rows, races, rmse, bias, mean sd, 68 and 90 percent coverage of the effective
    factor against the hindsight value, every race's rows from its own fold."""
    rows = []
    for name, d in data.items():
        f = _factor_rows(d["factors"])
        for (window, key), g in f.groupby(["window", by], sort=True):
            rows.append({"calibrator": name, "window": window, by: key, **_summary(g)})
    return pd.DataFrame(rows)


def _miscal(row, lvl) -> float:
    return abs(row[f"coverage_{lvl}"] - int(lvl) / 100)


def adopt(table: pd.DataFrame) -> pd.DataFrame:
    """The adoption rule, per factor kind: ridge (ridge_all for an event kind) against shrinkage
    in each belief window. Adopted when in BOTH windows its rmse is below shrinkage's and its
    |coverage - nominal| is no larger at 68 and at 90 percent. A kind either side lacks in a
    window is not adopted."""
    t = table.set_index(["calibrator", "window", "kind"])
    kinds = sorted(set(table.loc[table["calibrator"] == "shrinkage", "kind"]))
    out = []
    for kind in kinds:
        source = "ridge_all" if kind in EVENT_KINDS else "ridge"
        row = {"kind": kind, "learned": source}
        ok = True
        for w in WINDOWS:
            if ("shrinkage", w, kind) not in t.index or (source, w, kind) not in t.index:
                ok = False
                row[f"{w}:reason"] = "missing"
                continue
            s, r = t.loc[("shrinkage", w, kind)], t.loc[(source, w, kind)]
            lower = bool(r["rmse"] < s["rmse"])
            cov = all(_miscal(r, lvl) <= _miscal(s, lvl) for lvl in LEVELS)
            row.update({f"{w}:rmse_shrinkage": s["rmse"], f"{w}:rmse_learned": r["rmse"],
                        **{f"{w}:cov{lvl}_shrinkage": s[f"coverage_{lvl}"] for lvl in LEVELS},
                        **{f"{w}:cov{lvl}_learned": r[f"coverage_{lvl}"] for lvl in LEVELS},
                        f"{w}:rmse_lower": lower, f"{w}:coverage_no_worse": cov})
            ok = ok and lower and cov
        row["adopted"] = ok
        out.append(row)
    return pd.DataFrame(out)


def write_hybrid_arm(adopted: list[str], path=HYBRID_FILE) -> dict:
    """configs/sim/arms/belief_hybrid.yaml: the belief simulator with ridge (event targets on when
    an event kind is adopted) fitting the adopted kinds only, every other kind shrinkage's."""
    adopted = sorted(adopted)
    params = {"targets": "all" if any(k in EVENT_KINDS for k in adopted) else "degradation", "factors": adopted}
    over = {"base_config": "configs/sim/belief.yaml", "calibrator": {"name": "ridge", "params": params}}
    head = ("# Learned-calibrator arm \"belief_hybrid\", WRITTEN by `" + COMMAND + "` (src/sim/learned_calibrator.py)\n"
            "# from the per-factor adoption rule: ridge for the factor kinds in `factors` (lower leave-one-race-out\n"
            "# rmse than shrinkage and coverage no worse, in both belief windows), shrinkage for every other kind.\n"
            "# The belief simulator otherwise (base_config). Do not edit by hand; rerun the command.\n")
    Path(path).write_text(head + yaml.safe_dump(over, sort_keys=False))
    return params


def write_belief_learned(artifacts, sha256: str, decision: dict, path=None) -> Path:
    """configs/sim/belief_learned.yaml: configs/sim/belief.yaml with the trained calibrator (the
    artifacts and their sha256), the belief as it would be with the learned calibrator; usable as
    `--belief-config`. Rewritten by every run, as the sha256 moves with a retrain."""
    path = BELIEF_LEARNED if path is None else Path(path)
    cfg = yaml.safe_load(BELIEF_CONFIG.read_text()) or {}
    cfg["calibrator"] = {"name": "trained", "params": {"artifacts": str(artifacts), "sha256": sha256}}
    verdict = (f"ADOPTED as the default by the rule ({decision['default']})" if decision["default"] else
               "NOT adopted by the default rule: an option; configs/sim/belief.yaml keeps shrinkage")
    head = (f"# WRITTEN by `{COMMAND}` (src/sim/learned_calibrator.py): configs/sim/belief.yaml with the trained\n"
            "# learned calibrator (calibrators.TrainedCalibrator: the leave-one-out model for a calibration\n"
            f"# race, the all-races model for any other). {verdict}.\n"
            "# Do not edit by hand; rerun the command.\n")
    path.write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    return path


# --- belief measures and the default rule ---------------------------------------------------------
def _g(rows, window):
    return cross_race._g_window(rows, window, G_HORIZONS)


def _coverage(g: pd.DataFrame) -> dict:
    """Pooled 68 and 90 percent coverage of the pace-corrected real remaining time (calibration.aggregate's)."""
    t = g.assign(target=g["real_remaining_s"] - g["pace_offset_s"] * g["horizon"]).dropna(
        subset=["target", "pred_p05_s", "pred_p95_s", "pred_p16_s", "pred_p84_s"])
    if not len(t):
        return {"g_coverage_68": np.nan, "g_coverage_90": np.nan, "g_width_90_s": np.nan}
    return {"g_coverage_68": float(t["target"].between(t["pred_p16_s"], t["pred_p84_s"]).mean()),
            "g_coverage_90": float(t["target"].between(t["pred_p05_s"], t["pred_p95_s"]).mean()),
            "g_width_90_s": float((t["pred_p95_s"] - t["pred_p05_s"]).mean())}


def belief_table(data: dict, n_boot=N_BOOT) -> pd.DataFrame:
    """Per window and calibrator, every race's rows from its own fold: G mae_offset and signed bias
    at horizons 10-49 (race-weighted, with the race-bootstrap CI of the mae, and the paired
    difference against shrinkage with its CI), G's interval coverage, the stop-gain mae and signed
    bias (race-weighted) and the paired stop-gain difference against shrinkage."""
    rows = []
    base = data["shrinkage"]
    for w in WINDOWS:
        for name, d in data.items():
            g, gb = _g(d["g"], w), _g(base["g"], w)
            pg = cross_race.paired(g, gb, cross_race.G_KEYS, "err_offset_s", n_boot)
            s = d["stop_gain"][d["stop_gain"]["window"] == w]
            sb = base["stop_gain"][base["stop_gain"]["window"] == w]
            ps = cross_race.paired(s, sb, cross_race.SG_KEYS, "err_s", n_boot)
            rows.append({"window": w, "calibrator": name, "n_races": pg["n_races"], "g_rows": pg["n_rows"],
                         "g_mae_s": pg["mae"], "g_mae_lo": pg["mae_lo"], "g_mae_hi": pg["mae_hi"], "g_bias_s": pg["bias"],
                         "g_diff_s": pg["diff"], "g_diff_lo": pg["diff_lo"], "g_diff_hi": pg["diff_hi"],
                         **_coverage(g), "sg_rows": ps["n_rows"], "sg_mae_s": ps["mae"], "sg_mae_lo": ps["mae_lo"],
                         "sg_mae_hi": ps["mae_hi"], "sg_bias_s": ps["bias"], "sg_diff_s": ps["diff"],
                         "sg_diff_lo": ps["diff_lo"], "sg_diff_hi": ps["diff_hi"]})
    return pd.DataFrame(rows)


def belief_by_race(data: dict) -> pd.DataFrame:
    rows = []
    for name, d in data.items():
        for w in WINDOWS:
            g = _g(d["g"], w)
            s = d["stop_gain"][d["stop_gain"]["window"] == w]
            for race, t in g.groupby("race_id", sort=True):
                st = s[s["race_id"] == race]
                rows.append({"race_id": race, "window": w, "calibrator": name, "g_rows": int(len(t)),
                             "g_mae_s": float(t["err_offset_s"].abs().mean()), "g_bias_s": float(t["err_offset_s"].mean()),
                             **_coverage(t), "sg_rows": int(len(st)),
                             "sg_mae_s": float(st["err_s"].abs().mean()) if len(st) else np.nan,
                             "sg_bias_s": float(st["err_s"].mean()) if len(st) else np.nan})
    return pd.DataFrame(rows)


def g_by_horizon(data: dict, n_boot=N_BOOT) -> pd.DataFrame:
    """calibration.aggregate per calibrator: G by window and 10-lap horizon bucket."""
    parts = [calibration.aggregate(d["g"], n_boot=n_boot).reset_index().assign(calibrator=name)
             for name, d in data.items()]
    t = pd.concat(parts, ignore_index=True)
    keep = ["calibrator", "window", "horizon_bucket", "n", "mae_offset_s", "mae_offset_ci_lo", "mae_offset_ci_hi",
            "coverage_68", "coverage_90", "mean_width_90_s"]
    return t[[c for c in keep if c in t]]


def decide(belief: pd.DataFrame, candidates=DEFAULT_CANDIDATES) -> dict:
    """The default rule on the to_snapshot (belief) rows: the first candidate with G mae below
    shrinkage's, stop-gain mae no worse and G coverage no worse at 68 and 90 percent
    (|coverage - nominal|); None keeps shrinkage. Point estimates, race-weighted; the CIs are
    reported beside them (with 7 races a CI that excludes 0 is not required)."""
    t = belief[belief["window"] == BELIEF_WINDOW].set_index("calibrator")
    s = t.loc["shrinkage"]
    checks = {}
    for c in candidates:
        if c not in t.index:
            continue
        r = t.loc[c]
        chk = {"g_mae_lower": bool(r["g_mae_s"] < s["g_mae_s"]), "stop_gain_no_worse": bool(r["sg_mae_s"] <= s["sg_mae_s"]),
               "coverage_no_worse": all(abs(r[f"g_coverage_{lvl}"] - int(lvl) / 100)
                                        <= abs(s[f"g_coverage_{lvl}"] - int(lvl) / 100) for lvl in LEVELS)}
        chk["passes"] = all(chk.values())
        checks[c] = chk
    chosen = next((c for c in candidates if checks.get(c, {}).get("passes")), None)
    return {"default": chosen, "checks": checks}


# --- trained artifacts ------------------------------------------------------------------------------
def _commit() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    """The commit the command ran at, and whether src/ or configs/ differed from it (the hybrid arm
    file, which the command itself writes, aside)."""
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "src", "configs", f":!configs/sim/arms/{HYBRID_ARM}.yaml",
                              ":!configs/sim/belief_learned.yaml"))}


def train_artifacts(races, params: dict, profile, artifacts=ARTIFACTS, arm=HYBRID_ARM, eval_set=cross_race.EVAL_SET,
                    extra: dict | None = None) -> dict:
    """Fit the hybrid on the calibration races and save loo/<race_id>.json (every race but that
    one) and all.json (every race), with manifest.json. A held-out race is refused before any
    load. The dataset is build_dataset under the arm's config at the profile's snapshot laps, as
    in the folds, so loo/<race>.json is that race's fold model. Each fit is logged
    (compute.training_run)."""
    races = refuse_held_out(races, eval_set)
    out = Path(artifacts) if Path(artifacts).is_absolute() else ROOT / artifacts
    data = [cross_race.load_race(r, profile) for r in races]
    data = [replace(d, config=cross_race.arm_config(arm, d.config)) for d in data]
    ds = calibrators.build_dataset(data, laps_k=lambda d: jobs.laps_k_for(d.frame, d.race.total_laps, profile))
    if out.exists():
        shutil.rmtree(out)
    (out / "loo").mkdir(parents=True)
    fits = []
    for held in [*races, None]:
        train = sorted(r for r in races if r != held)
        cal = calibrators.CALIBRATORS.get("ridge")(**params)
        split = {"calibrator": "ridge", "artifact": "all" if held is None else f"loo/{held}", "train": train,
                 "held_out": [held] if held else [], "params": params}
        with training_run("calibrator_artifact", split) as run:
            cal.fit(ds.subset(train))
            if run is not None:
                run.log(cal.metrics())
        cal.save(out / ("all.json" if held is None else f"loo/{held}.json"))
        fits += [{"artifact": split["artifact"], **r} for r in cal.fit_table()]
    sources = {k: "ridge" if k in params["factors"] else "shrinkage"
               for k in (*DEGRADATION_KINDS, *EVENT_KINDS)}
    manifest = {"description": "The learned calibrator's trained artifacts (src/sim/learned_calibrator.py): the "
                               "per-factor hybrid of ridge and shrinkage. calibrators.TrainedCalibrator loads "
                               "loo/<race_id>.json for a race in `races` (fitted without it) and all.json for any "
                               "other race; nothing is trained at evaluation time.",
                "command": COMMAND, "races": races, "held_out_never_loaded": cross_race._shown(eval_set),
                "calibrator": {"name": "ridge", "params": params}, "arm": arm, "sources": sources,
                "dataset": {"rows": int(len(ds.rows)), "laps_k": "jobs.laps_k_for under the profile",
                            "profile": profile.name, "profile_hash": profile.hash},
                "sim_code_version": calibration.SIM_CODE_VERSION, **_commit(),
                "temporal_caveat": "a leave-one-out model is trained on races before and after the race it "
                                   "calibrates (none of that race); not strictly as of the race",
                **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float) + "\n")
    return {"manifest": manifest, "fits": pd.DataFrame(fits), "sha256": calibrators.artifact_sha256(out)}


# --- report -------------------------------------------------------------------------------------------
def _f(v, d=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}" if isinstance(v, float) else str(v)


def summary_md(races, adoption, factors, belief, by_race, decision, art) -> str:
    md = ["# Learned calibrator on the calibration races", "",
          f"Written by `{COMMAND}` (src/sim/learned_calibrator.py). Races (leave-one-race-out, calibration races "
          f"only): {', '.join(races)}. sim_code_version {calibration.SIM_CODE_VERSION}.", "",
          "## Per-factor adoption", "",
          "Per factor kind (pooled over compounds), every race's rows from its own fold; ridge_all for the "
          "event kinds. Adopted when in both belief windows the learned rmse is below shrinkage's and "
          "|coverage - nominal| is no larger at 68 and 90 percent.", "",
          "| factor | window | rmse shrinkage | rmse learned | cov68 shr | cov68 learned | cov90 shr | cov90 learned "
          "| adopted |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in adoption.iterrows():
        for w in WINDOWS:
            md.append(f"| {r['kind']} | {w} | {_f(r.get(f'{w}:rmse_shrinkage'), 4)} | {_f(r.get(f'{w}:rmse_learned'), 4)} "
                      f"| {_f(r.get(f'{w}:cov68_shrinkage'), 2)} | {_f(r.get(f'{w}:cov68_learned'), 2)} "
                      f"| {_f(r.get(f'{w}:cov90_shrinkage'), 2)} | {_f(r.get(f'{w}:cov90_learned'), 2)} "
                      f"| {'yes' if r['adopted'] else 'no'} |")
    md += ["", "## Factor error and coverage by calibrator", "",
           "| window | factor | calibrator | n | rmse | bias | mean sd | cov68 | cov90 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in factors.sort_values(["window", "kind", "calibrator"]).iterrows():
        md.append(f"| {r['window']} | {r['kind']} | {r['calibrator']} | {r['n']} | {_f(r['rmse'], 4)} | {_f(r['bias'], 4)} "
                  f"| {_f(r['mean_sd'], 4)} | {_f(r['coverage_68'], 2)} | {_f(r['coverage_90'], 2)} |")
    md += ["", "## Belief measures", "",
           "Replay of the teams' real remaining plans from each calibration point (G, horizons "
           f"{G_HORIZONS[0]}-{G_HORIZONS[1]}, pace-corrected, race-weighted; 95% race-bootstrap CIs), the paired "
           "difference against shrinkage, G's interval coverage, and the stop-gain error at the real pit stops.", "",
           "| window | calibrator | G mae s [CI] | G bias s | G diff vs shrinkage [CI] | G cov68 | G cov90 "
           "| stop-gain mae s [CI] | stop-gain bias s | stop-gain diff [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in belief.iterrows():
        md.append(f"| {r['window']} | {r['calibrator']} | {_f(r['g_mae_s'], 2)} [{_f(r['g_mae_lo'], 2)}, {_f(r['g_mae_hi'], 2)}] "
                  f"| {_f(r['g_bias_s'], 2)} | {_f(r['g_diff_s'], 2)} [{_f(r['g_diff_lo'], 2)}, {_f(r['g_diff_hi'], 2)}] "
                  f"| {_f(r['g_coverage_68'], 2)} | {_f(r['g_coverage_90'], 2)} "
                  f"| {_f(r['sg_mae_s'], 2)} [{_f(r['sg_mae_lo'], 2)}, {_f(r['sg_mae_hi'], 2)}] | {_f(r['sg_bias_s'], 2)} "
                  f"| {_f(r['sg_diff_s'], 2)} [{_f(r['sg_diff_lo'], 2)}, {_f(r['sg_diff_hi'], 2)}] |")
    md += ["", "## Per race (to_snapshot)", "",
           "| race | calibrator | G mae s | G bias s | G cov90 | stop-gain mae s | stop-gain bias s |", "|---|---|---|---|---|---|---|"]
    for _, r in by_race[by_race["window"] == BELIEF_WINDOW].sort_values(["race_id", "calibrator"]).iterrows():
        md.append(f"| {r['race_id']} | {r['calibrator']} | {_f(r['g_mae_s'], 2)} | {_f(r['g_bias_s'], 2)} "
                  f"| {_f(r['g_coverage_90'], 2)} | {_f(r['sg_mae_s'], 2)} | {_f(r['sg_bias_s'], 2)} |")
    md += ["", "## Default", "",
           "Rule (to_snapshot, race-weighted point estimates): G mae below shrinkage's, stop-gain mae no worse, "
           "|G coverage - nominal| no larger at 68 and 90 percent; hybrid first, then plain ridge.", ""]
    t = belief[belief["window"] == BELIEF_WINDOW].set_index("calibrator")
    for c, chk in decision["checks"].items():
        r, s = t.loc[c], t.loc["shrinkage"]
        md.append(f"- {c}: G mae {r['g_mae_s']:.3f} s vs {s['g_mae_s']:.3f} ({'lower' if chk['g_mae_lower'] else 'not lower'}); "
                  f"stop-gain mae {r['sg_mae_s']:.4f} s vs {s['sg_mae_s']:.4f} "
                  f"({'no worse' if chk['stop_gain_no_worse'] else 'worse'}); G coverage 68/90 "
                  f"{r['g_coverage_68']:.3f}/{r['g_coverage_90']:.3f} vs {s['g_coverage_68']:.3f}/{s['g_coverage_90']:.3f} "
                  f"({'no worse' if chk['coverage_no_worse'] else 'worse'}): "
                  f"{'passes' if chk['passes'] else 'does not pass'}")
    md += ["", (f"**Adopted as the belief's default: {decision['default']}.**" if decision["default"] else
                "**Not adopted: shrinkage stays the belief's default; the learned calibrator ships as an option, "
                "`--belief-config configs/sim/belief_learned.yaml` (calibrator `trained`).**"), "",
           f"Artifacts: `{ARTIFACTS}/` (sha256 {art['sha256']}): all.json and loo/<race_id>.json, manifest.json.", ""]
    return "\n".join(md)


def train(races=CALIBRATION_RACES, n_jobs=4, work=WORK_DIR, report_dir=REPORT_DIR, artifacts=ARTIFACTS,
          resume=False, profile_name=None, eval_set=cross_race.EVAL_SET, log=print) -> dict:
    races = refuse_held_out(races, eval_set)
    profile = profile_for(profile_name)
    work, report_dir = Path(work), Path(report_dir)
    if work.exists() and not resume:
        shutil.rmtree(work)
    log(f"folds: {list(ARMS.values())} on {races}")
    run_folds(ARMS.values(), races, profile, work, n_jobs)
    data = fold_outputs(ARMS, races, profile, work)
    factors = factor_table(data)
    adoption = adopt(factors)
    adopted = sorted(adoption.loc[adoption["adopted"], "kind"])
    log(f"adopted per factor: {adopted or 'none'}")
    params = write_hybrid_arm(adopted)
    hybrid_dir = work / "h" / "arms" / HYBRID_ARM
    if hybrid_dir.exists():    # the yaml was just rewritten: never reuse a hybrid fold
        shutil.rmtree(hybrid_dir)
    run_folds([HYBRID_ARM], races, profile, work, n_jobs)
    data.update(fold_outputs({"hybrid": HYBRID_ARM}, races, profile, work))
    factors = factor_table(data)
    belief = belief_table(data)
    by_race = belief_by_race(data)
    decision = decide(belief)
    if decision["default"] == "ridge":   # plain ridge passed where the hybrid did not: ship plain ridge
        params = {"targets": "degradation"}
    art = train_artifacts(races, params, profile, artifacts, eval_set=eval_set,
                          extra={"adopted": adopted, "decision": decision})
    write_belief_learned(artifacts, art["sha256"], decision)
    report_dir.mkdir(parents=True, exist_ok=True)
    factors.to_csv(report_dir / "factors.csv", index=False)
    factor_table(data, by="factor").to_csv(report_dir / "factors_by_compound.csv", index=False)
    adoption.to_csv(report_dir / "adoption.csv", index=False)
    belief.to_csv(report_dir / "belief.csv", index=False)
    by_race.to_csv(report_dir / "belief_by_race.csv", index=False)
    g_by_horizon(data).to_csv(report_dir / "g_by_horizon.csv", index=False)
    fold_fits = pd.concat([d["fits"].assign(arm=a) for a, d in data.items() if len(d["fits"])], ignore_index=True)
    fold_fits.to_csv(report_dir / "fold_fits.csv", index=False)
    art["fits"].to_csv(report_dir / "artifact_fits.csv", index=False)
    (report_dir / "summary.md").write_text(summary_md(races, adoption, factors, belief, by_race, decision, art))
    log(f"default: {decision['default'] or 'shrinkage stays'}; checks {decision['checks']}")
    log(f"{BELIEF_LEARNED.relative_to(ROOT)} written (the belief with the trained calibrator, sha256 {art['sha256']}); "
        + ("the rule adopts it: put its calibrator block into configs/sim/belief.yaml" if decision["default"] else
           "the rule keeps shrinkage: configs/sim/belief.yaml stays, the learned belief is an option (--belief-config)"))
    return {"adopted": adopted, "decision": decision, "sha256": art["sha256"], "belief": belief, "factors": factors}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.learned_calibrator", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="LORO evaluation, per-factor adoption, trained artifacts and report")
    t.add_argument("--races", nargs="+", default=list(CALIBRATION_RACES), help="calibration races (held-out refused)")
    t.add_argument("--jobs", type=int, default=4, help="fold worker processes")
    t.add_argument("--work", default=str(WORK_DIR), help="fold outputs (gitignored)")
    t.add_argument("--resume", action="store_true", help="reuse current fold outputs under --work")
    t.add_argument("--profile", help="compute profile, default local")
    t.add_argument("--as-of", action="store_true",
                   help="not implemented: train each race's model on earlier races only (docs/eval.md)")
    args = ap.parse_args(argv)
    if args.as_of:
        sys.exit("--as-of is not implemented: a strictly as-of model for race R would train on the races before R "
                 "only (race_ids.as_of); with 7 calibration races the 2019 and 2021 races have fewer than "
                 "RIDGE_MIN_RACES (5) earlier races, so every factor would fall back to shrinkage there")
    train(args.races, args.jobs, args.work, resume=args.resume, profile_name=args.profile)


if __name__ == "__main__":
    main()
