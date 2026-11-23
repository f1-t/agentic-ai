"""Every number printed on the 24 x 36 in PITWALL poster, computed from the evaluation pipeline's committed files.

    python poster/poster_stats.py --dev reports/eval/rerun/dev20 --test reports/eval/rerun/test9 \
        --fidelity reports/calibration/fidelity

Inputs (files only: nothing is loaded, simulated or scored, and nothing is imported from `src`):

- `--dev`, `--test`: out-dirs of `python -m src.eval.rerun` (development set dev20, report set test9), each with
  `provenance.json`, `scenarios.json`, `agent.csv` and `tables/` (`combined.csv`, `chart_regret_s.csv`,
  `tail.csv`, `summary.csv`, `paired_agent_minus_x.csv`, the pre-registered `reporting_paired.csv`,
  `reporting_tail.csv`, `reporting_counts.csv`, and for the report set `reporting_paired_by_era.csv`;
  `agent_latency.csv`). The held-out groups (fresh five, earlier four, all nine) are read from the reporting
  choices recorded in `provenance.json` (`configs/eval/reporting.yaml`, verbatim). Every paired difference,
  interval, verdict and tail share is recomputed here from `tables/combined.csv` and must equal the pipeline's
  own table, or the build stops. The test9 trajectories are read (text only) for the tyre-life prior tier.
- `--fidelity`: the fidelity scorecard's outputs: the adopted configuration's scorecard (`SCORECARD`, with its
  `ceiling_pooled.csv`), the ablation on the adopted base (`ABLATION`) and the earlier ablation (`ABLATION_EARLIER`,
  for the learned calibrator only).

Without `--test`, or without agent rows, the numbers that need them come back pending (`Num.pending`) and the
build prints a visibly marked placeholder for each.

Intervals: the percentile cluster bootstrap of `src.eval.compare.cluster_bootstrap` (2000 resamples, seed 0,
ratio of sums), re-implemented here; primary clusters (race, driver), race clusters beside. Verdicts follow
reporting.yaml: "ahead" if the primary interval is entirely below zero, "behind" if entirely above, else "level".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
EVAL_SET = PROJECT / "data" / "eval_set.json"
SCORECARD = "belief-guard-243bf40/main"  # the latest committed scorecard of the adopted configuration (eval-v3)
DEV_DEFAULT = "reports/eval/rerun/v3/dev20"
TEST_DEFAULT = "reports/eval/rerun/v3/test9"
ABLATION = "adopt-c67c9a7/ablation"     # each adopted choice switched back off, on the adopted base
ABLATION_EARLIER = "ablation"           # f7be251's ablation: the learned calibrator only
TIE = 1e-6            # src.eval.score.REGRET_TOL_S
TOL = 1e-9            # cross-check tolerance against the pipeline's own tables
ERAS = (("2019-20", 2019, 2020), ("2021", 2021, 2021), ("2022", 2022, 2022), ("2023", 2023, 2023),
        ("2024-25", 2024, 2025))        # src.eval.rerun.ERAS, checked against the era column of the rows
COMPARE = ("run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer")
TAIL_POL = ("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy")
GROUPS = {"fresh five": "fresh", "earlier four": "earlier", "all nine": "all"}
EXPECTED_MODEL = "qwen3-30b-a3b"
CIRCUIT = {"united_kingdom": "Britain", "united_arab_emirates": "Abu Dhabi", "emilia_romagna": "Imola",
           "saudi_arabia": "Saudi Arabia", "austrian": "Austria", "sakhir": "Sakhir", "monaco": "Monaco",
           "france": "France", "mexico": "Mexico", "miami": "Miami", "australia": "Australia", "belgium": "Belgium",
           "austria": "Austria"}
LEAK_WORDS = {"stint_limit": "tyre stint limit from the whole race",
              "pit_loss": "pit loss measured over the whole race",
              "frame_filter": "clean-lap filter fitted on every lap",
              "lap_window": "fit window cut by lap, not by clock",
              "race_length": "race length from the car’s own laps",
              "compound_list": "tyre list of the whole race",
              "gap_behind": "gap behind read from the next lap"}

LEAK_SHORT = {"stint_limit": "tyre stint limit", "pit_loss": "pit loss", "frame_filter": "clean-lap filter",
              "lap_window": "fit window by lap, not clock", "race_length": "race length", "compound_list": "tyre list",
              "gap_behind": "gap behind"}   # each taken from the whole race (or the next lap)

@dataclass
class Num:
    value: object
    text: str
    source: str
    how: str
    pending: bool = False
    tag: str = ""


@dataclass
class SetData:
    tag: str
    label: str
    root: Path | None = None
    base: Path | None = None
    rows: pd.DataFrame | None = None
    prov: dict = field(default_factory=dict)
    has_agent: bool = False
    fixture: bool = False

    @property
    def present(self):
        return self.rows is not None

    def f(self, name):
        return f"{self.label}: {name}"


def s(v, nd=2, sign=False):
    t = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return t.replace("-", "−")


def ci(lo, hi, nd=2):
    if pd.isna(lo) or pd.isna(hi):
        return "[no interval]"
    return f"[{s(lo, nd)}, {s(hi, nd)}]"


def pct(x):
    return f"{100 * x:.0f}%"


def race_name(rid: str) -> str:
    season, _, slug = str(rid).split("_", 2)
    return f"{CIRCUIT.get(slug, slug.replace('_', ' ').title())} {season}"


def era_of(rid: str) -> str:
    season = int(str(rid)[:4])
    return next((n for n, lo, hi in ERAS if lo <= season <= hi), str(season))


def verdict(lo, hi, ahead="ahead", behind="behind", level="level"):
    if hi < 0:
        return ahead
    if lo > 0:
        return behind
    return level


# --- statistics (as src.eval.compare) ---------------------------------------------------------------
def cluster_bootstrap(values, clusters, n_boot=2000, level=0.95, seed=0):
    values = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(pd.Series(list(clusters)), sort=True)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 2:
        return float("nan"), float("nan")
    sums = np.bincount(codes, weights=values, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    draws = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    means = (draws @ sums) / (draws @ counts)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def ok_rows(rows, pol):
    r = rows[(rows["policy"] == pol) & (rows["status"] == "ok")].copy()
    r["regret_s"] = pd.to_numeric(r["regret_s"], errors="coerce")
    return r.dropna(subset=["regret_s"])


def paired(rows, ref, other):
    a, o = ok_rows(rows, ref).set_index("scenario_id"), ok_rows(rows, other).set_index("scenario_id")
    both = a.index.intersection(o.index)
    diff = a.loc[both, "regret_s"] - o.loc[both, "regret_s"]
    ids = list(a.loc[both, ["race_id", "driver"]].itertuples(index=False, name=None))
    lo, hi = cluster_bootstrap(diff, ids)
    rlo, rhi = cluster_bootstrap(diff, list(a.loc[both, "race_id"]))
    return {"n": len(both), "mean": float(diff.mean()) if len(both) else float("nan"), "lo": lo, "hi": hi,
            "rlo": rlo, "rhi": rhi, "wins": int((diff < -TIE).sum()), "ties": int((diff.abs() <= TIE).sum()),
            "losses": int((diff > TIE).sum()), "clusters": len(set(ids)), "races": len(set(a.loc[both, "race_id"]))}


def tail(rows, pol):
    r = ok_rows(rows, pol)["regret_s"]
    out = {"scored": len(r), "median_s": r.median(), "worst_s": r.max()}
    out.update({f"over{t}": float((r > t).mean()) for t in (10, 20, 30)})
    return out


def _close(a, b, what):
    if pd.isna(a) and pd.isna(b):
        return
    if pd.isna(a) or pd.isna(b) or abs(float(a) - float(b)) > TOL * max(1.0, abs(float(b))):
        raise SystemExit(f"cross-check failed: {what}: poster {a!r} != pipeline {b!r}")


# --- loading -------------------------------------------------------------------------------------------
def load_set(tag, label, root, allow_dev_as_test=False) -> SetData:
    sd = SetData(tag, label)
    if root is None:
        return sd
    root = Path(root)
    sd.base = root if (root / "tables").is_dir() else root.parent
    sd.root = sd.base / "tables"
    comb = sd.root / "combined.csv"
    if not comb.exists():
        raise SystemExit(f"{comb} not found: the set needs its tables stage")
    rows = pd.read_csv(comb, low_memory=False)
    rows["regret_s"] = pd.to_numeric(rows["regret_s"], errors="coerce")
    eras = rows["race_id"].map(era_of)
    if "era" in rows and not (rows["era"].astype(str) == eras).all():
        raise SystemExit(f"{comb}: era column differs from the era rule copied from src.eval.rerun.ERAS")
    rows["era"] = eras
    sd.rows = rows
    sd.has_agent = bool((rows["policy"] == "agent").any())
    sd.prov = json.loads((sd.base / "provenance.json").read_text())
    kind = str(sd.prov.get("set_kind", ""))
    if tag == "h" and "development" in kind and not allow_dev_as_test:
        raise SystemExit(f"--test {root} is a {kind}: the held-out column takes a report set only")
    if tag == "d" and "development" not in kind:
        raise SystemExit(f"--dev {root} is a {kind}, not a development set")
    sd.fixture = tag == "h" and "development" in kind
    return sd


def group_races(prov, rows):
    """{group name: [race ids]} from the reporting choices recorded verbatim in provenance.json."""
    rep = yaml.safe_load((prov.get("reporting") or {}).get("text") or "{}")
    races = sorted(rows["race_id"].unique())
    out = {}
    for g in (rep.get("held_out") or {}).get("groups", []):
        if g.get("races") == "all":
            out[g["name"]] = races
        else:
            want = {(int(r["season"]), str(r["event"])) for r in g["races"]}
            out[g["name"]] = [r for r in races if (int(r[:4]), r.split("_", 2)[2]) in want]
            if len(out[g["name"]]) != len(want):
                raise SystemExit(f"group {g['name']}: races {out[g['name']]} do not match reporting.yaml")
    return out, rep


def model_id(prov):
    """The model id the agent pass ran with: provenance agent_run.passes[].env, else the top-level model block."""
    ids = set()
    for p in (prov.get("agent_run") or {}).get("passes", []):
        env = p.get("env") or {}
        b = str(env.get("LLM_BACKEND") or "").lower()
        ids.add({"voyager": env.get("VOYAGER_MODEL"), "anthropic": env.get("ANTHROPIC_MODEL"),
                 "claude": env.get("ANTHROPIC_MODEL")}.get(b, env.get("HF_MODEL")))
    m = prov.get("model") or {}
    if not ids and m.get("LLM_BACKEND"):
        ids.add(m.get("VOYAGER_MODEL") or m.get("ANTHROPIC_MODEL") or m.get("HF_MODEL"))
    ids.discard(None)
    return ids


def parse_cell(cell):
    x = [float(v) for v in re.findall(r"[-+]?\d+\.\d+", str(cell))]
    return tuple(x) if len(x) == 3 else (x[0], np.nan, np.nan)


# --- compute -----------------------------------------------------------------------------------------------
def compute(dev=None, test=None, fidelity=PROJECT / "reports" / "calibration" / "fidelity", noise_floor=None,
            allow_dev_as_test=False, scorecard=SCORECARD, ablation=ABLATION) -> dict[str, Num]:
    N: dict[str, Num] = {}

    def put(key, value, text, source, how):
        N[key] = Num(value, text, source, how)

    def pend(key, source, how, waits):
        N[key] = Num(None, "pending", source, how, pending=True, tag=waits)

    D = load_set("d", "development", dev)
    H = load_set("h", "test set", test, allow_dev_as_test)
    if not D.present:
        raise SystemExit("--dev is required")
    held = json.loads(EVAL_SET.read_text())
    held = list(held["races"] if isinstance(held, dict) else held)
    if H.present and not H.fixture and sorted(H.rows["race_id"].unique()) != sorted(held):
        raise SystemExit("--test rows do not cover exactly the races of data/eval_set.json")
    put("h_races_reg", len(held), str(len(held)), "data/eval_set.json", "race ids registered as held out")

    # ---- the groups: held-out fresh / earlier / all, development all
    groups = {}
    if H.present:
        gr, rep = group_races(H.prov, H.rows)
        for name, short in GROUPS.items():
            groups[f"h{short}"] = (H, name, gr[name])
    else:
        rep = yaml.safe_load((D.prov.get("reporting") or {}).get("text") or "{}")
        for g in (rep.get("held_out") or {}).get("groups", []):
            short = GROUPS[g["name"]]
            ids = held if g.get("races") == "all" else [r for r in held if (int(r[:4]), r.split("_", 2)[2]) in
                                                         {(int(x["season"]), str(x["event"])) for x in g["races"]}]
            groups[f"h{short}"] = (H, g["name"], ids)
    groups["d"] = (D, "development (all)", sorted(D.rows["race_id"].unique()))
    for gk, (S, name, ids) in groups.items():
        put(f"{gk}_nraces", len(ids), str(len(ids)), "reporting.yaml (provenance.json)", f"races of group {name}")
        put(f"{gk}_names", ids, ", ".join(race_name(r) for r in ids), "reporting.yaml (provenance.json)",
            f"races of group {name}: {', '.join(ids)}")
        put(f"{gk}_word", len(ids), {4: "four", 5: "five", 9: "nine"}.get(len(ids), str(len(ids))),
            "reporting.yaml (provenance.json)", f"races of group {name}, in words")

    # ---- paired: agent minus X per group, checked against tables/reporting_paired.csv
    for gk, (S, name, ids) in groups.items():
        keys = lambda x: [f"{gk}_{x}_{k}" for k in ("diff", "ci", "cir", "verdict", "n", "wtl")]  # noqa: E731
        if not S.present or not S.has_agent:
            for x in COMPARE:
                for k in keys(x):
                    pend(k, S.label, f"agent minus {x}, {name}", "held-out run" if S.tag == "h" else "dev agent pass")
            continue
        g = S.rows[S.rows["race_id"].isin(ids)]
        rp = pd.read_csv(S.root / "reporting_paired.csv")
        src = S.f("tables/reporting_paired.csv")
        for x in COMPARE:
            p = paired(g, "agent", x)
            c = rp[(rp["group"] == name) & (rp["policy"] == x)].iloc[0]
            for a, b in (("mean", "mean_diff"), ("lo", "ci_low_race_driver"), ("hi", "ci_high_race_driver"),
                         ("rlo", "ci_low_race"), ("rhi", "ci_high_race")):
                _close(p[a], c[b], f"{S.label} {name} agent minus {x} {a}")
            if (p["n"], p["wins"], p["ties"], p["losses"]) != (int(c["paired"]), int(c["reference_wins"]), int(c["ties"]),
                                                                 int(c["policy_wins"])):
                raise SystemExit(f"cross-check failed: {S.label} {name} agent minus {x} counts")
            v = verdict(p["lo"], p["hi"])
            if v != c["verdict"]:
                raise SystemExit(f"cross-check failed: {name} {x} verdict {v} != {c['verdict']}")
            how = (f"agent minus {x}, {name}: mean over {p['n']} scenarios both scored; 95% cluster bootstrap over "
                   f"{p['clusters']} (race, driver) clusters (race interval beside, {p['races']} races), 2000 resamples, "
                   f"seed 0; verdict from the (race, driver) interval; equals {src}")
            k = keys(x)
            nd = 2 if x == "belief_optimizer" else 1
            put(k[0], p["mean"], s(p["mean"], nd, sign=True), src, how)
            put(k[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], nd), src, how)
            put(k[2], (p["rlo"], p["rhi"]), ci(p["rlo"], p["rhi"], nd), src, how + " (race clusters)")
            put(k[3], v, v, src, how)
            put(k[4], p["n"], str(p["n"]), src, how)
            put(k[5], (p["wins"], p["ties"], p["losses"]), f"{p['wins']} / {p['ties']} / {p['losses']}", src, how)

    # ---- tail per group, checked against tables/reporting_tail.csv
    for gk in ("hfresh", "hall", "d"):
        S, name, ids = groups[gk]
        for pol in TAIL_POL:
            ks = [f"{gk}_t_{c}_{pol}" for c in ("over10", "over20", "over30", "median", "worst")]
            if not S.present or (pol == "agent" and not S.has_agent):
                for k in ks:
                    pend(k, S.label, f"tail {pol} {name}", "held-out run" if S.tag == "h" else "dev agent pass")
                continue
            t = tail(S.rows[S.rows["race_id"].isin(ids)], pol)
            rt = pd.read_csv(S.root / "reporting_tail.csv")
            c = rt[(rt["group"] == name) & (rt["policy"] == pol)].iloc[0]
            for a, b in (("over20", "share_over_20s (stated)"), ("over10", "share_over_10s"), ("over30", "share_over_30s"),
                         ("median_s", "median_s"), ("worst_s", "worst_s")):
                _close(t[a], c[b], f"{S.label} {name} tail {pol} {a}")
            src = S.f("tables/reporting_tail.csv")
            how = f"{pol}, {name}, over its {t['scored']} scored scenarios; equals {src}"
            put(ks[0], t["over10"], pct(t["over10"]), src, "share with regret_s > 10 s; " + how)
            put(ks[1], t["over20"], pct(t["over20"]), src, "share with regret_s > 20 s (the stated threshold); " + how)
            put(ks[2], t["over30"], pct(t["over30"]), src, "share with regret_s > 30 s; " + how)
            put(ks[3], t["median_s"], s(t["median_s"], 1), src, "median regret_s; " + how)
            put(ks[4], t["worst_s"], s(t["worst_s"], 0), src, "largest regret_s; " + how)

    # ---- held-out by era, checked against tables/reporting_paired_by_era.csv
    eras = sorted({era_of(r) for r in (groups["hall"][2])}, key=lambda e: [x[0] for x in ERAS].index(e))
    put("h_eras", eras, str(len(eras)), "data/eval_set.json", "eras of the held-out races (src.eval.rerun.ERAS)")
    for e in eras:
        ids = [r for r in groups["hall"][2] if era_of(r) == e]
        put(f"era_{e}_names", ids, ", ".join(race_name(r) for r in ids), "data/eval_set.json", f"held-out races, era {e}")
        for x in ("run_to_flag", "actual_strategy"):
            ks = [f"era_{e}_{x}_{k}" for k in ("diff", "ci", "verdict")]
            if not H.present or not H.has_agent:
                for k in ks:
                    pend(k, "held-out", f"era {e} agent minus {x}", "held-out run")
                continue
            p = paired(H.rows[H.rows["era"] == e], "agent", x)
            be = pd.read_csv(H.root / "reporting_paired_by_era.csv")
            c = be[(be["era"] == e) & (be["policy"] == x)].iloc[0]
            for a, b in (("mean", "mean_diff"), ("lo", "ci_low_race_driver"), ("hi", "ci_high_race_driver")):
                _close(p[a], c[b], f"era {e} agent minus {x} {a}")
            v = verdict(p["lo"], p["hi"])
            if v != c["verdict"]:
                raise SystemExit(f"cross-check failed: era {e} {x} verdict")
            src = H.f("tables/reporting_paired_by_era.csv")
            how = f"agent minus {x}, era {e} ({len(ids)} races, {p['n']} paired); equals {src}"
            put(ks[0], p["mean"], s(p["mean"], 1, sign=True), src, how)
            put(ks[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
            put(ks[2], v, v, src, how)

    # ---- per race, fresh five: agent and optimizer minus run_to_flag; tyre-life prior tier from the trajectories
    for r in groups["hall"][2]:
        ks = [f"pr_{r}_agent", f"pr_{r}_bo", f"pr_{r}_tier", f"pr_{r}_agent_ci", f"pr_{r}_n"]
        if not H.present or not H.has_agent:
            for k in ks:
                pend(k, "held-out", f"per race {r}", "held-out run")
            continue
        g = H.rows[H.rows["race_id"] == r]
        src = H.f("tables/combined.csv")
        for k, pol in ((ks[0], "agent"), (ks[1], "belief_optimizer")):
            p = paired(g, pol, "run_to_flag")
            how = (f"{pol} minus run_to_flag at {r}: mean over {p['n']} scenarios both scored, (race, driver) cluster "
                   "bootstrap 95% interval (computed here; no per-race pipeline table)")
            put(k, p["mean"], s(p["mean"], 1, sign=True), src, how)
            if pol == "agent":
                put(ks[3], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
                put(ks[4], p["n"], str(p["n"]), src, how)
        tiers = {}
        for fp in sorted((H.base / "trajectories").glob(f"{r}_*.jsonl")):
            for t in re.findall(r'"prior_source": "tyre_history_tier(\d)"', fp.read_text()):
                tiers[t] = tiers.get(t, 0) + 1
        put(ks[2], tiers, "/".join(sorted(tiers)) or "none", H.f(f"trajectories/{r}_*.jsonl"),
            f"tyre-life prior tiers in the tools' model reports (tyre_history_tierN counts {tiers}); 1 = same circuit, "
            "3 = the pool only, 4 = no usable earlier race")
    if H.present and H.has_agent:
        hist = [r for r in groups["hearlier"][2] if "1" in N[f"pr_{r}_tier"].value]
        put("earlier_hist_n", len(hist), {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(hist), str(len(hist))),
            H.f("trajectories"), "earlier-four races whose tools reported a same-circuit tyre-life prior (tier 1): "
            + ", ".join(hist))
        fresh_hist = [r for r in groups["hfresh"][2] if "1" in N[f"pr_{r}_tier"].value]
        put("fresh_hist_n", len(fresh_hist), str(len(fresh_hist)), H.f("trajectories"),
            "fresh-five races with a same-circuit tyre-life prior (tier 1): " + (", ".join(fresh_hist) or "none"))
        allt = {N[f"pr_{r}_tier"].text for r in groups["hfresh"][2]}
        put("fresh_tier", allt, " and ".join(sorted(allt)), H.f("trajectories"),
            "the tyre-life prior tiers seen at the fresh five, every report: " + ", ".join(
                f"{r}: {N[f'pr_{r}_tier'].value}" for r in groups["hfresh"][2]))
    else:
        for k in ("fresh_tier", "earlier_hist_n", "fresh_hist_n"):
            pend(k, "held-out", "tiers", "held-out run")

    # ---- reliability and cost (agent.csv, tables/agent_latency.csv)
    for S in (H, D):
        t = S.tag
        ks = ("ag_scored", "ag_ok", "ag_vol", "ag_noref", "ag_n", "lat", "tok", "ties_bo")
        if not S.present or not S.has_agent:
            for k in ks:
                pend(f"{t}_{k}", S.label, k, "held-out run" if t == "h" else "dev agent pass")
            continue
        a = pd.read_csv(S.base / "agent.csv")
        n = len(a)
        noref = int((a["status"] == "no_reference").sum())
        scored = n - noref
        okn = int((a["status"] == "ok").sum())
        sub = a["decision"].isin(["PIT", "STAY_OUT"])
        vol = int((sub & ~a["submitted_in_grace"].astype(str).str.lower().isin(("true", "1", "1.0"))).sum())
        src = S.f("agent.csv")
        put(f"{t}_ag_n", n, str(n), src, "agent rows (scenarios)")
        put(f"{t}_ag_noref", noref, str(noref), src, "rows with status no_reference (no legal plan for any policy)")
        put(f"{t}_ag_scored", scored, str(scored), src, "scenarios with a reference (agent rows less no_reference)")
        put(f"{t}_ag_ok", okn, f"{okn} of {scored}", src, f"agent rows ok (a legal, scored decision; {int(sub.sum())} "
            f"submitted, {int((a['status'] == 'invalid').sum())} invalid)")
        put(f"{t}_ag_vol", vol, str(vol), src, "decisions submitted outside a grace (forced-submission) turn")
        lat = pd.read_csv(S.root / "agent_latency.csv")
        iso = lat[lat["run"].str.startswith("isolated")].iloc[0]
        lsrc = S.f("tables/agent_latency.csv")
        put(f"{t}_lat", float(iso["latency_mean_s"]), f"{iso['latency_mean_s']:.0f} s", lsrc,
            f"mean decision time, isolated timing sample of {int(iso['episodes'])} episodes run one at a time")
        put(f"{t}_tok", float(iso["input_tokens_mean"]), f"{iso['input_tokens_mean'] / 1000:.0f}k", lsrc,
            "mean input tokens per decision, the same isolated sample")
        allrow = lat[lat["run"].str.contains("all scenarios")].iloc[0]
        put(f"{t}_calls", float(allrow["tool_calls_mean"]), f"{allrow['tool_calls_mean']:.0f}", lsrc,
            "mean tool calls per decision over the whole pass (row 'concurrent, all scenarios')")
        put(f"{t}_tok", float(allrow["input_tokens_mean"]), f"{allrow['input_tokens_mean'] / 1000:.0f}k", lsrc,
            "mean input tokens per decision over the whole pass (row 'concurrent, all scenarios')")
        allg = "hall" if t == "h" else "d"
        put(f"{t}_ties_bo", N[f"{allg}_belief_optimizer_wtl"].value[1], str(N[f"{allg}_belief_optimizer_wtl"].value[1]),
            N[f"{allg}_belief_optimizer_wtl"].source, "scenarios where the agent's regret equals the optimizer's (ties)")

    # ---- provenance: commit, tag, model, reporting hash, harness, configs
    for S in (H, D):
        t = S.tag
        if not S.present:
            for k in ("commit", "tag", "model", "rep", "harness", "fb_fixed", "fb_known", "sc", "edge", "bde",
                      "traffic", "hinge", "learned", "n", "races"):
                pend(f"{t}_{k}", "--test", k, "held-out run")
            continue
        P, src = S.prov, f"{S.label}: provenance.json"
        sc_ = S.rows.drop_duplicates("scenario_id")
        put(f"{t}_n", len(sc_), str(len(sc_)), S.f("tables/combined.csv"), "distinct scenario_id")
        put(f"{t}_races", sc_["race_id"].nunique(), str(sc_["race_id"].nunique()), S.f("tables/combined.csv"),
            "distinct race_id")
        put(f"{t}_commit", P.get("commit"), (P.get("commit") or "?")[:7] + (" (dirty)" if P.get("dirty") else ""),
            src, "commit of the run (and whether the tree was dirty)")
        put(f"{t}_tag", P.get("tag"), P.get("tag") or "", src, "git tag at the commit")
        ids = model_id(P)
        if S.has_agent:
            if ids != {EXPECTED_MODEL}:
                raise SystemExit(f"{src}: the agent pass ran {ids or 'no recorded model'}, not {EXPECTED_MODEL}")
            put(f"{t}_model", EXPECTED_MODEL, EXPECTED_MODEL, src, "model id from agent_run.passes[].env")
        else:
            pend(f"{t}_model", src, "model id", "agent pass")
        r = P.get("reporting") or {}
        if r.get("text") and hashlib.sha256(r["text"].encode()).hexdigest() != r.get("sha256"):
            raise SystemExit(f"{src}: the recorded reporting.yaml text does not hash to its recorded sha256")
        put(f"{t}_rep", r.get("sha256"), (r.get("sha256") or "?")[:8], src,
            "sha256 of configs/eval/reporting.yaml as recorded by the run (text verified against it)")
        fb = P.get("future_blindness") or {}
        dp = fb.get("default_part") or {}
        put(f"{t}_harness", dp, f"{dp.get('passed', 0)} passed, {dp.get('failed', 0)} failed" if dp else "see dev20",
            src, "the future-blindness harness's default part, run by `tables --harness`: " + str(fb.get(
                "default_part_summary")))
        put(f"{t}_fb_fixed", fb.get("fixed_leaks", []), str(len(fb.get("fixed_leaks", []))), src,
            "tests/future_blindness.py FIXED_LEAKS: " + ", ".join(fb.get("fixed_leaks", [])))
        put(f"{t}_fb_known", fb.get("known_leaks", []), str(len(fb.get("known_leaks", []))), src,
            "tests/future_blindness.py KNOWN_LEAKS (open leaks)")
        w = (P.get("world") or {}).get("choices") or {}
        b = (P.get("belief") or {}).get("choices") or {}
        if not w or not b:
            raise SystemExit(f"{src}: no world/belief choices recorded")
        flags = {"sc": (str(w.get("events", "none")) != "none", f"world events {w.get('events')} {w.get('events_params')}"),
                 "edge": (str(w.get("tyre_life_edge")) == "history_only", f"world tyre_life_edge {w.get('tyre_life_edge')}"),
                 "bde": (str(b.get("driver_effects")) not in ("None", "off", "False"),
                         f"belief driver_effects {b.get('driver_effects')}"),
                 "traffic": (str(w.get("traffic")) != "none" or str(b.get("traffic")) != "none",
                             f"traffic world {w.get('traffic')} belief {b.get('traffic')}"),
                 "hinge": (bool(w.get("hinge")), f"world hinge {w.get('hinge')}"),
                 "learned": ("learned" in str(b.get("calibrator")) or "trained" in str(b.get("calibrator")),
                             f"belief calibrator {b.get('calibrator')}")}
        for k, (on, how) in flags.items():
            put(f"{t}_{k}", on, "on" if on else "off", src + " (world/belief choices)", how)

    # ---- fidelity: the adopted configuration's scorecard, its noise floor, the ablations
    fid = Path(fidelity)
    sdir = fid / scorecard
    run = json.loads((sdir / "run.json").read_text())
    card = (sdir / "scorecard.md").read_text()
    fsrc = f"reports/calibration/fidelity/{scorecard}/run.json"
    hl = run["headline"]
    put("fid_races", len(run["races"]), str(len(run["races"])), fsrc, "races the scorecard ran: " + ", ".join(run["races"]))
    if run.get("failed"):
        raise SystemExit(f"{fsrc}: races failed: {run['failed']}")
    for key, col, nd in (("fid_close", "A_close_pair_acc", 3), ("fid_relgap", "A_rel_gap_mae_s", 1)):
        mm = re.search(rf"\| {col} \| ([-\d.]+) \|", card)
        if mm:
            _close(round(hl[col], 3), float(mm.group(1)), f"{col}: run.json vs scorecard.md")
        put(key, hl[col], s(hl[col], nd), fsrc, f"headline {col} (definition in scorecard.md)")
    fbs = run.get("future_blindness") or {}
    put("fid_fb", fbs.get("status"), str(fbs.get("status")), fsrc, "the scorecard's own future-blindness check (B5)")
    led = pd.read_csv(sdir / "ledger.csv")
    whole = led[(led["role"] == "judge") & (led["tag"] == "this race, whole")]["id"].drop_duplicates().tolist()
    words = {"fit_window": "lap-time fit", "clean_flag": "clean-lap filter", "ref_lap_s": "safety-car lap time",
             "pit_loss": "pit loss", "cliff_edge": "tyre life", "proxy_raced": "tyres raced",
             "driver_effects": "driver effects", "events": "safety cars in progress"}
    put("fid_ledger_words", whole, ", ".join(words.get(w, w.replace("_", " ")) for w in whole),
        f"reports/calibration/fidelity/{scorecard}/ledger.csv", "judge inputs tagged 'this race, whole': " + ", ".join(whole))
    cfile = Path(noise_floor) if noise_floor else sdir / "ceiling_pooled.csv"
    ceil = pd.read_csv(cfile)
    csrc = str(cfile.relative_to(PROJECT)) if cfile.is_absolute() and PROJECT in cfile.parents else str(cfile)
    lv0 = ceil[ceil["level"].astype(str).str.startswith("0")].iloc[0]
    lv4 = ceil[ceil["level"].astype(str).str.startswith("4")].iloc[0]
    lv3 = ceil[ceil["level"].astype(str).str.startswith("3")].iloc[0]
    _close(lv0["close_pair_acc"], hl["A_close_pair_acc"], "noise floor level 0 vs headline close pairs")
    _close(lv0["rel_gap_mae_s"], hl["A_rel_gap_mae_s"], "noise floor level 0 vs headline relative gap")
    how4 = "level 4 of the noise floor: the judge handed every stint's pace, the real stops, the SC/VSC laps and lap 1"
    put("ceil_close", lv4["close_pair_acc"], s(lv4["close_pair_acc"], 3), csrc, how4 + "; close-pair order")
    put("ceil_relgap", lv4["rel_gap_mae_s"], s(lv4["rel_gap_mae_s"], 1), csrc, how4 + "; relative gap error s")
    put("fid_close_ci", (lv0["close_pair_lo"], lv0["close_pair_hi"]), ci(lv0["close_pair_lo"], lv0["close_pair_hi"], 3),
        csrc, "level 0 (the judge), race-cluster 95% interval")
    put("ceil_close_ci", (lv4["close_pair_lo"], lv4["close_pair_hi"]), ci(lv4["close_pair_lo"], lv4["close_pair_hi"], 3),
        csrc, "level 4, race-cluster 95% interval")
    put("fid_relgap_ci", (lv0["rel_gap_lo"], lv0["rel_gap_hi"]), ci(lv0["rel_gap_lo"], lv0["rel_gap_hi"], 1), csrc,
        "level 0 relative gap, race-cluster 95% interval")
    put("ceil_relgap_ci", (lv4["rel_gap_lo"], lv4["rel_gap_hi"]), ci(lv4["rel_gap_lo"], lv4["rel_gap_hi"], 1), csrc,
        "level 4 relative gap, race-cluster 95% interval")
    put("ceil_relgap_sc", lv3["rel_gap_mae_s"], s(lv3["rel_gap_mae_s"], 1), csrc,
        "level 3 (stints, stops and SC/VSC laps as recorded): relative gap error s")
    put("ceil_pairs", int(lv0["close_pairs"]), str(int(lv0["close_pairs"])), csrc, "close pairs pooled")
    for name, adir, items in ((ablation, fid / ablation,
                               (("world_events_off", "A_close_pair_acc", 3), ("world_events_off", "A_rel_gap_mae_s", 1),
                                ("cliff_edge_history", "A_close_pair_acc", 3),
                                ("belief_no_driver_effects", "B2_regret_s", 1),
                                ("traffic_both", "A_close_pair_acc", 3), ("world_hinge", "A_close_pair_acc", 3))),
                              (ABLATION_EARLIER, fid / ABLATION_EARLIER, (("learned_trained", "B2_regret_s", 1),))):
        heads = pd.read_csv(adir / "headlines.csv").set_index("factor")
        abl = pd.read_csv(adir / "ablation.csv").set_index("factor")
        md = (adir / "ablation.md").read_text()
        m = re.search(r"Computed on main ([0-9a-f]{7,40})", md)
        asrc = f"reports/calibration/fidelity/{name}/ablation.csv"
        for factor, col, nd in items:
            d, lo, hi = parse_cell(abl.loc[factor, col])
            _close(round(heads.loc[factor, col] - heads.loc["base", col], 3), d, f"ablation {name} {factor} {col}")
            how = (f"{factor} minus the base ({name}{', main ' + m.group(1) if m else ''}), {col}, pooled; race-cluster "
                   f"bootstrap 95% interval; equals headlines.csv difference")
            put(f"abl_{factor}_{col}", (d, lo, hi), f"{s(d, nd, sign=True)} {ci(lo, hi, nd)}", asrc, how)
    put("fid_close_pct", hl["A_close_pair_acc"], f"{100 * hl['A_close_pair_acc']:.0f}%", fsrc,
        "headline A_close_pair_acc as a percentage")
    put("ceil_close_pct", lv4["close_pair_acc"], f"{100 * lv4['close_pair_acc']:.0f}%", csrc,
        "noise-floor level 4 close-pair order as a percentage")
    put("fid_b2regret", hl["B2_regret_s"], s(hl["B2_regret_s"], 1), fsrc,
        "headline B2_regret_s: judge time the predictive simulator's favourite plan loses to the judge's favourite, mean "
        "over snapshots of the calibration races")
    extra(N, put, pend, D, H, groups)
    v3(N, put, pend, D, H)
    N["_sets"] = Num((D, H), "", "", "")
    return N


MONACO = "2019_06_monaco"
SAME_CIRCUIT = {"austrian": "austria"}   # FastF1 event slugs naming one circuit two ways (2021 Austrian GP)


def extra(N, put, pend, D, H, groups):
    """The cautious readings: one-race sensitivity (Monaco), per-race means at the two fresh races that carry the
    shortfall, and development's lead split by whether the tyre-life history holds an earlier race at the circuit."""
    # without Monaco (all nine, earlier four): agent minus the teams and minus run_to_flag
    for gk in ("hall", "hearlier"):
        S, name, ids = groups[gk]
        for x in ("actual_strategy", "run_to_flag"):
            ks = [f"{gk}_nomon_{x}_{k}" for k in ("diff", "ci", "verdict")]
            if not S.present or not S.has_agent:
                for k in ks:
                    pend(k, "held-out", "without Monaco", "held-out run")
                continue
            p = paired(S.rows[S.rows["race_id"].isin([r for r in ids if r != MONACO])], "agent", x)
            src = S.f("tables/combined.csv")
            how = (f"agent minus {x}, {name} without {MONACO}: {p['n']} paired, (race, driver) cluster bootstrap "
                   "(computed here; a sensitivity check, not pre-registered)")
            put(ks[0], p["mean"], s(p["mean"], 1, sign=True), src, how)
            put(ks[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
            put(ks[2], verdict(p["lo"], p["hi"]), verdict(p["lo"], p["hi"]), src, how)
    # per-race mean regret (tables/summary.csv, checked against the rows)
    if H.present and H.has_agent:
        summ = pd.read_csv(H.root / "summary.csv")
        for r, pols in ((MONACO, ("actual_strategy",)), ("2022_03_australia", ("agent", "run_to_flag")),
                        ("2022_05_miami", ("agent", "run_to_flag"))):
            for pol in pols:
                rr = ok_rows(H.rows[H.rows["race_id"] == r], pol)["regret_s"]
                c = summ[(summ["race_id"] == r) & (summ["policy"] == pol)].iloc[0]
                _close(rr.mean(), c["mean_regret_s"], f"{r} {pol} mean vs summary.csv")
                src = H.f("tables/summary.csv")
                put(f"race_{r}_{pol}_mean", rr.mean(), s(rr.mean(), 1), src,
                    f"mean regret_s of {pol} at {r} over its {len(rr)} scored scenarios; equals summary.csv")
                put(f"race_{r}_{pol}_max", rr.max(), s(rr.max(), 0), src, f"largest regret_s of {pol} at {r}")
    else:
        for k in ("race_2019_06_monaco_actual_strategy_mean", "race_2019_06_monaco_actual_strategy_max",
                  "race_2022_03_australia_agent_mean", "race_2022_03_australia_run_to_flag_mean",
                  "race_2022_05_miami_agent_mean", "race_2022_05_miami_run_to_flag_mean"):
            pend(k, "held-out", "per race", "held-out run")
    # development lead split by circuit history in the tyre-life table (provenance as_of_tables.tyre_history)
    reads = ((D.prov.get("as_of_tables") or {}).get("tyre_history") or {}).get("reads") or {}
    circ = lambda r: SAME_CIRCUIT.get(r.split("_", 2)[2], r.split("_", 2)[2])  # noqa: E731
    races = sorted(D.rows["race_id"].unique())
    withh = [r for r in races if any(circ(x) == circ(r) for x in reads.get(r, []))]
    for lab, rs in (("with", withh), ("without", [r for r in races if r not in withh])):
        ks = [f"dev_hist_{lab}_{k}" for k in ("diff", "ci", "n")]
        if not D.has_agent:
            for k in ks:
                pend(k, "development", "history split", "dev agent pass")
            continue
        p = paired(D.rows[D.rows["race_id"].isin(rs)], "agent", "run_to_flag")
        src = "development: tables/combined.csv, provenance.json as_of_tables.tyre_history.reads"
        how = (f"agent minus run_to_flag at the {len(rs)} development races {lab} an earlier race at the same circuit in "
               f"their tyre-life history table ({', '.join(rs)}); (race, driver) cluster bootstrap (computed here)")
        put(ks[0], p["mean"], s(p["mean"], 1, sign=True), src, how)
        put(ks[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
        put(ks[2], len(rs), str(len(rs)), src, how)


def provenance_md(N: dict[str, Num], used: list[str], args_line: str) -> str:
    lines = ["# Numbers on the poster and where they come from", "",
             "Generated by `poster/build_poster.py` (" + args_line + "). Every printed number, its file and its "
             "computation. Each paired difference, interval, verdict and tail share was recomputed from "
             "`tables/combined.csv` and matched the pipeline's own table, or the build would have stopped. "
             "Rows marked **pending** print as a placeholder until the run named exists.", "",
             "| key | printed | file | computation |", "|---|---|---|---|"]
    for k in used:
        n = N[k]
        text = f"**pending** ({n.tag})" if n.pending else n.text.replace("|", "/")
        lines.append(f"| `{k}` | {text} | `{n.source}` | {n.how.replace('|', '/')} |")
    pend = sorted({N[k].tag for k in used if N[k].pending})
    lines += ["", "Pending inputs: " + (", ".join(pend) if pend else "none") + "."]
    return "\n".join(lines) + "\n"



LEVEL_POL = ("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy", "fixed_one_stop",
             "fixed_two_stop")
# the scorecard's adoption list, checked against the run's recorded world and belief choices
ADOPTED = (("Safety cars and the safety-car queue", lambda w, b: w.get("events") == "hybrid"
            and bool((w.get("sc_queue") or {}).get("enabled"))),
           ("Driver effects", lambda w, b: w.get("driver_effects") == "race" and b.get("driver_effects") == "race"),
           ("Tyre sets", lambda w, b: bool(w.get("tyre_inventory")) and bool(b.get("tyre_inventory"))),
           ("Thin-data guards", lambda w, b: bool((w.get("wear_guard") or {}).get("enabled"))
            and bool((b.get("belief_guard") or {}).get("enabled"))),
           ("Tyre-life history", lambda w, b: "as_of" in str(w.get("tyre_life_edge"))
            and "as_of" in str(b.get("tyre_life_edge"))))
NOT_ADOPTED = (("Traffic and overtaking difficulty", lambda w, b: w.get("traffic") == "none" and b.get("traffic") == "none"),
               ("Tyre-supplier guidance", None),
               ("Learned calibrator", lambda w, b: "learned" not in str(b.get("calibrator"))
                and "trained" not in str(b.get("calibrator"))),
               ("Wear hinge", lambda w, b: not w.get("hinge") and not b.get("hinge")))


def v3(N, put, pend, D, H):
    """The eval-v3 poster's extra numbers: per-policy mean regret on both sets, departures from the look-ahead,
    and the adoption list checked against the run's configuration."""
    for S in (H, D):
        t = S.tag
        if not S.present:
            continue
        chart = pd.read_csv(S.root / "chart_regret_s.csv")
        for pol in LEVEL_POL:
            r = ok_rows(S.rows, pol)
            ids = list(r[["race_id", "driver"]].itertuples(index=False, name=None))
            lo, hi = cluster_bootstrap(r["regret_s"], ids)
            c = chart[chart["policy"] == pol].iloc[0]
            for a, b in ((r["regret_s"].mean(), c["mean"]), (lo, c["ci_low"]), (hi, c["ci_high"])):
                _close(a, b, f"{S.label} {pol} mean regret vs chart_regret_s.csv")
            if len(r) != int(c["scenarios"]):
                raise SystemExit(f"cross-check failed: {S.label} {pol} scored")
            src = S.f("tables/chart_regret_s.csv")
            how = (f"mean regret_s of {pol} over its {len(r)} scored scenarios; (race, driver) cluster bootstrap 95% "
                   "interval; equals chart_regret_s.csv")
            put(f"{t}_mean_{pol}", float(r["regret_s"].mean()), s(r["regret_s"].mean(), 1), src, how)
            put(f"{t}_mci_{pol}", (lo, hi), ci(lo, hi, 1), src, how)
        # departures: the agent's scored plan differs from the look-ahead's top-ranked plan
        files = sorted((S.base / "run_support").glob("*departures.csv"))
        if not files:
            for k in ("dep_n", "dep_helped", "dep_hurt", "dep_net"):
                pend(f"{t}_{k}", S.label, "departures", "departures file")
            continue
        dep = pd.read_csv(files[0])
        a = ok_rows(S.rows, "agent").set_index("scenario_id")["regret_s"]
        b = ok_rows(S.rows, "belief_optimizer").set_index("scenario_id")["regret_s"]
        ok = dep[dep["status"] == "ok"].set_index("scenario_id")
        delta = (a - b).reindex(ok.index)
        if (delta - ok["delta_vs_bo"]).abs().max() > 1e-9:
            raise SystemExit(f"cross-check failed: {files[0].name} delta_vs_bo differs from combined.csv")
        dp = ok[ok["departs_plan"].astype(bool)]
        src = f"{S.label}: run_support/{files[0].name}, tables/combined.csv"
        how = ("decisions whose scored plan differs from the look-ahead's top-ranked plan (departs_plan); helped / hurt: "
               "agent regret lower / higher than belief_optimizer's by more than 0.05 s (outcome); delta checked "
               "against combined.csv")
        put(f"{t}_dep_n", len(dp), str(len(dp)), src, how)
        put(f"{t}_dep_helped", int((dp["outcome"] == "helped").sum()), str(int((dp["outcome"] == "helped").sum())), src, how)
        put(f"{t}_dep_hurt", int((dp["outcome"] == "hurt").sum()), str(int((dp["outcome"] == "hurt").sum())), src, how)
        net = float(dp["delta_vs_bo"].sum())
        put(f"{t}_dep_net", net, s(net, 1, sign=True), src, "sum of agent minus belief_optimizer regret over the departures")
        st = dp[dp["kind"].astype(str).str.startswith("stated")]
        hows = ("departures for which the agent stated a reason (kind 'stated: ...'): the scored plan differs from the "
                "look-ahead's top plan and the submission names another action, compound or stop lap; helped / hurt as above")
        put(f"{t}_sdep_n", len(st), str(len(st)), src, hows)
        put(f"{t}_sdep_helped", int((st["outcome"] == "helped").sum()), str(int((st["outcome"] == "helped").sum())), src,
            hows)
        put(f"{t}_sdep_hurt", int((st["outcome"] == "hurt").sum()), str(int((st["outcome"] == "hurt").sum())), src, hows)
    # the adoption list against the configuration the runs used
    for S in (H, D):
        if not S.present:
            continue
        w = (S.prov.get("world") or {}).get("choices") or {}
        b = (S.prov.get("belief") or {}).get("choices") or {}
        for lab, check in ADOPTED:
            if not check(w, b):
                raise SystemExit(f"{S.label}: '{lab}' is listed as adopted but the run's configuration does not have it")
        for lab, check in NOT_ADOPTED:
            if check is not None and not check(w, b):
                raise SystemExit(f"{S.label}: '{lab}' is listed as not adopted but the run's configuration has it")
    put("adopted", [a for a, _ in ADOPTED], ", ".join(a for a, _ in ADOPTED), "provenance.json world/belief choices",
        "adopted factors, each checked on in both runs' recorded configuration")
    put("not_adopted", [a for a, _ in NOT_ADOPTED], ", ".join(a for a, _ in NOT_ADOPTED),
        "provenance.json world/belief choices; docs/eval.md",
        "factors tested by the scorecard and not adopted; traffic, learned calibrator and hinge checked off in both runs' "
        "configuration (tyre-supplier guidance has no switch in the recorded choices)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test")
    ap.add_argument("--fidelity", default=str(PROJECT / "reports" / "calibration" / "fidelity"))
    a = ap.parse_args()
    N = compute(a.dev, a.test, a.fidelity)
    for k, n in N.items():
        if not k.startswith("_"):
            print(f"{k:40s} {('PENDING ' + n.tag) if n.pending else n.text:34s} {n.how[:80]}")
