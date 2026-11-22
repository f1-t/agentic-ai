"""Every number printed on the 24 x 36 in PITWALL poster, computed from the evaluation pipeline's files.

    python poster/poster_stats.py --dev <dir> [--test <dir>] --fidelity reports/calibration/fidelity

Inputs (files only: nothing is loaded, simulated or scored, and nothing is imported from `src`):

- `--dev`, `--test`: output directories of `python -m src.eval.rerun ... --out-dir <dir>` (development set
  `dev20`, report set `test9`). Each holds `scenarios.json`, `provenance.json` and `tables/` (written by the
  `tables` stage): `combined.csv` (every row), `summary.csv`, `chart_regret_s.csv`, `tail.csv`,
  `paired_<agent|belief_optimizer>_minus_x.csv`, and for a set with eras `tail_by_era.csv`,
  `chart_regret_s_by_era.csv`, `paired_<ref>_minus_x_era_<era>.csv`. Every number is recomputed here from
  `combined.csv` and must equal the pipeline's own table, or the build stops.
- `--fidelity`: the fidelity scorecard's outputs (`main/run.json`, `main/scorecard.md`, `main/ledger.csv`,
  `ablation/headlines.csv`, `ablation/ablation.csv`). The noise floor is read from a `ceiling_pooled.csv` under
  that directory when one exists, else from the table under "### The noise floor" in `docs/eval.md`.
- `data/eval_set.json`: the list of held-out race ids (ids only; no race is read).

When `--test` is absent, or a set has no agent rows, the numbers that need them are returned as pending
(`Num.pending`), and the build prints a visibly marked placeholder for each.

Intervals are the percentile cluster bootstrap of `src.eval.compare.cluster_bootstrap` (2000 resamples, seed 0,
ratio of sums), re-implemented here, with clusters (race, driver): "95% interval, resampling drivers within
races". `compute()` returns a dict of named `Num`s; each carries its file and computation for `numbers.md`.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
EVAL_SET = PROJECT / "data" / "eval_set.json"
DOCS_EVAL = PROJECT / "docs" / "eval.md"
TIE = 1e-6            # src.eval.score.REGRET_TOL_S
TOL = 1e-9            # cross-check tolerance against the pipeline's own tables
TAIL_S = (10, 20, 30)  # src.eval.rerun.TAIL_S
# src.eval.rerun.ERAS (copied: the build reads files only); checked against the era column of the rows
ERAS = (("2019-20", 2019, 2020), ("2021", 2021, 2021), ("2022", 2022, 2022), ("2023", 2023, 2023),
        ("2024-25", 2024, 2025))
# docs/eval.md "The registered split": the four held-out races of the earlier (d4) protocol, scored once under it
SCORED_BEFORE = ("2019_06_monaco", "2020_16_sakhir", "2025_11_austria", "2024_11_austria")
POLICIES = ("agent", "belief_optimizer", "actual_strategy", "one_stop_total", "run_to_flag", "fixed_one_stop",
            "fixed_two_stop")
COMPARE = ("run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer", "fixed_one_stop")
BO_COMPARE = ("run_to_flag", "one_stop_total", "actual_strategy", "fixed_one_stop")
EXPECTED_MODEL = "qwen3-30b-a3b"   # the decision model; the build stops if a provenance file names another
CIRCUIT = {"united_kingdom": "Britain", "united_arab_emirates": "Abu Dhabi", "emilia_romagna": "Imola",
           "saudi_arabia": "Saudi Arabia", "austrian": "Austria", "sakhir": "Sakhir", "monaco": "Monaco",
           "france": "France", "mexico": "Mexico", "miami": "Miami", "australia": "Australia", "belgium": "Belgium"}
# tests/future_blindness.py FIXED_LEAKS, in plain words (an unknown name prints as it is)
LEAK_WORDS = {"stint_limit": "tyre stint limit from the whole race",
              "pit_loss": "pit loss measured over the whole race",
              "frame_filter": "clean-lap filter fitted on every lap",
              "lap_window": "fit window cut by lap, not by clock",
              "race_length": "race length from the car’s own laps",
              "compound_list": "tyre list of the whole race",
              "gap_behind": "gap behind read from the next lap"}
# fidelity ledger ids of the judge's whole-race inputs, in plain words
LEDGER_WORDS = {"fit_window": "lap-time fit", "clean_flag": "clean-lap filter", "ref_lap_s": "safety-car lap time",
                "pit_loss": "pit loss", "cliff_edge": "tyre life", "proxy_raced": "tyres raced",
                "driver_effects": "driver effects", "events": "safety cars as recorded"}


@dataclass
class Num:
    value: object
    text: str            # as printed on the poster
    source: str          # file(s)
    how: str             # computation
    pending: bool = False
    tag: str = ""        # what a pending value waits for


@dataclass
class SetData:
    tag: str                       # "d" development, "h" held-out
    label: str
    root: Path | None
    rows: pd.DataFrame | None = None
    prov: dict = field(default_factory=dict)
    scen: dict = field(default_factory=dict)
    has_agent: bool = False

    @property
    def present(self):
        return self.rows is not None

    def f(self, name):
        return f"{self.label}: tables/{name}"


def s(v, nd=2, sign=False):
    """A number as printed: true minus sign, fixed decimals."""
    t = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return t.replace("-", "−")


def ci(lo, hi, nd=2):
    if pd.isna(lo) or pd.isna(hi):
        return "[no interval: one cluster]"
    return f"[{s(lo, nd)}, {s(hi, nd)}]"


def pct(x):
    return f"{100 * x:.0f}%"


def race_name(rid: str) -> str:
    season, _, slug = str(rid).split("_", 2)
    return f"{CIRCUIT.get(slug, slug.replace('_', ' ').title())} {season}"


def era_of(rid: str) -> str:
    season = int(str(rid)[:4])
    return next((n for n, lo, hi in ERAS if lo <= season <= hi), str(season))


# --- statistics (same computations as src.eval.compare) --------------------------------------------
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
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def ok_rows(rows, pol):
    r = rows[(rows["policy"] == pol) & (rows["status"] == "ok")].copy()
    r["regret_s"] = pd.to_numeric(r["regret_s"], errors="coerce")
    return r.dropna(subset=["regret_s"])


def paired(rows, ref, other):
    """`ref` minus `other` on the scenarios both scored ok (compare.paired_table)."""
    a, o = ok_rows(rows, ref).set_index("scenario_id"), ok_rows(rows, other).set_index("scenario_id")
    both = a.index.intersection(o.index)
    diff = a.loc[both, "regret_s"] - o.loc[both, "regret_s"]
    ids = list(a.loc[both, ["race_id", "driver"]].itertuples(index=False, name=None))
    lo, hi = cluster_bootstrap(diff, ids)
    return {"n": len(both), "mean": float(diff.mean()) if len(both) else float("nan"), "lo": lo, "hi": hi,
            "wins": int((diff < -TIE).sum()), "ties": int((diff.abs() <= TIE).sum()),
            "losses": int((diff > TIE).sum()), "clusters": len(set(ids))}


def policy_mean(rows, pol):
    r = ok_rows(rows, pol)
    ids = list(r[["race_id", "driver"]].itertuples(index=False, name=None))
    lo, hi = cluster_bootstrap(r["regret_s"], ids)
    return {"n": len(r), "mean": float(r["regret_s"].mean()) if len(r) else float("nan"), "lo": lo, "hi": hi}


def tail(rows, pol):
    r = ok_rows(rows, pol)["regret_s"]
    out = {"scored": len(r), "mean_s": r.mean(), "median_s": r.median(), "p90_s": r.quantile(0.9),
           "worst_s": r.max()}
    out.update({f"share_over_{t}s": float((r > t).mean()) for t in TAIL_S})
    return out


def _close(a, b, what):
    if (pd.isna(a) and pd.isna(b)):
        return
    if pd.isna(a) or pd.isna(b) or abs(float(a) - float(b)) > TOL * max(1.0, abs(float(b))):
        raise SystemExit(f"cross-check failed: {what}: poster {a!r} != pipeline {b!r}")


# --- loading ------------------------------------------------------------------------------------------
def load_set(tag, label, root, allow_dev_as_test=False) -> SetData:
    sd = SetData(tag, label, Path(root).resolve() if root else None)
    if root is None:
        return sd
    root = Path(root)
    tables = root / "tables" if (root / "tables").is_dir() else root
    base = root if (root / "tables").is_dir() else root.parent
    comb = tables / "combined.csv"
    if not comb.exists():
        raise SystemExit(f"{comb} not found: run `python -m src.eval.rerun tables --set <set> --out-dir {base}` first")
    sd.root = tables
    rows = pd.read_csv(comb, low_memory=False)
    rows["regret_s"] = pd.to_numeric(rows["regret_s"], errors="coerce")
    sd.rows = rows
    sd.has_agent = bool((rows["policy"] == "agent").any())
    pf, sf = base / "provenance.json", base / "scenarios.json"
    sd.prov = json.loads(pf.read_text()) if pf.exists() else {}
    if sf.exists():
        raw = json.loads(sf.read_text())
        sd.scen = {"skipped": len(raw.get("skipped") or []), "development": raw.get("development"),
                   "digest": raw.get("digest")}
    kind = sd.prov.get("set_kind") or ("development set" if sd.scen.get("development") else "report set")
    if tag == "h" and "development" in str(kind) and not allow_dev_as_test:
        raise SystemExit(f"--test {root} is a {kind}: the held-out column takes a report set only")
    if tag == "d" and "development" not in str(kind):
        raise SystemExit(f"--dev {root} is a {kind}, not a development set")
    sd.prov["_kind"] = kind
    eras = rows["race_id"].map(era_of)
    if "era" in rows and not (rows["era"].astype(str) == eras).all():
        raise SystemExit(f"{comb}: era column differs from the era rule copied from src.eval.rerun.ERAS")
    rows["era"] = eras
    return sd


def find_ceiling(fid: Path, explicit=None):
    """(DataFrame of the noise floor levels, source name)."""
    if explicit:
        return pd.read_csv(explicit), str(explicit)
    hits = sorted(fid.rglob("ceiling_pooled.csv")) if fid.exists() else []
    hits = [h for h in hits if ".cache" not in h.parts]
    if hits:
        return pd.read_csv(hits[0]), str(hits[0])
    text = DOCS_EVAL.read_text()
    m = re.search(r"### The noise floor.*?\n(\| level .*?)\n\n", text, re.S)
    if not m:
        raise SystemExit("no ceiling_pooled.csv under --fidelity and no noise-floor table in docs/eval.md")
    out = []
    for line in m.group(1).splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        nums = lambda c: [float(x) for x in re.findall(r"[-\d.]+", c)]  # noqa: E731
        cp, rg = nums(cells[1]), nums(cells[2])
        out.append({"level": cells[0], "close_pair_acc": cp[0], "close_pair_lo": cp[1] if len(cp) > 2 else np.nan,
                    "close_pair_hi": cp[2] if len(cp) > 2 else np.nan, "rel_gap_mae_s": rg[0],
                    "rel_gap_lo": rg[1] if len(rg) > 2 else np.nan, "rel_gap_hi": rg[2] if len(rg) > 2 else np.nan,
                    "decimals": True})
    hdr = text[m.start():m.start(1)]
    n = re.search(r"(\d+) close pairs", hdr)
    df = pd.DataFrame(out)
    df["close_pairs"] = int(n.group(1)) if n else np.nan
    return df, "docs/eval.md, \"The noise floor\" (ceiling_pooled.csv is not committed)"


def parse_cell(cell):
    """'+0.016 [+0.000, +0.038]' -> (0.016, 0.0, 0.038)."""
    x = [float(v) for v in re.findall(r"[-+]?\d+\.\d+", str(cell))]
    return tuple(x) if len(x) == 3 else (x[0], np.nan, np.nan)


# --- compute ----------------------------------------------------------------------------------------------
def compute(dev=None, test=None, fidelity=PROJECT / "reports" / "calibration" / "fidelity", noise_floor=None,
            allow_dev_as_test=False) -> dict[str, Num]:
    N: dict[str, Num] = {}

    def put(key, value, text, source, how):
        N[key] = Num(value, text, source, how)

    def pend(key, source, how, waits):
        N[key] = Num(None, "pending", source, how, pending=True, tag=waits)

    D = load_set("d", "development", dev)
    H = load_set("h", "held-out", test, allow_dev_as_test)
    if not D.present:
        raise SystemExit("--dev is required")

    # ---- the sets themselves
    held = json.loads(EVAL_SET.read_text())
    held = list(held["races"] if isinstance(held, dict) else held)
    put("h_races_reg", len(held), str(len(held)), "data/eval_set.json", "race ids registered as held out")
    before = [r for r in held if r in SCORED_BEFORE]
    if len(before) != len(SCORED_BEFORE):
        raise SystemExit("data/eval_set.json does not hold the four earlier held-out races")
    put("h_before", len(before), "four" if len(before) == 4 else str(len(before)), "data/eval_set.json, docs/eval.md",
        "held-out races scored once under the earlier protocol (d4): " + ", ".join(before))
    fresh = [r for r in held if r not in SCORED_BEFORE]
    put("h_fresh", len(fresh), "five" if len(fresh) == 5 else str(len(fresh)), "data/eval_set.json, docs/eval.md",
        "held-out races never examined (the fresh 2021-22 draw): " + ", ".join(fresh))
    put("h_fresh_names", fresh, ", ".join(race_name(r) for r in fresh), "data/eval_set.json", "race ids, names")
    put("h_before_names", before, ", ".join(race_name(r) for r in before), "data/eval_set.json", "race ids, names")
    era_set = held
    if H.present:
        got = sorted(H.rows["race_id"].unique())
        if allow_dev_as_test:          # a layout fixture: its own races stand in for the held-out ones
            era_set = got
        elif got != sorted(held):
            raise SystemExit(f"--test rows cover {got}, not the races of data/eval_set.json")
    order = [x[0] for x in ERAS]
    eras_h = sorted({era_of(r) for r in era_set}, key=lambda e: order.index(e) if e in order else 99)
    put("h_eras", eras_h, str(len(eras_h)), "data/eval_set.json", "eras of the held-out races (src.eval.rerun.ERAS)")
    for e in eras_h:
        rs = [r for r in era_set if era_of(r) == e]
        put(f"h_era_races_{e}", rs, ", ".join(race_name(r) for r in rs), "data/eval_set.json", f"held-out races of era {e}")

    for S in (D, H):
        t = S.tag
        if not S.present:
            for k in ("races", "n", "per_race", "skipped"):
                pend(f"{t}_{k}", "--test", "set size", "held-out run")
            continue
        rows = S.rows
        sc = rows.drop_duplicates("scenario_id")
        put(f"{t}_races", sc["race_id"].nunique(), str(sc["race_id"].nunique()), S.f("combined.csv"), "distinct race_id")
        put(f"{t}_n", len(sc), str(len(sc)), S.f("combined.csv"), "distinct scenario_id")
        per = sc.groupby("race_id").size()
        put(f"{t}_per_race", int(per.max()), str(int(per.max())), S.f("combined.csv"),
            f"most scenarios in one race (counts {sorted(set(per))})")
        sk = S.scen.get("skipped", 0)
        put(f"{t}_skipped", sk, str(sk), f"{S.label}: scenarios.json", "sampled points the sampler skipped")
        put(f"{t}_drivers", len(set(map(tuple, sc[["race_id", "driver"]].values))),
            str(len(set(map(tuple, sc[["race_id", "driver"]].values)))), S.f("combined.csv"), "(race, driver) clusters")

    # ---- per-policy level, tail and paired results
    for S in (D, H):
        t = S.tag
        waits = "held-out run" if t == "h" else "development agent pass"
        src = S.f("combined.csv") if S.present else "--test"
        for pol in POLICIES:
            need_agent = pol == "agent"
            if not S.present or (need_agent and not S.has_agent):
                w = "held-out run" if not S.present else waits
                for k in ("mean", "ci", "scored", "unavail", "median", "worst", "p90") + tuple(f"over{x}" for x in TAIL_S):
                    pend(f"{t}_{k}_{pol}", src, "needs the rows", w)
                continue
            rows = S.rows
            m = policy_mean(rows, pol)
            chart = pd.read_csv(S.root / "chart_regret_s.csv")
            c = chart[chart["policy"] == pol].iloc[0]
            _close(m["mean"], c["mean"], f"{S.label} {pol} mean")
            _close(m["lo"], c["ci_low"], f"{S.label} {pol} ci_low")
            _close(m["hi"], c["ci_high"], f"{S.label} {pol} ci_high")
            if m["n"] != int(c["scenarios"]):
                raise SystemExit(f"cross-check failed: {S.label} {pol} scored n")
            how = (f"mean regret_s of {pol} over its own ok rows (n {m['n']}); 95% cluster bootstrap over (race, driver); "
                   f"equals tables/chart_regret_s.csv")
            put(f"{t}_mean_{pol}", m["mean"], s(m["mean"], 1), src, how)
            put(f"{t}_ci_{pol}", (m["lo"], m["hi"]), ci(m["lo"], m["hi"], 1), src, how)
            allp = rows[rows["policy"] == pol]
            put(f"{t}_scored_{pol}", m["n"], str(m["n"]), src, f"{pol} rows with status ok")
            un = int((allp["status"] != "ok").sum())
            put(f"{t}_unavail_{pol}", un, str(un), src, f"{pol} rows not ok (unavailable, invalid, error)")
            tl = tail(rows, pol)
            tt = pd.read_csv(S.root / "tail.csv")
            tr = tt[tt["policy"] == pol].iloc[0]
            for col in ("mean_s", "median_s", "p90_s", "worst_s") + tuple(f"share_over_{x}s" for x in TAIL_S):
                _close(tl[col], tr[col], f"{S.label} tail {pol} {col}")
            howt = "over its own ok rows; equals tables/tail.csv"
            put(f"{t}_median_{pol}", tl["median_s"], s(tl["median_s"], 1), src, f"median regret_s of {pol} {howt}")
            put(f"{t}_p90_{pol}", tl["p90_s"], s(tl["p90_s"], 1), src, f"90th percentile regret_s of {pol} {howt}")
            put(f"{t}_worst_{pol}", tl["worst_s"], s(tl["worst_s"], 0), src, f"largest regret_s of {pol} {howt}")
            for x in TAIL_S:
                put(f"{t}_over{x}_{pol}", tl[f"share_over_{x}s"], pct(tl[f"share_over_{x}s"]), src,
                    f"share of {pol}'s ok rows with regret_s > {x} s {howt}")
            summ = pd.read_csv(S.root / "summary.csv")
            sr = summ[(summ["race_id"] == "ALL") & (summ["policy"] == pol)].iloc[0]
            if int(sr["scored"]) != m["n"]:
                raise SystemExit(f"cross-check failed: {S.label} {pol} scored vs summary.csv")
            _close(m["mean"], sr["mean_regret_s"], f"{S.label} {pol} mean vs summary.csv")

        # agent minus X
        for x in COMPARE:
            keys = [f"{t}_diff_{x}", f"{t}_dci_{x}", f"{t}_pn_{x}", f"{t}_wtl_{x}"]
            if not S.present or not S.has_agent:
                for k in keys:
                    pend(k, src, "agent minus " + x, "held-out run" if not S.present else waits)
                continue
            p = paired(S.rows, "agent", x)
            tab = pd.read_csv(S.root / "paired_agent_minus_x.csv")
            c = tab[tab["policy"] == x].iloc[0]
            for a, b in (("mean", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
                _close(p[a], c[b], f"{S.label} agent minus {x} {a}")
            if (p["n"], p["wins"], p["ties"], p["losses"]) != (int(c["paired"]), int(c["reference_wins"]),
                                                                 int(c["ties"]), int(c["policy_wins"])):
                raise SystemExit(f"cross-check failed: {S.label} agent minus {x} counts")
            how = (f"agent minus {x}, mean over {p['n']} scenarios both scored ok; 95% cluster bootstrap over "
                   f"{p['clusters']} (race, driver) clusters, 2000 resamples, seed 0; equals tables/paired_agent_minus_x.csv")
            put(keys[0], p["mean"], s(p["mean"], 1, sign=True), src, how)
            put(keys[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
            put(keys[2], p["n"], str(p["n"]), src, how)
            put(keys[3], (p["wins"], p["ties"], p["losses"]), f"{p['wins']} / {p['ties']} / {p['losses']}", src,
                f"scenarios agent better / tied (|diff| <= 1e-6 s) / {x} better")

        # optimizer (no LLM) minus X: the planning layer on its own
        for x in BO_COMPARE:
            keys = [f"{t}_bo_diff_{x}", f"{t}_bo_dci_{x}", f"{t}_bo_pn_{x}"]
            if not S.present:
                for k in keys:
                    pend(k, src, "belief_optimizer minus " + x, "held-out run")
                continue
            p = paired(S.rows, "belief_optimizer", x)
            fpath = S.root / "paired_belief_optimizer_minus_x.csv"
            note = "no pipeline table (agent rows present): computed here only"
            if fpath.exists():
                c = pd.read_csv(fpath)
                c = c[c["policy"] == x].iloc[0]
                for a, b in (("mean", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
                    _close(p[a], c[b], f"{S.label} belief_optimizer minus {x} {a}")
                if p["n"] != int(c["paired"]):
                    raise SystemExit(f"cross-check failed: {S.label} belief_optimizer minus {x} n")
                note = "equals tables/paired_belief_optimizer_minus_x.csv"
            how = (f"belief_optimizer minus {x}, mean over {p['n']} scenarios both scored ok; (race, driver) cluster "
                   f"bootstrap; {note}")
            put(keys[0], p["mean"], s(p["mean"], 1, sign=True), src, how)
            put(keys[1], (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
            put(keys[2], p["n"], str(p["n"]), src, how)

        # the agent's reliability, cost and its call against the optimizer's
        rel = ("ag_ok", "ag_invalid", "ag_grace", "ag_vol_pct", "ag_lat_med", "ag_tok_mean", "ag_calls_mean",
               "same_call", "diff_call_bo_better", "diff_call_n", "same_plan")
        if not S.present or not S.has_agent:
            for k in rel:
                pend(f"{t}_{k}", src, "agent rows", "held-out run" if not S.present else waits)
        else:
            a = S.rows[S.rows["policy"] == "agent"].set_index("scenario_id")
            n = len(a)
            okn = int((a["status"] == "ok").sum())
            put(f"{t}_ag_ok", okn, f"{okn} of {n}", src, "agent rows with status ok (a legal, scored decision)")
            inv = int((a["status"] == "invalid").sum())
            put(f"{t}_ag_invalid", inv, str(inv), src, "agent rows with status invalid")
            grace = a["submitted_in_grace"].astype(str).str.lower().isin(("true", "1", "1.0"))
            put(f"{t}_ag_grace", int(grace.sum()), str(int(grace.sum())), src, "agent rows submitted in a grace turn")
            vol = int(((a["status"] == "ok") & ~grace).sum())
            put(f"{t}_ag_vol_pct", vol / n, pct(vol / n), src, "agent ok rows not submitted in a grace turn / agent rows")
            lat = pd.to_numeric(a["latency_s"], errors="coerce")
            put(f"{t}_ag_lat_med", float(lat.median()), f"{lat.median():.0f} s", src, "median latency_s (episode wall time)")
            tok = pd.to_numeric(a["input_tokens"], errors="coerce")
            put(f"{t}_ag_tok_mean", float(tok.mean()), f"{tok.mean() / 1000:.0f}k", src, "mean input_tokens per decision")
            calls = pd.to_numeric(a["tool_calls"], errors="coerce")
            put(f"{t}_ag_calls_mean", float(calls.mean()), f"{calls.mean():.0f}", src, "mean tool_calls per decision")
            bo = S.rows[S.rows["policy"] == "belief_optimizer"].set_index("scenario_id").reindex(a.index)
            bo_pit = pd.to_numeric(bo["first_pit_lap"], errors="coerce") == pd.to_numeric(bo["lap"], errors="coerce") + 1
            same = (a["decision"] == "PIT") == bo_pit
            put(f"{t}_same_call", int(same.sum()), f"{int(same.sum())} of {n}", src,
                "agent PIT/STAY_OUT equals the belief_optimizer's immediate action (first_pit_lap == lap + 1)")
            d = a["regret_s"] - bo["regret_s"]
            dd = d[~same].dropna()
            put(f"{t}_diff_call_n", int((~same).sum()), str(int((~same).sum())), src, "scenarios where the calls differ")
            put(f"{t}_diff_call_bo_better", int((dd > TIE).sum()), str(int((dd > TIE).sum())), src,
                "of those, belief_optimizer regret lower by more than 1e-6 s")
            sp = int((a["plan"] == bo["plan"]).sum())
            put(f"{t}_same_plan", sp, f"{sp} of {n}", src, "agent plan identical to the belief_optimizer plan")

    # ---- by era (held-out)
    for e in eras_h:
        keys = [f"h_era_n_{e}", f"h_era_agent_{e}", f"h_era_d_actual_{e}", f"h_era_dci_actual_{e}",
                f"h_era_d_rtf_{e}", f"h_era_dci_rtf_{e}", f"h_era_teams_{e}"]
        if not H.present:
            for k in keys:
                pend(k, "--test", f"era {e}", "held-out run")
            continue
        g = H.rows[H.rows["era"] == e]
        src = H.f("combined.csv")
        put(keys[0], g["scenario_id"].nunique(), str(g["scenario_id"].nunique()), src, f"scenarios of era {e}")
        tm = ok_rows(g, "actual_strategy")["regret_s"]
        put(keys[6], float(tm.mean()), s(tm.mean(), 1), src, f"mean actual_strategy regret_s, era {e}")
        if not H.has_agent:
            for k in keys[1:6]:
                pend(k, src, f"era {e}", "held-out agent run")
            continue
        am = ok_rows(g, "agent")["regret_s"]
        put(keys[1], float(am.mean()), s(am.mean(), 1), src, f"mean agent regret_s, era {e}")
        tab = pd.read_csv(H.root / f"paired_agent_minus_x_era_{e}.csv")
        for x, kd, kc in (("actual_strategy", keys[2], keys[3]), ("run_to_flag", keys[4], keys[5])):
            p = paired(g, "agent", x)
            c = tab[tab["policy"] == x].iloc[0]
            for a_, b_ in (("mean", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
                _close(p[a_], c[b_], f"era {e} agent minus {x} {a_}")
            how = f"agent minus {x}, era {e}, {p['n']} paired; equals tables/paired_agent_minus_x_era_{e}.csv"
            put(kd, p["mean"], s(p["mean"], 1, sign=True), src, how)
            put(kc, (p["lo"], p["hi"]), ci(p["lo"], p["hi"], 1), src, how)
        te = pd.read_csv(H.root / "tail_by_era.csv")
        tr = te[(te["era"] == e) & (te["policy"] == "agent")].iloc[0]
        _close(am.mean(), tr["mean_s"], f"era {e} agent mean vs tail_by_era.csv")

    # ---- provenance: commit, model, configs, future-blindness, as-of tables
    for S in (D, H):
        t = S.tag
        if not S.present:
            for k in ("commit", "model", "world_sc", "fb_known", "harness", "asof_none", "st_sc", "st_edge", "st_hinge",
                      "st_wtraffic", "st_bde", "st_learned"):
                pend(f"{t}_{k}", "--test provenance.json", k, "held-out run")
            continue
        P, src = S.prov, f"{S.label}: provenance.json"
        c = (P.get("commit") or "unknown")[:7] + (" (uncommitted changes)" if P.get("dirty") else "")
        put(f"{t}_commit", P.get("commit"), c, src, "commit the tables were made on (and whether the tree was dirty)")
        put(f"{t}_tag", P.get("tag"), P.get("tag") or "", src, "git tag at that commit, if any")
        model = P.get("model") or {}
        mid = {"voyager": model.get("VOYAGER_MODEL"), "anthropic": model.get("ANTHROPIC_MODEL"),
               "claude": model.get("ANTHROPIC_MODEL")}.get(str(model.get("LLM_BACKEND")).lower(), model.get("HF_MODEL"))
        if S.has_agent:
            if mid and mid != EXPECTED_MODEL:
                raise SystemExit(f"{src}: the agent ran {mid}, not {EXPECTED_MODEL}: check the run and EXPECTED_MODEL")
            put(f"{t}_model", mid, mid or EXPECTED_MODEL, src, "model id the agent stage recorded")
        else:
            pend(f"{t}_model", src, "model id the agent stage records", "agent run")
        wpath = str((P.get("world") or {}).get("path", "default"))
        sc_on = False
        if wpath.startswith("default"):
            how = "world config 'default': the race's own SimConfig, no SC/VSC (docs/eval.md \"Who sees what\")"
        else:
            cfg = _read_config(wpath, (P.get("world") or {}).get("file_sha256"))
            sc_on = bool(cfg) and str((cfg.get("events") or {}).get("name", "none")) != "none"
            how = f"events in the world config {Path(wpath).name}: {(cfg or {}).get('events')}"
        put(f"{t}_world_sc", sc_on, "with" if sc_on else "without", src, how)
        fb = P.get("future_blindness") or {}
        put(f"{t}_fb_fixed", fb.get("fixed_leaks", []), str(len(fb.get("fixed_leaks", []))), src,
            "tests/future_blindness.py FIXED_LEAKS at the commit (regression checks)")
        put(f"{t}_fb_known", fb.get("known_leaks", []), str(len(fb.get("known_leaks", []))), src,
            "tests/future_blindness.py KNOWN_LEAKS at the commit (leaks still open)")
        dp = fb.get("default_part")
        if dp:
            put(f"{t}_harness", dp, f"{dp.get('passed', 0)} passed, {dp.get('failed', 0)} failed", src,
                "the harness's default part, run by `tables --harness`")
        else:
            pend(f"{t}_harness", src, "`tables --harness` records the harness counts", "tables --harness")
        reads = ((P.get("as_of_tables") or {}).get("tyre_history") or {}).get("reads") or {}
        races = sorted(S.rows["race_id"].unique())
        none = [r for r in races if r in reads and len(reads[r]) == 0]
        put(f"{t}_asof_none", none, str(len(none)), src,
            "races of the set whose tyre-life history table (as_of_tables.tyre_history.reads) holds no earlier race: "
            + (", ".join(none) or "none"))
        few = [r for r in races if r in reads and len(reads[r]) <= 1]
        put(f"{t}_asof_few", few, str(len(few)), src, "races whose tyre-life history holds at most one earlier race: "
            + (", ".join(few) or "none"))
        ms = sorted(pd.to_numeric(S.rows["max_stops"], errors="coerce").dropna().unique())
        if len(ms) != 1:
            raise SystemExit(f"{S.label}: rows mix max_stops {ms}")
        put(f"{t}_max_stops", int(ms[0]), {1: "one", 2: "two", 3: "three"}.get(int(ms[0]), str(int(ms[0]))),
            S.f("combined.csv"), "max_stops column: future stops the hindsight reference searches")
        # which scorecard factors the configs of this run switch on
        wcfg = {} if wpath.startswith("default") else (_read_config(wpath, (P.get("world") or {}).get("file_sha256")) or {})
        bp = (P.get("belief") or {})
        bcfg = _read_config(str(bp.get("path", "")), bp.get("file_sha256"))
        wsrc = (f"{src}: world config {'default (docs/eval.md: no SC/VSC, traffic off, linear_fuel with driver effects, '
                'cliff edge history)' if wpath.startswith('default') else Path(wpath).name}")
        bsrc = f"{src}: belief config {Path(str(bp.get('path', '?'))).name} (sha256 {bp.get('file_sha256')})"
        deg = (wcfg.get("degradation") or {})
        flags = {"st_sc": (sc_on, wsrc, "SC/VSC in the judge (world events not none)"),
                 "st_edge": (str((wcfg.get("wear_cliff") or {}).get("world_edge", "history")) == "history_only", wsrc,
                             "judge's tyre-life edge from history only (wear_cliff.world_edge history_only)"),
                 "st_hinge": (str(deg.get("name", "linear_fuel")) == "linear_fuel_hinge", wsrc,
                              "judge's wear fit with the hinge (degradation linear_fuel_hinge)"),
                 "st_wtraffic": (str((wcfg.get("traffic") or {}).get("name", "none")) != "none", wsrc,
                                 "traffic in the judge (world traffic not none)")}
        if bcfg is not None:
            bde = ((bcfg.get("degradation") or {}).get("params") or {}).get("driver_effects")
            flags["st_bde"] = (bde not in (None, False, "off"), bsrc, "driver effects in the predictive simulator")
            cal = str((bcfg.get("calibrator") or {}).get("name", ""))
            flags["st_learned"] = ("learned" in cal or "trained" in cal, bsrc, f"learned calibrator (calibrator {cal})")
        for k, (on, fsrc_, how) in flags.items():
            put(f"{t}_{k}", bool(on), "yes" if on else "no", fsrc_, how)
        for k in ("st_bde", "st_learned"):
            if f"{t}_{k}" not in N:
                pend(f"{t}_{k}", bsrc, "belief config file not found", "belief config file")
    if not H.present:
        pend("h_max_stops", "--test", "max_stops", "held-out run")
        pend("h_fb_fixed", "--test", "fixed leaks", "held-out run")
        pend("h_asof_few", "--test", "as-of reads", "held-out run")
        pend("h_tag", "--test", "tag", "held-out run")

    # ---- fidelity scorecard
    fid = Path(fidelity)
    run = json.loads((fid / "main" / "run.json").read_text())
    card = (fid / "main" / "scorecard.md").read_text()
    m = re.search(r"Computed on main ([0-9a-f]{7,40})", card)
    fcommit = m.group(1) if m else "unknown"
    fsrc = "reports/calibration/fidelity/main/run.json"
    put("fid_commit", fcommit, fcommit, "reports/calibration/fidelity/main/scorecard.md", "commit the scorecard was computed on")
    put("fid_races", len(run["races"]), str(len(run["races"])), fsrc, "races the scorecard ran: " + ", ".join(run["races"]))
    hl = run["headline"]
    for key, col, nd, text in (("fid_close", "A_close_pair_acc", 3, None), ("fid_relgap", "A_rel_gap_mae_s", 1, None),
                               ("fid_spearman", "A_spearman", 2, None), ("fid_b2rho", "B2_rank_rho", 2, None),
                               ("fid_b2regret", "B2_regret_s", 1, None), ("fid_pclose", "P_close_pair_acc", 2, None)):
        mm = re.search(rf"\| {col} \| ([-\d.]+) \|", card)
        if mm:
            _close(round(hl[col], 3), float(mm.group(1)), f"{col}: run.json vs scorecard.md")
        put(key, hl[col], s(hl[col], nd), fsrc, f"headline {col} (scorecard.md: definition)")
    fbs = run.get("future_blindness") or {}
    put("fid_fb_status", fbs.get("status"), str(fbs.get("status")), fsrc,
        f"the scorecard's own future-blindness check (B5): {fbs.get('new_leak_differences')} new differences, "
        f"known leaks {fbs.get('known_leaks')}")
    for S in (D, H):
        if S.present and sorted(S.prov.get("future_blindness", {}).get("fixed_leaks", [])) != sorted(fbs.get("fixed_leaks", [])):
            print(f"note: {S.label} provenance fixed leaks differ from the scorecard's: "
                  f"{S.prov.get('future_blindness', {}).get('fixed_leaks')} vs {fbs.get('fixed_leaks')}")
    # the judge's whole-race inputs (its hindsight ledger)
    led = pd.read_csv(fid / "main" / "ledger.csv")
    whole = led[(led["role"] == "judge") & (led["tag"] == "this race, whole")]["id"].drop_duplicates().tolist()
    put("fid_ledger_n", len(whole), str(len(whole)), "reports/calibration/fidelity/main/ledger.csv",
        "judge inputs tagged 'this race, whole': " + ", ".join(whole))
    put("fid_ledger_words", whole, ", ".join(LEDGER_WORDS.get(w, w.replace("_", " ")) for w in whole),
        "reports/calibration/fidelity/main/ledger.csv", "the same, in words")

    # noise floor (ceiling)
    ceil, csrc = find_ceiling(fid, noise_floor)
    lv0 = ceil[ceil["level"].astype(str).str.startswith("0")].iloc[0]
    lv4 = ceil[ceil["level"].astype(str).str.startswith("4")].iloc[0]
    rnd = "decimals" in ceil
    _close(round(hl["A_close_pair_acc"], 3) if rnd else hl["A_close_pair_acc"], lv0["close_pair_acc"],
           "noise floor level 0 close pairs vs the scorecard headline")
    if abs(hl["A_rel_gap_mae_s"] - lv0["rel_gap_mae_s"]) > (0.05 if rnd else 0.01):
        raise SystemExit("cross-check failed: noise floor level 0 relative gap vs the scorecard headline")
    howc = ("level 4 of the noise floor: the judge's replay handed every stint's pace, the real stops, the SC/VSC laps "
            "and lap 1 as recorded (what is left: non-clean green laps and lap noise)")
    put("ceil_close", lv4["close_pair_acc"], s(lv4["close_pair_acc"], 3), csrc, howc + "; close-pair order")
    put("ceil_relgap", lv4["rel_gap_mae_s"], s(lv4["rel_gap_mae_s"], 1), csrc, howc + "; relative gap error, s")
    put("fid_close_ci", (lv0["close_pair_lo"], lv0["close_pair_hi"]), ci(lv0["close_pair_lo"], lv0["close_pair_hi"], 3),
        csrc, "level 0 (the judge) race-cluster 95% interval")
    put("ceil_close_ci", (lv4["close_pair_lo"], lv4["close_pair_hi"]), ci(lv4["close_pair_lo"], lv4["close_pair_hi"], 3),
        csrc, "level 4 race-cluster 95% interval")
    put("fid_relgap_ci", (lv0["rel_gap_lo"], lv0["rel_gap_hi"]), ci(lv0["rel_gap_lo"], lv0["rel_gap_hi"], 1), csrc,
        "level 0 relative gap race-cluster 95% interval")
    put("ceil_relgap_ci", (lv4["rel_gap_lo"], lv4["rel_gap_hi"]), ci(lv4["rel_gap_lo"], lv4["rel_gap_hi"], 1), csrc,
        "level 4 relative gap race-cluster 95% interval")
    lv3 = ceil[ceil["level"].astype(str).str.startswith("3")].iloc[0]
    put("ceil_relgap_sc", lv3["rel_gap_mae_s"], s(lv3["rel_gap_mae_s"], 1), csrc,
        "level 3 (up to SC/VSC laps as recorded): relative gap error, s")
    if "close_pairs" in ceil and pd.notna(lv0.get("close_pairs")):
        put("ceil_pairs", int(lv0["close_pairs"]), str(int(lv0["close_pairs"])), csrc, "close pairs pooled")

    # ablation: change when a factor is switched on alone, pooled, race-cluster interval
    heads = pd.read_csv(fid / "ablation" / "headlines.csv").set_index("factor")
    abl = pd.read_csv(fid / "ablation" / "ablation.csv").set_index("factor")
    for col in ("A_close_pair_acc", "A_rel_gap_mae_s", "B2_rank_rho", "B2_regret_s"):
        _close(round(heads.loc["base", col], 3), float(abl.loc["base", col]), f"ablation base {col}")
        if abs(heads.loc["base", col] - hl[col]) > 1e-9:
            print(f"note: the ablation base {col} differs from the scorecard headline (different runs)")
    asrc = "reports/calibration/fidelity/ablation/ablation.csv"
    for factor, col, nd in (("world_events_hybrid", "A_close_pair_acc", 3), ("world_events_hybrid", "A_rel_gap_mae_s", 1),
                            ("belief_driver_effects", "B2_rank_rho", 3), ("belief_driver_effects", "B2_regret_s", 1),
                            ("cliff_edge_history_only", "A_close_pair_acc", 3), ("wear_cliff_off", "B2_regret_s", 1),
                            ("world_traffic", "A_close_pair_acc", 3), ("world_traffic", "B2_regret_s", 1),
                            ("learned_trained", "B2_regret_s", 1), ("world_no_hinge", "A_close_pair_acc", 3)):
        d, lo, hi = parse_cell(abl.loc[factor, col])
        _close(round(heads.loc[factor, col] - heads.loc["base", col], 3), d, f"ablation {factor} {col}")
        how = (f"{factor} minus base, {col}, pooled over the races every configuration ran; race-cluster bootstrap 95% "
               f"interval (2000 draws); equals ablation/headlines.csv difference")
        put(f"abl_{factor}_{col}", (d, lo, hi), f"{s(d, nd, sign=True)} {ci(lo, hi, nd)}", asrc, how)
        put(f"abl_{factor}_{col}_pt", d, s(d, nd, sign=True), asrc, how)
    m = re.search(r"(\d+) calibration races", (fid / "ablation" / "ablation.md").read_text())
    put("abl_races", int(m.group(1)) if m else None, m.group(1) if m else "?",
        "reports/calibration/fidelity/ablation/ablation.md", "races the ablation ran")
    m = re.search(r"Computed on main ([0-9a-f]{7,40})", (fid / "ablation" / "ablation.md").read_text())
    put("abl_commit", m.group(1) if m else None, m.group(1) if m else "unknown",
        "reports/calibration/fidelity/ablation/ablation.md", "commit the ablation was computed on")
    N["_sets"] = Num((D, H), "", "", "")
    return N


def _read_config(path, sha12=None):
    """A SimConfig yaml by its recorded path, else the repo file of that name whose sha256 matches."""
    import hashlib
    for p in (Path(path), PROJECT / "configs" / "sim" / Path(path).name):
        if p.is_file() and (not sha12 or hashlib.sha256(p.read_bytes()).hexdigest()[:12] == sha12):
            return yaml.safe_load(p.read_text()) or {}
    return None


def verdict(lo, hi, better="better than", worse="worse than", level="no clear difference from"):
    if hi < 0:
        return better
    if lo > 0:
        return worse
    return level


def provenance_md(N: dict[str, Num], used: list[str], args_line: str) -> str:
    lines = ["# Numbers on the poster and where they come from", "",
             "Generated by `poster/build_poster.py` (" + args_line + "). Every printed number, its file and its "
             "computation; cross-checks against the pipeline's own tables passed, or the build would have stopped. "
             "Rows marked **pending** print as a placeholder until the run named in the last column exists.", "",
             "| key | printed | file | computation |", "|---|---|---|---|"]
    for k in used:
        n = N[k]
        text = f"**pending** ({n.tag})" if n.pending else n.text.replace("|", "/")
        lines.append(f"| `{k}` | {text} | `{n.source}` | {n.how.replace('|', '/')} |")
    pend = sorted({N[k].tag for k in used if N[k].pending})
    lines += ["", "Pending inputs: " + (", ".join(pend) if pend else "none") + "."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test")
    ap.add_argument("--fidelity", default=str(PROJECT / "reports" / "calibration" / "fidelity"))
    ap.add_argument("--noise-floor")
    a = ap.parse_args()
    N = compute(a.dev, a.test, a.fidelity, a.noise_floor)
    for k, n in N.items():
        if not k.startswith("_"):
            print(f"{k:42s} {('PENDING ' + n.tag) if n.pending else n.text:30s} {n.how[:90]}")
