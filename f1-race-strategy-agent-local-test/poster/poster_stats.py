"""Every number printed on the 24 x 36 in PITWALL poster, computed from the committed result files.

Reads only `reports/eval/d4_*.csv` (held-out, report set, tag d4-final-eval, one agent pass) and
`reports/eval/dev6_run2_*.csv` (development, calibration races). Nothing is loaded, simulated or
scored: no race data, no simulator, no model.

    python poster/poster_stats.py          # prints the provenance table (also written by build_poster.py)

`compute()` returns a dict of named values; each value carries the file it comes from and how it
was computed, so the build can write the provenance table (`poster/numbers.md`).

Intervals are the percentile cluster bootstrap of `src.eval.compare.cluster_bootstrap` (2000
resamples, seed 0, ratio of sums), re-implemented here with no import from `src`, with clusters
(race, driver): "95% interval, resampling drivers within races". The pooled paired results are
checked against the committed `*_compare_regret_s_paired.csv` and the per-race ones against
`d4_compare_per_race_regret_s_paired.csv`; a mismatch stops the build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EVAL = PROJECT / "reports" / "eval"
TIE = 1e-6   # src.eval.score.REGRET_TOL_S
HELD = "d4"
DEV = "dev6_run2"
HELD_RACES = ["2019_06_monaco", "2020_16_sakhir", "2024_11_austria", "2025_11_austria"]
RACE_NAMES = {"2019_06_monaco": "Monaco 2019", "2020_16_sakhir": "Sakhir 2020",
              "2024_11_austria": "Austria 2024", "2025_11_austria": "Austria 2025"}
COMPARATORS = ["fixed_one_stop", "actual_strategy", "belief_optimizer"]
# Fairer rule baselines being scored after d4-final-eval. Their rows are picked up automatically
# once they are in the combined CSVs under these policy names (rename here to match the files);
# until then the poster prints "pending" in their slot.
FAIR_RULES = [("run_to_flag", "Run to the flag"), ("one_stop_total", "One stop in total")]


@dataclass
class Num:
    value: float
    text: str          # as printed on the poster
    source: str        # file(s) under reports/eval/
    how: str           # computation


def load(prefix: str) -> pd.DataFrame:
    d = pd.read_csv(EVAL / f"{prefix}_combined.csv")
    d["lap"] = d["lap"].astype(int)
    return d


def cluster_bootstrap(values, clusters, n_boot=2000, level=0.95, seed=0):
    """Same computation as src.eval.compare.cluster_bootstrap."""
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


def paired(rows: pd.DataFrame, other: str, mask=None) -> dict:
    """agent minus `other` on scenarios both scored ok (src.eval.compare.paired_table)."""
    ok = rows[rows["status"] == "ok"]
    a = ok[ok["policy"] == "agent"].set_index("scenario_id")
    o = ok[ok["policy"] == other].set_index("scenario_id")
    both = a.index.intersection(o.index, sort=False)
    if mask is not None:
        both = both[mask(a.loc[both])]
    diff = a.loc[both, "regret_s"].astype(float) - o.loc[both, "regret_s"].astype(float)
    ids = list(a.loc[both, ["race_id", "driver"]].itertuples(index=False, name=None))
    lo, hi = cluster_bootstrap(diff, ids)
    return {"n": len(both), "mean": float(diff.mean()), "lo": lo, "hi": hi,
            "wins": int((diff < -TIE).sum()), "ties": int((diff.abs() <= TIE).sum()),
            "losses": int((diff > TIE).sum()), "clusters": len(set(ids))}


def common_pair(rows: pd.DataFrame, other: str):
    """agent and `other` regret on the scenarios both scored ok, with (race, driver) cluster ids."""
    ok = rows[rows["status"] == "ok"]
    a = ok[ok["policy"] == "agent"].set_index("scenario_id")
    o = ok[ok["policy"] == other].set_index("scenario_id")
    both = a.index.intersection(o.index, sort=False)
    cl = list(a.loc[both, ["race_id", "driver"]].itertuples(index=False, name=None))
    return a.loc[both, "regret_s"].astype(float), o.loc[both, "regret_s"].astype(float), cl


def p90_diff(ra, rt, cl, n_boot=2000, seed=0):
    """90th percentile of ra minus that of rt, with a (race, driver) cluster-bootstrap interval."""
    codes, _ = pd.factorize(pd.Series(cl), sort=True)
    k = int(codes.max()) + 1
    groups = [np.where(codes == i)[0] for i in range(k)]
    va, vt = np.asarray(ra), np.asarray(rt)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ii = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        d.append(np.quantile(va[ii], .9) - np.quantile(vt[ii], .9))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return float(np.quantile(va, .9) - np.quantile(vt, .9)), (float(lo), float(hi))


def s(v, nd=2, sign=False):
    """Number as printed: true minus sign, fixed decimals."""
    t = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return t.replace("-", "−")


def ci(lo, hi, nd=2):
    return f"[{s(lo, nd)}, {s(hi, nd)}]"


def _check(mine: dict, committed: pd.Series, label: str):
    for k_m, k_c in (("mean", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
        if abs(mine[k_m] - float(committed[k_c])) > 1e-9:
            raise SystemExit(f"{label}: {k_m} {mine[k_m]} != committed {committed[k_c]}")
    if mine["n"] != int(committed["paired"]) or mine["ties"] != int(committed["ties"]):
        raise SystemExit(f"{label}: n/ties differ from the committed file")


def compute() -> dict[str, Num]:
    N: dict[str, Num] = {}
    held, dev = load(HELD), load(DEV)
    HC, DC = f"{HELD}_combined.csv", f"{DEV}_combined.csv"

    def put(key, value, text, source, how):
        N[key] = Num(value, text, source, how)

    for tag, rows, src in (("h", held, HC), ("d", dev, DC)):
        a = rows[rows["policy"] == "agent"]
        put(f"{tag}_n", len(a), str(len(a)), src, "agent rows (one per scenario)")
        put(f"{tag}_races", a["race_id"].nunique(), str(a["race_id"].nunique()), src, "distinct race_id")
        per = a.groupby("race_id").size()
        put(f"{tag}_per_race", int(per.iloc[0]), str(int(per.iloc[0])), src,
            f"scenarios per race (all equal: {sorted(set(per))})")
        assert per.nunique() == 1
        sc = int((a["status"] == "ok").sum())
        put(f"{tag}_scored", sc, f"{sc} / {len(a)}", src, "agent rows with status ok")
        grace = a["submitted_in_grace"].fillna(0).astype(bool)
        vol = int(((a["status"] == "ok") & ~grace).sum())
        put(f"{tag}_voluntary", vol, str(vol), src, "agent ok rows with submitted_in_grace false")
        put(f"{tag}_grace", int(grace.sum()), str(int(grace.sum())), src, "agent rows with submitted_in_grace true")
        put(f"{tag}_voluntary_pct", 100 * vol / len(a), f"{100 * vol / len(a):.0f}%", src,
            "voluntary / scenarios")
        put(f"{tag}_latency_median", float(a["latency_s"].median()), f"{a['latency_s'].median():.0f} s", src,
            "median latency_s, agent rows (episode wall time per decision)")
        put(f"{tag}_tokens_mean", float(a["input_tokens"].mean()), f"{a['input_tokens'].mean() / 1000:.0f}k", src,
            "mean input_tokens per decision, summed over the episode's model turns")
        for pol in ["agent"] + COMPARATORS:
            r = rows[(rows["policy"] == pol) & (rows["status"] == "ok")]
            put(f"{tag}_mean_{pol}", float(r["regret_s"].mean()), s(r["regret_s"].mean()), src,
                f"mean regret_s of {pol}, its own ok rows (n {len(r)})")
        present = set(rows["policy"])
        for pol in COMPARATORS + [p for p, _ in FAIR_RULES if p in present]:
            p = paired(rows, pol)
            committed = pd.read_csv(EVAL / f"{HELD if tag == 'h' else DEV}_compare_regret_s_paired.csv")
            if (committed["policy"] == pol).any():
                _check(p, committed[committed["policy"] == pol].iloc[0], f"{tag} {pol}")
            cfile = f"{HELD if tag == 'h' else DEV}_compare_regret_s_paired.csv"
            how = (f"agent minus {pol}, mean over {p['n']} paired ok scenarios; 95% cluster bootstrap over "
                   f"{p['clusters']} (race, driver) clusters, 2000 resamples, seed 0; equals {cfile}")
            put(f"{tag}_diff_{pol}", p["mean"], s(p["mean"], sign=True), src, how)
            put(f"{tag}_ci_{pol}", (p["lo"], p["hi"]), ci(p["lo"], p["hi"]), src, how)
            put(f"{tag}_n_{pol}", p["n"], str(p["n"]), src, f"paired ok scenarios, agent and {pol}")
            put(f"{tag}_ties_{pol}", p["ties"], str(p["ties"]), src, f"|agent - {pol}| <= 1e-6 s")
            put(f"{tag}_wins_{pol}", (p["wins"], p["losses"]), f"{p['wins']} / {p['losses']}", src,
                f"scenarios agent better / {pol} better")

    # per-race paired differences and agent means (held-out)
    per_committed = pd.read_csv(EVAL / f"{HELD}_compare_per_race_regret_s_paired.csv")
    for race in HELD_RACES:
        sub = held[held["race_id"] == race]
        a = sub[(sub["policy"] == "agent") & (sub["status"] == "ok")]
        put(f"h_race_mean_agent_{race}", float(a["regret_s"].mean()), f"{a['regret_s'].mean():.1f}", HC,
            f"mean agent regret_s at {race} ({len(a)} scenarios)")
        for pol in COMPARATORS:
            p = paired(sub, pol)
            c = per_committed[(per_committed["race"] == race) & (per_committed["cluster"] == "driver")
                              & (per_committed["policy"] == pol)].iloc[0]
            _check(p, c, f"{race} {pol}")
            put(f"h_race_diff_{pol}_{race}", p["mean"], s(p["mean"], 1, sign=True), HC,
                f"agent minus {pol} at {race}, {p['n']} paired scenarios; equals {HELD}_compare_per_race_regret_s_paired.csv")
        for pol in ("belief_optimizer", "actual_strategy", "fixed_one_stop"):
            r = sub[(sub["policy"] == pol) & (sub["status"] == "ok")]
            put(f"h_race_mean_{pol}_{race}", float(r["regret_s"].mean()), f"{r['regret_s'].mean():.1f}", HC,
                f"mean {pol} regret_s at {race} (n {len(r)})")
    a = held[(held["policy"] == "agent")]
    put("h_mean_agent_1", float(a["regret_s"].mean()), f"{a['regret_s'].mean():.1f}", HC, "mean agent regret_s, all 100")

    # fixed one-stop before / after the driver's first stop (tyre_age < lap <=> a stop has been made)
    for tag, rows, src in (("h", held, HC), ("d", dev, DC)):
        for part, m in (("before", lambda df: (df["tyre_age"] >= df["lap"]).values),
                        ("after", lambda df: (df["tyre_age"] < df["lap"]).values)):
            p = paired(rows, "fixed_one_stop", mask=m)
            how = (f"agent minus fixed_one_stop on paired ok scenarios where the driver has "
                   f"{'not yet stopped (tyre_age >= lap)' if part == 'before' else 'already stopped (tyre_age < lap)'}; "
                   f"n {p['n']}, driver-cluster bootstrap as above")
            put(f"{tag}_f1s_{part}", p["mean"], s(p["mean"], sign=True), src, how)
            put(f"{tag}_f1s_{part}_ci", (p["lo"], p["hi"]), ci(p["lo"], p["hi"]), src, how)
            put(f"{tag}_f1s_{part}_n", p["n"], str(p["n"]), src, how)

    # agent's immediate call vs the belief optimizer's (held-out)
    ag = held[held["policy"] == "agent"].set_index("scenario_id")
    bo = held[held["policy"] == "belief_optimizer"].set_index("scenario_id").loc[ag.index]
    bo_pit = (bo["first_pit_lap"] == bo["lap"] + 1)
    ag_pit = ag["decision"] == "PIT"
    same = ag_pit == bo_pit
    put("h_same_call", int(same.sum()), str(int(same.sum())), HC,
        "agent decision (PIT/STAY_OUT) equals belief_optimizer's immediate action (its plan's first_pit_lap == lap + 1)")
    d = ag["regret_s"] - bo["regret_s"]
    dd = d[~same]
    put("h_diff_call_n", int((~same).sum()), str(int((~same).sum())), HC, "scenarios where the immediate calls differ")
    put("h_diff_call_bo_better", int((dd > TIE).sum()), str(int((dd > TIE).sum())), HC,
        "of those, belief_optimizer regret lower by more than 1e-6 s")
    put("h_diff_call_mean", float(dd.mean()), s(dd.mean()), HC, "mean (agent - belief_optimizer) regret_s where calls differ")

    # pit-now decision metrics (held-out)
    truth = ag["pit_now_opt"].astype(str).map({"True": 1.0, "False": 0.0})
    put("h_pitnow_base", float(truth.mean()), f"{100 * truth.mean():.0f}%", HC,
        "share of scenarios with pit_now_opt (reference's first stop at the end of this lap)")
    agree = float((ag_pit.astype(float) == truth).mean())
    put("h_pitnow_agree_agent", agree, f"{100 * agree:.0f}%", HC, "agent PIT/STAY_OUT equals pit_now_opt")
    put("h_pitnow_agree_stay", float(1 - truth.mean()), f"{100 * (1 - truth.mean()):.0f}%", HC,
        "always STAY_OUT equals pit_now_opt")
    brier = float(((ag["p_pit"].astype(float) - truth) ** 2).mean())
    put("h_brier_agent", brier, f"{brier:.3f}", HC, "mean (p_pit - pit_now_opt)^2, agent rows")
    put("h_brier_stay", float(truth.mean()), f"{truth.mean():.3f}", HC, "Brier of constant p_pit = 0 (always stay out) = base rate")
    put("h_agent_pits", int(ag_pit.sum()), str(int(ag_pit.sum())), HC, "agent PIT decisions")
    put("h_agent_pits_right", int((ag_pit & (truth == 1)).sum()), str(int((ag_pit & (truth == 1)).sum())), HC,
        "agent PIT decisions where pit_now_opt")

    # Austria 2024 note
    aut = held[held["race_id"] == "2024_11_austria"]
    ref = aut[aut["policy"] == "agent"]
    ref_soft = int(ref["opt_plan"].str.contains("SOFT").sum())
    put("aut24_ref_soft", ref_soft, f"{ref_soft} of {len(ref)}", HC, "2024 Austria scenarios whose hindsight reference plan (opt_plan) has a SOFT stint")
    act = aut[(aut["policy"] == "actual_strategy") & (aut["status"] == "ok")]
    act_soft = act[act["plan"].str.contains("SOFT")]
    put("aut24_actual_soft_drivers", act_soft["driver"].nunique(),
        f"{act_soft['driver'].nunique()} of {aut['driver'].nunique()}", HC,
        f"sampled 2024 Austria drivers whose actual remaining strategy (actual_strategy plan) has a SOFT stint "
        f"({len(act_soft)} of {len(act)} scored rows; driver {', '.join(sorted(act_soft['driver'].unique()))})")
    cur_soft = int((ref["compound"] == "SOFT").sum())
    put("aut24_current_soft", cur_soft, str(cur_soft), HC, "2024 Austria scenarios with the driver currently on SOFT")

    # the lead comparison: agent against the teams' actual strategy, whole distribution on the
    # scenarios both scored (held-out 98, development 147)
    for tag, rows, src in (("h", held, HC), ("d", dev, DC)):
        ra, rt, cl = common_pair(rows, "actual_strategy")
        n = len(ra)
        put(f"{tag}_tail_n", n, str(n), src, "scenarios scored ok for both agent and actual_strategy")
        for who, r in (("agent", ra), ("actual", rt)):
            put(f"{tag}_tail_mean_{who}", float(r.mean()), f"{r.mean():.2f}", src, f"mean regret_s, {who}, common scenarios")
            put(f"{tag}_tail_median_{who}", float(np.median(r)), f"{np.median(r):.1f}", src, f"median regret_s, {who}, common scenarios")
            put(f"{tag}_tail_p90_{who}", float(np.quantile(r, .9)), f"{np.quantile(r, .9):.1f}", src,
                f"90th percentile of regret_s (numpy linear), {who}, common scenarios")
            put(f"{tag}_tail_max_{who}", float(r.max()), f"{r.max():.1f}", src, f"largest regret_s, {who}, common scenarios")
            for thr in (1, 20, 30):
                ind = (r <= thr) if thr == 1 else (r > thr)
                word = "within 1 s of" if thr == 1 else f"more than {thr} s from"
                put(f"{tag}_tail_{thr}_{who}", float(ind.mean()), f"{100 * ind.mean():.0f}%", src,
                    f"share of common scenarios {word} the hindsight reference, {who}")
        for thr in (1, 20):
            ia = ((ra <= 1) if thr == 1 else (ra > thr)).astype(float)
            it = ((rt <= 1) if thr == 1 else (rt > thr)).astype(float)
            lo, hi = cluster_bootstrap(ia - it, cl)
            dd = float(ia.mean() - it.mean())
            how = (f"difference in share (agent minus actual), percentage points; 95% interval: cluster bootstrap of the "
                   f"paired indicator difference over (race, driver), 2000 resamples, seed 0")
            put(f"{tag}_tail_{thr}_diff", dd, f"{100 * dd:+.0f} pts".replace("-", "−"), src, how)
            put(f"{tag}_tail_{thr}_ci", (lo, hi), f"[{100 * lo:.0f}, {100 * hi:.0f}]".replace("-", "−"), src, how)
        p90d, (lo, hi) = p90_diff(ra, rt, cl)
        how = ("90th percentile of agent regret minus that of actual_strategy, common scenarios; 95% interval: "
               "resample (race, driver) clusters with replacement, 2000 times, seed 0, recompute both percentiles")
        put(f"{tag}_tail_p90_diff", p90d, s(p90d, 1, sign=True), src, how)
        put(f"{tag}_tail_p90_ci", (lo, hi), ci(lo, hi, 1), src, how)
        put(f"{tag}_tail_curve", None, "(figure)", src, "share of common scenarios with regret_s above x, x = 0..50 s")
        N[f"{tag}_tail_curve"].value = (ra.values, rt.values)

    # held-out without Austria 2024 (robustness)
    x = held[held["race_id"] != "2024_11_austria"]
    xa = x[(x["policy"] == "agent") & (x["status"] == "ok")]
    put("h_noaut_mean_agent", float(xa["regret_s"].mean()), s(xa["regret_s"].mean()), HC,
        f"mean agent regret_s without 2024_11_austria ({len(xa)} scenarios)")
    for pol in COMPARATORS:
        p = paired(x, pol)
        how = f"agent minus {pol} without 2024_11_austria, {p['n']} paired scenarios, driver-cluster bootstrap"
        put(f"h_noaut_diff_{pol}", p["mean"], s(p["mean"], sign=True), HC, how)
        put(f"h_noaut_ci_{pol}", (p["lo"], p["hi"]), ci(p["lo"], p["hi"]), HC, how)
    ag_all = held[held["policy"] == "agent"]
    share = ag_all.loc[ag_all["race_id"] == "2024_11_austria", "regret_s"].sum() / ag_all["regret_s"].sum()
    put("aut24_share_agent_regret", float(share), f"{100 * share:.0f}%", HC,
        "2024 Austria's share of the summed agent regret_s over the 100 held-out scenarios")
    f2 = aut[(aut["policy"] == "fixed_two_stop") & (aut["status"] == "ok")]["regret_s"]
    put("aut24_mean_fixed_two_stop", float(f2.mean()), f"{f2.mean():.1f}", HC, f"mean fixed_two_stop regret_s at 2024 Austria (n {len(f2)})")
    soft = act_soft.iloc[0]
    last = json.loads(soft["plan"])[-1]
    put("aut24_soft_stint", last[1], str(last[1]), HC,
        f"the only SOFT stint in the sampled drivers' actual strategies: {soft['driver']}'s final stint, {last[1]} laps "
        f"(plan {soft['plan']} from lap {soft['lap']})")
    same_plan = int((ag["plan"] == bo["plan"]).sum())
    put("h_same_plan", same_plan, str(same_plan), HC, "agent plan string identical to belief_optimizer plan")

    circuits = held["race_id"].str.split("_", n=2).str[2].nunique()
    put("h_circuits", circuits, str(circuits), HC, "distinct circuits among held-out race_id (Austria twice)")
    ms = sorted(held["max_stops"].unique())
    assert ms == [2], ms
    put("max_stops", 2, "two", HC, "max_stops column (future stops searched by the hindsight reference), 2 on every row")
    trio = [N[f"h_race_mean_{p}_2024_11_austria"].value for p in ("agent", "belief_optimizer", "actual_strategy")]
    put("aut24_about", float(np.mean(trio)), f"{np.mean(trio):.0f}", HC,
        f"2024 Austria mean regret of agent / belief_optimizer / actual_strategy: "
        f"{' / '.join(f'{x:.2f}' for x in trio)} s, all about this value")

    # development -> held-out ratios
    for pol in ("agent", "belief_optimizer", "actual_strategy"):
        r = N[f"h_mean_{pol}"].value / N[f"d_mean_{pol}"].value
        put(f"ratio_{pol}", r, f"×{r:.1f}", f"{HC}, {DC}", f"held-out mean / development mean, {pol}")
    return N


def provenance_md(N: dict[str, Num], used: list[str]) -> str:
    lines = ["# Numbers on the poster and where they come from", "",
             "Generated by `poster/build_poster.py` from the committed result files in `reports/eval/`. "
             "Held-out = `d4_*` (report set, 4 races, 100 scenarios, one agent pass, tag `d4-final-eval`); "
             "development = `dev6_run2_*` (6 calibration races, 150 scenarios).", "",
             "| key | printed | file | computation |", "|---|---|---|---|"]
    for k in used:
        n = N[k]
        lines.append(f"| `{k}` | {n.text} | `{n.source}` | {n.how} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    N = compute()
    for k, n in N.items():
        print(f"{k:42s} {n.text:22s} {n.how[:110]}")
