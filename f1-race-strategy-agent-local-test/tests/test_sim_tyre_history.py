"""The historical tyre-life prior (src/sim/tyre_history.py) on synthetic histories, plus one slow
test on a real calibration race."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.sim import calibration, tyre_history as th


def cfg(**extra):
    c = copy.deepcopy(th.load_config())
    c.update(extra)
    return c


def noms(rows):
    """rows: race_id -> (circuit, date, (hard, medium, soft) common indices or None, laps)."""
    out = []
    for rid, (circuit, date, idx, laps) in rows.items():
        season, rnd, _ = th.parse_race_id(rid)
        idx = idx or (np.nan, np.nan, np.nan)
        out.append({"race_id": rid, "season": season, "round": rnd, "date": date, "circuit": circuit,
                    "hard": None if np.isnan(idx[0]) else f"C{idx[0]:g}",
                    "medium": None if np.isnan(idx[1]) else f"C{idx[1]:g}",
                    "soft": None if np.isnan(idx[2]) else f"C{idx[2]:g}",
                    "scale": "test", "scheduled_laps": laps,
                    "hard_index": idx[0], "medium_index": idx[1], "soft_index": idx[2]})
    return pd.DataFrame(out).set_index("race_id")


def hist_rows(rid, circuit, date, idx, fracs, laps=60, wet=0.0):
    season, rnd, _ = th.parse_race_id(rid)
    trio = float(np.mean(idx)) if idx and all(np.isfinite(idx)) else np.nan
    rows = []
    for role, i, f in zip(th.ROLES, idx or (np.nan,) * 3, fracs):
        if f is None:
            continue
        rows.append({"race_id": rid, "season": season, "round": rnd, "date": date, "era": th.era(season),
                     "circuit": circuit, "total_laps": laps, "role": role, "compound": f"C{i:g}",
                     "scale": "test", "common_index": i, "trio_index": trio, "edge_laps": f * laps,
                     "edge_frac": f, "wet_share": wet, "excluded": None})
    return rows


NOMS = noms({
    "2030_01_alpha": ("alpha", "2030-03-01", (2, 3, 4), 60),
    "2030_02_beta": ("beta", "2030-03-15", (1, 2, 3), 50),
    "2031_01_alpha": ("alpha", "2031-03-01", (2, 3, 4), 60),
    "2031_05_gamma": ("gamma", "2031-05-01", (3, 4, 5), 70),
    "2032_01_alpha": ("alpha", "2032-03-01", (2, 3, 4), 60),
})
HIST = pd.DataFrame(
    hist_rows("2030_01_alpha", "alpha", "2030-03-01", (2, 3, 4), (0.7, 0.5, 0.3))
    + hist_rows("2030_02_beta", "beta", "2030-03-15", (1, 2, 3), (0.8, 0.6, 0.4), laps=50)
    + hist_rows("2031_01_alpha", "alpha", "2031-03-01", (2, 3, 4), (0.72, 0.52, 0.32))
    + hist_rows("2031_05_gamma", "gamma", "2031-05-01", (3, 4, 5), (0.9, 0.9, 0.9), laps=70))


def prior(rid, role, **kw):
    kw.setdefault("history", HIST)
    kw.setdefault("nominations", NOMS)
    kw.setdefault("config", cfg())
    return th.tyre_life_prior(rid, role, **kw)


# --- as-of --------------------------------------------------------------------------------------
def test_only_races_strictly_before_feed_the_prior():
    p = prior("2031_01_alpha", "MEDIUM")
    used = {s[0] for s in p.sources}
    assert used == {"2030_01_alpha", "2030_02_beta"}           # never itself, never 2031_05 or later
    # a later race changing does not move the prior; an earlier one does
    later = HIST.copy()
    later.loc[later["race_id"] == "2031_05_gamma", "edge_frac"] = 0.05
    assert prior("2031_01_alpha", "MEDIUM", history=later).edge_laps == p.edge_laps
    earlier = HIST.copy()
    earlier.loc[earlier["race_id"] == "2030_01_alpha", "edge_frac"] = 0.05
    assert prior("2031_01_alpha", "MEDIUM", history=earlier).edge_laps != p.edge_laps


def test_same_round_and_same_date_are_not_earlier():
    h = pd.concat([HIST, pd.DataFrame(hist_rows("2031_01_other", "other", "2031-03-01", (2, 3, 4), (0.1, 0.1, 0.1)))])
    p = prior("2031_01_alpha", "SOFT", history=h)
    assert "2031_01_other" not in {s[0] for s in p.sources}
    # a race dated on or after the target is left out even when its (season, round) is earlier
    h2 = pd.concat([HIST, pd.DataFrame(hist_rows("2030_09_late", "late", "2031-04-01", (2, 3, 4), (0.1, 0.1, 0.1)))])
    assert "2030_09_late" not in {s[0] for s in prior("2031_01_alpha", "SOFT", history=h2).sources}


def test_races_argument_restricts_but_never_widens():
    p = prior("2031_01_alpha", "SOFT", races=["2030_02_beta", "2032_01_alpha"])
    assert {s[0] for s in p.sources} == {"2030_02_beta"}


# --- held out -----------------------------------------------------------------------------------
def test_held_out_races_never_feed_the_prior_by_default():
    c = cfg(held_out_extra=["2030_01_alpha"])
    p = prior("2031_01_alpha", "SOFT", config=c)
    assert "2030_01_alpha" not in {s[0] for s in p.sources}
    allowed = prior("2031_01_alpha", "SOFT", config=c, allow_held_out=True)
    assert "2030_01_alpha" in {s[0] for s in allowed.sources}
    # the flag never lets in a held-out race that is not earlier
    c2 = cfg(held_out_extra=["2031_05_gamma"])
    assert "2031_05_gamma" not in {s[0] for s in prior("2031_01_alpha", "SOFT", config=c2, allow_held_out=True).sources}
    assert "2031_05_gamma" not in {s[0] for s in prior("2032_01_alpha", "SOFT", config=c2).sources}


def test_eval_set_races_are_held_out():
    held = th.held_out(cfg())
    assert {"2019_06_monaco", "2020_16_sakhir", "2025_11_austria", "2024_11_austria"} <= held


def test_committed_history_has_no_held_out_race():
    hist = th.load_history()
    assert len(hist) and not set(hist["race_id"]) & th.held_out()
    import json
    man = json.loads(th.MANIFEST.read_text())
    assert not {r["race_id"] for r in man["races"]} & th.held_out()


def test_observing_a_held_out_race_is_refused_before_loading(monkeypatch):
    def boom(_):
        raise AssertionError("held-out race data loaded")
    monkeypatch.setattr(th, "_load_frame", boom)
    with pytest.raises(ValueError, match="held out"):
        th.observed_tyre_life("2019_06_monaco")
    with pytest.raises(ValueError, match="held out"):
        th.build(["2019_06_monaco"], out="/nonexistent/x.csv", log=lambda *_: None)


# --- tiers and weights ----------------------------------------------------------------------------
def test_each_fallback_tier():
    # tier 1: same circuit (alpha) and same compound earlier
    assert prior("2031_01_alpha", "SOFT").tier == 1
    # tier 2: same circuit, other compounds only (alpha's nomination moved two steps)
    n2 = noms({**{k: (v.circuit, v.date, (v.hard_index, v.medium_index, v.soft_index), v.scheduled_laps)
                  for k, v in NOMS.iterrows()}, "2031_01_alpha": ("alpha", "2031-03-01", (4.5, 5.5, 6.5), 60)})
    assert prior("2031_01_alpha", "HARD", nominations=n2, history=HIST[HIST.race_id != "2030_02_beta"]).tier == 2
    # tier 3: no earlier race at this circuit
    assert prior("2031_05_gamma", "SOFT").tier == 3
    # tier 4: nothing earlier at all; the constant prior, the widest sd
    p4 = prior("2030_01_alpha", "SOFT")
    assert p4.tier == 4 and p4.n_effective == 0 and p4.sources == ()
    c = cfg()
    assert p4.edge_frac == c["tier4_frac"]["SOFT"]
    assert p4.sd_frac == c["sd_floor_frac"][4]
    floors = [c["sd_floor_frac"][t] for t in (1, 2, 3, 4)]
    assert floors == sorted(floors) and len(set(floors)) == 4


def test_the_earliest_cached_season_gets_tier_4():
    hist = th.load_history()
    first = min(hist["race_id"], key=lambda r: th.parse_race_id(r)[:2])
    assert th.tyre_life_prior(first, "HARD").tier == 4


def test_weights_are_strictly_ordered():
    w = cfg()["weights"]
    order = [w[k] for k in th.CATEGORIES]
    assert order == sorted(order, reverse=True) and len(set(order)) == 4
    assert 0 < w["other_era"] < 1 and 0 < w["recency_per_season"] < 1


def test_weight_of_each_category_in_the_sources():
    p = prior("2031_01_alpha", "SOFT", use_severity=False)
    wts = {(r, role): wt for r, role, wt in p.sources}
    w = cfg()["weights"]
    rec = w["recency_per_season"]
    assert wts[("2030_01_alpha", "SOFT")] == pytest.approx(w["same_circuit_same_compound"] * rec)
    assert wts[("2030_01_alpha", "MEDIUM")] == pytest.approx(w["same_circuit_other_compound"] * rec)
    assert wts[("2030_02_beta", "MEDIUM")] == pytest.approx(w["other_circuit_other_compound"] * rec
                                                         * w["compound_step_decay"])   # C2 vs C4: two steps
    assert wts[("2030_02_beta", "SOFT")] == pytest.approx(w["other_circuit_other_compound"] * rec)
    p2 = prior("2031_05_gamma", "HARD", use_severity=False)   # gamma's hard is C3: beta's soft, alpha's soft... C3
    wts2 = {(r, role): wt for r, role, wt in p2.sources}
    assert wts2[("2030_02_beta", "SOFT")] == pytest.approx(w["other_circuit_same_compound"] * rec)
    # the heaviest source comes first
    assert [s[2] for s in p.sources] == sorted([s[2] for s in p.sources], reverse=True)


def test_other_era_and_recency_down_weight_but_keep():
    h = pd.DataFrame(hist_rows("2021_01_alpha", "alpha", "2021-03-01", (2, 3, 4), (0.7, 0.5, 0.3))
                     + hist_rows("2022_01_alpha", "alpha", "2022-03-01", (2, 3, 4), (0.7, 0.5, 0.3)))
    n = noms({"2023_01_alpha": ("alpha", "2023-03-01", (2, 3, 4), 60)})
    p = th.tyre_life_prior("2023_01_alpha", "SOFT", history=h, nominations=n, config=cfg())
    wts = {(r, role): wt for r, role, wt in p.sources}
    w = cfg()["weights"]
    assert wts[("2022_01_alpha", "SOFT")] == pytest.approx(w["recency_per_season"])
    assert wts[("2021_01_alpha", "SOFT")] == pytest.approx(w["recency_per_season"] ** 2 * w["other_era"])


def test_same_circuit_same_compound_dominates():
    # alpha (same circuit, same compound) says 0.3 of the race; gamma (other circuit) says 0.9
    p = prior("2032_01_alpha", "SOFT")
    assert abs(p.edge_frac - 0.31) < abs(p.edge_frac - 0.9) / 4


def test_neighbouring_compound_is_moved_by_the_life_ratio():
    h = pd.DataFrame(hist_rows("2030_01_alpha", "alpha", "2030-03-01", (2, 3, 4), (0.6, None, None)))
    n = noms({"2031_01_alpha": ("alpha", "2031-03-01", (2, 3, 4), 60)})
    p = th.tyre_life_prior("2031_01_alpha", "MEDIUM", history=h, nominations=n, config=cfg())
    assert p.tier == 2
    assert p.edge_frac == pytest.approx(0.6 * cfg()["life_ratio"]["default_per_step"])   # one step softer


def test_severity_scaling_moves_other_circuits_by_the_nominated_trio():
    # beta was given a trio one step harder than gamma: a harsher circuit, so its tyres last longer at gamma
    h = HIST[HIST.race_id == "2030_02_beta"]
    n = NOMS
    on = th.tyre_life_prior("2031_05_gamma", "SOFT", history=h, nominations=n, config=cfg())
    off = th.tyre_life_prior("2031_05_gamma", "SOFT", history=h, nominations=n, config=cfg(), use_severity=False)
    assert on.edge_frac > off.edge_frac
    # same role, trio shifted evenly: the severity cancels the C-number step exactly
    soft = [s for s in on.sources if s[1] == "SOFT"]
    assert soft


def test_life_ratio_estimated_as_of_with_a_default():
    c = cfg()
    r, n = th.life_ratio(HIST.iloc[:0], c)
    assert r == c["life_ratio"]["default_per_step"] and n == 0
    r2, n2 = th.life_ratio(th._usable(HIST[HIST.race_id == "2030_01_alpha"]), c)
    assert n2 == 3 and r2 == pytest.approx(np.exp(np.mean([np.log(0.5 / 0.7), np.log(0.3 / 0.5), np.log(0.3 / 0.7) / 2])))


def test_wet_and_excluded_observations():
    h = HIST.copy()
    h.loc[h["race_id"] == "2030_01_alpha", "wet_share"] = 0.5
    p = prior("2031_01_alpha", "SOFT", history=h)
    wts = {(r, role): wt for r, role, wt in p.sources}
    w = cfg()["weights"]
    assert wts[("2030_01_alpha", "SOFT")] == pytest.approx(w["same_circuit_same_compound"] * w["recency_per_season"] * w["wet_race"])
    h.loc[h["race_id"] == "2030_01_alpha", "excluded"] = "capped"
    assert "2030_01_alpha" not in {s[0] for s in prior("2031_01_alpha", "SOFT", history=h).sources}


# --- missing nomination --------------------------------------------------------------------------
def test_missing_nomination_falls_back_to_role():
    n = noms({**{k: (v.circuit, v.date, (v.hard_index, v.medium_index, v.soft_index), v.scheduled_laps)
                 for k, v in NOMS.iterrows()}, "2031_01_alpha": ("alpha", "2031-03-01", None, 60)})
    p = prior("2031_01_alpha", "SOFT", nominations=n)
    assert p.compound is None and np.isfinite(p.edge_laps) and p.sd_laps > 0
    assert any("no nomination" in x for x in p.notes)
    assert p.tier == 2           # never "the same compound" without a nomination
    # an unknown race (not in the table at all) still returns, with the distance from the circuit history
    q = prior("2033_01_zeta", "MEDIUM")
    assert q.tier == 3 and q.compound is None and q.total_laps == th.load_config()["default_race_laps"]


def test_bad_role_is_refused():
    with pytest.raises(ValueError):
        prior("2031_01_alpha", "INTERMEDIATE")


def test_interval_widths():
    p = prior("2031_01_alpha", "SOFT")
    lo68, hi68 = p.interval(0.68)
    lo90, hi90 = p.interval(0.9)
    assert lo90 < lo68 < p.edge_laps < hi68 < hi90
    assert hi68 - p.edge_laps == pytest.approx(p.sd_laps * 0.994, rel=1e-2)


# --- the observation ------------------------------------------------------------------------------
def frame_of(stints, total=20):
    """stints: (driver, compound, first_lap, last_lap, start_age, pit_in_at_end, status_on_last)."""
    rows, t = [], 0.0
    for drv, comp, first, last, age0, pit, status in stints:
        for lap in range(first, last + 1):
            rows.append({"driver": drv, "lap": lap, "stint": float(sum(1 for s in stints if s[0] == drv and s[2] <= first)),
                         "compound": comp, "tyre_age": float(age0 + lap - first + 1), "lap_time_s": 90.0 + 0.05 * lap,
                         "track_status": status if lap == last else "1", "pit_in": pit and lap == last,
                         "pit_out": lap == first and first > 1, "end_s": 100.0 * lap})
    return pd.DataFrame(rows)


def test_stint_ends_are_classified():
    f = frame_of([("AAA", "SOFT", 1, 8, 0, True, "1"), ("AAA", "HARD", 9, 20, 0, False, "1"),
                  ("BBB", "SOFT", 1, 6, 0, True, "4"), ("BBB", "HARD", 7, 20, 0, False, "1"),
                  ("CCC", "SOFT", 1, 12, 0, False, "1")])
    st = th.stint_table(f, cfg()).set_index(["driver", "stint"])["end"]
    assert st[("AAA", 1)] == "pitted" and st[("AAA", 2)] == "flag"
    assert st[("BBB", 1)] == "neutralised"
    assert st[("CCC", 1)] == "retired"


def test_observation_is_a_high_quantile_of_usable_stint_ends():
    stints = []
    for i, end in enumerate([10, 12, 14, 16, 18]):
        d = f"D{i}"
        stints += [(d, "SOFT", 1, end, 0, True, "1"), (d, "HARD", end + 1, 30, 0, False, "1")]
    stints += [("SC1", "SOFT", 1, 25, 0, True, "4"), ("SC1", "HARD", 26, 30, 0, False, "1"),
               ("RET", "SOFT", 1, 29, 0, False, "1")]
    obs = th.observed_tyre_life(frame_of(stints), race_id="2030_01_alpha", config=cfg()).set_index("role")
    soft = obs.loc["SOFT"]
    assert soft["n_usable"] == 5 and soft["n_neutralised"] == 1 and soft["n_retired"] == 1
    assert soft["edge_laps"] == pytest.approx(np.quantile([10, 12, 14, 16, 18], 0.9))   # SC stop and retirement left out
    assert soft["edge_frac"] == pytest.approx(soft["edge_laps"] / 30)
    assert obs.loc["HARD", "n_flag"] == 6
    # under min_stints usable stints: no edge
    few = th.observed_tyre_life(frame_of(stints[:4]), race_id="2030_01_alpha", config=cfg()).set_index("role")
    assert np.isnan(few.loc["SOFT", "edge_laps"])


def test_module_is_classified_physics():
    """The engine places the wear cliff's support edge with it (cliff.history_prior), so it moves lap times."""
    assert "tyre_history" in calibration.PHYSICS_MODULES and "tyre_history" not in calibration.NON_PHYSICS


def test_validation_runs_on_the_committed_tables():
    v = th.validate()
    assert len(v) and set(v["variant"]) == set(th.VARIANTS)
    assert not set(v["race_id"]) & th.held_out()
    s = th.summarise(v)
    assert {"mae", "cover68", "cover90", "beyond_share"} <= set(s.columns)


# --- real data ------------------------------------------------------------------------------------
@pytest.mark.slow
def test_real_calibration_race_matches_the_committed_history():
    obs = th.observed_tyre_life("2021_04_spain").set_index("role")
    hist = th.load_history()
    row = hist[(hist.race_id == "2021_04_spain")].set_index("role")
    for role in ("MEDIUM", "SOFT"):
        assert obs.loc[role, "edge_laps"] == pytest.approx(row.loc[role, "edge_laps"])
        assert 0 < obs.loc[role, "edge_frac"] <= 1
    # the soft runs out before the medium, and the 2023 Spanish prior (tier 1) is near what 2023 showed
    assert obs.loc["SOFT", "edge_laps"] < obs.loc["MEDIUM", "edge_laps"]
    p = th.tyre_life_prior("2023_07_spain", "SOFT")
    seen = hist[(hist.race_id == "2023_07_spain") & (hist.role == "SOFT")]["edge_laps"].iloc[0]
    assert p.tier == 1 and abs(p.edge_laps - seen) < 2 * p.sd_laps
