"""SC/VSC rates by circuit and era (SPEC 0011b) and the lap-1 term of the "rate" model. Offline,
synthetic data only."""
import hashlib
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.contracts import RaceParams, Snapshot
from src.sim import SimConfig, agent_tools, event_rates as er
from src.sim.calibrators import CALIBRATORS, Calibrator
from src.sim.config import EventModel
from src.sim.events import EVENTS

RACE = RaceParams(race_id="syn", total_laps=50, pit_loss_s=20.0)
# events.RateEvents draws before SPEC 0011b (computed on the base branch, feat/sim-cross-race)
BASE_DIGEST = "848d2848ece45a7678b441e66dc376c0ea0af9c9887338252c32fb26eebfab26"


def _rate_digest(**extra):
    race = RACE
    circuit = race.model_copy(update={"sc_rate_per_lap": 0.04, "sc_mean_laps": 4.0, "vsc_rate_per_lap": 0.03,
                                      "vsc_mean_laps": 2.0})
    h = hashlib.sha256()
    for r in (race, circuit):
        snaps = [Snapshot(race=r, driver="X", compound="SOFT"),
                 Snapshot(race=r, driver="X", laps_completed=12, compound="SOFT", tyre_age=12, sc_laps=[11, 12],
                          track_status="sc"),
                 Snapshot(race=r, driver="X", compound="SOFT", track_status="vsc")]
        for params in ({}, {"p_sc": 0.1, "p_vsc": 0.1}, {"p_sc": 0.0, "p_vsc": 0.0}):
            model = EVENTS.get("rate")(**params, **extra)
            for snap in snaps:
                for k in range(100):
                    s = model.draw(snap, np.random.default_rng([7, k]))
                    h.update(repr((s.sc_laps, s.vsc_laps)).encode())
    return h.hexdigest()


def draws(model, snap, n, seed=0):
    return [model.draw(snap, np.random.default_rng([seed, k])) for k in range(n)]


def start(race=RACE):
    return Snapshot(race=race, driver="X", compound="SOFT")


def periods_table(races):
    """A neutralised_periods table from {race_id: (total_laps, [(kind, start, end)], wet)}."""
    rows = []
    for rid, (total, periods, wet) in races.items():
        if not periods:
            rows.append({"race_id": rid, "kind": "", "start_lap": np.nan, "end_lap": np.nan, "total_laps": total,
                         "wet": wet})
        rows += [{"race_id": rid, "kind": k, "start_lap": s, "end_lap": e, "total_laps": total, "wet": wet}
                 for k, s, e in periods]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rich():
    """256 synthetic races (4 per circuit and season, 2018-2025) drawn by RateEvents."""
    periods, feats = er.synthetic_periods(range(2018, 2026), repeats=4)
    return er.stats_from_periods(periods, feats)


# --- the lap-1 term ----------------------------------------------------------------------
def test_lap1_none_is_byte_identical_to_the_base():
    assert _rate_digest() == BASE_DIGEST
    assert _rate_digest(p_sc_lap1=None, p_vsc_lap1=None) == BASE_DIGEST


def test_lap1_equal_to_p_gives_the_same_draws():
    a = draws(EVENTS.get("rate")(p_sc=0.05, p_vsc=0.04), start(), 200)
    b = draws(EVENTS.get("rate")(p_sc=0.05, p_vsc=0.04, p_sc_lap1=0.05, p_vsc_lap1=0.04), start(), 200)
    assert a == b


def test_lap1_term_applies_on_lap_1_at_k0_only():
    model = EVENTS.get("rate")(p_sc=0.01, p_vsc=0.01, p_sc_lap1=1.0, p_vsc_lap1=0.0)
    assert all(1 in s.sc_laps for s in draws(model, start(), 100))
    later = Snapshot(race=RACE, driver="X", laps_completed=5, compound="SOFT", tyre_age=5, standing_start=False)
    plain = EVENTS.get("rate")(p_sc=0.01, p_vsc=0.01)
    assert draws(model, later, 100) == draws(plain, later, 100)
    only = EVENTS.get("rate")(p_sc=0.0, p_vsc=0.0, p_vsc_lap1=0.5)
    got = draws(only, start(), 2000)
    assert all(not s.sc_laps and (not s.vsc_laps or s.vsc_laps[0] == 1) for s in got)
    assert np.mean([bool(s.vsc_laps) for s in got]) == pytest.approx(0.5, abs=0.04)
    # a race with circuit rates keeps its p on lap 1 for a kind whose lap-1 param is None
    circuit = RACE.model_copy(update={"sc_rate_per_lap": 0.0, "vsc_rate_per_lap": 0.0})
    assert all(s.sc_laps == [] for s in draws(EVENTS.get("rate")(p_vsc_lap1=0.3), start(circuit), 300))


def test_lap1_params_validate():
    EventModel(p_sc_lap1=0.2, p_vsc_lap1=0.1)
    with pytest.raises(ValidationError):
        EventModel(p_sc_lap1=0.7, p_vsc_lap1=0.5)
    with pytest.raises(ValidationError):
        EventModel(p_sc_lap1=0.99)           # with p_vsc 0.02 on lap 1
    with pytest.raises(ValidationError):
        EventModel(p_sc_lap1=-0.1)
    model = EVENTS.get("rate")(p_sc_lap1=0.9)
    with pytest.raises(ValueError, match="lap-1"):
        model.draw(start(RACE.model_copy(update={"vsc_rate_per_lap": 0.2})), np.random.default_rng(0))


# --- input -----------------------------------------------------------------------------------
def test_stats_from_periods_counts_and_laps_at_risk():
    t = periods_table({
        "2021_04_spain": (66, [("sc", 8, 10), ("vsc", 1, 2), ("red_flag", 30, 31)], False),
        "2021_05_monaco": (78, [], False),
        "2022_03_australia": (58, [("sc", 1, 3), ("sc", 20, 21)], True),
        "2022_04_bahrain": (57, [("vsc", 10, 11), ("sc", 55, 57)], False),
    })
    s = er.stats_from_periods(t).set_index("race_id")
    sp = s.loc["2021_04_spain"]
    assert (sp["n_sc"], sp["laps_sc"], sp["lap1_sc"], sp["n_vsc"], sp["laps_vsc"], sp["lap1_vsc"]) == (1, 3, 0, 1, 2, 1)
    assert (sp["cens_sc"], sp["cens_vsc"]) == (0, 0)
    bh = s.loc["2022_04_bahrain"]
    assert (bh["cens_sc"], bh["cens_vsc"]) == (1, 0)   # the SC ends on the last lap
    assert (sp["n_red_flag"], sp["neutralised_laps"], sp["at_risk"]) == (1, 7, 59)
    assert (sp["era"], sp["circuit"], sp["circuit_type"]) == ("2018-2021", "catalunya", "permanent")
    mo = s.loc["2021_05_monaco"]
    assert (mo["n_sc"], mo["n_vsc"], mo["at_risk"], mo["circuit_type"]) == (0, 0, 78, "street")
    au = s.loc["2022_03_australia"]
    assert bool(au["wet"]) and au["era"] == "2022+" and (au["n_sc"], au["laps_sc"], au["lap1_sc"]) == (2, 5, 1)
    assert list(s.index) == ["2021_04_spain", "2021_05_monaco", "2022_03_australia", "2022_04_bahrain"]   # date order
    with pytest.raises(ValueError, match="unknown period kinds"):
        er.stats_from_periods(periods_table({"2021_04_spain": (66, [("yellow", 3, 4)], False)}))


def test_circuit_info_map_features_and_unknown():
    ids = ["2021_04_spain", "2023_06_monaco", "2024_02_atlantis"]
    c = er.circuit_info(ids).set_index("race_id")
    assert tuple(c.loc["2023_06_monaco"]) == ("monaco", "street")
    assert tuple(c.loc["2024_02_atlantis"]) == ("atlantis", er.UNKNOWN)
    feats = pd.DataFrame({"race_id": ["2021_Spain", "2024_02_atlantis"], "circuit": ["x", ""],
                          "circuit_type": ["Street", "permanent"]})
    c = er.circuit_info(ids, feats).set_index("race_id")
    assert tuple(c.loc["2021_04_spain"]) == ("x", "street")            # alias, lower-cased
    assert tuple(c.loc["2024_02_atlantis"]) == ("atlantis", "permanent")   # blank circuit keeps the slug


def test_load_falls_back_to_factors_then_nothing(tmp_path):
    s, source = er.load_stats(tmp_path / "absent.csv", er.FACTORS, tmp_path / "absent_features.csv")
    assert source == str(er.FACTORS)
    sp = s.set_index("race_id").loc["2021_04_spain"]
    assert (sp["n_sc"], sp["laps_sc"], sp["lap1_sc"], sp["n_vsc"], sp["at_risk"], bool(sp["wet"])) == (1, 3, 0, 0, 63,
                                                                                                         False)
    s, source = er.load_stats(tmp_path / "absent.csv", tmp_path / "absent_factors.csv", None)
    assert source == "none" and len(s) == 0
    path = tmp_path / "periods.csv"
    periods_table({"2021_04_spain": (66, [("sc", 8, 10)], False)}).to_csv(path, index=False)
    s, source = er.load_stats(path, er.FACTORS, None)
    assert source == str(path) and s["n_sc"].tolist() == [1]


# --- fit -------------------------------------------------------------------------------------
def test_no_data_falls_back_to_the_prior():
    f = er.fit(pd.DataFrame(columns=er.STAT_COLUMNS))
    row = {"era": "2022+", "circuit_type": "street", "circuit": "monaco"}
    ev = EventModel()
    for c in er.CANDIDATES:
        r = f.rates(row, c)
        assert (r["p_sc"], r["d_sc"], r["p_vsc"], r["d_vsc"]) == (ev.p_sc, ev.d_sc, ev.p_vsc, ev.d_vsc)
        assert r["p_sc_lap1"] is None and r["p_vsc_lap1"] is None


def test_unseen_nodes_take_their_parent(rich):
    f = er.fit(rich)
    new = {"era": "2022+", "circuit_type": "street", "circuit": "never_raced"}
    assert f.plain["sc"].p("circuit", new) == f.plain["sc"].p("circuit_type", new)
    odd = {"era": "2022+", "circuit_type": er.UNKNOWN, "circuit": "never_raced"}
    assert f.plain["sc"].p("circuit", odd) == f.plain["sc"].p("era", odd)


def test_rich_data_goes_to_the_mle():
    races = {}
    for i in range(1500):   # 1500 races per circuit against a prior of a few races
        when = f"{2022 + i // 90}_{i % 90 + 1:02d}"
        races[f"{when}_synthetic_a{i}"] = (50, [("sc", 10, 12)] if i % 2 else [], False)
        races[f"{when}_synthetic_b{i}"] = (50, [("sc", 10, 12), ("sc", 30, 31), ("vsc", 40, 40)], False)
    feats = pd.DataFrame([{"race_id": r, "circuit": "a" if "_a" in r else "b", "circuit_type": "permanent"}
                          for r in races])
    s = er.stats_from_periods(periods_table(races), feats)
    f = er.fit(s)
    for c in ("a", "b"):
        part = s[s["circuit"] == c]
        mle = part["n_sc"].sum() / part["at_risk"].sum()
        got = f.plain["sc"].p("circuit", {"era": "2022+", "circuit_type": "permanent", "circuit": c})
        assert got == pytest.approx(mle, rel=0.01)
    assert f.plain["sc"].d_era["2022+"] == pytest.approx(s["laps_sc"].sum() / s["n_sc"].sum())
    assert f.plain["sc"].strength["circuit"] >= er.K0


def test_strength_is_floored_and_full_pooling_without_excess_variance():
    g = pd.DataFrame({"N": [5.0, 35.0], "L": [1000.0, 1000.0], "parent": [0.02, 0.02]})
    assert er._strength(g, 50.0, er.K0) == er.K0           # large spread: the floor
    same = pd.DataFrame({"N": [20.0, 20.0], "L": [1000.0, 1000.0], "parent": [0.02, 0.02]})
    assert math.isinf(er._strength(same, 50.0, er.K0))


def test_planted_circuit_effects_are_recovered(rich):
    rec = er.recovery_table(rich, er.synthetic_truth())
    for kind in er.KINDS:
        r = rec[rec["kind"] == kind]
        shrunk = (r["shrunk"] - r["planted"]).abs().mean()
        assert shrunk < (r["global"] - r["planted"]).abs().mean()
        assert shrunk < (r["mle"] - r["planted"]).abs().mean()
    sc = rec[rec["kind"] == "sc"]
    assert np.corrcoef(sc["planted"], sc["shrunk"])[0, 1] > 0.85
    for e in er.ERAS:   # the street effect shows at the circuit-type level
        t = sc[sc["era"] == e].groupby("circuit_type")["type_level"].first()
        assert t["street"] > t["permanent"]
    top = sc[sc["era"] == "2022+"].sort_values("shrunk")["circuit"].iloc[-1]
    assert top == "street1"   # the planted highest SC rate
    d = er.recovery_d_lap1(rich).set_index(["kind", "quantity"])
    for (kind, q), r in d.iterrows():
        if q == "lap-1 share":
            assert r["recovered"] == pytest.approx(r["planted"], abs=0.03)
        else:
            assert r["recovered"] == pytest.approx(r["planted"], rel=0.15)


def test_wet_races_stay_out_of_the_dry_rates(rich):
    wet = rich[rich["wet"]]
    assert len(wet) > 0
    a = er.fit(er.dry(rich))
    b = er.fit(er.dry(pd.concat([rich, wet.assign(n_sc=wet["n_sc"] + 5)])))
    assert a.plain["sc"].rates["global"] == b.plain["sc"].rates["global"]
    c = er.fit(er.dry(rich, include_wet=True))
    assert c.plain["sc"].rates["global"][()] > a.plain["sc"].rates["global"][()]   # wet at 3x the rate


# --- likelihood and selection ------------------------------------------------------------------
def test_nll_pieces():
    assert er.starts_nll(0, 50, 0.0) == 0.0
    assert math.isinf(er.starts_nll(1, 50, 0.0))
    assert er.starts_nll(2, 50, 0.02) == pytest.approx(1.0 - 2 * math.log(0.02))
    assert er.durations_nll(2, 6, 3.0) == pytest.approx(-(2 * math.log(1 / 3) + 4 * math.log(2 / 3)))
    assert er.durations_nll(3, 3, 1.0) == 0.0
    assert er.lap1_nll(1, 0.2) == pytest.approx(-math.log(0.2))
    assert er.lap1_nll(0, 0.2) == pytest.approx(-math.log(0.8))
    # lengths 2 and 4, the 4 still running at the flag: log q + log(1-q) for the 2, P(D >= 4) for the 4
    q = 1 / 3
    assert er.durations_nll(2, 6, 3.0, censored=1) == pytest.approx(-(math.log(q) + 4 * math.log(1 - q)))


def test_durations_enter_the_race_nll_and_censoring_lengthens_d():
    row = {"at_risk": 50, **{f"{c}_vsc": 0 for c in ("n", "laps", "lap1", "cens")},
           "n_sc": 1, "laps_sc": 4, "lap1_sc": 0, "cens_sc": 0}
    rates = {"p_sc": 0.02, "d_sc": 3.0, "p_vsc": 0.02, "d_vsc": 2.0, "p_sc_lap1": None, "p_vsc_lap1": None}
    base = er.race_nll(row, rates)
    assert er.race_nll(row, {**rates, "d_sc": 4.0}) < base            # length 4 is likelier under d 4
    assert er.race_nll({**row, "cens_sc": 1}, rates) < base            # censored: P(D >= 4) > P(D = 4)
    races = {"2022_01_bahrain": (57, [("sc", 10, 12)], False), "2022_02_saudi_arabia": (50, [("sc", 46, 50)], False)}
    s = er.stats_from_periods(periods_table(races))
    f = er.fit(s).plain["sc"]
    assert f.d_era["2022+"] == pytest.approx(1 + (8 - 2) / 1)         # 8 laps, 2 periods, 1 complete
    assert f.d_era["2022+"] > s["laps_sc"].sum() / s["n_sc"].sum()     # longer than the naive mean
    late = periods_table({"2022_02_saudi_arabia": races["2022_02_saudi_arabia"]})
    only_censored = er.fit(er.stats_from_periods(late))
    assert only_censored.plain["sc"].d_global == EventModel().d_sc     # no complete period: the prior


def test_selection_rule():
    nll = pd.DataFrame([{"race_id": f"r{i}", "candidate": c, "nll": 10.0 + (-0.5 if c == "circuit" else 0.0)
                         + (-0.5 if c == "circuit_lap1" else 0.0) + 0.01 * i}
                        for i in range(30) for c in er.CANDIDATES])
    t, chosen = er.selection(nll, n_boot=200)
    assert chosen == "circuit"   # tie with circuit_lap1: the simpler
    assert not t.set_index("candidate").loc["era", "passes"]   # difference 0: the CI is not below 0
    nll.loc[(nll["candidate"] == "circuit") & (nll["race_id"] == "r0"), "nll"] = math.inf
    t, chosen = er.selection(nll, n_boot=200)
    assert chosen == "circuit_lap1" and not t.set_index("candidate").loc["circuit", "passes"]
    # a race infinite under every candidate cannot tell them apart: dropped and counted
    void = pd.DataFrame([{"race_id": "r_void", "candidate": c, "nll": math.inf} for c in er.CANDIDATES])
    t2, chosen2 = er.selection(pd.concat([nll, void]), n_boot=200)
    assert chosen2 == chosen and (t2["n_dropped"] == 1).all() and (t2["n_races"] == 30).all()


def test_forward_folds_select_circuit_effects_on_the_rich_world(rich):
    t, chosen, fw, lo = er.both_schemes(rich)
    dry = rich[~rich["wet"]]
    assert set(lo["race_id"]) == set(dry["race_id"])                          # wet races are not scored
    assert set(fw["race_id"]) == set(dry.loc[dry["season"] >= er.FORWARD_FROM_SEASON, "race_id"])
    assert chosen == "circuit_lap1"
    t = t.set_index(["scheme", "candidate"])
    assert t.loc[("forward", "circuit"), "passes"] and t.loc[("loro", "circuit"), "passes"]
    assert t.loc[("forward", "circuit_lap1"), "chosen"]


def test_forward_folds_fit_on_earlier_races_only(rich):
    dry = rich[~rich["wet"]].reset_index(drop=True)
    fw = er.forward(rich).set_index(["race_id", "candidate"])["nll"]
    cut = len(dry) // 2
    target = dry.iloc[cut]
    changed = dry.assign(n_sc=np.where(dry.index >= cut, dry["n_sc"] + 3, dry["n_sc"]))
    again = er.forward(changed).set_index(["race_id", "candidate"])["nll"]
    f = er.fit(dry.iloc[:cut])   # the races dated before the target
    for c in er.CANDIDATES:
        assert fw[(target["race_id"], c)] == er.race_nll(target, f.rates(target, c))
        assert again[(dry.iloc[cut - 1]["race_id"], c)] == fw[(dry.iloc[cut - 1]["race_id"], c)]


def test_folds_keep_global_without_effects():
    flat = {f"c{i}": (er.PERMANENT, 1.0, 1.0) for i in range(6)}
    periods, feats = er.synthetic_periods(range(2022, 2026), circuits=flat, repeats=2, lap1={}, wet_share=0.0)
    s = er.stats_from_periods(periods, feats)
    t, chosen, _, _ = er.both_schemes(s, n_boot=500)
    assert chosen == "global"
    assert not t[(t["candidate"] != "global")]["passes"].any()


# --- as-of and the calibrator ---------------------------------------------------------------------
def test_as_of_uses_only_earlier_races(rich):
    cal = er.EventRatesCalibrator(level="circuit_lap1").fit(rich)
    dry = rich[~rich["wet"]].reset_index(drop=True)
    target = dry.iloc[len(dry) // 2]
    got = cal.rates_for(target["race_id"])
    later = dry.index >= len(dry) // 2   # itself and every race after it
    changed = dry.assign(n_sc=np.where(later, dry["n_sc"] + 4, dry["n_sc"]),
                         lap1_vsc=np.where(later, 1, dry["lap1_vsc"]))
    assert er.EventRatesCalibrator(level="circuit_lap1").fit(changed).rates_for(target["race_id"]) == got
    earlier = dry.assign(n_sc=np.where(~later, dry["n_sc"] + 1, dry["n_sc"]))
    assert er.EventRatesCalibrator(level="circuit_lap1").fit(earlier).rates_for(target["race_id"]) != got
    first = cal.rates_for(dry.iloc[0]["race_id"])
    ev = EventModel()
    assert (first["p_sc"], first["d_sc"], first["p_vsc"], first["d_vsc"]) == (ev.p_sc, ev.d_sc, ev.p_vsc, ev.d_vsc)
    table = er.as_of_rates(rich)
    n_train = table[table["candidate"] == "global"].set_index("race_id")["n_train"]
    for _, r in rich.iterrows():
        key = (r["season"], r["round"])
        assert n_train[r["race_id"]] == int(((dry["season"] * 100 + dry["round"]) < key[0] * 100 + key[1]).sum())


def test_calibrator_fills_race_params_and_the_tools_read_circuit(rich):
    assert "event_rates" in CALIBRATORS
    cal = CALIBRATORS.get("event_rates")(level="circuit").fit(rich)
    race = RaceParams(race_id="2026_01_synthetic_street1", total_laps=50, pit_loss_s=20.0)
    filled = cal.race_params(race, circuit="street1", circuit_type="street")
    r = cal.rates_for(race.race_id, "street1", "street")
    assert (filled.sc_rate_per_lap, filled.sc_mean_laps, filled.vsc_rate_per_lap, filled.vsc_mean_laps) == (
        r["p_sc"], r["d_sc"], r["p_vsc"], r["d_vsc"])
    cfg = SimConfig(events={"name": "rate"})
    out = agent_tools.event_rates(Snapshot(race=filled, driver="X", compound="SOFT"), cfg)
    assert out["source"] == agent_tools.EVENT_RATES_CIRCUIT and out["p_sc"] == r["p_sc"]
    assert agent_tools.event_rates(Snapshot(race=race, driver="X", compound="SOFT"), cfg)["source"] != "circuit"
    assert cal.event_params(race.race_id) == {}
    lap1 = er.EventRatesCalibrator(level="circuit_lap1").fit(rich).event_params(race.race_id)
    assert set(lap1) == {"p_sc_lap1", "p_vsc_lap1"}
    EventModel(**lap1)
    assert "p_sc_lap1" not in out and "p_vsc_lap1" not in out          # None: the block is unchanged
    shown = agent_tools.event_rates(Snapshot(race=filled, driver="X", compound="SOFT"),
                                    SimConfig(events={"name": "rate", "params": lap1}))
    assert (shown["p_sc_lap1"], shown["p_vsc_lap1"]) == (lap1["p_sc_lap1"], lap1["p_vsc_lap1"])
    assert shown["p_sc_lap1"] > shown["p_sc"]                          # first-lap risk above the per-lap rate
    one = agent_tools.event_rates(Snapshot(race=filled, driver="X", compound="SOFT"),
                                  SimConfig(events={"name": "rate", "params": {"p_sc_lap1": 0.1}}))
    assert one["p_sc_lap1"] == 0.1 and "p_vsc_lap1" not in out and "p_vsc_lap1" not in one
    with pytest.raises(TypeError, match="not a lap-time calibrator"):
        cal.calibrate(None)
    with pytest.raises(ValueError, match="unknown event_rates level"):
        er.EventRatesCalibrator(level="county")
    with pytest.raises(RuntimeError, match="fit"):
        er.EventRatesCalibrator().rates_for(race.race_id)


def test_save_load_round_trip(tmp_path):
    cal = er.EventRatesCalibrator(level="era", k0=8)
    cal.save(tmp_path / "c.json")
    back = Calibrator.load(tmp_path / "c.json")
    assert type(back) is er.EventRatesCalibrator and (back.level, back.k0) == ("era", 8.0)
    assert back.report()["params"] == {"level": "era", "k0": 8, "include_wet": False}


# --- report and guard ------------------------------------------------------------------------------
def test_synthetic_report_goes_to_tmp_only(tmp_path):
    periods, feats = er.synthetic_periods(range(2020, 2024))
    stats = er.stats_from_periods(periods, feats)
    with pytest.raises(ValueError, match="synthetic race"):
        er.report(stats, er.OUT_DIR, n_boot=50)
    with pytest.raises(ValueError, match="synthetic race"):
        er.report(stats, er.REPORTS_DIR, n_boot=50)
    assert not er.OUT_DIR.exists()
    path = er.report(stats, tmp_path / "rates", "synthetic", n_boot=50, truth=er.synthetic_truth())
    names = {p.name for p in path.parent.iterdir()}
    assert names == {"summary.md", "selection.csv", "forward_nll.csv", "loro_nll.csv", "rates.csv", "strengths.csv",
                     "recovery.csv"}
    assert set(pd.read_csv(path.parent / "selection.csv")["scheme"]) == {"forward", "loro"}
    assert "Chosen (forward):" in path.read_text()


def test_cli_synthetic(tmp_path, capsys):
    from src.sim import calibrate
    assert calibrate.main(["rates", "--synthetic", "--out", str(tmp_path), "--n-boot", "50",
                           "--seasons", "2021", "2022"]) == 0
    assert (tmp_path / "summary.md").is_file() and "written to" in capsys.readouterr().out
