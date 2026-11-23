"""The supplier's pre-race tables (src/sim/prerace.py) and the switches that read them: off is the
earlier behaviour bit for bit; on, the readers never touch a race's data and the agent's block names
no race."""
import re

import pandas as pd
import pytest

from src.sim import agent_tools, prerace, provenance, tyre_history as th
from src.sim.calibration import config_hash
from src.sim.config import PreRace, SimConfig


def test_tables_have_one_row_per_nominated_weekend():
    noms = th.load_nominations().index
    for t in (prerace.load_ratings(), prerace.load_guidance()):
        assert sorted(t.index) == sorted(noms)


def test_guidance_published_before_the_race_or_noted():
    g = prerace.load_guidance()
    for rid, r in g.iterrows():
        if r["published_date"]:
            assert r["published_date"] <= r["race_date"], rid
            if r["published_date"] == r["race_date"]:
                assert "local time" in r["notes"].lower() or "saturday" in r["notes"].lower(), rid


def test_ratings_in_range():
    t = prerace.load_ratings()
    vals = t[list(prerace.RATING_KEYS)].stack()
    assert ((vals >= 1) & (vals <= 5)).all()


def test_strategies_parse_and_windows_give_stint_lengths():
    s = prerace.strategies_of("2019_10_united_kingdom")
    assert s[0]["stops"] == 2 and s[0]["sequence"] == ["SOFT", "SOFT", "HARD"]
    assert s[0]["windows"] == [(13.0, 13.0), (26.0, 26.0)]
    life = prerace.guidance_life("2019_10_united_kingdom", 52)
    # the longest stint per role over the three options: HARD 52 - 26 (option 1's last stint),
    # SOFT 13, MEDIUM 15 (option 2's first stint)
    assert life == {"HARD": 26.0, "SOFT": 13.0, "MEDIUM": 15.0}


def test_unstated_sequence_keeps_the_stop_count():
    s = prerace.strategies_of("2021_04_spain")
    assert s and s[0]["stops"] == 2 and s[0]["sequence"] is None
    assert prerace.expected_stops("2021_04_spain") == 2
    assert prerace.guidance_life("2021_04_spain", 66, "stated") == {}


def test_stated_mode_splits_the_distance_by_role_life_weight():
    life = prerace.guidance_life("2021_09_austrian", 71, "stated")
    w = prerace.ROLE_LIFE_WEIGHT
    assert life["HARD"] == pytest.approx(71 * w["HARD"] / (w["MEDIUM"] + w["HARD"]))
    assert prerace.guidance_life("2021_09_austrian", 71, "windows") == {}


def test_off_is_the_earlier_prior_and_hash():
    assert config_hash(SimConfig()) == config_hash(SimConfig(prerace=PreRace()))
    assert config_hash(SimConfig(prerace=PreRace(tyre_life_guidance="stated"))) != config_hash(SimConfig())
    for rid in ("2022_06_spain", "2023_07_spain", "2021_20_qatar"):
        for role in th.ROLES:
            a = th.tyre_life_prior(rid, role)
            b = th.tyre_life_prior(rid, role, ratings="off", guidance="off")
            assert (a.edge_laps, a.sd_laps) == (b.edge_laps, b.sd_laps)


@pytest.mark.parametrize("kw", [{"ratings": "supplement"}, {"ratings": "replace"}, {"guidance": "stated"},
                                {"guidance": "windows"}])
def test_switches_read_no_later_race(kw):
    """Truncating the history to the races before the target leaves the switched prior unchanged."""
    hist = th.load_history()
    rid = "2022_12_france"
    from src.sim.race_ids import as_of
    earlier = set(as_of(rid, list(hist["race_id"].unique())))
    cut = hist[hist["race_id"].isin(earlier)]
    for role in th.ROLES:
        a = th.tyre_life_prior(rid, role, **kw)
        b = th.tyre_life_prior(rid, role, history=cut, **kw)
        assert a.edge_laps == pytest.approx(b.edge_laps) and a.sd_laps == pytest.approx(b.sd_laps)


def test_world_prior_never_moves():
    from src.sim import cliff
    wc = SimConfig().wear_cliff
    on = PreRace(tyre_life_ratings="supplement", tyre_life_guidance="stated")
    assert cliff.history_prior("2022_12_france", 53, wc, role="world", prerace=on) == \
        cliff.history_prior("2022_12_france", 53, wc, role="world")


def test_registered_as_pre_race():
    for name in ("circuit_ratings", "prerace_guidance"):
        a = provenance.POLICY_SIDE[name]
        assert a.kind == "pre_race" and provenance.recorded(name) == []
        assert provenance.races_for(name, "2022_12_france") == []


def test_agent_block_names_no_race(monkeypatch):
    monkeypatch.setenv("AGENT_PRERACE", "on")
    assert agent_tools.prerace_shown()
    noms = pd.read_csv(th.NOMINATIONS)
    words = set(noms["circuit"].str.lower()) | {"grand prix", "pirelli"}
    for rid in noms["race_id"]:
        block = agent_tools.prerace_block(rid)
        assert block is not None
        text = str(block).lower()
        assert not [w for w in words if w in text], rid
        assert not re.search(r"\b(19|20)\d\d\b", text), rid


def test_agent_block_off_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_PRERACE", raising=False)
    assert not agent_tools.prerace_shown()
    monkeypatch.setenv("AGENT_PRERACE", "maybe")
    with pytest.raises(ValueError):
        agent_tools.prerace_shown()
