"""WearCliff.world_edge "as_of": the judge's tyre-life edge is the historical prior extended by the
ages run up to the scored snapshot only (cliff.as_of_world_cliffs, engine.cliffs_for). No cache."""
import pandas as pd

from src.contracts import RaceParams
from src.sim import SimConfig, cliff, fidelity
from src.sim.config import WearCliff

RACE = RaceParams(race_id="2021_04_spain", total_laps=60, pit_loss_s=20.0, min_stint_laps=1,
                  compounds=["SOFT", "MEDIUM", "HARD"])


def _laps(max_age):
    return pd.DataFrame({"driver": "AAA", "lap": range(1, max_age + 1), "compound": "HARD",
                         "tyre_age": range(1, max_age + 1)})


def test_as_of_extends_the_prior_by_ages_seen_by_the_snapshot_only(monkeypatch):
    monkeypatch.setattr(cliff, "history_prior", lambda *a, **k: {"HARD": (30.0, 3.0, "t"), "MEDIUM": (25.0, 3.0, "t")})
    wc = WearCliff(world_edge="as_of")
    whole = _laps(50)
    only = cliff.world_cliffs(whole, None, RACE, WearCliff(world_edge="history_only"))
    hist = cliff.world_cliffs(whole, None, RACE, WearCliff(world_edge="history"))
    at0 = cliff.as_of_world_cliffs(whole, whole.iloc[0:0], None, RACE, wc)
    mid = cliff.as_of_world_cliffs(whole, whole[whole["lap"] <= 35], None, RACE, wc)
    early = cliff.as_of_world_cliffs(whole, whole[whole["lap"] <= 20], None, RACE, wc)
    assert {c: x.age for c, x in at0.items()} == {c: x.age for c, x in only.items()}   # lap 0: history_only
    assert early["HARD"].age == 30.0 and mid["HARD"].age == 35.0 and hist["HARD"].age == 50.0
    assert "seen by the snapshot" in mid["HARD"].source


def test_the_ledger_tags_as_of_as_known_at_the_snapshot():
    row = {r["id"]: r for r in fidelity.ledger(SimConfig(wear_cliff={"world_edge": "as_of"}), "judge")}["cliff_edge"]
    assert row["tag"] == "this race, to snapshot"
