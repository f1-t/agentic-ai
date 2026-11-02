"""Race-state snapshot passed to the strategy agent at a decision point.

RaceSnapshot is the agent's view. The shared contract is contracts.Snapshot (the snapshot
builder's output, what the eval scores from); from_contract builds the agent's view from
it and to_contract goes back. RaceSnapshot retires once agent.py reads the contract.

    contracts.Snapshot              RaceSnapshot
    race_id <season>_<rr>_<slug>    season, event (the slug)
    laps_completed                  lap
    race.total_laps                 total_laps
    tyre_age                        tyre_age_laps
    track_status green / sc / vsc   track_status GREEN / SC / VSC
    weather.rain_probability,       rain_probability, air_temp_c, track_temp_c
      air_temp_c, track_temp_c
    field.nearby                    nearby_cars [{driver, position, gap_s}], gap_s < 0 ahead
    recent_lap_times_s              recent_laps [{lap, lap_time_s}], oldest first
    race.min_stint_laps             min_stint_laps (R2, docs/rules.md)
    sets_available / race.compounds available_compounds
    context                         context

available_compounds are the inventory-legal compounds: those with a set in sets_available,
else race.compounds when the snapshot is unconstrained (the default three when the race
names none). The stint history, SC laps and sets are not in the agent's view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
_STATUS_TO_AGENT = {"green": "GREEN", "sc": "SC", "vsc": "VSC"}


@dataclass
class RaceSnapshot:
    season: int
    event: str
    lap: int
    total_laps: int
    driver: str
    position: int
    compound: str
    tyre_age_laps: int
    gap_ahead_s: float | None = None
    gap_behind_s: float | None = None
    track_status: str = "GREEN"
    rain_probability: float | None = None
    air_temp_c: float | None = None
    track_temp_c: float | None = None
    available_compounds: list[str] = field(default_factory=lambda: list(DEFAULT_COMPOUNDS))
    nearby_cars: list[dict[str, Any]] = field(default_factory=list)
    recent_laps: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    min_stint_laps: int | None = None

    @property
    def laps_remaining(self) -> int:
        return max(0, self.total_laps - self.lap)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["laps_remaining"] = self.laps_remaining
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceSnapshot":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_contract(cls, snapshot) -> "RaceSnapshot":
        """The agent's view of a contracts.Snapshot (module docstring)."""
        from .sim.race_ids import canonical_race_id, parse_race_id

        race = snapshot.race
        try:
            season, _, event = parse_race_id(race.race_id)
        except ValueError:
            head, _, rest = canonical_race_id(race.race_id).partition("_")
            season, event = (int(head), rest or race.race_id) if head.isdigit() else (0, race.race_id)
        if snapshot.sets_available is not None:
            compounds = sorted({t.compound for t in snapshot.sets_available})
        else:
            compounds = list(race.compounds) or list(DEFAULT_COMPOUNDS)
        k, recent = snapshot.laps_completed, snapshot.recent_lap_times_s
        return cls(
            season=season, event=event, lap=k, total_laps=race.total_laps, driver=snapshot.driver,
            position=snapshot.position, compound=snapshot.compound, tyre_age_laps=snapshot.tyre_age,
            gap_ahead_s=snapshot.gap_ahead_s, gap_behind_s=snapshot.gap_behind_s,
            track_status=_STATUS_TO_AGENT[snapshot.track_status],
            rain_probability=snapshot.weather.rain_probability, air_temp_c=snapshot.weather.air_temp_c,
            track_temp_c=snapshot.weather.track_temp_c, available_compounds=compounds,
            nearby_cars=[c.model_dump() for c in snapshot.field.nearby],
            recent_laps=[{"lap": k - len(recent) + 1 + i, "lap_time_s": t} for i, t in enumerate(recent)],
            context=dict(snapshot.context), min_stint_laps=race.min_stint_laps)

    def to_contract(self, race=None):
        """A contracts.Snapshot with what this view carries. `race` (contracts.RaceParams)
        supplies the race constants; without it they are rebuilt from the view, with
        pit_loss_s 0 (R5: the simulator owns pit loss, the snapshot record does not) and
        min_stint_laps from the view or the RaceParams default. The stint history, SC laps
        and sets are not in the view, so they come back empty (sets: unconstrained)."""
        from .contracts import FieldState, NearbyCar, RaceParams, Snapshot, Weather

        if race is None:
            extra = {} if self.min_stint_laps is None else {"min_stint_laps": self.min_stint_laps}
            race = RaceParams(race_id=f"{self.season}_{self.event}", total_laps=self.total_laps, pit_loss_s=0.0,
                              compounds=list(self.available_compounds), **extra)
        status = {v: k for k, v in _STATUS_TO_AGENT.items()}.get(str(self.track_status).upper(), "green")
        return Snapshot(
            race=race, driver=self.driver, laps_completed=self.lap, compound=self.compound,
            tyre_age=self.tyre_age_laps, track_status=status, position=self.position,
            gap_ahead_s=self.gap_ahead_s, gap_behind_s=self.gap_behind_s,
            weather=Weather(rain_probability=self.rain_probability, air_temp_c=self.air_temp_c,
                            track_temp_c=self.track_temp_c),
            field=FieldState(nearby=[NearbyCar(**c) for c in self.nearby_cars]),
            recent_lap_times_s=[float(r["lap_time_s"]) for r in self.recent_laps],
            context={str(k): str(v) for k, v in self.context.items()}, standing_start=self.lap == 0)
