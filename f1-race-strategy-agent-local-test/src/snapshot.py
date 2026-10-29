"""Race-state snapshot passed to the strategy agent at a decision point."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    available_compounds: list[str] = field(default_factory=lambda: ["SOFT", "MEDIUM", "HARD"])
    nearby_cars: list[dict[str, Any]] = field(default_factory=list)
    recent_laps: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

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
