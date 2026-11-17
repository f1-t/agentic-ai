"""Deterministic, fixed-step bird's-eye driving engine on sampled circuit centerlines.

Coordinates are metres in a local track plane. The engine is deliberately independent of
rendering, so a UI can draw ``GameState.cars`` and ``TrackGeometry`` at any frame rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


@dataclass(frozen=True)
class TrackGeometry:
    name: str
    points: tuple[tuple[float, float], ...]
    width_m: float = 12.0
    source: str = "unknown"
    _cumulative: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError("track geometry needs at least three points")
        if self.width_m <= 0:
            raise ValueError("track width must be positive")
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in self.points):
            raise ValueError("track points must be finite")
        pts = self.points if self.points[0] == self.points[-1] else (*self.points, self.points[0])
        object.__setattr__(self, "points", tuple(pts))
        lengths = [0.0]
        for a, b in zip(pts, pts[1:]):
            lengths.append(lengths[-1] + _distance(a, b))
        if lengths[-1] <= 0:
            raise ValueError("track length must be positive")
        object.__setattr__(self, "_cumulative", tuple(lengths))

    @property
    def length_m(self):
        return self._cumulative[-1]

    def at(self, distance_m: float) -> tuple[float, float, float]:
        """Return centerline (x, y, heading) at wrapped distance in metres."""
        d = float(distance_m) % self.length_m
        for i in range(len(self.points) - 1):
            start, end = self._cumulative[i], self._cumulative[i + 1]
            if d <= end:
                a, b = self.points[i], self.points[i + 1]
                f = (d - start) / max(end - start, 1e-9)
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]),
                        math.atan2(b[1] - a[1], b[0] - a[0]))
        return (*self.points[0], 0.0)

    @classmethod
    def load(cls, path: str | Path) -> "TrackGeometry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(name=raw["name"], points=tuple(map(tuple, raw["points"])),
                   width_m=raw.get("width_m", 12.0), source=raw.get("source", "unknown"))


@dataclass
class CarState:
    driver: str
    distance_m: float = 0.0
    lateral_m: float = 0.0
    speed_mps: float = 0.0
    heading_error_rad: float = 0.0
    lap: int = 0
    elapsed_s: float = 0.0
    off_track: bool = False
    retired: bool = False

    @property
    def progress_m(self):
        return self.distance_m


@dataclass(frozen=True)
class Action:
    throttle: float = 0.0  # -1 brake, +1 accelerate
    steering: float = 0.0  # -1 left, +1 right

    def __post_init__(self):
        if not (-1 <= self.throttle <= 1 and -1 <= self.steering <= 1):
            raise ValueError("throttle and steering must be in [-1, 1]")


@dataclass
class GameState:
    tick: int = 0
    cars: dict[str, CarState] = field(default_factory=dict)
    finished: bool = False


class GameEngine:
    """Fixed-step car dynamics with track edges, lap timing and simple car contact.

    This is a playable simulation model, not a homologated vehicle/physics model. Its
    parameters are configurable and units are SI; callers should advance in small steps.
    """

    def __init__(self, track: TrackGeometry, *, laps: int = 1, dt: float = 0.02,
                 max_speed_mps: float = 95.0, acceleration_mps2: float = 12.0,
                 braking_mps2: float = 24.0, grip_mps2: float = 24.0,
                 car_width_m: float = 2.0, collision_penalty_s: float = 1.0):
        if laps < 1 or dt <= 0 or max_speed_mps <= 0:
            raise ValueError("laps, dt, and max speed must be positive")
        self.track, self.laps, self.dt = track, int(laps), float(dt)
        self.max_speed_mps, self.acceleration_mps2 = max_speed_mps, acceleration_mps2
        self.braking_mps2, self.grip_mps2 = braking_mps2, grip_mps2
        self.car_width_m, self.collision_penalty_s = car_width_m, collision_penalty_s
        self.state = GameState()
        self._contacts: set[tuple[str, str]] = set()

    @classmethod
    def for_race(cls, race_id: str, **options) -> "GameEngine":
        """Create an engine using this project's race-id/schedule/circuit mapping."""
        from .tracks import load_race_track
        return cls(load_race_track(race_id), **options)

    def add_car(self, driver: str, *, distance_m: float = 0.0, lateral_m: float = 0.0,
                speed_mps: float = 0.0) -> CarState:
        if driver in self.state.cars:
            raise ValueError(f"driver {driver!r} already exists")
        car = CarState(driver, distance_m % self.track.length_m, lateral_m,
                       min(max(speed_mps, 0.0), self.max_speed_mps))
        self.state.cars[driver] = car
        return car

    def step(self, actions: dict[str, Action | dict] | None = None) -> GameState:
        """Advance all active cars by dt seconds and return the live state."""
        if self.state.finished:
            return self.state
        actions = actions or {}
        for driver, car in self.state.cars.items():
            if car.retired:
                continue
            raw = actions.get(driver, Action())
            action = raw if isinstance(raw, Action) else Action(**raw)
            car.heading_error_rad += action.steering * 2.4 * self.dt
            car.heading_error_rad = max(-math.pi / 2, min(math.pi / 2, car.heading_error_rad))
            lateral_speed = car.speed_mps * math.sin(car.heading_error_rad)
            car.lateral_m += lateral_speed * self.dt
            car.heading_error_rad *= max(0.0, 1.0 - 1.2 * self.dt)
            accel = action.throttle * (self.acceleration_mps2 if action.throttle >= 0 else self.braking_mps2)
            _, _, before = self.track.at(car.distance_m - 2.0)
            _, _, after = self.track.at(car.distance_m + 2.0)
            turn = (after - before + math.pi) % (2 * math.pi) - math.pi
            curvature = abs(turn) / 4.0
            corner_speed = (self.max_speed_mps if curvature < 1e-4 else
                            min(self.max_speed_mps, math.sqrt(self.grip_mps2 / curvature)))
            car.speed_mps = min(corner_speed, max(0.0, car.speed_mps + accel * self.dt))
            advance = max(0.0, car.speed_mps * math.cos(car.heading_error_rad)) * self.dt
            total_distance = car.distance_m + advance
            completed = int(total_distance // self.track.length_m)
            car.distance_m = total_distance % self.track.length_m
            car.lap += completed
            car.elapsed_s += self.dt
            half_width = self.track.width_m / 2 - self.car_width_m / 2
            car.off_track = abs(car.lateral_m) > half_width
            if car.off_track:
                car.speed_mps = max(0.0, car.speed_mps - self.braking_mps2 * self.dt)
                car.lateral_m = math.copysign(min(abs(car.lateral_m), self.track.width_m), car.lateral_m)
            # Recover heading when steering is released; offset gently returns to center.
            if abs(action.steering) < 0.05:
                car.lateral_m *= max(0.0, 1.0 - 0.03 * self.dt)
        self._resolve_contacts()
        self.state.tick += 1
        self.state.finished = any(c.lap >= self.laps for c in self.state.cars.values())
        return self.state

    def _resolve_contacts(self):
        cars = [c for c in self.state.cars.values() if not c.retired]
        contacts = set()
        for i, a in enumerate(cars):
            for b in cars[i + 1:]:
                delta = min((a.distance_m - b.distance_m) % self.track.length_m,
                            (b.distance_m - a.distance_m) % self.track.length_m)
                pair = tuple(sorted((a.driver, b.driver)))
                if delta < self.car_width_m * 1.25 and abs(a.lateral_m - b.lateral_m) < self.car_width_m:
                    contacts.add(pair)
                    a.speed_mps *= 0.85
                    b.speed_mps *= 0.85
                    if pair not in self._contacts:
                        a.elapsed_s += self.collision_penalty_s
                        b.elapsed_s += self.collision_penalty_s
        self._contacts = contacts

    def telemetry(self) -> dict[str, dict[str, float | int | bool]]:
        """Renderer/API-friendly snapshot keyed by driver."""
        result = {}
        for name, car in self.state.cars.items():
            x, y, heading = self.track.at(car.distance_m)
            x -= math.sin(heading) * car.lateral_m
            y += math.cos(heading) * car.lateral_m
            result[name] = {"x": x, "y": y, "heading_rad": heading + car.heading_error_rad,
                            "speed_mps": car.speed_mps, "distance_m": car.distance_m,
                            "lap": car.lap, "elapsed_s": car.elapsed_s, "off_track": car.off_track}
        return result
