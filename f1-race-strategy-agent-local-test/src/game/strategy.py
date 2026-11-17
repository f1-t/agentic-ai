"""Bridge the existing race-strategy simulator into game-session timing overlays."""
from __future__ import annotations


def evaluate_strategy(simulator, snapshot, plan, *, seed=0, n_samples=1, scenario=None):
    """Return the project's SimResult with per-lap timing for a game HUD/replay.

    The track engine owns vehicle motion; the existing simulator remains the authority for
    tyre/race-strategy estimates. Keeping this adapter thin preserves both APIs and avoids
    duplicating the project's validated race model.
    """
    return simulator.simulate(snapshot, plan, seed=seed, n_samples=n_samples,
                              scenario=scenario, return_laps=True)


def lap_time_overlay(result) -> list[float]:
    """Extract finite lap-time targets from a valid simulation result."""
    if not result.valid or result.lap_times_s is None:
        raise ValueError("a valid simulation result with lap times is required")
    return [float(value) for value in result.lap_times_s]
