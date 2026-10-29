from dataclasses import dataclass
from math import isfinite

from .domain import Plan, PolicyUnavailable, Snapshot, Stint, validate_plan


@dataclass(frozen=True, slots=True)
class LegacySimulator:
    degradation: tuple[tuple[str, float, float], ...]

    def pace(self, compound: str, tyre_age: int) -> float:
        for name, base, slope in self.degradation:
            if name == compound:
                return base + slope * tyre_age
        raise ValueError(f"No legacy tyre model for {compound}.")

    def __call__(self, snapshot: Snapshot, plan: Plan, seed: int) -> float:
        validate_plan(snapshot, plan)
        times = []
        for index, stint in enumerate(plan.stints):
            age = snapshot.tyre_age if index == 0 else 0
            seconds = sum(
                self.pace(stint.compound, age + lap) for lap in range(1, stint.laps + 1)
            )
            times.append(round(seconds, 1))
        return round(sum(times) + snapshot.pit_loss_s * plan.num_stops, 1)


@dataclass(frozen=True, slots=True)
class LegacyContext:
    snapshot: Snapshot
    simulator: LegacySimulator
    actual_stints: tuple[Stint, ...]
    race: str
    driver: str


def load_legacy_context(driver: str = "HAM", completed_laps: int = 0) -> LegacyContext:
    from .. import tools

    info = tools.get_race_info()
    if type(completed_laps) is not int or not 0 <= completed_laps <= info["total_laps"]:
        raise ValueError("The snapshot lap must be between zero and the race length.")
    driver = driver.upper()
    actual = tools.get_driver_stints(driver)
    if "error" in actual:
        raise PolicyUnavailable(actual["error"])
    session = tools._load_session()
    laps = session.laps.pick_drivers(driver).sort_values("LapNumber")
    expected = list(range(1, info["total_laps"] + 1))
    if laps["LapNumber"].tolist() != expected:
        raise PolicyUnavailable(
            "The cached driver has missing laps or did not finish the full race."
        )
    row = laps.iloc[max(0, completed_laps - 1)]
    stint_rows = laps[laps["Stint"] == row["Stint"]]
    if stint_rows.empty or stint_rows["Compound"].nunique() != 1:
        raise PolicyUnavailable(
            "The current cached stint has inconsistent compound data."
        )
    first_lap = int(stint_rows["LapNumber"].min())
    tyre_age = float(row["TyreLife"]) - int(completed_laps == 0)
    if not isfinite(tyre_age) or not tyre_age.is_integer() or tyre_age < 0:
        raise PolicyUnavailable("The current tyre age is missing or invalid.")
    history = tuple(Stint(s["compound"], s["laps"]) for s in actual["stints"])
    models = []
    for compound in info["compounds_available"]:
        fitted = tools.estimate_tyre_degradation(compound)
        if "error" not in fitted:
            models.append((compound, fitted["base_laptime_s"], fitted["deg_per_lap_s"]))
    state = Snapshot(
        total_laps=info["total_laps"],
        completed_laps=completed_laps,
        current_compound=row["Compound"],
        tyre_age=int(tyre_age),
        current_stint_laps=max(0, completed_laps - first_lap + 1),
        used_compounds=tuple(
            laps[laps["LapNumber"] <= max(1, completed_laps)]["Compound"].unique()
        ),
        available_compounds=tuple(c for c, _, _ in models),
        min_stint_laps=info["min_stint_laps"],
        pit_loss_s=info["pit_loss_seconds"],
    )
    return LegacyContext(
        state, LegacySimulator(tuple(models)), history, info["race"], driver
    )
