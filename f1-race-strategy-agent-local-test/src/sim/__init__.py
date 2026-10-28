"""Race strategy simulator package.

    from src.sim import Simulator, SimConfig
    sim = Simulator(lap_frame(session.laps))
    result = sim.simulate(snapshot, [["MEDIUM", 40], ["SOFT", 26]])
"""
from ..contracts import Plan, RaceParams, SimResult, Snapshot  # noqa: F401
from .config import SimConfig  # noqa: F401
from .engine import Simulator  # noqa: F401
