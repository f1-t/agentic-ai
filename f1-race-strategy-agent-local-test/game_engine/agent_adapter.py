from __future__ import annotations
import json
from pathlib import Path
from src.agent import run_agent
from src.contracts import Snapshot

class StrategyAgentAdapter:
    def __init__(self, backend=None): self.backend=backend
    def decide(self, snapshot_path):
        data=json.loads(Path(snapshot_path).read_text()); snap=Snapshot.model_validate(data)
        rec,traj=run_agent(snap,backend=self.backend,verbose=True)
        return rec,traj
