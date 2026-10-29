"""Compute profiles: every number that scales a run (samples, folds, epochs, workers) is a field here.

    profile = load_profile()            # --profile, else SIM_PROFILE, else "local"
    profile.sim.n_samples, profile.g.laps_every, profile.h.mlp.epochs

Profiles are YAML files in configs/compute/<name>.yaml. "local" is the desktop (Ryzen 9850X3D,
RTX 5080) and the default; "sol" is the big-run profile for the ASU Sol cluster.
"""
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

PROFILE_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "compute"
DEFAULT_PROFILE = "local"
PROFILE_ENV = "SIM_PROFILE"
HASH_EXCLUDE = {"wandb", "build"}   # sections that do not change a job's output


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimCompute(_Cfg):
    n_samples: int = Field(gt=0)
    max_stops: int = Field(default=2, ge=1)          # plan sweep depth
    min_stint: int | None = None                      # None: RaceParams.min_stint_laps
    compare_n_samples: int | None = None              # None: same as n_samples


class ParallelCompute(_Cfg):
    n_jobs: int = Field(default=1, ge=1)              # in-process pool size
    eval_chunk_size: int = Field(gt=0)                # eval items per chunk


class GCompute(_Cfg):
    laps_every: int = Field(gt=0)                     # snapshot laps k: 0, every, 2*every, ...
    sc_adjacent: bool = True                          # plus the lap before and after each SC / VSC period
    windows: list[Literal["full", "to_snapshot", "pre_race"]]
    n_boot: int = Field(gt=0)                         # bootstrap draws in the aggregate
    # N sweep. g_race: k_subsample, the snapshots at multiples of N (None = lap 0 only);
    # h_fold: SimConfig.calibration_every, the cadence (None = pre-race only)
    cadences: list[int | None]


class MlpCompute(_Cfg):
    ensemble: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    patience: int = Field(gt=0)
    hidden_sizes: list[int]
    weight_decay: float = Field(ge=0)
    device: Literal["auto", "cpu", "cuda"] = "cpu"    # auto: cuda if available, else cpu


class GbmCompute(_Cfg):
    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)


class HCompute(_Cfg):
    loro_folds: int | None = None                     # None: every race in the pool, else a seeded subsample
    loso: bool = False                                # also leave-one-season-out folds
    mlp: MlpCompute
    gbm: GbmCompute
    seed: int = 0


class BuildCompute(_Cfg):
    workers: int = Field(gt=0)
    retries: int = Field(ge=0)
    rate_pause_s: float = Field(ge=0)


class AgentCompute(_Cfg):
    """The live agent's budget in the eval (eval.policies agent policy, src/agent.py run_agent)."""
    max_steps: int = Field(default=10, gt=0)          # model turns per decision, agent.DEFAULT_MAX_STEPS


class WandbCompute(_Cfg):
    mode: Literal["online", "offline", "disabled"] = "disabled"
    project: str = "f1-strategy-sim"
    tags: list[str] = Field(default_factory=list)


class ComputeProfile(_Cfg):
    name: str
    sim: SimCompute
    parallel: ParallelCompute
    g: GCompute
    h: HCompute
    build: BuildCompute
    wandb: WandbCompute
    agent: AgentCompute = Field(default_factory=AgentCompute)

    @property
    def hash(self) -> str:
        """First 12 hex digits of sha256 over the profile as sorted JSON, without the wandb and build sections."""
        data = self.model_dump(mode="json", exclude=HASH_EXCLUDE)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def device(self) -> str:
        """The torch device name for the MLP: "auto" resolves to cuda when available, else cpu."""
        want = self.h.mlp.device
        if want != "auto":
            return want
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


def profile_path(name: str) -> Path:
    return Path(name) if name.endswith((".yaml", ".yml")) else PROFILE_DIR / f"{name}.yaml"


def profile_names() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str | None = None) -> ComputeProfile:
    """The named profile, else $SIM_PROFILE, else "local". A name ending in .yaml is a path."""
    name = name or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    path = profile_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no compute profile {name!r} at {path}; available: {profile_names()}")
    with open(path) as f:
        return ComputeProfile.model_validate(yaml.safe_load(f))


@contextlib.contextmanager
def wandb_run(profile: ComputeProfile, job: str, config: dict | None = None):
    """A wandb run for the job, or None when wandb is disabled or not installed.

    wandb is imported here and nowhere else, so nothing else needs it. Offline mode writes
    to ./wandb for `wandb sync` later.
    """
    if profile.wandb.mode == "disabled":
        yield None
        return
    try:
        import wandb
    except ImportError:
        log.warning("wandb is not installed, continuing without it")
        yield None
        return
    run = wandb.init(project=profile.wandb.project, mode=profile.wandb.mode, job_type=job,
                     tags=sorted({*profile.wandb.tags, profile.name}),
                     config={"profile": profile.name, "profile_hash": profile.hash, **(config or {})})
    try:
        yield run
    finally:
        run.finish()
