"""Shared test setup: no test logs to wandb (compute.training_run reads WANDB_MODE)."""
import os

import pytest

os.environ["WANDB_MODE"] = "disabled"   # set at import too, before any worker process starts


@pytest.fixture(autouse=True, scope="session")
def _wandb_disabled():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        yield
