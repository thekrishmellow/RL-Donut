"""Scripted development bots for Phase 5 training and evaluation."""

import numpy as np
from rl_donut.rl.stand_in import Policy, PolicyContext


class RandomBot:
    """Samples uniform random valid actions for any action_nvec."""

    def __init__(self, name: str = "random_bot"):
        self.name = name

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None) -> np.ndarray:
        n = obs.shape[0]
        nvec = ctx.env.action_nvec if ctx is not None else (3, 2, 16, 2)
        rng = ctx.rng if ctx is not None else np.random.default_rng()
        actions = np.zeros((n, len(nvec)), dtype=np.int64)
        for i, dim in enumerate(nvec):
            actions[:, i] = rng.integers(0, dim, size=n)
        return actions


class StaticTargetBot:
    """Stands still, does not thrust, aims right, and does not shoot."""

    def __init__(self, name: str = "static_target_bot"):
        self.name = name

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None) -> np.ndarray:
        n = obs.shape[0]
        # move=1 (none), jet=0 (off), aim=0, shoot=0 (no)
        actions = np.zeros((n, 4), dtype=np.int64)
        actions[:, 0] = 1
        return actions

