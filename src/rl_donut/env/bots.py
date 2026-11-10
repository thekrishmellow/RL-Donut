"""Scripted bots: the fixed yardstick that self-play ratings can't inflate.

Each follows the `Policy` interface. Actions are in each agent's own frame, where the enemy
side is always to the right (aim 0).
"""

import numpy as np

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC
from rl_donut.env.policy import PolicyContext

NO_MOVE = 1


class RandomBot:
    """Every discrete part uniform at random, aim uniform in [-pi, pi]."""

    name = "random"

    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        n = len(obs)
        discrete = np.stack([ctx.rng.integers(0, k, size=n) for k in DISCRETE_NVEC], axis=-1).astype(np.int64)
        return discrete, ctx.rng.uniform(-np.pi, np.pi, size=(n, CONTINUOUS_DIM)).astype(np.float32)


class StaticTargetBot:
    """Stands still on its spawn, never uses the jetpack or shoots, aims at the enemy side."""

    name = "static_target"

    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        n = len(obs)
        discrete = np.zeros((n, len(DISCRETE_NVEC)), dtype=np.int64)
        discrete[:, 0] = NO_MOVE
        return discrete, np.zeros((n, CONTINUOUS_DIM), dtype=np.float32)
