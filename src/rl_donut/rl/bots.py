"""The same scripted bots the environment ships, under the names the trainer already uses."""

import numpy as np

from rl_donut.env.bots import RandomBot as _RandomBot
from rl_donut.env.bots import StaticTargetBot as _StaticTargetBot
from rl_donut.env.policy import PolicyContext


def _name(name, default: str) -> str:
    return name if isinstance(name, str) else default


class RandomBot:
    """Uniform discrete actions and a uniform aim. Accepts a leftover action tuple as its name."""

    def __init__(self, name: str = "random_bot"):
        self.name = _name(name, "random_bot")
        self._inner = _RandomBot()

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None):
        if ctx is None:
            raise ValueError("RandomBot needs the match context")
        return self._inner.act(obs, ctx)


class StaticTargetBot:
    """Stands still, jetpack off, no shots, aim toward the enemy side."""

    def __init__(self, name: str = "static_target_bot"):
        self.name = _name(name, "static_target_bot")
        self._inner = _StaticTargetBot()

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None):
        if ctx is None:
            raise ValueError("StaticTargetBot needs the match context")
        return self._inner.act(obs, ctx)
