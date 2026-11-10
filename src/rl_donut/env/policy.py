"""One interface for everything that controls agents: learners, pool snapshots, scripted bots.

A policy gets the observation rows of the slots it controls and returns their actions, in
each agent's own frame (handoff guide, section 4.3). Scripted bots may also read the true
game state through `ctx.env`; that makes them "privileged", a yardstick rather than a fair
opponent. Policies that remember things per slot notice a new match through
`ctx.env.world.episode`, which counts up each time a match restarts.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC
from rl_donut.env.match_env import MatchEnv


@dataclass
class PolicyContext:
    env: MatchEnv
    slots: np.ndarray            # (n, 2) (match, agent) of each row
    rng: np.random.Generator


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        """obs (n, obs_dim) -> discrete (n, len(action_nvec)) int64, continuous (n, action_cont_dim) float32."""
        ...


def team_slots(env: MatchEnv, team: int) -> np.ndarray:
    """(match, agent) of every agent of `team` in every match, match-major."""
    agents = np.nonzero(env.team == team)[0]
    return np.stack(np.meshgrid(np.arange(env.n_matches), agents, indexing="ij"), axis=-1).reshape(-1, 2)


def play_matches(env: MatchEnv, team_a: Policy, team_b: Policy, seed: int = 0) -> list[dict]:
    """Reset `env` and play one match in every slot of the batch, `team_a` controlling team A's
    agents and `team_b` team B's. Returns each match's episode info (see `MatchEnv.step`).
    To swap sides, call again with the policies exchanged."""
    rng = np.random.default_rng(seed)
    contexts = [(policy, PolicyContext(env, team_slots(env, team), rng)) for team, policy in enumerate((team_a, team_b))]
    discrete = np.zeros((env.n_matches, env.n_agents, len(DISCRETE_NVEC)), dtype=np.int64)
    continuous = np.zeros((env.n_matches, env.n_agents, CONTINUOUS_DIM))
    obs = env.reset()
    while True:
        for policy, ctx in contexts:
            m, a = ctx.slots.T
            d, c = policy.act(obs[m, a], ctx)
            if np.shape(d) != (len(m), len(DISCRETE_NVEC)) or np.shape(c) != (len(m), CONTINUOUS_DIM):
                raise ValueError(f"{policy.name} returned actions of the wrong shape")
            discrete[m, a], continuous[m, a] = d, c
        r = env.step(discrete, continuous)
        if r.done.all():
            return r.info
        obs = r.obs
