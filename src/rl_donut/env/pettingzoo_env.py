"""PettingZoo `ParallelEnv` over one match, for outside tools (training uses the batched `MatchEnv`).

Agents are named `a_0`, `b_0` (and `a_1`, `b_1` in 2v2). Observations and actions are in each
agent's own frame, as in `MatchEnv`: team B's are mirrored. One match is one episode: every
agent is truncated when the timer runs out, and none is ever terminated, because deaths
respawn. Importing this module needs `pettingzoo` and `gymnasium`; `rl_donut.env` itself
doesn't import it.
"""

import functools

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv

from rl_donut.env.actions import DISCRETE_NVEC
from rl_donut.env.config import EngineConfig
from rl_donut.env.match import MatchEvents
from rl_donut.env.match_env import STATE_DIM, MatchEnv
from rl_donut.env.observe import OBS_DIM
from rl_donut.mapgen import load_split


def kills_minus_deaths(events: MatchEvents) -> np.ndarray:
    """Placeholder reward until Phase 5's: +1 per kill (pit falls credited) and -1 per death."""
    return events.kills - events.deaths


@functools.cache
def _split(name: str):
    return load_split(name)


class RLDonutParallelEnv(ParallelEnv):
    metadata = {"name": "rl_donut_v0", "render_modes": []}

    def __init__(self, team_size: int = 1, split: str = "train", reward_fn=kills_minus_deaths,
                 cfg: EngineConfig | None = None, maps=None):
        """`reward_fn(events)` gets the step's `MatchEvents` (a batch of one match) and returns
        one reward per agent, in `MatchEnv` agent order (team A first)."""
        self.team_size, self.reward_fn, self.cfg = team_size, reward_fn, cfg
        self.maps = list(maps) if maps is not None else _split(split)
        self.possible_agents = [f"{side}_{i}" for side in ("a", "b") for i in range(team_size)]
        self.agents = []
        self.render_mode = None
        self.env = None
        self._next_obs = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gymnasium.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gymnasium.spaces.Dict({"discrete": gymnasium.spaces.MultiDiscrete(DISCRETE_NVEC),
                                      "aim": gymnasium.spaces.Box(-np.pi, np.pi, (1,), np.float64)})

    @functools.cached_property
    def state_space(self):
        return gymnasium.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)

    def reset(self, seed=None, options=None):
        """A seed restarts the map deck and random streams; without one, the next map is dealt."""
        if seed is not None or self.env is None:
            self.env = MatchEnv(1, self.team_size, seed=0 if seed is None else seed, cfg=self.cfg, maps=self.maps)
            obs = self.env.reset()[0]
        elif self._next_obs is not None:
            obs = self._next_obs    # MatchEnv already started the next match when the last one ended
        else:
            obs = self.env.reset()[0]
        self._next_obs = None
        self.agents = list(self.possible_agents)
        return {agent: obs[i] for i, agent in enumerate(self.agents)}, {agent: {} for agent in self.agents}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("the match is over (or never started); call reset()")
        if set(actions) != set(self.agents):
            raise ValueError(f"need one action for each of {self.agents}")
        discrete = np.stack([np.asarray(actions[a]["discrete"]) for a in self.possible_agents])[None]
        aim = np.stack([np.reshape(actions[a]["aim"], 1) for a in self.possible_agents])[None]
        r = self.env.step(discrete, aim)
        rewards = np.asarray(self.reward_fn(r.events), dtype=np.float64).reshape(len(self.agents))
        done = bool(r.done[0])
        names = self.agents
        obs = {agent: r.final_obs[0, i] for i, agent in enumerate(names)}
        infos = {agent: {} for agent in names}
        if done:
            infos = {agent: self._episode_info(r.info[0], i) for i, agent in enumerate(names)}
            self._next_obs = r.obs[0]
            self.agents = []
        return (obs, {agent: float(rewards[i]) for i, agent in enumerate(names)},
                dict.fromkeys(names, False), dict.fromkeys(names, done), infos)

    def state(self) -> np.ndarray:
        """The true state in team A's frame (`MatchEnv.get_state()[:, 0]`), not limited by sight."""
        return self.env.get_state()[0, 0]

    def _episode_info(self, info: dict, i: int) -> dict:
        """Score and outcome from this agent's team's side, and its episode stats."""
        team = int(self.env.team[i])
        score = info["score"]
        return {"map_seed": info["map_seed"], "team": team, "score": (int(score[team]), int(score[1 - team])),
                "outcome": int(info["outcome"][team]),
                "stats": {name: float(v[i]) for name, v in info["agents"].items()}}
