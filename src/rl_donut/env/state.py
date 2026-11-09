"""Engine state for many matches at once.

Every per-agent quantity is a numpy array of shape (matches, agents); every per-match
quantity has shape (matches,). Agents are ordered team A first, then team B, so agent
`i` and agent `i + team_size` are mirror partners.

Coordinates are continuous, in tiles: x grows to the right, y grows downward (like grid
rows), and an agent's position is the centre of its hitbox. Mirroring a position is
therefore `x -> width - x`.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.config import EngineConfig
from rl_donut.mapgen import GeneratedMap, Spawn

TEAM_A, TEAM_B = 0, 1


def spawn_centre(spawn: Spawn, cfg: EngineConfig) -> tuple[float, float]:
    """Hitbox centre of an agent standing in a spawn's clearance box, feet on its floor."""
    clearance = cfg.map.regions
    return spawn.x + clearance.clearance_w / 2, spawn.y + clearance.clearance_h - cfg.hitbox_h / 2


@dataclass(frozen=True)
class AgentView:
    """One agent's state, for tests and debugging."""
    match: int
    agent: int
    team: int
    x: float
    y: float
    vx: float
    vy: float
    alive: bool


@dataclass
class World:
    cfg: EngineConfig
    seeds: np.ndarray       # (M,) map seed of each match
    grids: np.ndarray       # (M, H, W) bool, True = rock
    kill_y: np.ndarray      # (M,)
    spawns: np.ndarray      # (M, 2, per_team, 2) spawn hitbox centres as (x, y), per team
    tick: np.ndarray        # (M,) ticks played
    team: np.ndarray        # (A,) team of each agent slot
    x: np.ndarray           # (M, A)
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    alive: np.ndarray       # (M, A) bool

    @classmethod
    def from_maps(cls, maps: list[GeneratedMap], cfg: EngineConfig | None = None) -> "World":
        cfg = cfg or EngineConfig()
        if not maps:
            raise ValueError("need at least one map")
        shape = (cfg.map.height, cfg.map.width)
        if any(m.grid.shape != shape for m in maps):
            raise ValueError(f"every map must be {shape[0]} x {shape[1]} to share one batch")
        spawns = np.array([[[spawn_centre(s, cfg) for s in team]
                            for team in (m.spawns.team_a, m.spawns.team_b)] for m in maps])
        n, k = len(maps), cfg.n_agents
        world = cls(
            cfg=cfg,
            seeds=np.array([m.seed for m in maps]),
            grids=np.stack([m.grid for m in maps]).astype(bool),
            kill_y=np.array([m.kill_y for m in maps], dtype=np.float64),
            spawns=spawns,
            tick=np.zeros(n, dtype=np.int64),
            team=np.repeat([TEAM_A, TEAM_B], cfg.team_size),
            x=np.zeros((n, k)), y=np.zeros((n, k)), vx=np.zeros((n, k)), vy=np.zeros((n, k)),
            alive=np.ones((n, k), dtype=bool),
        )
        world.reset()
        return world

    @classmethod
    def single(cls, m: GeneratedMap, cfg: EngineConfig | None = None) -> "World":
        return cls.from_maps([m], cfg)

    @property
    def n_matches(self) -> int:
        return len(self.seeds)

    def reset(self, matches: np.ndarray | None = None) -> None:
        """Start the given matches (default: all) from tick 0, agent `i` of each team on spawn `i`.

        Team B's spawns mirror team A's, so starting positions are mirror images.
        """
        matches = np.arange(self.n_matches) if matches is None else np.asarray(matches)
        slot = np.arange(self.cfg.n_agents) % self.cfg.team_size
        start = self.spawns[matches[:, None], self.team[None, :], slot[None, :]]
        self.x[matches], self.y[matches] = start[..., 0], start[..., 1]
        self.vx[matches] = self.vy[matches] = 0.0
        self.alive[matches] = True
        self.tick[matches] = 0

    def agent(self, match: int, agent: int) -> AgentView:
        return AgentView(match, agent, int(self.team[agent]), float(self.x[match, agent]),
                         float(self.y[match, agent]), float(self.vx[match, agent]),
                         float(self.vy[match, agent]), bool(self.alive[match, agent]))
