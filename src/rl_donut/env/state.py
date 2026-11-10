"""Engine state for many matches at once.

Every per-agent quantity is a numpy array of shape (matches, agents); every per-match
quantity has shape (matches,). Agents are ordered team A first, then team B, so agent
`i` and agent `i + team_size` are mirror partners.

Coordinates are continuous, in tiles: x grows to the right, y grows downward (like grid
rows), and an agent's position is the centre of its hitbox. Mirroring a position is
therefore `x -> width - x`. Aim angles are radians with direction (cos, sin) in these
coordinates, so 0 points right and pi/2 points down; mirroring an angle is `a -> pi - a`.
"""

from dataclasses import dataclass, field

import numpy as np

from rl_donut.env.config import EngineConfig
from rl_donut.mapgen import GeneratedMap, Spawn

TEAM_A, TEAM_B = 0, 1


def spawn_centre(spawn: Spawn, cfg: EngineConfig) -> tuple[float, float]:
    """Hitbox centre of an agent standing in a spawn's clearance box, feet on its floor."""
    clearance = cfg.map.regions
    return spawn.x + clearance.clearance_w / 2, spawn.y + clearance.clearance_h - cfg.hitbox_h / 2


def map_arrays(maps: list[GeneratedMap], cfg: EngineConfig):
    """Per-match map data: seeds, rock grids, kill lines and spawn hitbox centres."""
    shape = (cfg.map.height, cfg.map.width)
    if any(m.grid.shape != shape for m in maps):
        raise ValueError(f"every map must be {shape[0]} x {shape[1]} to share one batch")
    spawns = np.array([[[spawn_centre(s, cfg) for s in team]
                        for team in (m.spawns.team_a, m.spawns.team_b)] for m in maps])
    return (np.array([m.seed for m in maps]), np.stack([m.grid for m in maps]).astype(bool),
            np.array([m.kill_y for m in maps], dtype=np.float64), spawns)


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
    fuel: float
    alive: bool
    aim: float
    ammo: int
    cooldown: int
    reload: int
    health: float
    respawn: int
    protection: int


@dataclass
class World:
    cfg: EngineConfig
    seeds: np.ndarray       # (M,) map seed of each match
    grids: np.ndarray       # (M, H, W) bool, True = rock
    kill_y: np.ndarray      # (M,)
    spawns: np.ndarray      # (M, 2, per_team, 2) spawn hitbox centres as (x, y), per team
    tick: np.ndarray        # (M,) ticks played
    score: np.ndarray       # (M, 2) kills per team
    team: np.ndarray        # (A,) team of each agent slot
    x: np.ndarray           # (M, A)
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    fuel: np.ndarray        # (M, A) jetpack fuel units
    alive: np.ndarray       # (M, A) bool
    aim: np.ndarray         # (M, A) radians
    ammo: np.ndarray        # (M, A) rounds in the magazine
    cooldown: np.ndarray    # (M, A) ticks until the rifle can fire again
    reload: np.ndarray      # (M, A) ticks of reloading left; 0 = not reloading
    health: np.ndarray      # (M, A)
    since_damage: np.ndarray  # (M, A) ticks since last damaged; regeneration waits on it
    last_damager: np.ndarray  # (M, A) agent that last damaged this one, -1 = none since respawn
    respawn: np.ndarray     # (M, A) ticks until a dead agent respawns; 0 with alive False = out of play
    protection: np.ndarray  # (M, A) ticks of spawn protection left
    physics_scale: np.ndarray  # (M,) gravity and thrust multiplier of each match
    # Random numbers (rifle spread) come from one stream per match, seeded from
    # (rng_seed, map seed, episode), so a match never depends on the rest of its batch.
    # The physics scale has its own stream so turning it on leaves spread unchanged.
    rng_seed: int = 0
    episode: np.ndarray = field(default=None)   # (M,) episode number of each match, from 0
    rngs: list = field(default_factory=list)    # (M,) numpy Generators

    @classmethod
    def from_maps(cls, maps: list[GeneratedMap], cfg: EngineConfig | None = None, rng_seed: int = 0) -> "World":
        cfg = cfg or EngineConfig()
        if not maps:
            raise ValueError("need at least one map")
        seeds, grids, kill_y, spawns = map_arrays(maps, cfg)
        n, k = len(maps), cfg.n_agents
        world = cls(
            cfg=cfg, seeds=seeds, grids=grids, kill_y=kill_y, spawns=spawns,
            tick=np.zeros(n, dtype=np.int64), score=np.zeros((n, 2), dtype=np.int64),
            team=np.repeat([TEAM_A, TEAM_B], cfg.team_size),
            x=np.zeros((n, k)), y=np.zeros((n, k)), vx=np.zeros((n, k)), vy=np.zeros((n, k)),
            fuel=np.zeros((n, k)), alive=np.ones((n, k), dtype=bool),
            aim=np.zeros((n, k)), ammo=np.zeros((n, k), dtype=np.int64),
            cooldown=np.zeros((n, k), dtype=np.int64), reload=np.zeros((n, k), dtype=np.int64),
            health=np.zeros((n, k)), since_damage=np.zeros((n, k), dtype=np.int64),
            last_damager=np.zeros((n, k), dtype=np.int64), respawn=np.zeros((n, k), dtype=np.int64),
            protection=np.zeros((n, k), dtype=np.int64), physics_scale=np.ones(n),
            rng_seed=rng_seed, episode=np.full(n, -1, dtype=np.int64), rngs=[None] * n,
        )
        world.reset()
        return world

    @classmethod
    def single(cls, m: GeneratedMap, cfg: EngineConfig | None = None, rng_seed: int = 0) -> "World":
        return cls.from_maps([m], cfg, rng_seed)

    @property
    def n_matches(self) -> int:
        return len(self.seeds)

    def load_maps(self, matches, maps: list[GeneratedMap]) -> None:
        """Put new maps into the given matches and start them; the rest of the batch is untouched."""
        matches = np.asarray(matches)
        if len(matches) != len(maps):
            raise ValueError("need one map per match")
        if not len(maps):
            return
        self.seeds[matches], self.grids[matches], self.kill_y[matches], self.spawns[matches] = map_arrays(maps, self.cfg)
        self.episode[matches] = -1
        self.reset(matches)

    def reset(self, matches: np.ndarray | None = None) -> None:
        """Start the given matches (default: all) from tick 0, agent `i` of each team on spawn `i`.

        Team B's spawns mirror team A's, so starting positions are mirror images. Team A aims
        right and team B left; every rifle starts loaded and ready, every agent at full health
        and without spawn protection.
        """
        matches = np.arange(self.n_matches) if matches is None else np.asarray(matches)
        slot = np.arange(self.cfg.n_agents) % self.cfg.team_size
        start = self.spawns[matches[:, None], self.team[None, :], slot[None, :]]
        self.x[matches], self.y[matches] = start[..., 0], start[..., 1]
        self.vx[matches] = self.vy[matches] = 0.0
        self.fuel[matches] = self.cfg.physics.fuel_capacity
        self.alive[matches] = True
        self.aim[matches] = np.where(self.team == TEAM_A, 0.0, np.pi)
        self.ammo[matches] = self.cfg.rifle.magazine
        self.cooldown[matches] = self.reload[matches] = 0
        self.health[matches] = self.cfg.life.max_health
        self.since_damage[matches] = 0
        self.last_damager[matches] = -1
        self.respawn[matches] = self.protection[matches] = 0
        self.tick[matches] = 0
        self.score[matches] = 0
        lo, hi = self.cfg.physics.scale_range
        for m in matches:
            self.episode[m] += 1
            key = [self.rng_seed, int(self.seeds[m]), int(self.episode[m])]
            self.rngs[m] = np.random.default_rng(key)
            self.physics_scale[m] = lo if lo == hi else np.random.default_rng(key + [1]).uniform(lo, hi)

    def agent(self, match: int, agent: int) -> AgentView:
        return AgentView(match, agent, int(self.team[agent]), float(self.x[match, agent]),
                         float(self.y[match, agent]), float(self.vx[match, agent]),
                         float(self.vy[match, agent]), float(self.fuel[match, agent]),
                         bool(self.alive[match, agent]), float(self.aim[match, agent]),
                         int(self.ammo[match, agent]), int(self.cooldown[match, agent]),
                         int(self.reload[match, agent]), float(self.health[match, agent]),
                         int(self.respawn[match, agent]), int(self.protection[match, agent]))
