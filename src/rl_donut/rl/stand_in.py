"""Stand-in environment matching the Phase 3/4 MatchEnv contract.

This environment lets Team B develop Phase 5 (rewards, PPO, curriculum) and Phase 6
(self-play, pool, Elo) in parallel while Team A finishes the full engine and interface.
At the end of Phase 4, MatchEnv replaces StandInEnv with zero changes to training code.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Protocol

import numpy as np

from rl_donut.env.config import EngineConfig
from rl_donut.env.collision import on_ground
from rl_donut.env.physics import physics_tick
from rl_donut.env.state import World, TEAM_A, TEAM_B
from rl_donut.mapgen import GeneratedMap, MapConfig, load_split
from rl_donut.mapgen.geometry import line_of_sight

SPEC_VERSION = "0.1.0-stand-in"


@dataclass
class Events:
    """Raw events on one decision step, shape (M, A) unless noted."""
    damage_dealt: np.ndarray     # (M, A) float32
    damage_taken: np.ndarray     # (M, A) float32
    kills: np.ndarray            # (M, A) int64
    deaths: np.ndarray           # (M, A) int64 (0 or 1)
    fell: np.ndarray             # (M, A) bool
    shots: np.ndarray            # (M, A) int64
    hits: np.ndarray             # (M, A) int64
    outcome: np.ndarray          # (M, 2) int64: +1 win, -1 loss, 0 draw per team; only non-zero when done


@dataclass
class StepResult:
    obs: np.ndarray              # (M, A, obs_dim) float32
    events: Events
    done: np.ndarray             # (M,) bool
    final_obs: np.ndarray        # (M, A, obs_dim) float32 (valid where done is True)
    info: list[dict[str, Any]]   # per match summary stats


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray, ctx: "PolicyContext") -> np.ndarray:
        ...


@dataclass
class PolicyContext:
    env: Any
    slots: np.ndarray            # (N, 2) array of (match_idx, agent_idx)
    rng: np.random.Generator


def compute_observations(world: World, health: np.ndarray, ticks_left: np.ndarray,
                         max_match_ticks: int, n_rays: int = 16, ray_range: float = 20.0) -> np.ndarray:
    """Extract egocentric observation vectors for all agents, shape (M, A, obs_dim) float32."""
    n_matches = world.n_matches
    n_agents = world.cfg.n_agents
    width, height = world.cfg.map.width, world.cfg.map.height
    p = world.cfg.physics

    grounded = on_ground(world).astype(np.float32)
    health_norm = (health / 100.0).astype(np.float32)
    fuel_norm = (world.fuel / p.fuel_capacity).astype(np.float32)
    vx_norm = (world.vx / p.run_speed).astype(np.float32)
    vy_norm = (world.vy / p.max_fall_speed).astype(np.float32)
    x_norm = (world.x / width).astype(np.float32)
    y_norm = (world.y / height).astype(np.float32)
    time_norm = np.broadcast_to((ticks_left / max_match_ticks)[:, None], (n_matches, n_agents)).astype(np.float32)

    # 16 rays around the agent checking distance to walls
    ray_angles = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
    ray_cos = np.cos(ray_angles)
    ray_sin = np.sin(ray_angles)
    ray_obs = np.ones((n_matches, n_agents, n_rays), dtype=np.float32)

    for m in range(n_matches):
        grid = world.grids[m]
        for a in range(n_agents):
            if not world.alive[m, a]:
                continue
            ax, ay = world.x[m, a], world.y[m, a]
            for r in range(n_rays):
                for dist in range(1, int(ray_range) + 1):
                    tx, ty = ax + dist * ray_cos[r], ay + dist * ray_sin[r]
                    col, row = int(math.floor(tx)), int(math.floor(ty))
                    if not (0 <= col < width and 0 <= row < height) or grid[row, col]:
                        ray_obs[m, a, r] = dist / ray_range
                        break

    # Enemy relative positions (up to 2 enemy slots)
    max_enemies = 2
    enemy_obs = np.zeros((n_matches, n_agents, max_enemies, 4), dtype=np.float32)
    for m in range(n_matches):
        for a in range(n_agents):
            team_a = world.team[a]
            enemies = [e for e in range(n_agents) if world.team[e] != team_a]
            for idx, e in enumerate(enemies[:max_enemies]):
                if world.alive[m, e] and world.alive[m, a]:
                    dx = world.x[m, e] - world.x[m, a]
                    dy = world.y[m, e] - world.y[m, a]
                    dist = math.hypot(dx, dy)
                    visible = line_of_sight(world.grids[m], (world.x[m, a], world.y[m, a]),
                                            (world.x[m, e], world.y[m, e]))
                    if visible:
                        enemy_obs[m, a, idx] = [1.0, dx / 30.0, dy / 30.0, health[m, e] / 100.0]

    # Flatten observation parts: 8 self + n_rays + max_enemies * 4
    self_features = np.stack([
        health_norm, fuel_norm, vx_norm, vy_norm, grounded, x_norm, y_norm, time_norm
    ], axis=-1)
    enemy_flat = enemy_obs.reshape(n_matches, n_agents, max_enemies * 4)
    obs = np.concatenate([self_features, ray_obs, enemy_flat], axis=-1).astype(np.float32)
    return obs


class StandInEnv:
    """Batched environment satisfying Section 4 of docs/handoff_phase5_6.md."""

    def __init__(
        self,
        n_matches: int = 4,
        team_size: int = 1,
        split: str = "train",
        seed: int = 0,
        map_filter: Callable[[GeneratedMap], bool] | None = None,
        cfg: EngineConfig | None = None,
        match_duration_seconds: float = 60.0,
        decision_interval_ticks: int = 2,
    ):
        self.spec_version = SPEC_VERSION
        self.n_matches = n_matches
        self.team_size = team_size
        self.n_agents = 2 * team_size
        self.team = np.repeat([TEAM_A, TEAM_B], team_size)
        self.split_name = split
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or EngineConfig(team_size=team_size)
        self.decision_interval = decision_interval_ticks
        self.match_duration_ticks = int(match_duration_seconds * self.cfg.tick_rate)
        self.action_nvec = (3, 2, 16, 2)  # move (3), jet (2), aim (16), shoot (2)

        # Pre-load split maps
        all_maps = load_split(split)
        if map_filter is not None:
            all_maps = [m for m in all_maps if map_filter(m)]
        if not all_maps:
            raise ValueError(f"no maps matched filter in split {split}")
        self.maps = all_maps
        self.map_filter = map_filter

        # Compute obs dimension
        n_rays = 16
        self.obs_dim = 8 + n_rays + 2 * 4  # self (8) + rays (16) + 2 enemies * 4 = 32

        # Episode stats trackers
        self.ticks_played = np.zeros(n_matches, dtype=np.int64)
        self.health = np.full((n_matches, self.n_agents), 100.0, dtype=np.float32)
        self.current_maps = self._sample_maps(n_matches)
        self.world = World.from_maps(self.current_maps, self.cfg)

    def _sample_maps(self, count: int) -> list[GeneratedMap]:
        indices = self.rng.choice(len(self.maps), size=count, replace=True)
        return [self.maps[i] for i in indices]

    def reset(self) -> np.ndarray:
        """Reset all matches onto newly drawn maps from split."""
        self.current_maps = self._sample_maps(self.n_matches)
        self.world = World.from_maps(self.current_maps, self.cfg)
        self.world.reset()
        self.ticks_played[:] = 0
        self.health[:] = 100.0
        return compute_observations(
            self.world, self.health, self.match_duration_ticks - self.ticks_played,
            self.match_duration_ticks
        )

    def get_state(self) -> np.ndarray:
        """Global state representation for centralized critic in 2v2."""
        # Flatten all agent positions, velocities, health, and match time
        M, A = self.n_matches, self.n_agents
        state = np.concatenate([
            self.world.x / self.cfg.map.width,
            self.world.y / self.cfg.map.height,
            self.world.vx / self.cfg.physics.run_speed,
            self.world.vy / self.cfg.physics.max_fall_speed,
            self.health / 100.0,
            (self.match_duration_ticks - self.ticks_played)[:, None] / self.match_duration_ticks
        ], axis=-1).astype(np.float32)
        return state

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance all matches by 1 decision step (decision_interval physics ticks)."""
        actions = np.asarray(actions, dtype=np.int64)
        M, A = self.n_matches, self.n_agents
        assert actions.shape == (M, A, len(self.action_nvec)), f"unexpected action shape: {actions.shape}"

        move_action = actions[..., 0] - 1  # 0,1,2 -> -1, 0, 1
        jet_action = actions[..., 1].astype(bool)
        aim_action = actions[..., 2]
        shoot_action = actions[..., 3].astype(bool)

        aim_rad = aim_action * (2 * math.pi / self.action_nvec[2])

        # Event accumulators for this decision step
        damage_dealt = np.zeros((M, A), dtype=np.float32)
        damage_taken = np.zeros((M, A), dtype=np.float32)
        kills = np.zeros((M, A), dtype=np.int64)
        deaths = np.zeros((M, A), dtype=np.int64)
        fell = np.zeros((M, A), dtype=bool)
        shots = np.zeros((M, A), dtype=np.int64)
        hits = np.zeros((M, A), dtype=np.int64)

        for _ in range(self.decision_interval):
            # 1. Physics tick
            physics_tick(self.world, move_action, jet_action)
            self.ticks_played += 1

            # 2. Falls through pits below kill_y
            below = (self.world.y >= self.world.kill_y[:, None]) & self.world.alive
            if below.any():
                deaths[below] += 1
                fell[below] = True
                self.world.alive[below] = False
                self.health[below] = 0.0

            # 3. Simple hitscan shooting simulation
            firing = shoot_action & self.world.alive
            shots[firing] += 1

            for m in range(M):
                for a in range(A):
                    if not firing[m, a]:
                        continue
                    theta = aim_rad[m, a]
                    ax, ay = self.world.x[m, a], self.world.y[m, a]
                    target_x = ax + 30.0 * math.cos(theta)
                    target_y = ay + 30.0 * math.sin(theta)

                    # Find nearest enemy intersected
                    team_a = self.world.team[a]
                    best_enemy = None
                    best_dist = 30.0

                    for e in range(A):
                        if self.world.team[e] == team_a or not self.world.alive[m, e]:
                            continue
                        ex, ey = self.world.x[m, e], self.world.y[m, e]
                        # Rough point-to-line segment distance
                        dx, dy = ex - ax, ey - ay
                        proj = dx * math.cos(theta) + dy * math.sin(theta)
                        if 0 < proj < best_dist:
                            perp = abs(-dx * math.sin(theta) + dy * math.cos(theta))
                            if perp < max(self.cfg.hitbox_w, self.cfg.hitbox_h):
                                if line_of_sight(self.world.grids[m], (ax, ay), (ex, ey)):
                                    best_dist = proj
                                    best_enemy = e

                    if best_enemy is not None:
                        dmg = min(15.0, self.health[m, best_enemy])
                        damage_dealt[m, a] += dmg
                        damage_taken[m, best_enemy] += dmg
                        hits[m, a] += 1
                        self.health[m, best_enemy] -= dmg
                        if self.health[m, best_enemy] <= 0:
                            self.world.alive[m, best_enemy] = False
                            deaths[m, best_enemy] += 1
                            kills[m, a] += 1

            # Instant respawn for dead agents in stand-in
            dead = ~self.world.alive
            if dead.any():
                for m in range(M):
                    for a in range(A):
                        if dead[m, a]:
                            # Respawn on own spawn
                            slot = a % self.team_size
                            sx = self.world.spawns[m, self.world.team[a], slot, 0]
                            sy = self.world.spawns[m, self.world.team[a], slot, 1]
                            self.world.x[m, a] = sx
                            self.world.y[m, a] = sy
                            self.world.vx[m, a] = 0.0
                            self.world.vy[m, a] = 0.0
                            self.world.fuel[m, a] = self.cfg.physics.fuel_capacity
                            self.world.alive[m, a] = True
                            self.health[m, a] = 100.0

        # Check match completion (time limit)
        done = self.ticks_played >= self.match_duration_ticks

        outcome = np.zeros((M, 2), dtype=np.int64)
        if done.any():
            for m in np.nonzero(done)[0]:
                team_kills = [kills[m, self.world.team == TEAM_A].sum(),
                              kills[m, self.world.team == TEAM_B].sum()]
                if team_kills[0] > team_kills[1]:
                    outcome[m] = [1, -1]
                elif team_kills[1] > team_kills[0]:
                    outcome[m] = [-1, 1]
                else:
                    outcome[m] = [0, 0]

        events = Events(
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            kills=kills,
            deaths=deaths,
            fell=fell,
            shots=shots,
            hits=hits,
            outcome=outcome,
        )

        # Observations before any auto-reset
        final_obs = compute_observations(
            self.world, self.health, np.zeros(M, dtype=np.int64), self.match_duration_ticks
        )

        info: list[dict[str, Any]] = [{} for _ in range(M)]

        # Auto-reset completed matches onto new maps
        if done.any():
            ended_indices = np.nonzero(done)[0]
            new_maps = self._sample_maps(len(ended_indices))
            for i, m in enumerate(ended_indices):
                info[m] = {
                    "match_ticks": int(self.ticks_played[m]),
                    "kills_team_a": int(kills[m, self.world.team == TEAM_A].sum()),
                    "kills_team_b": int(kills[m, self.world.team == TEAM_B].sum()),
                }
                self.current_maps[m] = new_maps[i]
                self.world.grids[m] = new_maps[i].grid
                self.world.kill_y[m] = new_maps[i].kill_y
                for team in (0, 1):
                    for slot in range(self.team_size):
                        sp = new_maps[i].spawns.team_a[slot] if team == 0 else new_maps[i].spawns.team_b[slot]
                        self.world.spawns[m, team, slot] = [sp.x + 1.0, sp.y + 2.2]
                self.ticks_played[m] = 0
                self.health[m] = 100.0
            self.world.reset(ended_indices)

        next_obs = compute_observations(
            self.world, self.health, self.match_duration_ticks - self.ticks_played,
            self.match_duration_ticks
        )

        return StepResult(
            obs=next_obs,
            events=events,
            done=done,
            final_obs=final_obs,
            info=info,
        )
