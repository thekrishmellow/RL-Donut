"""Scripted bots: the fixed yardstick that self-play ratings can't inflate.

Each follows the `Policy` interface. Actions are in each agent's own frame, where the enemy
side is always to the right (aim 0). `TurretBot` holds still and shoots what it can see.
`RusherBot` flies toward the nearest living enemy, using the map generator's own route
planner, and shoots on the way.
"""

from types import SimpleNamespace

import numpy as np

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC
from rl_donut.env.collision import on_ground
from rl_donut.env.navigation import Navigator, RouteBot
from rl_donut.env.perception import Reaction, controls_to_actions, look, nearest_living
from rl_donut.env.policy import PolicyContext

NO_MOVE = 1
# Rebuilt flight plans are expensive, and a map seed never changes its grid.
_NAVIGATORS: dict[tuple, Navigator] = {}


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


def cached_navigator(world, match: int) -> Navigator:
    """The flight planner for one match's map, built once per seed."""
    cfg = world.cfg
    seed = int(world.seeds[match])
    key = (seed, hash(world.grids[match].tobytes()), cfg.hitbox_w, cfg.hitbox_h, cfg.map.climb_budget)
    nav = _NAVIGATORS.get(key)
    if nav is None:
        nav = Navigator(SimpleNamespace(seed=seed, grid=world.grids[match].copy()), cfg)
        _NAVIGATORS[key] = nav
    return nav


def floor_segment(nav: Navigator, x: float, y: float) -> int | None:
    """Reachable floor segment under (x, y), or the nearest standing spot's segment."""
    grid = nav.map.grid
    height, width = grid.shape
    col = int(np.clip(np.floor(x), 0, width - 1))
    row = int(np.clip(np.floor(y), 0, height - 1))
    for r in range(row, height):
        if grid[r, col]:
            box = nav.standing_box(x, r - nav.cfg.hitbox_h / 2 - 1e-3)
            if box is not None and nav.segments[box] >= 0:
                return int(nav.segments[box])
            break
    spots = nav.main_spots()
    if not len(spots):
        return None
    centres_x = spots[:, 1] + nav.box_w / 2
    centres_y = spots[:, 0] + nav.box_h - nav.cfg.hitbox_h / 2
    k = int(np.argmin(np.abs(centres_x - x) + 0.25 * np.abs(centres_y - y)))
    return int(nav.segments[tuple(spots[k])])


class _Shooter:
    """Shared aim: a visible point plus Gaussian error, fired only after `reaction` sights."""

    def __init__(self, aim_sigma_deg: float, reaction: int):
        self.sigma = np.radians(aim_sigma_deg)
        self.reaction_needed = reaction
        self.reaction = Reaction()

    def aim(self, ctx: PolicyContext):
        seen = look(ctx)
        held = self.reaction.seen_for(ctx, seen)
        w = ctx.env.world
        m, i = ctx.slots.T
        n = len(m)
        aim = np.zeros(n)
        visible = seen.enemy >= 0
        aim[visible] = np.arctan2(seen.aim_y[visible] - w.y[m[visible], i[visible]],
                                  seen.aim_x[visible] - w.x[m[visible], i[visible]])
        if self.sigma:
            aim = aim + ctx.rng.normal(0, self.sigma, n)
        trigger = visible & seen.in_range & (held >= self.reaction_needed)
        return seen, aim, trigger


class TurretBot(_Shooter):
    """Stands its ground and shoots enemies that come into sight."""

    name = "turret"

    def __init__(self, aim_sigma_deg: float = 2.0, reaction: int = 2):
        super().__init__(aim_sigma_deg, reaction)

    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        _, aim, trigger = self.aim(ctx)
        n = len(obs)
        return controls_to_actions(ctx, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool), aim, trigger)


class RusherBot(_Shooter):
    """Flies toward the nearest living enemy and shoots it once it is in sight."""

    name = "rusher"

    def __init__(self, aim_sigma_deg: float = 2.0, reaction: int = 1):
        super().__init__(aim_sigma_deg, reaction)
        self._routes: dict[tuple[int, int], tuple[int, RouteBot]] = {}

    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        w = ctx.env.world
        m, i = ctx.slots.T
        n = len(m)
        grounded = on_ground(w)[m, i]
        enemy, ex, ey = nearest_living(ctx)
        _, aim, trigger = self.aim(ctx)
        move = np.zeros(n, dtype=np.int64)
        jet = np.zeros(n, dtype=bool)
        for k in range(n):
            if enemy[k] < 0 or not w.alive[m[k], i[k]]:
                continue
            move[k], jet[k] = self._chase(ctx, k, w, m[k], i[k], float(ex[k]), float(ey[k]), bool(grounded[k]))
        return controls_to_actions(ctx, move, jet, aim, trigger)

    def _chase(self, ctx, k, w, match, agent, ex, ey, grounded) -> tuple[int, bool]:
        x, y = float(w.x[match, agent]), float(w.y[match, agent])
        nav = cached_navigator(w, match)
        goal = floor_segment(nav, ex, ey)
        here = nav.standing_box(x, y)
        here_seg = None if here is None else int(nav.segments[here])
        toward = 0 if abs(ex - x) < 0.4 else (1 if ex > x else -1)
        if goal is None or here_seg == goal:
            return toward, False
        bot = self._planner(match, agent, int(w.episode[match]), nav, goal)
        step, thrust = bot.decide(x, y, float(w.vy[match, agent]), float(w.fuel[match, agent]), grounded)
        if bot.status == "flying":
            return int(step), bool(thrust)
        return toward, y > ey + 1.5 and float(w.fuel[match, agent]) > 10

    def _planner(self, match, agent, episode, nav, goal) -> RouteBot:
        key = (int(match), int(agent))
        stored = self._routes.get(key)
        if stored is None or stored[0] != episode:
            bot = RouteBot(nav, goal)
            self._routes[key] = (episode, bot)
            return bot
        bot = stored[1]
        if bot.goal != goal:
            bot.goal, bot.status, bot.waypoints, bot.segment = goal, "flying", [], -1
        return bot
