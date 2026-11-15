"""Roam the map when the observation does not contain an enemy.

Sight in this game is line of sight from the body, so turning the gun does not
reveal anyone. The agent has to visit other floors. While no enemy is seen or
remembered, it flies a loop of the map's reachable floors with the same route
planner the rusher uses, and it does not shoot. The moment an enemy is in the
observation, the learned aim and trigger are left alone.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.bots import cached_navigator
from rl_donut.env.collision import on_ground
from rl_donut.env.navigation import RouteBot
from rl_donut.env.observe import FEATURE_NAMES
from rl_donut.env.perception import controls_to_actions
from rl_donut.env.policy import PolicyContext

# Used only when a map has a single floor and there is no route to fly.
PATROL_STEPS = 30
HOP_STEPS = 6

_ENEMY_FLAGS = tuple(
    i for i, name in enumerate(FEATURE_NAMES)
    if name.endswith(("_seen_by_me", "_seen_by_team", "_remembered"))
)


def enemy_in_view(obs: np.ndarray) -> np.ndarray:
    """(n,) bool. True when any enemy slot is seen or still remembered."""
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[-1] <= max(_ENEMY_FLAGS):
        return np.ones(obs.shape[0], dtype=bool)
    return obs[:, _ENEMY_FLAGS].max(axis=1) > 0.5


def _far_end(nav, segment: int, x: float) -> float | None:
    """Centre x of the standing spot on `segment` farthest from `x`."""
    spots = nav.main_spots()
    on_floor = spots[nav.segments[spots[:, 0], spots[:, 1]] == segment]
    if len(on_floor) == 0:
        return None
    centres = np.array([nav.standing_centre((int(r), int(c)))[0] for r, c in on_floor])
    return float(centres[np.argmax(np.abs(centres - x))])


def _floor_tour(nav) -> list[int]:
    """Main floors from left to right, so a loop walks the cave rather than one ledge."""
    spots = nav.main_spots()
    if len(spots) == 0:
        return []
    seg_at = nav.segments[spots[:, 0], spots[:, 1]]
    ids = [int(s) for s in np.unique(seg_at) if int(s) >= 0]

    def centre_x(segment: int) -> float:
        cols = spots[seg_at == segment][:, 1]
        return float(cols.mean()) if len(cols) else 0.0

    ids.sort(key=centre_x)
    return ids


@dataclass
class _Tour:
    episode: int
    index: int
    bot: RouteBot
    # After landing on a floor, walk to its far end before flying to the next one.
    scan_x: float | None = None


class MapRoamer:
    """One route per agent. A finished floor advances to the next floor in the tour."""

    def __init__(self):
        self._tours: dict[tuple[int, int], _Tour] = {}

    def apply(
        self,
        obs: np.ndarray,
        discrete: np.ndarray,
        continuous: np.ndarray,
        ctx: PolicyContext | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        discrete = np.array(discrete, copy=True)
        continuous = np.array(continuous, dtype=np.float32, copy=True)
        hidden = ~enemy_in_view(obs)
        if not np.any(hidden):
            return discrete, continuous
        if ctx is None:
            return _pace(discrete, continuous, hidden, 0)
        w = ctx.env.world
        m, i = ctx.slots.T
        grounded = on_ground(w)[m, i]
        move = np.zeros(len(m), dtype=np.int64)
        jet = np.zeros(len(m), dtype=bool)
        aim = np.zeros(len(m), dtype=np.float64)
        paced = np.zeros(len(m), dtype=bool)
        for k in np.nonzero(hidden)[0]:
            if not w.alive[m[k], i[k]]:
                continue
            step = self._toward_next_floor(
                int(m[k]), int(i[k]),
                float(w.x[m[k], i[k]]), float(w.y[m[k], i[k]]),
                float(w.vy[m[k], i[k]]), float(w.fuel[m[k], i[k]]),
                bool(grounded[k]), int(w.episode[m[k]]), w,
            )
            if step is None:
                paced[k] = True
                continue
            move[k], jet[k], aim[k] = step
        routed = hidden & ~paced
        if np.any(routed):
            encoded_d, encoded_c = controls_to_actions(
                ctx, move, jet, aim, np.zeros(len(m), dtype=bool)
            )
            discrete[routed] = encoded_d[routed]
            continuous[routed] = encoded_c[routed]
        if np.any(paced):
            tick = int(w.tick[m[0]]) // w.cfg.match.decision_ticks
            discrete, continuous = _pace(discrete, continuous, paced, tick)
        return discrete, continuous

    def _toward_next_floor(self, match, agent, x, y, vy, fuel, grounded, episode, world):
        nav = cached_navigator(world, match)
        floors = _floor_tour(nav)
        if len(floors) < 2:
            return None
        key = (match, agent)
        tour = self._tours.get(key)
        if tour is None or tour.episode != episode or tour.bot.nav is not nav:
            here = nav.standing_box(x, y)
            here_seg = None if here is None else int(nav.segments[here])
            index = 0
            if here_seg in floors:
                index = (floors.index(here_seg) + 1) % len(floors)
            tour = _Tour(episode, index, RouteBot(nav, floors[index]))
            self._tours[key] = tour
        if tour.scan_x is not None:
            if abs(x - tour.scan_x) < 1.0:
                tour.scan_x = None
                tour.index = (tour.index + 1) % len(floors)
                tour.bot = RouteBot(nav, floors[tour.index])
            else:
                step = 1 if tour.scan_x > x else -1
                return step, False, 0.0 if step > 0 else np.pi
        elif tour.bot.status != "flying":
            tour.scan_x = _far_end(nav, floors[tour.index], x)
            if tour.scan_x is None or abs(x - tour.scan_x) < 1.0:
                tour.scan_x = None
                tour.index = (tour.index + 1) % len(floors)
                tour.bot = RouteBot(nav, floors[tour.index])
        step, thrust = tour.bot.decide(x, y, vy, fuel, grounded)
        if tour.bot.status != "flying":
            tour.scan_x = _far_end(nav, floors[tour.index], x)
            if tour.scan_x is not None and abs(x - tour.scan_x) >= 1.0:
                step = 1 if tour.scan_x > x else -1
                return step, False, 0.0 if step > 0 else np.pi
            tour.scan_x = None
            tour.index = (tour.index + 1) % len(floors)
            tour.bot = RouteBot(nav, floors[tour.index])
            step, thrust = tour.bot.decide(x, y, vy, fuel, grounded)
        if step > 0:
            aim = 0.0
        elif step < 0:
            aim = np.pi
        elif thrust:
            aim = -np.pi / 2
        else:
            aim = 0.0
        return int(step), bool(thrust), aim


def _pace(discrete, continuous, hidden, step: int):
    """Walk back and forth when the map has nowhere else to fly."""
    leg = step // PATROL_STEPS
    discrete = np.array(discrete, copy=True)
    discrete[hidden, 0] = 2 if leg % 2 else 0
    discrete[hidden, 1] = int((step % PATROL_STEPS) < HOP_STEPS)
    discrete[hidden, 2] = 0
    return discrete, continuous


def circulate(
    obs: np.ndarray,
    discrete: np.ndarray,
    continuous: np.ndarray,
    ctx: PolicyContext | None = None,
    roamer: MapRoamer | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roam where the enemy is absent. `roamer` keeps the route from one decision to the next."""
    owner = roamer if roamer is not None else MapRoamer()
    return owner.apply(obs, discrete, continuous, ctx)
