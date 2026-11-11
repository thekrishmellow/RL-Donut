"""What a scripted bot is allowed to know, and the aim it should take.

Shooting uses the same sight as an agent: a living enemy whose centre or a hitbox corner is
in line of sight. Movement may use the true position of the nearest living enemy, which is
what makes a bot a yardstick rather than a fair opponent. Aim is in world coordinates;
`controls_to_actions` turns it into each agent's own frame.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.actions import Controls, encode
from rl_donut.env.policy import PolicyContext
from rl_donut.env.raycast import segments
from rl_donut.env.visibility import target_points


@dataclass(frozen=True)
class Look:
    """One row per slot in `ctx.slots`."""
    enemy: np.ndarray       # agent index, or -1 when nothing is seen
    distance: np.ndarray    # tiles to that enemy; inf when none
    aim_x: np.ndarray       # world point to aim at (centre, or the first visible corner)
    aim_y: np.ndarray
    in_range: np.ndarray    # the aim point is visible and within rifle range


def look(ctx: PolicyContext) -> Look:
    """Nearest enemy each slot can see, and a point on it that a shot can reach."""
    w = ctx.env.world
    m, i = ctx.slots.T
    n = len(m)
    visible = ctx.env.vis.own[m, i]
    dist = np.where(visible, np.hypot(w.x[m] - w.x[m, i][:, None], w.y[m] - w.y[m, i][:, None]), np.inf)
    nearest = dist.argmin(axis=1)
    seen = np.isfinite(dist.min(axis=1))
    nearest = np.where(seen, nearest, 0)

    points = target_points(w.cfg)
    p = len(points)
    aim_x = w.x[m, nearest][:, None] + points[:, 0]
    aim_y = w.y[m, nearest][:, None] + points[:, 1]
    clear, _ = segments(w.grids, np.repeat(m, p), np.repeat(w.x[m, i], p), np.repeat(w.y[m, i], p),
                        aim_x.ravel(), aim_y.ravel())
    clear = clear.reshape(n, p) & seen[:, None]
    which = clear.argmax(axis=1)
    rows = np.arange(n)
    gap = np.hypot(aim_x[rows, which] - w.x[m, i], aim_y[rows, which] - w.y[m, i])
    return Look(enemy=np.where(seen, nearest, -1),
                distance=np.where(seen, dist[rows, nearest], np.inf),
                aim_x=aim_x[rows, which], aim_y=aim_y[rows, which],
                in_range=clear.any(axis=1) & (gap <= w.cfg.rifle.range))


def nearest_living(ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True position of the nearest living enemy, seen or not. Index is -1 when none lives."""
    w = ctx.env.world
    m, i = ctx.slots.T
    enemy = w.alive[m] & (w.team[None, :] != w.team[i][:, None])
    dist = np.where(enemy, np.hypot(w.x[m] - w.x[m, i][:, None], w.y[m] - w.y[m, i][:, None]), np.inf)
    nearest = dist.argmin(axis=1)
    found = np.isfinite(dist.min(axis=1))
    nearest = np.where(found, nearest, 0)
    rows = np.arange(len(m))
    return np.where(found, nearest, -1), w.x[m[rows], nearest], w.y[m[rows], nearest]


class Reaction:
    """How many decisions in a row the same enemy has been seen. A new match starts it over."""

    def __init__(self):
        self._episode: dict[tuple[int, int], int] = {}
        self._enemy: dict[tuple[int, int], int] = {}
        self._count: dict[tuple[int, int], int] = {}

    def seen_for(self, ctx: PolicyContext, seen: Look) -> np.ndarray:
        m, a = ctx.slots.T
        episode = ctx.env.world.episode
        out = np.zeros(len(m), dtype=np.int64)
        for k in range(len(m)):
            key = (int(m[k]), int(a[k]))
            now, enemy = int(episode[m[k]]), int(seen.enemy[k])
            if self._episode.get(key) != now or self._enemy.get(key) != enemy or enemy < 0:
                self._count[key] = 0
            self._episode[key], self._enemy[key] = now, enemy
            if enemy >= 0:
                self._count[key] += 1
            out[k] = self._count[key]
        return out


def controls_to_actions(ctx: PolicyContext, move, jet, aim, trigger) -> tuple[np.ndarray, np.ndarray]:
    """World-frame controls for `ctx.slots` -> actions in each agent's own frame."""
    mirror = ctx.env.mirror[ctx.slots[:, 1]]
    discrete, continuous = encode(Controls(np.asarray(move), np.asarray(jet), np.asarray(aim, dtype=np.float64),
                                           np.asarray(trigger)), mirror=mirror)
    return discrete, continuous.astype(np.float32)
