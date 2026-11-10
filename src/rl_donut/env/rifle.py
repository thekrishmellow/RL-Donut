"""Aiming and the hitscan rifle.

A shot has no travelling bullet: on the tick it is fired, a ray from the shooter's hitbox
centre hits the nearest enemy hitbox within range, unless rock is in the way. Rock is
judged by the map generator's own `line_of_sight` (M40), so "can't see" and "can't be
shot" use the same rule. Teammates are neither hit nor in the way.
"""

import math
from dataclasses import dataclass

import numpy as np

from rl_donut.env.state import World
from rl_donut.mapgen import line_of_sight


@dataclass(frozen=True)
class Shot:
    match: int
    shooter: int
    angle: float            # direction the shot left in, spread included
    target: int             # agent hit, or -1 for a miss
    hit: tuple[float, float] | None   # where the ray entered the target's hitbox


def ray_box(ox: float, oy: float, dx: float, dy: float,
            x0: float, y0: float, x1: float, y1: float) -> float | None:
    """Distance along the unit ray (ox, oy) + t (dx, dy), t >= 0, to where it enters the
    box [x0, x1] x [y0, y1]; 0 when it starts inside; None when it misses."""
    t_near, t_far = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
        if d == 0:
            if not lo <= o <= hi:
                return None
            continue
        a, b = (lo - o) / d, (hi - o) / d
        t_near, t_far = max(t_near, min(a, b)), min(t_far, max(a, b))
        if t_near > t_far:
            return None
    return t_near


def fire(world: World, aim: np.ndarray, trigger: np.ndarray) -> list[Shot]:
    """Advance every rifle by one tick and resolve the shots fired on it.

    `aim` (radians) and `trigger` (bool) have shape (matches, agents). Living agents turn to
    their aim instantly; dead agents neither turn nor fire. A rifle fires when the trigger is
    held, it isn't cooling down or reloading, and the magazine isn't empty; firing the last
    round starts the reload.
    """
    cfg, rifle = world.cfg, world.cfg.rifle
    alive = world.alive
    world.aim[...] = np.where(alive, aim, world.aim)

    world.cooldown[...] = np.maximum(world.cooldown - 1, 0)
    reloading = world.reload > 0
    world.reload[reloading] -= 1
    world.ammo[reloading & (world.reload == 0)] = rifle.magazine

    shooting = np.asarray(trigger, dtype=bool) & alive & (world.cooldown == 0) & (world.reload == 0) & (world.ammo > 0)
    world.ammo[shooting] -= 1
    world.cooldown[shooting] = cfg.shot_interval_ticks
    world.reload[shooting & (world.ammo == 0)] = cfg.reload_ticks

    spread = math.radians(rifle.spread_deg)
    hw, hh = cfg.hitbox_w / 2, cfg.hitbox_h / 2
    shots = []
    for m, i in zip(*np.nonzero(shooting)):
        m, i = int(m), int(i)
        angle = float(world.aim[m, i]) + (world.rngs[m].uniform(-spread, spread) if spread else 0.0)
        ox, oy, dx, dy = float(world.x[m, i]), float(world.y[m, i]), math.cos(angle), math.sin(angle)
        best, target = rifle.range, -1
        for j in np.nonzero(alive[m] & (world.team != world.team[i]))[0]:
            ex, ey = float(world.x[m, j]), float(world.y[m, j])
            t = ray_box(ox, oy, dx, dy, ex - hw, ey - hh, ex + hw, ey + hh)
            if t is not None and t <= best:
                best, target = t, int(j)
        hit = None
        if target >= 0:
            hit = (ox + best * dx, oy + best * dy)
            # Rock that hides the nearest enemy also hides everyone behind it on this ray.
            if not line_of_sight(world.grids[m], (ox, oy), hit):
                target, hit = -1, None
        shots.append(Shot(m, i, angle, target, hit))
    return shots
