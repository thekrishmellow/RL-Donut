"""Moving agent hitboxes against the tile grid, for every match at once.

Motion is resolved one axis at a time, horizontal first: the box moves, and if its leading
edge entered a rock tile it snaps back to that tile's edge and its speed on that axis
becomes 0. Because no box moves a full tile in one tick, the leading edge can reach at most
one new column (or row) per tick, so checking that one line of tiles is exact.

Edges that only touch do not overlap. Snapped positions such as `21 - 0.4` are not exact in
floating point, so tile ranges are computed with a tolerance of EPS tiles.

Outside the map, the left side, right side and top are solid. Below the bottom edge is air:
the bottom border row is solid except at pits, so pits are the only way down there.
"""

import numpy as np

from rl_donut.env.state import World

EPS = 1e-9
# Feet within this distance of a tile edge count as resting on it.
GROUND_TOLERANCE = 1e-6


def tile_solid(grids: np.ndarray, match: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Whether tile (row, col) of each match is solid, applying the outside-the-map rules."""
    height, width = grids.shape[1:]
    in_cols = (col >= 0) & (col < width)
    inside = in_cols & (row >= 0) & (row < height)
    below = in_cols & (row >= height)
    value = grids[match, np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)]
    return np.where(inside, value, ~below)


def span(centre: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    """First and last tile index a box of half-size `half` overlaps along one axis."""
    return np.floor(centre - half + EPS).astype(np.int64), np.floor(centre + half - EPS).astype(np.int64)


def line_blocked(world: World, match: np.ndarray, fixed: np.ndarray, first: np.ndarray, last: np.ndarray,
                 horizontal: bool, count: int) -> np.ndarray:
    """Whether any tile from `first` to `last` along a column (horizontal move) or row is solid."""
    blocked = np.zeros(fixed.shape, dtype=bool)
    for k in range(count):
        other = first + k
        row, col = (other, fixed) if horizontal else (fixed, other)
        blocked |= (other <= last) & tile_solid(world.grids, match, row, col)
    return blocked


def move_axis(world: World, delta: np.ndarray, horizontal: bool, active: np.ndarray) -> None:
    cfg = world.cfg
    half_move, half_side = (cfg.hitbox_w / 2, cfg.hitbox_h / 2) if horizontal else (cfg.hitbox_h / 2, cfg.hitbox_w / 2)
    pos, side, vel = (world.x, world.y, world.vx) if horizontal else (world.y, world.x, world.vy)
    match = np.broadcast_to(np.arange(world.n_matches)[:, None], pos.shape)
    delta = np.where(active, delta, 0.0)

    target = pos + delta
    forward = delta > 0
    edge = np.where(forward, target + half_move, target - half_move)
    line = np.where(forward, np.floor(edge - EPS), np.floor(edge + EPS)).astype(np.int64)
    first, last = span(side, half_side)
    count = int(np.ceil(2 * half_side)) + 1
    hit = (delta != 0) & line_blocked(world, match, line, first, last, horizontal, count)

    snapped = np.where(forward, line - half_move, line + 1 + half_move)
    pos[...] = np.where(hit, snapped, target)
    vel[hit] = 0.0


def move(world: World, dt: float) -> None:
    """Move every living agent by its velocity for `dt` seconds, horizontal then vertical."""
    dx, dy = world.vx * dt, world.vy * dt
    active = world.alive
    if (np.abs(dx[active]) >= 1).any() or (np.abs(dy[active]) >= 1).any():
        raise ValueError("an agent moved a full tile in one tick; collision would miss walls")
    move_axis(world, dx, horizontal=True, active=active)
    move_axis(world, dy, horizontal=False, active=active)


def on_ground(world: World) -> np.ndarray:
    """(matches, agents) bool: feet rest on a tile edge with rock directly beneath the hitbox."""
    cfg = world.cfg
    feet = world.y + cfg.hitbox_h / 2
    row = np.round(feet).astype(np.int64)
    at_edge = np.abs(feet - row) < GROUND_TOLERANCE
    match = np.broadcast_to(np.arange(world.n_matches)[:, None], feet.shape)
    first, last = span(world.x, cfg.hitbox_w / 2)
    supported = line_blocked(world, match, row, first, last, horizontal=False,
                             count=int(np.ceil(cfg.hitbox_w)) + 1)
    return world.alive & at_edge & supported


def overlaps_rock(world: World) -> np.ndarray:
    """(matches, agents) bool: any tile under any part of the hitbox is solid. Checks every tile."""
    cfg = world.cfg
    match = np.arange(world.n_matches)[:, None]
    x0, x1 = span(world.x, cfg.hitbox_w / 2)
    y0, y1 = span(world.y, cfg.hitbox_h / 2)
    out = np.zeros(world.x.shape, dtype=bool)
    for dy in range(int(np.ceil(cfg.hitbox_h)) + 1):
        for dx in range(int(np.ceil(cfg.hitbox_w)) + 1):
            row, col = y0 + dy, x0 + dx
            out |= (row <= y1) & (col <= x1) & tile_solid(world.grids, match, row, col)
    return out
