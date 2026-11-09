"""Spawn points: standing spots on the main reachable floor, mirrored between two teams.

Team A spawns on the left half and team B at the mirror images, so both teams get identical
geometry. Spawns are picked in random order from the SPAWNS stream; a spot is kept only if
it is far enough from pits, the centre line and the team's other spawns, and no spawn can
see any enemy spawn.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.geometry import box_centre, line_of_sight
from rl_donut.mapgen.pits import Pit, pit_columns
from rl_donut.mapgen.reach import Reachability
from rl_donut.mapgen.seeding import Stream, make_rng


@dataclass(frozen=True)
class Spawn:
    # Clearance-box anchor: the agent's box covers rows y..y+h-1 and columns x..x+w-1.
    y: int
    x: int


@dataclass(frozen=True)
class Spawns:
    team_a: tuple[Spawn, ...]
    team_b: tuple[Spawn, ...]


def mirror(spawn: Spawn, width: int, w: int) -> Spawn:
    return Spawn(spawn.y, width - w - spawn.x)


def candidates(grid: np.ndarray, reach: Reachability, pits: tuple[Pit, ...], cfg: MapConfig) -> list[Spawn]:
    """Main-group standing spots on the left side that respect the pit and centre gaps."""
    width = grid.shape[1]
    w = cfg.regions.clearance_w
    sc = cfg.spawns
    in_main = (reach.segments >= 0) & reach.main[np.maximum(reach.segments, 0)]
    near_pit = pit_columns(pits, width, sc.pit_gap)
    limit = width // 2 - sc.centre_gap
    return [Spawn(int(y), int(x)) for y, x in zip(*np.nonzero(in_main))
            if x + w <= limit and not near_pit[x: x + w].any()]


def sees(grid: np.ndarray, a: Spawn, b: Spawn, cfg: MapConfig) -> bool:
    """Clear line between box centres in either direction (centres can sit on grid lines,
    where the walk's start tile depends on direction)."""
    w, h = cfg.regions.clearance_w, cfg.regions.clearance_h
    ca, cb = box_centre((a.y, a.x), w, h), box_centre((b.y, b.x), w, h)
    return line_of_sight(grid, ca, cb) or line_of_sight(grid, cb, ca)


def place_spawns(grid: np.ndarray, reach: Reachability, pits: tuple[Pit, ...], seed: int,
                 cfg: MapConfig, attempt: int = 0) -> Spawns:
    rng = make_rng(seed, Stream.SPAWNS, attempt)
    width = grid.shape[1]
    w = cfg.regions.clearance_w
    sc = cfg.spawns
    pool = candidates(grid, reach, pits, cfg)
    chosen: list[Spawn] = []
    for i in rng.permutation(len(pool)):
        s = pool[i]
        if any(np.hypot(s.y - c.y, s.x - c.x) < sc.spacing for c in chosen):
            continue
        enemy = mirror(s, width, w)
        if any(sees(grid, s, mirror(c, width, w), cfg) or sees(grid, c, enemy, cfg) for c in chosen + [s]):
            continue
        chosen.append(s)
        if len(chosen) == sc.per_team:
            break
    return Spawns(tuple(chosen), tuple(mirror(s, width, w) for s in chosen))
