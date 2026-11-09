"""Bottom pits: shafts cut up through the bottom edge; falling through one kills the agent.

One pit is centred on the seam; two are a mirrored pair. A pit may only open through floor
in the main reachable group, so it sits on a walked path. Every pit must keep the map one
region, keep the centre-crossing rule, and must not strand floors or create traps. The
shaft has no floor, so the flight model treats falling in as leaving the map, not a trap.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.platforms import reach_of, region_check
from rl_donut.mapgen.reach import Reachability
from rl_donut.mapgen.seeding import Stream, make_rng
from rl_donut.mapgen.terrain import AIR


@dataclass(frozen=True)
class Pit:
    x0: int
    width: int
    # Top row of the opening: the lowest air row above the pit's columns.
    top: int


@dataclass(frozen=True)
class PitResult:
    grid: np.ndarray
    pits: tuple[Pit, ...]
    requested: int
    carved: np.ndarray     # rock tiles turned into air by pit shafts


def lowest_air(grid: np.ndarray) -> np.ndarray:
    """Row of the lowest air tile above the bottom row in every column, -1 if none."""
    air = grid[:-1] == AIR
    flipped = air[::-1].argmax(axis=0)
    rows = air.shape[0] - 1 - flipped
    return np.where(air.any(axis=0), rows, -1)


def main_floor_columns(grid: np.ndarray, reach: Reachability, cfg: MapConfig) -> np.ndarray:
    """Columns whose lowest air rests on floor belonging to the main group."""
    w, h = cfg.regions.clearance_w, cfg.regions.clearance_h
    low = lowest_air(grid)
    in_main = (reach.segments >= 0) & reach.main[np.maximum(reach.segments, 0)]
    out = np.zeros(grid.shape[1], dtype=bool)
    for y, x in zip(*np.nonzero(in_main)):
        bottom = y + h - 1
        for col in range(x, x + w):
            if low[col] == bottom:
                out[col] = True
    return out


def pit_spans(rng: np.random.Generator, n: int, width: int, cfg: MapConfig) -> list[tuple[int, int]] | None:
    """Random (x0, width) spans for one centred pit or a mirrored pair, or None if impossible."""
    lo, hi = cfg.pits.width
    pw = int(rng.integers(lo, hi + 1))
    if n == 1:
        if (width - pw) % 2:
            pw = pw + 1 if pw + 1 <= hi else pw - 1
        if (width - pw) % 2 or pw < lo:
            return None
        return [((width - pw) // 2, pw)]
    # Pair: keep a gap of at least one clearance box between the pit and its mirror image.
    gap = cfg.regions.clearance_w
    last = (width - gap) // 2 - pw
    if last < cfg.border:
        return None
    x0 = int(rng.integers(cfg.border, last + 1))
    return [(x0, pw), (width - x0 - pw, pw)]


def place_pits(grid: np.ndarray, seed: int, cfg: MapConfig, attempt: int = 0) -> PitResult:
    rng = make_rng(seed, Stream.PITS, attempt)
    lo, hi = cfg.pits.count
    requested = int(rng.integers(lo, hi + 1))
    nothing = np.zeros(grid.shape, dtype=bool)
    if requested == 0:
        return PitResult(grid, (), 0, nothing)

    height, width = grid.shape
    reach = reach_of(grid, cfg)
    stranded = int(reach.sizes[~reach.main].sum())
    low = lowest_air(grid)
    depth_ok = (low >= 0) & (height - 1 - low <= cfg.pits.max_depth)
    usable = main_floor_columns(grid, reach, cfg) & depth_ok
    usable[: cfg.border] = usable[width - cfg.border:] = False

    for _ in range(cfg.pits.tries):
        spans = pit_spans(rng, requested, width, cfg)
        if spans is None or not all(usable[x0: x0 + pw].all() for x0, pw in spans):
            continue
        trial = grid.copy()
        for x0, pw in spans:
            for col in range(x0, x0 + pw):
                trial[low[col] + 1:, col] = AIR
        ok, _, _ = region_check(trial, cfg)
        if not ok:
            continue
        trial_reach = reach_of(trial, cfg)
        if int(trial_reach.sizes[~trial_reach.main].sum()) > stranded:
            continue
        if reach.trap_free and not trial_reach.trap_free:
            continue
        pits = tuple(Pit(x0, pw, int(low[x0: x0 + pw].min())) for x0, pw in spans)
        return PitResult(trial, pits, requested, trial != grid)
    return PitResult(grid, (), requested, nothing)


def pit_columns(pits: tuple[Pit, ...], width: int, margin: int) -> np.ndarray:
    """Columns within `margin` of any pit; platforms may not use them."""
    out = np.zeros(width, dtype=bool)
    for p in pits:
        out[max(p.x0 - margin, 0): p.x0 + p.width + margin] = True
    return out
