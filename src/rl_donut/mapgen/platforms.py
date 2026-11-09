"""Floating platforms: ledges where the reachability check finds stranded floors, plus cover.

Platforms are 1-tile-thick rock strips, mirrored across the centre line, with clear air
around them. Every placement must keep the map one region and keep the centre-crossing
rule; ledges must also increase the reachable part of the map, and cover platforms must
never reduce it. Between ledges and cover, small pockets of air that no reachable flight
can occupy are filled with rock.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.reach import Reachability, analyse_reachability, occupiable_tiles
from rl_donut.mapgen.regions import analyse, crossing_runs, crossings_ok, seam_column, widest
from rl_donut.mapgen.seeding import Stream, make_rng
from rl_donut.mapgen.terrain import AIR, ROCK


@dataclass(frozen=True)
class Platform:
    row: int
    x0: int
    width: int
    kind: str          # "ledge" (restores reachability) or "cover"


@dataclass(frozen=True)
class PlatformResult:
    grid: np.ndarray
    platforms: tuple[Platform, ...]
    unreachable_filled: np.ndarray     # air no reachable flight could occupy, turned into rock
    reachability: Reachability
    crossings: int
    widest_crossing: int


def placement(grid: np.ndarray, row: int, x0: int, pw: int, cfg: MapConfig,
              forbidden: np.ndarray | None = None) -> list[tuple[int, int, int]] | None:
    """Platform strips (row, x0, width) for a mirrored placement, or None if it doesn't fit.

    A placement too close to the centre line becomes one centred platform instead of a pair.
    Strips may not touch `forbidden` columns (those above pits).
    """
    height, width = grid.shape
    margin = cfg.regions.clearance_w
    mirror_x0 = width - x0 - pw
    if abs(mirror_x0 - x0) < pw + 2 * margin:
        pw += (width - pw) % 2
        strips = [(row, (width - pw) // 2, pw)]
    else:
        strips = [(row, x0, pw), (row, mirror_x0, pw)]

    pc = cfg.platforms
    for r, x, sw in strips:
        top, bottom = r - pc.clear_above, r + pc.clear_below
        left, right = x - margin, x + sw - 1 + margin
        if top < cfg.border or bottom > height - 1 - cfg.border:
            return None
        if left < cfg.border or right > width - 1 - cfg.border:
            return None
        if (grid[top: bottom + 1, left: right + 1] != AIR).any():
            return None
        if forbidden is not None and forbidden[x: x + sw].any():
            return None
    return strips


def region_check(grid: np.ndarray, cfg: MapConfig) -> tuple[bool, int, int]:
    _, labels, sizes, main = analyse(grid, cfg.regions)
    runs = crossing_runs(labels == main, seam_column(grid.shape[1], cfg.regions.clearance_w))
    ok = len(sizes) == 1 and crossings_ok(len(runs), widest(runs), cfg.regions)
    return ok, len(runs), widest(runs)


def with_strips(grid: np.ndarray, strips) -> np.ndarray:
    out = grid.copy()
    for r, x, sw in strips:
        out[r, x: x + sw] = ROCK
    return out


def reachable_spots(reach: Reachability) -> int:
    return int(reach.sizes[reach.main].sum())


def reach_of(grid: np.ndarray, cfg: MapConfig) -> Reachability:
    return analyse_reachability(grid, cfg.regions.clearance_w, cfg.regions.clearance_h, cfg.climb_budget)


def segment_centre(reach: Reachability, segment: int) -> tuple[int, int]:
    ys, xs = np.nonzero(reach.segments == segment)
    return int(ys[0]), int(round(xs.mean()))


def nearest_reachable(reach: Reachability, y: int, x: int) -> tuple[int, int]:
    in_main = reach.main[np.maximum(reach.segments, 0)] & (reach.segments >= 0)
    ys, xs = np.nonzero(in_main)
    i = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
    return int(ys[i]), int(xs[i])


def fill_unreachable(grid: np.ndarray, reach: Reachability, crossings: int, wide: int, cfg: MapConfig):
    """Fill small unreachable air with rock; keep the map unchanged if the fill is too large
    or would break the region or centre-crossing rules."""
    nothing = np.zeros(grid.shape, dtype=bool)
    occupiable = occupiable_tiles(grid, reach, cfg.regions.clearance_w, cfg.regions.clearance_h,
                                  cfg.climb_budget)
    filled = (grid == AIR) & ~occupiable
    if not filled.any() or filled.sum() > cfg.max_unreachable_fill * (grid == AIR).sum():
        return grid, reach, crossings, wide, nothing
    trial = grid.copy()
    trial[filled] = ROCK
    ok, c, wd = region_check(trial, cfg)
    if not ok:
        return grid, reach, crossings, wide, nothing
    return trial, reach_of(trial, cfg), c, wd, filled


def place_platforms(grid: np.ndarray, seed: int, cfg: MapConfig, attempt: int = 0,
                    forbidden: np.ndarray | None = None) -> PlatformResult:
    """`forbidden` marks columns no platform may use (above pits)."""
    rng = make_rng(seed, Stream.PLATFORMS, attempt)
    pc = cfg.platforms
    h = cfg.regions.clearance_h
    height, width = grid.shape
    placed: list[Platform] = []
    _, crossings, wide = region_check(grid, cfg)
    reach = reach_of(grid, cfg)

    # Ledges: bridge the largest stranded floor to the nearest reachable one, halfway between.
    given_up: set[int] = set()
    for _ in range(pc.max_needed):
        if reach.fully_reachable:
            break
        stranded = [s for s in np.nonzero(~reach.main)[0] if s not in given_up]
        if not stranded:
            break
        target = max(stranded, key=lambda s: reach.sizes[s])
        sy, sx = segment_centre(reach, target)
        my, mx = nearest_reachable(reach, sy, sx)
        row = (sy + my) // 2 + h
        col = (sx + mx) // 2
        improved = None
        evaluations = 0
        for _ in range(pc.tries):
            pw = int(rng.integers(pc.width[0], pc.width[1] + 1))
            strips = placement(grid, row + int(rng.integers(-2, 3)),
                               col - pw // 2 + int(rng.integers(-3, 4)), pw, cfg, forbidden)
            if not strips:
                continue
            trial = with_strips(grid, strips)
            ok, c, wd = region_check(trial, cfg)
            if not ok:
                continue
            trial_reach = reach_of(trial, cfg)
            evaluations += 1
            if reachable_spots(trial_reach) > reachable_spots(reach):
                improved = (trial, strips, trial_reach, c, wd)
                break
            if evaluations >= pc.max_evaluations:
                break
        if improved is None:
            given_up.add(int(target))
            continue
        grid, strips, reach, crossings, wide = improved
        placed.extend(Platform(r, x, sw, "ledge") for r, x, sw in strips)

    grid, reach, crossings, wide, unreachable_filled = fill_unreachable(grid, reach, crossings, wide, cfg)

    # Cover: scaled by how much air the map has; each must keep the map's reachable part intact.
    n_cover = min(pc.max_cover, int(round((grid == AIR).sum() / pc.cover_area)))
    for _ in range(n_cover):
        for _ in range(pc.tries):
            pw = int(rng.integers(pc.width[0], pc.width[1] + 1))
            row = int(rng.integers(cfg.border + pc.clear_above, height - cfg.border - pc.clear_below))
            x0 = int(rng.integers(cfg.border, width // 2))
            strips = placement(grid, row, x0, pw, cfg, forbidden)
            if not strips:
                continue
            trial = with_strips(grid, strips)
            ok, c, wd = region_check(trial, cfg)
            if not ok:
                continue
            trial_reach = reach_of(trial, cfg)
            keeps_reach = reachable_spots(trial_reach) >= reachable_spots(reach)
            if keeps_reach and trial_reach.stranded_fraction <= reach.stranded_fraction:
                grid, reach, crossings, wide = trial, trial_reach, c, wd
                placed.extend(Platform(r, x, sw, "cover") for r, x, sw in strips)
                break

    return PlatformResult(grid=grid, platforms=tuple(placed), unreachable_filled=unreachable_filled,
                          reachability=reach, crossings=crossings, widest_crossing=wide)
