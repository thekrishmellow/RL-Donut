"""Make the map one connected play space for an agent-sized box.

Positions are box anchors: anchor (y, x) means the clearance box covers rows y..y+h-1 and
columns x..x+w-1. This step is fully determined by the input grid and uses no randomness.
"""

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import RegionConfig
from rl_donut.mapgen.terrain import AIR, ROCK

BLOCKED = np.int32(1 << 20)
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class RegionResult:
    grid: np.ndarray
    dug: np.ndarray        # rock tiles turned into air by tunnels
    filled: np.ndarray     # air tiles turned into rock as dead air
    tunnels: int
    crossings: int
    # Tallest opening across the centre line, in anchor rows.
    widest_crossing: int

    def crossings_ok(self, cfg: RegionConfig) -> bool:
        return crossings_ok(self.crossings, self.widest_crossing, cfg)


def box_rock_counts(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    """Number of rock tiles inside the box at every anchor; anchors whose box leaves the map are BLOCKED."""
    height, width = grid.shape
    out = np.full((height, width), BLOCKED, dtype=np.int32)
    if height < h or width < w:
        return out
    c = np.pad(grid.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(height - h + 1)[:, None]
    xs = np.arange(width - w + 1)[None, :]
    out[: height - h + 1, : width - w + 1] = c[ys + h, xs + w] - c[ys, xs + w] - c[ys + h, xs] + c[ys, xs]
    return out


def label_regions(fits: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """4-connected components of usable positions, labelled in row-major discovery order."""
    height, width = fits.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    sizes = []
    for y0, x0 in zip(*np.nonzero(fits)):
        if labels[y0, x0] >= 0:
            continue
        k = len(sizes)
        labels[y0, x0] = k
        queue = deque([(y0, x0)])
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy, dx in NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and fits[ny, nx] and labels[ny, nx] < 0:
                    labels[ny, nx] = k
                    queue.append((ny, nx))
        sizes.append(size)
    return labels, sizes


def cheapest_tunnel(counts: np.ndarray, sources: np.ndarray, targets: np.ndarray,
                    rock_cost: float, allowed: np.ndarray) -> list[tuple[int, int]] | None:
    """Dijkstra over anchors: each step costs 1 plus rock_cost per rock tile in the box entered."""
    dist = np.full(counts.shape, np.inf)
    prev = {}
    heap = []
    for y, x in zip(*np.nonzero(sources & allowed)):
        dist[y, x] = 0.0
        heap.append((0.0, int(y), int(x)))
    heapq.heapify(heap)
    height, width = counts.shape
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if targets[y, x]:
            path = [(y, x)]
            while (y, x) in prev:
                y, x = prev[(y, x)]
                path.append((y, x))
            return path[::-1]
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and allowed[ny, nx]:
                nd = d + 1.0 + rock_cost * counts[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (nd, ny, nx))
    return None


def carve(grid: np.ndarray, dug: np.ndarray, path, w: int, h: int, mirror: bool) -> None:
    width = grid.shape[1]
    for y, x in path:
        anchors = [(y, x), (y, width - w - x)] if mirror else [(y, x)]
        for ay, ax in anchors:
            box = (slice(ay, ay + h), slice(ax, ax + w))
            dug[box] |= grid[box] == ROCK
            grid[box] = AIR


def covered_by_boxes(anchors: np.ndarray, w: int, h: int) -> np.ndarray:
    """Tiles covered by a box at any of the given anchors."""
    out = np.zeros_like(anchors, dtype=bool)
    height, width = anchors.shape
    for dy in range(h):
        for dx in range(w):
            out[dy:, dx:] |= anchors[: height - dy, : width - dx]
    return out


def seam_column(width: int, w: int) -> int:
    """Anchor column whose box straddles the centre line; it is its own mirror image."""
    return (width - w) // 2


def crossing_runs(main: np.ndarray, column: int) -> list[tuple[int, int]]:
    """Vertical runs (start_row, end_row inclusive) of main-region anchors in the seam column."""
    col = main[:, column]
    runs = []
    y = 0
    while y < len(col):
        if col[y]:
            start = y
            while y + 1 < len(col) and col[y + 1]:
                y += 1
            runs.append((start, y))
        y += 1
    return runs


def widest(runs: list[tuple[int, int]]) -> int:
    return max((end - start + 1 for start, end in runs), default=0)


def crossings_ok(crossings: int, widest_crossing: int, cfg: RegionConfig) -> bool:
    return crossings >= cfg.min_crossings or widest_crossing >= cfg.wide_crossing


def _analyse(grid: np.ndarray, cfg: RegionConfig):
    counts = box_rock_counts(grid, cfg.clearance_w, cfg.clearance_h)
    labels, sizes = label_regions(counts == 0)
    main = int(np.argmax(sizes)) if sizes else -1
    return counts, labels, sizes, main


def connect_regions(grid: np.ndarray, cfg: RegionConfig, border: int, mirror: bool = True) -> RegionResult:
    grid = grid.copy()
    height, width = grid.shape
    w, h = cfg.clearance_w, cfg.clearance_h
    dug = np.zeros(grid.shape, dtype=bool)

    # Anchors whose box stays inside the border; tunnels never dig the frame.
    allowed = np.zeros(grid.shape, dtype=bool)
    allowed[border: height - border - h + 1, border: width - border - w + 1] = True

    tunnels = 0
    counts, labels, sizes, main = _analyse(grid, cfg)
    while main >= 0 and tunnels < cfg.max_tunnels:
        others = [k for k, s in enumerate(sizes) if k != main and s >= cfg.min_region]
        if not others:
            break
        k = max(others, key=lambda r: sizes[r])
        path = cheapest_tunnel(counts, labels == k, labels == main, cfg.rock_cost, allowed)
        if path is None:
            break
        carve(grid, dug, path, w, h, mirror)
        tunnels += 1
        counts, labels, sizes, main = _analyse(grid, cfg)

    column = seam_column(width, w)
    runs = crossing_runs(labels == main, column) if main >= 0 else []
    while main >= 0 and not crossings_ok(len(runs), widest(runs), cfg) and tunnels < cfg.max_tunnels:
        seam_ok = allowed[:, column].copy()
        for start, end in runs:
            seam_ok[max(start - cfg.crossing_gap, 0): end + cfg.crossing_gap + 1] = False
        if not seam_ok.any():
            break
        route_allowed = allowed.copy()
        route_allowed[:, column] = seam_ok
        main_mask = labels == main
        xs = np.arange(width)[None, :]
        path = cheapest_tunnel(counts, main_mask & (xs < column), main_mask & (xs > column),
                               cfg.rock_cost, route_allowed)
        if path is None:
            break
        carve(grid, dug, path, w, h, mirror)
        tunnels += 1
        counts, labels, sizes, main = _analyse(grid, cfg)
        runs = crossing_runs(labels == main, column)

    filled = np.zeros(grid.shape, dtype=bool)
    if main >= 0:
        playable = covered_by_boxes(labels == main, w, h)
        filled = (grid == AIR) & ~playable
        grid[filled] = ROCK

    return RegionResult(grid=grid, dug=dug, filled=filled, tunnels=tunnels,
                        crossings=len(runs), widest_crossing=widest(runs))
