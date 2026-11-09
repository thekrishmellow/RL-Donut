"""Line of sight on the tile grid, shared by spawn placement and (later) hitscan weapons.

Coordinates are continuous tile units: x grows right, y grows down, and tile [row, col]
covers x in [col, col + 1) and y in [row, row + 1).
"""

import math

import numpy as np

from rl_donut.mapgen.terrain import ROCK


def box_centre(anchor: tuple[int, int], w: int, h: int) -> tuple[float, float]:
    """(x, y) centre of the clearance box at anchor (row, col)."""
    y, x = anchor
    return x + w / 2, y + h / 2


def line_of_sight(grid: np.ndarray, start: tuple[float, float], end: tuple[float, float]) -> bool:
    """True when the segment from start to end (x, y) touches no rock tile.

    Exact grid walk (Amanatides & Woo): visits every tile the segment passes through, in
    order. When the segment passes exactly through a tile corner, both tiles beside the
    corner are checked, so a line can't slip between two diagonal rocks.
    """
    height, width = grid.shape
    (x0, y0), (x1, y1) = start, end
    col, row = math.floor(x0), math.floor(y0)
    end_col, end_row = math.floor(x1), math.floor(y1)

    def solid(r: int, c: int) -> bool:
        return not (0 <= r < height and 0 <= c < width) or grid[r, c] == ROCK

    if solid(row, col):
        return False
    dx, dy = x1 - x0, y1 - y0
    step_c = 1 if dx > 0 else -1
    step_r = 1 if dy > 0 else -1
    # Parameter t (0..1 along the segment) at which the next vertical / horizontal grid line is crossed.
    t_max_c = ((col + (step_c > 0)) - x0) / dx if dx else math.inf
    t_max_r = ((row + (step_r > 0)) - y0) / dy if dy else math.inf
    t_delta_c = abs(1 / dx) if dx else math.inf
    t_delta_r = abs(1 / dy) if dy else math.inf

    while (row, col) != (end_row, end_col):
        if min(t_max_c, t_max_r) > 1:
            break
        if math.isclose(t_max_c, t_max_r):
            if solid(row, col + step_c) or solid(row + step_r, col):
                return False
            col += step_c
            row += step_r
            t_max_c += t_delta_c
            t_max_r += t_delta_r
        elif t_max_c < t_max_r:
            col += step_c
            t_max_c += t_delta_c
        else:
            row += step_r
            t_max_r += t_delta_r
        if solid(row, col):
            return False
    return True
