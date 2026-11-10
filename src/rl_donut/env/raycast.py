"""Many line-of-sight checks at once on the matches' rock grids.

`segments` runs the same exact grid walk as `mapgen.line_of_sight` (Amanatides & Woo,
including its rule for lines passing exactly through a tile corner) for a whole batch of
segments in lockstep, so its answers are identical, and it also reports where the first rock
is entered. Each loop iteration advances every unfinished segment by one tile.
"""

import numpy as np


def segments(grids: np.ndarray, match: np.ndarray, x0: np.ndarray, y0: np.ndarray,
             x1: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Walk N segments (x0, y0) -> (x1, y1), segment k on `grids[match[k]]` (True = rock).

    Returns (clear, t), both (N,): `clear` is `line_of_sight` for each segment; `t` is the
    fraction of the segment at which the first rock tile is entered (0 when the start is in
    rock, inf when clear). Outside the grid counts as rock.
    """
    match = np.asarray(match, dtype=np.int64)
    x0, y0, x1, y1 = (np.asarray(v, dtype=np.float64) for v in (x0, y0, x1, y1))
    n = len(x0)
    height, width = grids.shape[1:]
    # A one-tile rock frame: a walk that starts inside stops on the frame before it can leave,
    # so steps need no bounds checks, and a tile is one flat index that moves by 1 or a row.
    row_len = width + 2
    solid = np.pad(grids, ((0, 0), (1, 1), (1, 1)), constant_values=True).ravel()

    col, row = np.floor(x0).astype(np.int64), np.floor(y0).astype(np.int64)
    end_col, end_row = np.floor(x1).astype(np.int64), np.floor(y1).astype(np.int64)
    dx, dy = x1 - x0, y1 - y0
    step_c = np.where(dx > 0, 1, -1)
    step_r = np.where(dy > 0, 1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_max_c = np.where(dx != 0, ((col + (step_c > 0)) - x0) / dx, np.inf)
        t_max_r = np.where(dy != 0, ((row + (step_r > 0)) - y0) / dy, np.inf)
        t_delta_c = np.where(dx != 0, np.abs(1 / dx), np.inf)
        t_delta_r = np.where(dy != 0, np.abs(1 / dy), np.inf)

    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    tile = match * (height + 2) * row_len + (row + 1) * row_len + (col + 1)
    end = match * (height + 2) * row_len + (end_row + 1) * row_len + (end_col + 1)
    blocked = ~inside
    blocked[inside] = solid[tile[inside]]
    t = np.where(blocked, 0.0, np.inf)

    live = ~blocked & ((row != end_row) | (col != end_col)) & (np.minimum(t_max_c, t_max_r) <= 1)
    k = np.nonzero(live)[0]
    tile, end, tc, tr = tile[k], end[k], t_max_c[k], t_max_r[k]
    dc, dr, sc, sr = t_delta_c[k], t_delta_r[k], step_c[k], step_r[k] * row_len
    while len(k):
        near = np.minimum(tc, tr)
        far = np.maximum(tc, tr)
        # math.isclose(tc, tr) with default tolerances; both are >= 0 and never both inf here
        diag = (tc == tr) | ((far - near <= 1e-9 * far) & (far < np.inf))
        by_col = ~diag & (tc < tr)
        move_c, move_r = diag | by_col, diag | ~by_col
        hit = diag & (solid[tile + sc] | solid[tile + sr])
        tile = tile + np.where(move_c, sc, 0) + np.where(move_r, sr, 0)
        hit |= solid[tile]
        crossing = np.where(move_c, tc, tr)
        tc = np.where(move_c, tc + dc, tc)
        tr = np.where(move_r, tr + dr, tr)
        if hit.any():
            blocked[k[hit]] = True
            t[k[hit]] = np.minimum(crossing[hit], 1.0)    # rounding can put an end-tile crossing just past 1
        keep = ~hit & (tile != end) & (np.minimum(tc, tr) <= 1)
        if not keep.all():
            k, tile, end, tc, tr, dc, dr, sc, sr = (a[keep] for a in (k, tile, end, tc, tr, dc, dr, sc, sr))
    return ~blocked, t
