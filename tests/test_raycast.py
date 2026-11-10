import math

import numpy as np
import pytest

from rl_donut.env.raycast import segments
from rl_donut.mapgen import load_split, line_of_sight


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:100]


def random_segments(maps, n, rng):
    """Most endpoints in air tiles, the rest anywhere on (and just off) the grid; a third snapped
    to whole or half tiles so axis-aligned lines, exact tile corners and zero-length segments
    are well covered."""
    h, w = maps[0].grid.shape
    match = rng.integers(len(maps), size=n)
    pts = rng.uniform([-2, -2, -2, -2], [w + 2, h + 2, w + 2, h + 2], size=(n, 4))
    air = [np.argwhere(m.grid == 0) for m in maps]
    for end in (0, 2):
        for k in np.nonzero(rng.random(n) < 0.8)[0]:
            r, c = air[match[k]][rng.integers(len(air[match[k]]))]
            pts[k, end:end + 2] = c + rng.random(), r + rng.random()
    snap = rng.random(n) < 1 / 3
    pts[snap] = np.round(pts[snap] * 2) / 2
    pts[:50, 2:] = pts[:50, :2]
    return match, pts


def test_matches_line_of_sight_exactly(maps):
    grids = np.stack([m.grid == 1 for m in maps])
    match, pts = random_segments(maps, 100_000, np.random.default_rng(0))
    clear, t = segments(grids, match, *pts.T)
    expected = [line_of_sight(maps[k].grid, (a, b), (c, d)) for k, (a, b, c, d) in zip(match, pts)]
    np.testing.assert_array_equal(clear, expected)
    assert clear.sum() > 10_000 and (~clear).sum() > 10_000
    assert np.isinf(t[clear]).all() and (t[~clear] >= 0).all() and (t[~clear] <= 1).all()


def test_t_is_where_the_first_rock_is_entered(maps):
    grids = np.stack([m.grid == 1 for m in maps])
    match, pts = random_segments(maps, 5_000, np.random.default_rng(1))
    clear, t = segments(grids, match, *pts.T)
    checked = 0
    for k, (x0, y0, x1, y1), tk in zip(match, pts, t):
        length = math.hypot(x1 - x0, y1 - y0)
        if not 0 < tk < 1 or length == 0:
            continue
        eps = 1e-6 / length
        point = lambda s: (x0 + s * (x1 - x0), y0 + s * (y1 - y0))
        assert line_of_sight(maps[k].grid, (x0, y0), point(tk - eps))
        assert not line_of_sight(maps[k].grid, (x0, y0), point(tk + eps))
        checked += 1
    assert checked > 1000


def test_hand_cases():
    grid = np.zeros((1, 10, 10), dtype=bool)
    grid[0, 4, 6] = True
    grid[0, 5, 5] = True                       # two rocks touching only at the corner (6, 5)
    run = lambda *seg: segments(grid, [0], *[[v] for v in seg])
    clear, t = run(0.5, 4.5, 9.5, 4.5)         # along row 4, into the rock at x = 6
    assert not clear[0] and t[0] == pytest.approx(5.5 / 9)
    assert run(0.5, 2.5, 9.5, 2.5)[0][0]       # open row
    assert not run(4.0, 3.0, 8.0, 7.0)[0][0]   # through the corner between the two rocks
    clear, t = run(6.5, 4.5, 0.5, 0.5)         # starts inside rock
    assert not clear[0] and t[0] == 0
    clear, t = run(2.5, 2.5, 2.5, 2.5)         # zero length in air
    assert clear[0] and np.isinf(t[0])
    clear, t = run(8.5, 8.5, 8.5, 12.0)        # leaves the grid: outside counts as rock
    assert not clear[0] and t[0] == pytest.approx(1.5 / 3.5)


def test_each_segment_uses_its_own_match():
    grids = np.zeros((2, 6, 6), dtype=bool)
    grids[1, :, 3] = True
    clear, _ = segments(grids, [0, 1], [0.5, 0.5], [2.5, 2.5], [5.5, 5.5], [2.5, 2.5])
    assert clear.tolist() == [True, False]


def test_empty_batch():
    clear, t = segments(np.zeros((1, 4, 4), dtype=bool), [], [], [], [], [])
    assert clear.shape == t.shape == (0,)
