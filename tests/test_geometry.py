import numpy as np
import pytest

from rl_donut.mapgen import ROCK
from rl_donut.mapgen.geometry import box_centre, line_of_sight


def empty(height=10, width=10):
    return np.zeros((height, width), dtype=np.uint8)


def test_box_centre():
    assert box_centre((4, 7), 2, 3) == (8.0, 5.5)


def test_clear_line_in_open_space():
    assert line_of_sight(empty(), (0.5, 0.5), (9.5, 7.2))


def test_wall_blocks_the_line():
    grid = empty()
    grid[:, 5] = ROCK
    assert not line_of_sight(grid, (1.5, 4.5), (8.5, 4.5))


def test_single_tile_on_the_path_blocks():
    grid = empty()
    grid[3, 3] = ROCK
    assert not line_of_sight(grid, (0.5, 0.5), (6.5, 6.5))      # diagonal through tile centres


def test_tile_beside_the_path_does_not_block():
    grid = empty()
    grid[3, 5] = ROCK
    assert line_of_sight(grid, (0.5, 0.5), (6.5, 6.5))


def test_line_cannot_slip_between_diagonal_rocks():
    grid = empty()
    grid[4, 5] = grid[5, 4] = ROCK
    assert not line_of_sight(grid, (2.0, 2.0), (8.0, 8.0))      # passes exactly through the corner (5, 5)


def test_corner_clip_that_point_sampling_misses():
    # The segment enters tile [2, 3] at x ~ 3.905 and leaves at x = 4: a 0.1-tile sliver.
    grid = empty()
    grid[2, 3] = ROCK
    start, end = (0.5, 0.2), (8.5, 4.43)
    assert not line_of_sight(grid, start, end)
    t = np.arange(0, 1, 0.25 / np.hypot(8, 4.23))[:, None]
    pts = np.array(start) + t * (np.array(end) - np.array(start))
    assert not grid[pts[:, 1].astype(int), pts[:, 0].astype(int)].any()    # quarter-tile sampling misses it


def test_start_or_end_inside_rock_is_blocked():
    grid = empty()
    grid[0, 0] = grid[9, 9] = ROCK
    assert not line_of_sight(grid, (0.5, 0.5), (5.5, 5.5))
    assert not line_of_sight(grid, (5.5, 5.5), (9.5, 9.5))


def test_leaving_the_map_counts_as_blocked():
    assert not line_of_sight(empty(), (5.5, 5.5), (12.0, 5.5))


@pytest.mark.parametrize("seed", range(5))
def test_matches_dense_sampling_when_nothing_is_near_a_corner(seed):
    # Random maps: whenever very dense sampling finds rock, the exact walk must too.
    rng = np.random.default_rng(seed)
    grid = (rng.random((20, 20)) < 0.15).astype(np.uint8)
    for _ in range(200):
        a, b = rng.uniform(0, 20, 2), rng.uniform(0, 20, 2)
        t = np.linspace(0, 1, 4000)[:, None]
        pts = a + t * (b - a)
        sampled_clear = not grid[pts[:, 1].astype(int), pts[:, 0].astype(int)].any()
        if not sampled_clear:
            assert not line_of_sight(grid, tuple(a), tuple(b))
