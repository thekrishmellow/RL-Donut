import numpy as np
import pytest

from rl_donut.mapgen import AIR, ROCK, MapConfig, RegionConfig, box_rock_counts, label_regions
from rl_donut.mapgen.regions import (
    cheapest_tunnel, connect_regions, covered_by_boxes, crossing_runs, crossings_ok, seam_column,
)

CFG = MapConfig()
RC = CFG.regions


def rock_frame(height, width):
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = ROCK
    return grid


# --- box positions ---

def test_box_counts_rock_inside_each_box():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = ROCK
    counts = box_rock_counts(grid, 2, 3)
    assert counts[0, 1] == 1 and counts[0, 2] == 1
    assert counts[0, 0] == 0
    assert counts[3, 0] >= 1 << 20          # box would leave the map


def test_one_tile_high_tunnel_has_no_positions():
    grid = np.ones((7, 10), dtype=np.uint8)
    grid[3, 1:9] = AIR
    assert not (box_rock_counts(grid, 2, 3) == 0).any()


def test_label_regions_finds_separate_rooms():
    grid = rock_frame(8, 13)
    grid[:, 6] = ROCK
    labels, sizes = label_regions(box_rock_counts(grid, 2, 3) == 0)
    assert len(sizes) == 2 and sizes[0] == sizes[1]


# --- tunnels ---

def test_cheapest_tunnel_prefers_air_over_digging():
    grid = rock_frame(12, 20)
    grid[1:11, 9] = ROCK                    # wall between two rooms
    grid[1:5, 9] = AIR                      # gap at the top
    counts = box_rock_counts(grid, 2, 3)
    xs = np.arange(20)[None, :]
    fits = counts == 0
    path = cheapest_tunnel(counts, fits & (xs < 8), fits & (xs > 10), 12.0, np.ones_like(fits))
    assert path is not None
    assert all(counts[y, x] == 0 for y, x in path)   # went through the gap, dug nothing


def longest_vertical_run(path):
    longest = run = 0
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        run = run + 1 if x0 == x1 else 0
        longest = max(longest, run)
    return longest


def solid_rock_route(vertical_run_cost):
    counts = np.full((30, 30), 6, dtype=np.int32)
    sources = np.zeros(counts.shape, dtype=bool)
    targets = np.zeros(counts.shape, dtype=bool)
    sources[25, 15] = True
    targets[2, 15] = True
    return cheapest_tunnel(counts, sources, targets, 12.0, np.ones(counts.shape, dtype=bool),
                           vertical_run_cost)


def test_plain_tunnel_digs_a_straight_shaft():
    path = solid_rock_route(1.0)
    assert longest_vertical_run(path) == 23


def test_staircase_tunnel_never_digs_two_vertical_steps_in_a_row():
    path = solid_rock_route(RC.vertical_run_cost)
    assert path[0] == (25, 15) and path[-1] == (2, 15)
    assert longest_vertical_run(path) == 1


def test_tunnel_digs_through_a_sealed_wall():
    grid = rock_frame(10, 20)
    grid[:, 9:11] = ROCK
    result = connect_regions(grid, RegionConfig(min_region=1, min_crossings=1), border=1, mirror=False)
    labels, sizes = label_regions(box_rock_counts(result.grid, 2, 3) == 0)
    assert len(sizes) == 1
    assert result.dug.any() and result.tunnels >= 1


# --- generated maps ---

@pytest.fixture(scope="module")
def maps(generated_maps):
    return generated_maps


def test_every_map_is_one_region(maps):
    for m in maps:
        _, sizes = label_regions(box_rock_counts(m.grid, RC.clearance_w, RC.clearance_h) == 0)
        assert len(sizes) == 1, m.seed


def test_no_dead_air_remains(maps):
    for m in maps:
        labels, _ = label_regions(box_rock_counts(m.grid, RC.clearance_w, RC.clearance_h) == 0)
        playable = covered_by_boxes(labels == 0, RC.clearance_w, RC.clearance_h)
        assert not ((m.grid == AIR) & ~playable).any(), m.seed


def test_maps_stay_symmetric(maps):
    for m in maps:
        assert np.array_equal(m.grid, m.grid[:, ::-1]), m.seed


def test_border_is_never_dug(maps):
    for m in maps:
        dug = m.layers["dug"]
        assert not (dug[0].any() or dug[-1].any() or dug[:, 0].any() or dug[:, -1].any())
        g = m.grid
        assert (g[0] == ROCK).all() and (g[-1][~m.layers["pits"][-1]] == ROCK).all()
        assert (g[:, 0] == ROCK).all() and (g[:, -1] == ROCK).all()


def test_every_map_meets_the_crossing_rule(maps):
    for m in maps:
        assert crossings_ok(m.crossings, m.widest_crossing, RC), m.seed


def test_recorded_crossings_match_the_grid(maps):
    for m in maps:
        labels, _ = label_regions(box_rock_counts(m.grid, RC.clearance_w, RC.clearance_h) == 0)
        runs = crossing_runs(labels == 0, seam_column(CFG.width, RC.clearance_w))
        assert len(runs) == m.crossings


def test_layers_match_grid_changes(maps):
    for m in maps:
        assert (m.grid[m.layers["dug"] & ~m.layers["unreachable"]] == AIR).all()
        assert (m.grid[m.layers["filled"] & ~m.layers["pits"]] == ROCK).all()


def test_seam_column_is_its_own_mirror():
    for width in (96, 97):
        col = seam_column(width, 2)
        assert width - 2 - col in (col, col + 1)


@pytest.mark.parametrize("kwargs", [
    {"clearance_w": 0}, {"min_region": 0}, {"rock_cost": -1.0},
    {"min_crossings": 0}, {"crossing_gap": -1}, {"max_tunnels": 0}, {"wide_crossing": 0},
    {"vertical_run_cost": 0.5},
])
def test_region_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RegionConfig(**kwargs)
