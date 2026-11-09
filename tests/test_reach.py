import numpy as np
import pytest

from rl_donut.mapgen import AIR, ROCK, MapConfig
from rl_donut.mapgen.reach import (
    analyse_reachability, floor_segments, occupiable_tiles, standing_spots, transitive_closure,
)

CFG = MapConfig()
W, H = CFG.regions.clearance_w, CFG.regions.clearance_h
BUDGET = CFG.climb_budget


def room(height, width):
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = ROCK
    return grid


def step_map(step, shelf_left=22):
    """Open room with a floor on the left and a raised shelf, `step` tiles higher, on the right."""
    grid = room(40, 34)
    grid[-1 - step:, shelf_left:] = ROCK
    return grid


def reach_of(grid):
    return analyse_reachability(grid, W, H, BUDGET)


# --- standing spots and floor segments ---

def test_standing_spots_rest_on_rock():
    grid = room(10, 12)
    fits, supported = standing_spots(grid, W, H)
    ys, _ = np.nonzero(supported)
    assert set(ys) == {10 - 1 - H}           # box bottom sits right above the floor row
    assert fits[supported].all()


def test_floor_segments_split_on_gaps_and_rows():
    supported = np.zeros((4, 10), dtype=bool)
    supported[1, 1:4] = True
    supported[1, 6:9] = True
    supported[3, 2:5] = True
    segments, sizes = floor_segments(supported)
    assert list(sizes) == [3, 3, 3]
    assert len(np.unique(segments[supported])) == 3


def test_transitive_closure_follows_chains():
    reach = np.zeros((3, 3), dtype=bool)
    reach[0, 1] = reach[1, 2] = True
    closure = transitive_closure(reach)
    assert closure[0, 2] and not closure[2, 0]
    assert closure.diagonal().all()


# --- the flight model ---

def test_flat_room_is_fully_reachable():
    r = reach_of(room(12, 30))
    assert r.fully_reachable and r.trap_free and r.stranded_fraction == 0.0


def test_low_step_can_be_climbed_both_ways():
    r = reach_of(step_map(BUDGET - 2))
    assert len(r.sizes) == 2 and r.fully_reachable


def test_step_taller_than_the_climb_budget_cannot_be_climbed():
    r = reach_of(step_map(BUDGET + 2))
    assert len(r.sizes) == 2 and not r.fully_reachable
    assert r.closure.sum() == 3              # each reaches itself, plus the fall off the shelf


def test_unreachable_high_shelf_is_not_a_trap():
    # Larger low floor is the main group; you can drop out of the shelf, so nothing traps you.
    r = reach_of(step_map(BUDGET + 2, shelf_left=26))
    assert not r.fully_reachable and r.trap_free
    assert r.acceptable(max_stranded=0.5) and not r.acceptable(max_stranded=0.0)


def test_low_floor_under_a_tall_step_is_a_trap():
    # Larger shelf is the main group; dropping to the low floor means you can never get back.
    r = reach_of(step_map(BUDGET + 2, shelf_left=8))
    assert not r.fully_reachable and not r.trap_free
    assert not r.acceptable(max_stranded=0.99)


# --- occupiable air ---

def test_all_air_is_occupiable_in_a_low_room():
    grid = room(12, 30)
    occupiable = occupiable_tiles(grid, reach_of(grid), W, H, BUDGET)
    assert np.array_equal(occupiable, grid == AIR)


def test_air_far_above_the_main_floor_is_not_occupiable():
    grid = room(40, 30)
    occupiable = occupiable_tiles(grid, reach_of(grid), W, H, BUDGET)
    assert not occupiable[1:5].any()          # ~34 tiles above the floor, out of jetpack range
    assert occupiable[-2, 1:-1].all()


# --- generated maps ---

@pytest.fixture(scope="module")
def maps(generated_maps):
    return generated_maps[:30]


def test_recorded_reachability_matches_the_grid(maps):
    for m in maps:
        r = reach_of(m.grid)
        assert (m.fully_reachable, m.trap_free) == (r.fully_reachable, r.trap_free)
        assert m.stranded_fraction == pytest.approx(r.stranded_fraction)


def test_most_maps_are_acceptable(maps):
    ok = [m.trap_free and m.stranded_fraction <= CFG.max_stranded for m in maps]
    assert np.mean(ok) >= 0.85


@pytest.mark.parametrize("kwargs", [{"max_stranded": -0.1}, {"max_stranded": 1.0},
                                    {"max_unreachable_fill": 1.0}, {"climb_budget": 0}])
def test_map_config_rejects_invalid_reachability_values(kwargs):
    with pytest.raises(ValueError):
        MapConfig(**kwargs)
