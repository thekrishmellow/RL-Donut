from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import AIR, MapConfig, PitConfig, box_rock_counts, generate_map, label_regions
from rl_donut.mapgen.pits import Pit, lowest_air, pit_columns, pit_spans, place_pits
from rl_donut.mapgen.platforms import reach_of

CFG = MapConfig()


def floor_room():
    """Open room on a 3-tile-thick floor; 16 rows tall so it meets the centre-crossing rule."""
    grid = np.ones((CFG.height, CFG.width), dtype=np.uint8)
    grid[CFG.height - 19: CFG.height - 3, 1:-1] = AIR
    return grid


# --- building blocks ---

def test_lowest_air_finds_the_bottom_of_each_column():
    grid = floor_room()
    low = lowest_air(grid)
    assert (low[1:-1] == CFG.height - 4).all()
    assert low[0] == -1 and low[-1] == -1


def test_centred_span_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(20):
        (x0, pw), = pit_spans(rng, 1, CFG.width, CFG)
        assert x0 + pw == CFG.width - x0
        assert CFG.pits.width[0] <= pw <= CFG.pits.width[1]


def test_pair_spans_are_mirrored_and_separate():
    rng = np.random.default_rng(0)
    for _ in range(20):
        (a, pw), (b, pw2) = pit_spans(rng, 2, CFG.width, CFG)
        assert pw == pw2 and b == CFG.width - a - pw
        assert a + pw + CFG.regions.clearance_w <= b


def test_pit_columns_include_the_margin():
    cols = pit_columns((Pit(10, 4, 40),), 30, 2)
    assert list(np.nonzero(cols)[0]) == list(range(8, 16))


# --- placement on a simple room ---

@pytest.mark.parametrize("count", [1, 2])
def test_pits_cut_through_the_floor_of_a_room(count):
    cfg = replace(CFG, pits=replace(CFG.pits, count=(count, count)))
    result = place_pits(floor_room(), 0, cfg)
    assert len(result.pits) == count and result.requested == count
    for p in result.pits:
        assert (result.grid[p.top:, p.x0: p.x0 + p.width] == AIR).all()
    assert np.array_equal(result.grid, result.grid[:, ::-1])


def test_shaft_deeper_than_max_depth_is_refused():
    cfg = replace(CFG, pits=replace(CFG.pits, count=(2, 2), max_depth=2))
    result = place_pits(floor_room(), 0, cfg)        # floor is 3 tiles thick
    assert result.pits == () and result.requested == 2
    assert np.array_equal(result.grid, floor_room())


def test_zero_pits_changes_nothing():
    cfg = replace(CFG, pits=replace(CFG.pits, count=(0, 0)))
    result = place_pits(floor_room(), 0, cfg)
    assert result.pits == () and result.requested == 0 and not result.carved.any()


def test_falling_into_a_pit_is_not_a_trap():
    cfg = replace(CFG, pits=replace(CFG.pits, count=(1, 1)))
    reach = reach_of(place_pits(floor_room(), 0, cfg).grid, cfg)
    assert reach.fully_reachable and reach.trap_free


# --- generated maps ---

@pytest.fixture(scope="module")
def maps():
    return [generate_map(s, CFG) for s in range(40)]


def test_pits_open_through_the_bottom_edge(maps):
    for m in maps:
        for p in m.pits:
            assert (m.grid[-1, p.x0: p.x0 + p.width] == AIR).all()
            assert CFG.height - 1 - p.top <= CFG.pits.max_depth + 1


def test_pits_are_symmetric_and_counted(maps):
    for m in maps:
        assert len(m.pits) in (0, m.pits_requested)
        spans = {(p.x0, p.width) for p in m.pits}
        assert spans == {(CFG.width - x - w, w) for x, w in spans}
        assert np.array_equal(m.layers["pits"], m.layers["pits"][:, ::-1])


def test_maps_with_pits_stay_one_region_and_trap_free(maps):
    for m in maps:
        if m.pits:
            _, sizes = label_regions(box_rock_counts(m.grid, 2, 3) == 0)
            assert len(sizes) == 1
            assert m.trap_free


def test_no_platform_above_a_pit(maps):
    for m in maps:
        cols = pit_columns(m.pits, CFG.width, CFG.pits.platform_margin)
        for p in m.platforms:
            assert not cols[p.x0: p.x0 + p.width].any()


def test_some_maps_get_pits(maps):
    assert sum(bool(m.pits) for m in maps) >= 10


@pytest.mark.parametrize("kwargs", [{"tries": 0}, {"platform_margin": -1}])
def test_pit_config_rejects_invalid_search_values(kwargs):
    with pytest.raises(ValueError):
        PitConfig(**kwargs)
