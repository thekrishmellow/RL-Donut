from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import AIR, ROCK, MapConfig, PlatformConfig, box_rock_counts, generate_map, label_regions
from rl_donut.mapgen.platforms import placement, place_platforms

CFG = MapConfig()
PC = CFG.platforms


def open_map():
    grid = np.zeros((CFG.height, CFG.width), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = ROCK
    return grid


# --- placement rules ---

def test_placement_is_a_mirrored_pair():
    strips = placement(open_map(), 20, 10, 6, CFG)
    assert strips == [(20, 10, 6), (20, CFG.width - 16, 6)]


def test_placement_near_the_centre_becomes_one_centred_platform():
    strips = placement(open_map(), 20, CFG.width // 2 - 4, 6, CFG)
    assert len(strips) == 1
    _, x0, pw = strips[0]
    assert x0 + pw == CFG.width - x0           # symmetric about the centre line


def test_placement_needs_clear_air_around_it():
    grid = open_map()
    grid[20 - PC.clear_above, 12] = ROCK       # rock inside the standing room above
    assert placement(grid, 20, 10, 6, CFG) is None


def test_placement_stays_inside_the_border():
    assert placement(open_map(), 2, 10, 6, CFG) is None
    assert placement(open_map(), 20, 1, 6, CFG) is None


# --- generated maps ---

@pytest.fixture(scope="module")
def maps():
    return [generate_map(s, CFG) for s in range(30)]


def test_platforms_are_rock_strips_and_match_the_layer(maps):
    for m in maps:
        mask = np.zeros(m.grid.shape, dtype=bool)
        for p in m.platforms:
            mask[p.row, p.x0: p.x0 + p.width] = True
            assert PC.width[0] <= p.width <= PC.width[1] + 1     # +1: centred platforms fix parity
        assert (m.grid[mask] == ROCK).all()
        assert np.array_equal(mask, m.layers["platforms"])


def test_platforms_keep_standing_room_and_side_margins(maps):
    margin = CFG.regions.clearance_w
    for m in maps:
        for p in m.platforms:
            above = m.grid[p.row - PC.clear_above: p.row, p.x0 - margin: p.x0 + p.width + margin]
            assert (above == AIR).all()


def test_platform_sets_are_symmetric(maps):
    for m in maps:
        spans = {(p.row, p.x0, p.width) for p in m.platforms}
        assert spans == {(r, CFG.width - x - w, w) for r, x, w in spans}


def test_platforms_keep_one_region(maps):
    for m in maps:
        _, sizes = label_regions(box_rock_counts(m.grid, CFG.regions.clearance_w, CFG.regions.clearance_h) == 0)
        assert len(sizes) == 1


def test_unreachable_fill_is_small_and_solid(maps):
    for m in maps:
        filled = m.layers["unreachable"]
        assert (m.grid[filled] == ROCK).all()
        # Upper bound on the air at fill time (cover platforms are placed after the fill).
        air_before = (m.grid == AIR).sum() + filled.sum() + m.layers["platforms"].sum()
        assert filled.sum() <= CFG.max_unreachable_fill * air_before


def test_fill_can_be_disabled():
    cfg = replace(CFG, max_unreachable_fill=0.0)
    assert not any(generate_map(s, cfg).layers["unreachable"].any() for s in range(30))


def test_no_platforms_when_none_are_needed_or_allowed():
    cfg = replace(CFG, platforms=replace(PC, max_cover=0))
    result = place_platforms(open_map(), 0, cfg)
    assert result.platforms == ()


def test_platforms_are_reproducible():
    a, b = generate_map(11), generate_map(11)
    assert a.platforms == b.platforms


@pytest.mark.parametrize("kwargs", [{"width": (1, 4)}, {"width": (6, 4)}, {"clear_above": 0},
                                    {"max_cover": -1}, {"tries": 0}, {"max_evaluations": 0}])
def test_platform_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PlatformConfig(**kwargs)
