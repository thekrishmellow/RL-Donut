import hashlib
from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import GENERATOR_VERSION, ROCK, MapConfig, PitConfig, add_border, generate_map, terrain_grid

CFG = MapConfig()


@pytest.mark.parametrize("thickness", [1, 2])
def test_border_is_solid_on_all_four_sides(thickness):
    cfg = replace(CFG, border=thickness)
    grid = generate_map(5, cfg).grid
    t = thickness
    for edge in (grid[:t, :], grid[-t:, :], grid[:, :t], grid[:, -t:]):
        assert (edge == ROCK).all()


def test_border_leaves_interior_untouched():
    terrain = terrain_grid(5, CFG).grid
    grid = add_border(terrain, 1)
    assert np.array_equal(grid[1:-1, 1:-1], terrain[1:-1, 1:-1])


def test_add_border_does_not_modify_its_input():
    grid = np.zeros((10, 10), dtype=np.uint8)
    add_border(grid, 1)
    assert not grid.any()


def test_generated_maps_stay_symmetric():
    for seed in range(20):
        grid = generate_map(seed, CFG).grid
        assert np.array_equal(grid, grid[:, ::-1])


def test_map_records_metadata():
    game_map = generate_map(8, CFG, attempt=2)
    assert (game_map.seed, game_map.attempt) == (8, 2)
    assert game_map.generator_version == GENERATOR_VERSION
    assert game_map.kill_y == CFG.height + CFG.pits.kill_margin
    assert game_map.rock_fraction == pytest.approx(game_map.grid.mean())


def test_kill_margin_is_configurable():
    cfg = replace(CFG, pits=PitConfig(kill_margin=5.0))
    assert generate_map(0, cfg).kill_y == CFG.height + 5.0


def test_generation_is_reproducible():
    assert np.array_equal(generate_map(77).grid, generate_map(77).grid)


@pytest.mark.parametrize("kwargs", [
    {"count": (-1, 2)},
    {"count": (0, 3)},
    {"count": (2, 1)},
    {"width": (1, 4)},
    {"width": (5, 3)},
    {"max_depth": 0},
    {"kill_margin": -1.0},
])
def test_pit_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PitConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"border": 0}, {"width": 4, "border": 1}])
def test_map_config_rejects_invalid_border(kwargs):
    with pytest.raises(ValueError):
        MapConfig(**kwargs)


def test_golden_generated_map_seed_42():
    game_map = generate_map(42)
    assert int(game_map.grid.sum()) == 2700
    assert (game_map.tunnels, game_map.crossings, game_map.widest_crossing) == (1, 2, 8)
    assert [p.kind for p in game_map.platforms] == ["cover"] * 4
    assert game_map.fully_reachable and game_map.trap_free
    assert hashlib.sha256(game_map.grid.tobytes()).hexdigest()[:16] == "8fe72e69aae02aba"
