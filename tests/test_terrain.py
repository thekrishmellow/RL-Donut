import hashlib
from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import (
    AIR, ROCK, MapConfig, TerrainConfig, map_noise, rock_neighbours, sample_params, smooth, terrain_grid,
    threshold,
)

CFG = MapConfig()
UNSMOOTHED = replace(CFG, terrain=replace(CFG.terrain, smooth_passes=0))


def test_grid_contains_only_rock_and_air():
    grid = terrain_grid(3, CFG).grid
    assert grid.dtype == np.uint8
    assert grid.shape == (CFG.height, CFG.width)
    assert set(np.unique(grid)) <= {AIR, ROCK}


# --- thresholding ---

FRACTIONS = [0.3, 0.4, 0.475, 0.55, 0.65]


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_threshold_hits_rock_fraction_exactly_without_mirroring(fraction):
    cfg = replace(CFG, terrain=replace(CFG.terrain, mirror=False))
    grid = threshold(map_noise(9, cfg), fraction, height_bias=0.15)
    assert int(grid.sum()) == round(fraction * grid.size)


# Mirrored tiles share a noise value, so rock comes in pairs and an odd target is one tile over.
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_threshold_is_within_one_tile_with_mirroring(fraction):
    grid = threshold(map_noise(9, CFG), fraction, height_bias=0.15)
    assert 0 <= int(grid.sum()) - round(fraction * grid.size) <= 1


def test_unsmoothed_terrain_matches_sampled_fraction():
    for seed in range(20):
        terrain = terrain_grid(seed, UNSMOOTHED)
        assert abs(int(terrain.grid.sum()) - round(terrain.params.rock_fraction * terrain.grid.size)) <= 1


def test_rock_is_the_highest_noise():
    noise = map_noise(9, CFG)
    grid = threshold(noise, 0.45, height_bias=0.0)
    assert noise[grid == ROCK].min() >= noise[grid == AIR].max()


def test_sampled_params_cover_configured_ranges():
    params = [sample_params(s, CFG) for s in range(500)]
    fractions = np.array([p.rock_fraction for p in params])
    biases = np.array([p.height_bias for p in params])
    assert 0.40 <= fractions.min() < 0.41 and 0.54 < fractions.max() <= 0.55
    assert 0.0 <= biases.min() < 0.01 and 0.29 < biases.max() <= 0.3


def test_params_are_reproducible_and_change_on_retry():
    assert sample_params(7, CFG) == sample_params(7, CFG)
    assert sample_params(7, CFG) != sample_params(7, CFG, attempt=1)


def test_height_bias_puts_more_rock_at_the_bottom():
    noise = map_noise(11, CFG)
    half = CFG.height // 2
    flat = threshold(noise, 0.45, height_bias=0.0)
    biased = threshold(noise, 0.45, height_bias=0.3)
    assert biased[half:].mean() - flat[half:].mean() > 0.05
    assert biased.sum() == flat.sum()


def test_no_bias_is_balanced_on_average():
    cfg = replace(UNSMOOTHED, terrain=replace(UNSMOOTHED.terrain, height_bias=(0.0, 0.0)))
    half = cfg.height // 2
    grids = [terrain_grid(s, cfg).grid for s in range(200)]
    assert abs(np.mean([g[half:].mean() - g[:half].mean() for g in grids])) < 0.03


# --- mirroring ---

@pytest.mark.parametrize("width", [96, 97])
def test_mirrored_maps_are_left_right_symmetric(width):
    cfg = replace(CFG, width=width)
    for seed in range(20):
        grid = terrain_grid(seed, cfg).grid
        assert grid.shape == (cfg.height, width)
        assert np.array_equal(grid, grid[:, ::-1])


def test_mirror_off_gives_asymmetric_maps():
    cfg = replace(CFG, terrain=replace(CFG.terrain, mirror=False))
    assert not np.array_equal(terrain_grid(0, cfg).grid, terrain_grid(0, cfg).grid[:, ::-1])


# --- smoothing ---

def test_neighbour_count_treats_outside_as_rock():
    counts = rock_neighbours(np.zeros((3, 3), dtype=np.uint8))
    assert counts[1, 1] == 0
    assert counts[0, 0] == 5
    assert counts[0, 1] == 3


def test_smoothing_removes_lone_specks():
    grid = np.zeros((9, 9), dtype=np.uint8)
    grid[4, 4] = ROCK
    assert smooth(grid, passes=1)[4, 4] == AIR

    grid = np.ones((9, 9), dtype=np.uint8)
    grid[4, 4] = AIR
    assert smooth(grid, passes=1)[4, 4] == ROCK


def test_smoothing_leaves_ties_unchanged():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[1:4, 1] = ROCK
    grid[1, 2] = ROCK          # centre (2, 2) now has exactly 4 rock neighbours
    assert rock_neighbours(grid)[2, 2] == 4
    assert smooth(grid, passes=1)[2, 2] == grid[2, 2]


def test_zero_passes_is_identity():
    grid = threshold(map_noise(4, CFG), 0.45, 0.1)
    assert np.array_equal(smooth(grid, passes=0), grid)


def test_smoothed_maps_have_almost_no_rough_tiles():
    rough = []
    for seed in range(100):
        grid = terrain_grid(seed, CFG).grid
        counts = rock_neighbours(grid)
        rough.append(int(((grid == ROCK) & (counts <= 2)).sum() + ((grid == AIR) & (counts >= 6)).sum()))
    assert np.mean(rough) <= 3


def test_smoothing_keeps_rock_fraction_close_to_target():
    drift = [terrain_grid(s, CFG).final_rock_fraction - sample_params(s, CFG).rock_fraction for s in range(200)]
    assert max(abs(d) for d in drift) < 0.05


# --- config ---

@pytest.mark.parametrize("kwargs", [
    {"rock_fraction": (0.0, 0.5)},
    {"rock_fraction": (0.6, 0.5)},
    {"rock_fraction": (0.4, 1.0)},
    {"height_bias": (-0.1, 0.2)},
    {"height_bias": (0.3, 0.1)},
    {"smooth_passes": -1},
    {"smooth_rock_min": 3, "smooth_air_max": 3},
    {"smooth_rock_min": 9},
])
def test_terrain_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)


def test_golden_terrain_seed_42():
    terrain = terrain_grid(42, CFG)
    assert terrain.params.rock_fraction == pytest.approx(0.5296764213836348, abs=1e-12)
    assert terrain.params.height_bias == pytest.approx(0.2308075478177168, abs=1e-12)
    assert int(terrain.grid.sum()) == 2516
    assert hashlib.sha256(terrain.grid.tobytes()).hexdigest()[:16] == "13e9e96b9a03f6e8"
