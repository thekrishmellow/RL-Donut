import hashlib

import numpy as np
import pytest

from rl_donut.mapgen import (
    AIR, ROCK, MapConfig, Stream, TerrainConfig, fbm, make_rng, sample_params, terrain_grid, threshold,
)

CFG = MapConfig()


def test_grid_contains_only_rock_and_air():
    grid, _ = terrain_grid(3, CFG)
    assert grid.dtype == np.uint8
    assert grid.shape == (CFG.height, CFG.width)
    assert set(np.unique(grid)) <= {AIR, ROCK}


@pytest.mark.parametrize("fraction", [0.3, 0.4, 0.475, 0.55, 0.65])
def test_threshold_hits_rock_fraction_exactly(fraction):
    noise = fbm(CFG.width, CFG.height, CFG.noise, make_rng(9, Stream.TERRAIN))
    grid = threshold(noise, fraction, height_bias=0.15)
    assert int(grid.sum()) == round(fraction * grid.size)


def test_rock_is_the_highest_noise():
    noise = fbm(CFG.width, CFG.height, CFG.noise, make_rng(9, Stream.TERRAIN))
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
    noise = fbm(CFG.width, CFG.height, CFG.noise, make_rng(11, Stream.TERRAIN))
    half = CFG.height // 2
    flat = threshold(noise, 0.45, height_bias=0.0)
    biased = threshold(noise, 0.45, height_bias=0.3)
    gain_bottom = biased[half:].mean() - flat[half:].mean()
    assert gain_bottom > 0.05
    assert biased.sum() == flat.sum()


def test_no_bias_is_balanced_on_average():
    cfg = MapConfig(terrain=TerrainConfig(height_bias=(0.0, 0.0)))
    half = cfg.height // 2
    diffs = [g[half:].mean() - g[:half].mean() for g, _ in (terrain_grid(s, cfg) for s in range(200))]
    assert abs(np.mean(diffs)) < 0.03


@pytest.mark.parametrize("kwargs", [
    {"rock_fraction": (0.0, 0.5)},
    {"rock_fraction": (0.6, 0.5)},
    {"rock_fraction": (0.4, 1.0)},
    {"height_bias": (-0.1, 0.2)},
    {"height_bias": (0.3, 0.1)},
])
def test_terrain_config_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)


def test_golden_terrain_seed_42():
    grid, params = terrain_grid(42, CFG)
    assert params.rock_fraction == pytest.approx(0.5296764213836348, abs=1e-12)
    assert params.height_bias == pytest.approx(0.2308075478177168, abs=1e-12)
    assert int(grid.sum()) == 2441
    assert hashlib.sha256(grid.tobytes()).hexdigest()[:16] == "56acb287c7ca443c"
