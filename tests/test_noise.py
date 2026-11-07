import numpy as np
import pytest

from rl_donut.mapgen import MapConfig, NoiseConfig, Stream, fbm, make_rng, perlin_grid
from rl_donut.mapgen.noise import PERLIN_2D_BOUND, fade_quintic, perlin_at, random_gradients


def terrain_rng(seed=42):
    return make_rng(seed, Stream.TERRAIN)


def test_perlin_is_zero_at_lattice_points():
    grads = random_gradients(terrain_rng(), 6, 6)
    y, x = np.mgrid[0:5, 0:5].astype(float)
    assert np.allclose(perlin_at(x, y, grads), 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_perlin_stays_within_theoretical_bound(seed):
    grads = random_gradients(terrain_rng(seed), 5, 5)
    points = make_rng(seed, Stream.DOMAIN).uniform(0, 3.999, size=(2, 20_000))
    assert np.abs(perlin_at(points[0], points[1], grads)).max() <= PERLIN_2D_BOUND


@pytest.mark.parametrize("cell", [8.0, 16.0, 32.0])
def test_single_octave_is_smooth_between_neighbouring_tiles(cell):
    for seed in range(20):
        noise = perlin_grid(96, 48, cell, cell, terrain_rng(seed))
        max_step = max(np.abs(np.diff(noise, axis=0)).max(), np.abs(np.diff(noise, axis=1)).max())
        assert max_step * cell < 2.5


def test_quintic_fade_has_flat_ends():
    eps = 1e-6
    assert fade_quintic(np.array(0.0)) == 0.0
    assert fade_quintic(np.array(1.0)) == 1.0
    assert fade_quintic(np.array(0.5)) == pytest.approx(0.5)
    assert (fade_quintic(np.array(eps)) - 0.0) / eps == pytest.approx(0.0, abs=1e-6)
    assert (1.0 - fade_quintic(np.array(1 - eps))) / eps == pytest.approx(0.0, abs=1e-6)


def test_fbm_shape_and_range():
    cfg = MapConfig()
    noise = fbm(cfg.width, cfg.height, cfg.noise, terrain_rng())
    assert noise.shape == (cfg.height, cfg.width)
    assert np.abs(noise).max() <= PERLIN_2D_BOUND


def test_fbm_is_reproducible_and_seed_dependent():
    cfg = NoiseConfig()
    a = fbm(96, 48, cfg, terrain_rng(5))
    assert np.array_equal(a, fbm(96, 48, cfg, terrain_rng(5)))
    assert not np.allclose(a, fbm(96, 48, cfg, terrain_rng(6)))


def test_adding_octaves_keeps_the_large_scale_shape():
    one = fbm(96, 48, NoiseConfig(octaves=1), terrain_rng())
    four = fbm(96, 48, NoiseConfig(octaves=4), terrain_rng())
    assert np.corrcoef(one.ravel(), four.ravel())[0, 1] > 0.8


@pytest.mark.parametrize("kwargs", [
    {"octaves": 0},
    {"persistence": 0.0},
    {"persistence": 1.5},
    {"lacunarity": 1.0},
    {"cell_w": 4.0, "octaves": 4},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)


# Golden values for seed 42 with default config (numpy 2.5.4).
GOLDEN_FBM_42 = {
    (0, 0): -0.05210286334986344,
    (24, 48): -0.31203713141769024,
    (47, 95): -0.15433238489494447,
    (10, 70): -0.1734391112026909,
}


def test_golden_fbm_values():
    cfg = MapConfig()
    noise = fbm(cfg.width, cfg.height, cfg.noise, terrain_rng(42))
    for (row, col), expected in GOLDEN_FBM_42.items():
        assert noise[row, col] == pytest.approx(expected, abs=1e-12)
