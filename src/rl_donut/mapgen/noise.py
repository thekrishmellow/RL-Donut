"""Vectorized 2D Perlin gradient noise and fractal Brownian motion (FBM)."""

import numpy as np

from rl_donut.mapgen.config import NoiseConfig

# Largest magnitude 2D Perlin noise can reach with unit-length gradients.
PERLIN_2D_BOUND = np.sqrt(0.5)


def fade_quintic(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def fade_linear(t: np.ndarray) -> np.ndarray:
    return t


FADES = {"quintic": fade_quintic, "linear": fade_linear}


def random_gradients(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Unit vectors at uniformly random angles, shape (rows, cols, 2)."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(rows, cols))
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def perlin_at(x: np.ndarray, y: np.ndarray, grads: np.ndarray, fade=fade_quintic) -> np.ndarray:
    """Evaluate Perlin noise at points given in gradient-cell units."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    def corner_dot(dy, dx):
        g = grads[y0 + dy, x0 + dx]
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)

    u = fade(fx)
    v = fade(fy)
    top = corner_dot(0, 0) + u * (corner_dot(0, 1) - corner_dot(0, 0))
    bottom = corner_dot(1, 0) + u * (corner_dot(1, 1) - corner_dot(1, 0))
    return top + v * (bottom - top)


def perlin_grid(width: int, height: int, cell_w: float, cell_h: float,
                rng: np.random.Generator, fade=fade_quintic) -> np.ndarray:
    """One octave of Perlin noise sampled at tile centres, shape (height, width)."""
    cols = int(np.ceil(width / cell_w)) + 1
    rows = int(np.ceil(height / cell_h)) + 1
    grads = random_gradients(rng, rows, cols)
    xs = (np.arange(width) + 0.5) / cell_w
    ys = (np.arange(height) + 0.5) / cell_h
    x, y = np.meshgrid(xs, ys)
    return perlin_at(x, y, grads, fade)


def fbm(width: int, height: int, cfg: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """Sum of octaves, each finer and weaker than the last, normalized to [-bound, bound]."""
    fade = FADES[cfg.fade]
    total = np.zeros((height, width))
    amplitude = 1.0
    weight = 0.0
    for octave in range(cfg.octaves):
        scale = cfg.lacunarity ** octave
        total += amplitude * perlin_grid(width, height, cfg.cell_w / scale, cfg.cell_h / scale, rng, fade)
        weight += amplitude
        amplitude *= cfg.persistence
    return total / weight
