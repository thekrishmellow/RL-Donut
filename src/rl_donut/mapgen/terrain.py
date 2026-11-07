"""Noise -> mirrored, smoothed rock/air tile grid."""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.noise import fbm
from rl_donut.mapgen.seeding import Stream, make_rng

AIR = np.uint8(0)
ROCK = np.uint8(1)


@dataclass(frozen=True)
class MapParams:
    """Concrete values drawn for one map from the ranges in MapConfig."""
    rock_fraction: float
    height_bias: float


@dataclass(frozen=True)
class Terrain:
    grid: np.ndarray
    params: MapParams

    @property
    def final_rock_fraction(self) -> float:
        """Rock share after smoothing; differs slightly from params.rock_fraction."""
        return float(self.grid.mean())


def sample_params(seed: int, cfg: MapConfig, attempt: int = 0) -> MapParams:
    rng = make_rng(seed, Stream.PARAMS, attempt)
    return MapParams(
        rock_fraction=float(rng.uniform(*cfg.terrain.rock_fraction)),
        height_bias=float(rng.uniform(*cfg.terrain.height_bias)),
    )


def map_noise(seed: int, cfg: MapConfig, attempt: int = 0) -> np.ndarray:
    """FBM noise for the full map; with mirroring, only the left half is generated and reflected."""
    rng = make_rng(seed, Stream.TERRAIN, attempt)
    if not cfg.terrain.mirror:
        return fbm(cfg.width, cfg.height, cfg.noise, rng)
    half_width = (cfg.width + 1) // 2
    left = fbm(half_width, cfg.height, cfg.noise, rng)
    right = left[:, : cfg.width - half_width][:, ::-1]
    return np.hstack([left, right])


def threshold(noise: np.ndarray, rock_fraction: float, height_bias: float) -> np.ndarray:
    """Mark the highest-valued tiles as rock so ``rock_fraction`` of the grid is solid.

    Exact for unique noise values. Mirrored noise repeats every value, so rock is added in
    pairs and an odd target count comes out one tile over.
    """
    height = noise.shape[0]
    depth = (np.arange(height) / max(height - 1, 1) - 0.5)[:, None]
    biased = noise + height_bias * depth
    n_rock = int(round(rock_fraction * biased.size))
    if n_rock == 0:
        return np.zeros(noise.shape, dtype=np.uint8)
    cutoff = np.sort(biased, axis=None)[biased.size - n_rock]
    return np.where(biased >= cutoff, ROCK, AIR)


def rock_neighbours(grid: np.ndarray) -> np.ndarray:
    """Count of rock tiles among the 8 neighbours; tiles outside the map count as rock."""
    height, width = grid.shape
    padded = np.pad(grid, 1, constant_values=ROCK).astype(np.int8)
    return sum(
        padded[1 + dy: 1 + dy + height, 1 + dx: 1 + dx + width]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )


def smooth(grid: np.ndarray, passes: int, rock_min: int = 5, air_max: int = 3) -> np.ndarray:
    for _ in range(passes):
        counts = rock_neighbours(grid)
        grid = np.where(counts >= rock_min, ROCK, np.where(counts <= air_max, AIR, grid))
    return grid


def terrain_grid(seed: int, cfg: MapConfig, attempt: int = 0) -> Terrain:
    params = sample_params(seed, cfg, attempt)
    grid = threshold(map_noise(seed, cfg, attempt), params.rock_fraction, params.height_bias)
    t = cfg.terrain
    grid = smooth(grid, t.smooth_passes, t.smooth_rock_min, t.smooth_air_max)
    return Terrain(grid=grid, params=params)
