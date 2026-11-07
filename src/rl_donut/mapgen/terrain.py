"""Noise -> rock/air tile grid with an exact, per-seed rock fraction."""

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


def sample_params(seed: int, cfg: MapConfig, attempt: int = 0) -> MapParams:
    rng = make_rng(seed, Stream.PARAMS, attempt)
    return MapParams(
        rock_fraction=float(rng.uniform(*cfg.terrain.rock_fraction)),
        height_bias=float(rng.uniform(*cfg.terrain.height_bias)),
    )


def threshold(noise: np.ndarray, rock_fraction: float, height_bias: float) -> np.ndarray:
    """Mark the highest-valued tiles as rock so exactly ``rock_fraction`` of the grid is solid."""
    height = noise.shape[0]
    depth = (np.arange(height) / max(height - 1, 1) - 0.5)[:, None]
    biased = noise + height_bias * depth
    n_rock = int(round(rock_fraction * biased.size))
    if n_rock == 0:
        return np.zeros(noise.shape, dtype=np.uint8)
    cutoff = np.sort(biased, axis=None)[biased.size - n_rock]
    return np.where(biased >= cutoff, ROCK, AIR)


def terrain_grid(seed: int, cfg: MapConfig, attempt: int = 0) -> tuple[np.ndarray, MapParams]:
    params = sample_params(seed, cfg, attempt)
    noise = fbm(cfg.width, cfg.height, cfg.noise, make_rng(seed, Stream.TERRAIN, attempt))
    return threshold(noise, params.rock_fraction, params.height_bias), params
