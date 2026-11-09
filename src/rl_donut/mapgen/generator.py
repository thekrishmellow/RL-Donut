"""Full map pipeline: seed -> finished map. Each Phase 1 step adds a stage here."""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.seeding import GENERATOR_VERSION
from rl_donut.mapgen.terrain import ROCK, MapParams, terrain_grid


@dataclass(frozen=True)
class GeneratedMap:
    seed: int
    attempt: int
    generator_version: int
    # Tile grid indexed [row, col]; row 0 is the top, so y grows downward.
    grid: np.ndarray
    params: MapParams
    # Rock fraction after smoothing, before the border was added.
    terrain_rock_fraction: float
    # World y (in tiles) past which an agent whose whole hitbox is below it is dead.
    kill_y: float

    @property
    def rock_fraction(self) -> float:
        return float(self.grid.mean())


def add_border(grid: np.ndarray, thickness: int) -> np.ndarray:
    out = grid.copy()
    out[:thickness, :] = ROCK
    out[-thickness:, :] = ROCK
    out[:, :thickness] = ROCK
    out[:, -thickness:] = ROCK
    return out


def generate_map(seed: int, cfg: MapConfig | None = None, attempt: int = 0) -> GeneratedMap:
    cfg = cfg or MapConfig()
    terrain = terrain_grid(seed, cfg, attempt)
    grid = add_border(terrain.grid, cfg.border)
    return GeneratedMap(
        seed=seed,
        attempt=attempt,
        generator_version=GENERATOR_VERSION,
        grid=grid,
        params=terrain.params,
        terrain_rock_fraction=terrain.final_rock_fraction,
        kill_y=cfg.height + cfg.pits.kill_margin,
    )
