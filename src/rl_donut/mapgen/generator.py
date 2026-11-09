"""Full map pipeline: seed -> finished map. Each Phase 1 step adds a stage here."""

from dataclasses import dataclass, field

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.pits import Pit, pit_columns, place_pits
from rl_donut.mapgen.platforms import Platform, place_platforms
from rl_donut.mapgen.regions import connect_regions
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
    tunnels: int
    # Separate routes across the centre line, and the tallest of them in anchor rows.
    crossings: int
    widest_crossing: int
    # Pits actually cut, and how many this seed asked for; validation rejects a shortfall.
    pits: tuple[Pit, ...]
    pits_requested: int
    platforms: tuple[Platform, ...]
    # Jetpack reachability between standing spots (see reach.py).
    fully_reachable: bool
    trap_free: bool
    stranded_fraction: float
    # Boolean masks of what each stage changed, for previews and debugging.
    layers: dict[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)

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
    regions = connect_regions(grid, cfg.regions, cfg.border, mirror=cfg.terrain.mirror)
    pits = place_pits(regions.grid, seed, cfg, attempt)
    forbidden = pit_columns(pits.pits, cfg.width, cfg.pits.platform_margin)
    platforms = place_platforms(pits.grid, seed, cfg, attempt, forbidden)
    platform_mask = (platforms.grid != pits.grid) & ~platforms.unreachable_filled
    return GeneratedMap(
        seed=seed,
        attempt=attempt,
        generator_version=GENERATOR_VERSION,
        grid=platforms.grid,
        params=terrain.params,
        terrain_rock_fraction=terrain.final_rock_fraction,
        kill_y=cfg.height + cfg.pits.kill_margin,
        tunnels=regions.tunnels,
        crossings=platforms.crossings,
        widest_crossing=platforms.widest_crossing,
        pits=pits.pits,
        pits_requested=pits.requested,
        platforms=platforms.platforms,
        fully_reachable=platforms.reachability.fully_reachable,
        trap_free=platforms.reachability.trap_free,
        stranded_fraction=platforms.reachability.stranded_fraction,
        layers={"dug": regions.dug, "filled": regions.filled, "pits": pits.carved,
                "platforms": platform_mask, "unreachable": platforms.unreachable_filled},
    )
