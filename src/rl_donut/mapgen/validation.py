"""Accept or reject finished maps, and retry rejected seeds.

`validate` returns the reasons a map is rejected (empty means valid). `generate_valid_map`
tries attempts 0, 1, 2, ... for a seed until one passes. A seed's design settings
(MapParams) stay fixed across attempts; only the geometry is re-drawn.
"""

from dataclasses import replace

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.generator import GeneratedMap, generate_map
from rl_donut.mapgen.platforms import region_check
from rl_donut.mapgen.terrain import ROCK

# Rejection reasons.
TRAP = "trap"                  # a reachable floor with no way back
STRANDED = "stranded"          # more unreachable floor than cfg.max_stranded
PITS = "pits"                  # fewer pits than the seed requested
SPAWNS = "spawns"              # fewer spawns than cfg.spawns.per_team
ASYMMETRIC = "asymmetric"      # grid is not a left/right mirror image
REGIONS = "regions"            # not one region, or the centre-crossing rule fails
BORDER = "border"              # frame broken somewhere other than a pit


class MapGenerationError(RuntimeError):
    pass


def border_intact(game_map: GeneratedMap, cfg: MapConfig) -> bool:
    grid, t = game_map.grid, cfg.border
    if not ((grid[:t] == ROCK).all() and (grid[:, :t] == ROCK).all() and (grid[:, -t:] == ROCK).all()):
        return False
    bottom = grid[-t:]
    return bool((bottom[~game_map.layers["pits"][-t:]] == ROCK).all())


def validate(game_map: GeneratedMap, cfg: MapConfig) -> list[str]:
    reasons = []
    if not game_map.trap_free:
        reasons.append(TRAP)
    if game_map.stranded_fraction > cfg.max_stranded:
        reasons.append(STRANDED)
    if len(game_map.pits) != game_map.pits_requested:
        reasons.append(PITS)
    if len(game_map.spawns.team_a) < cfg.spawns.per_team:
        reasons.append(SPAWNS)
    # Safety re-checks of guarantees earlier steps already make.
    if cfg.terrain.mirror and not np.array_equal(game_map.grid, game_map.grid[:, ::-1]):
        reasons.append(ASYMMETRIC)
    if not region_check(game_map.grid, cfg)[0]:
        reasons.append(REGIONS)
    if not border_intact(game_map, cfg):
        reasons.append(BORDER)
    return reasons


def generate_valid_map(seed: int, cfg: MapConfig | None = None) -> GeneratedMap:
    """First valid map for the seed; records why each earlier attempt was rejected."""
    cfg = cfg or MapConfig()
    rejections = []
    for attempt in range(cfg.max_attempts):
        game_map = generate_map(seed, cfg, attempt)
        reasons = validate(game_map, cfg)
        if not reasons:
            return replace(game_map, rejections=tuple(rejections))
        rejections.append(tuple(reasons))
    raise MapGenerationError(f"seed {seed}: no valid map in {cfg.max_attempts} attempts; "
                             f"rejections: {rejections}")
