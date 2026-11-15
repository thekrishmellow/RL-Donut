"""Fair spawn-distance curriculum.

The official spawns hide the teams from each other, and a policy trained there deals no
damage. These rungs keep the real cave and move the two spawns closer, then widen the gap
once the policy wins half its matches. The observation stays the fair 330 numbers.
"""

from dataclasses import dataclass, replace

import numpy as np

from rl_donut.env.config import EngineConfig
from rl_donut.env.state import spawn_centre
from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.generator import GeneratedMap
from rl_donut.mapgen.geometry import line_of_sight
from rl_donut.mapgen.spawns import Spawn, Spawns
from rl_donut.mapgen.terrain import ROCK


@dataclass(frozen=True)
class SpawnRung:
    """A band of distances between the two spawn centres. Official spawns leave the map alone."""

    name: str
    min_gap: float | None = None
    max_gap: float | None = None
    require_sight: bool = False

    @property
    def official(self) -> bool:
        return self.min_gap is None


# Rifle range is 30. The first two rungs are inside it and in sight, so the lesson is aim.
# The third is past it, so the agent has to close ground. The last rung is the real spawn.
RUNG_LADDER: tuple[SpawnRung, ...] = (
    SpawnRung("tiles_8", 6.0, 12.0, require_sight=True),
    SpawnRung("tiles_14", 12.0, 16.0, require_sight=True),
    SpawnRung("tiles_20", 16.0, 28.0, require_sight=True),
    SpawnRung("tiles_40", 32.0, 48.0, require_sight=False),
    SpawnRung("official"),
)


@dataclass
class SpawnCurriculum:
    """Walks the ladder. A win rate at or above `gate` opens the next rung."""

    gate: float = 0.5
    rungs: tuple[SpawnRung, ...] = RUNG_LADDER
    index: int = 0

    def __post_init__(self):
        if not self.rungs:
            raise ValueError("the ladder needs a rung")
        if not 0.0 <= self.gate <= 1.0:
            raise ValueError("gate must be a win rate")
        if not 0 <= self.index < len(self.rungs):
            raise ValueError("index is outside the ladder")

    @property
    def rung(self) -> SpawnRung:
        return self.rungs[self.index]

    @property
    def finished(self) -> bool:
        return self.rung.official

    def consider(self, win_rate: float) -> bool:
        """Advance one rung when `win_rate` clears the gate. The official rung stays put."""
        if self.index >= len(self.rungs) - 1 or win_rate < self.gate:
            return False
        self.index += 1
        return True


def standing_anchors(grid: np.ndarray) -> list[Spawn]:
    """Clearance boxes whose interior is air and whose floor row is rock."""
    cfg = MapConfig()
    box_h, box_w = cfg.regions.clearance_h, cfg.regions.clearance_w
    rock = np.asarray(grid) == ROCK
    height, width = rock.shape
    spots = []
    for row in range(1, height - box_h - 1):
        occupied = np.any(rock[row:row + box_h, :-1] | rock[row:row + box_h, 1:], axis=0)
        floor = rock[row + box_h, :-1] & rock[row + box_h, 1:]
        for col in np.flatnonzero(~occupied & floor):
            if 0 < col < width - box_w:
                spots.append(Spawn(int(row), int(col)))
    return spots


def _same_floor_pairs(game_map: GeneratedMap):
    """Standing spots that share a floor, left then right, with the gap between centres."""
    engine = EngineConfig()
    by_row: dict[int, list[Spawn]] = {}
    for spot in standing_anchors(game_map.grid):
        by_row.setdefault(spot.y, []).append(spot)
    for group in by_row.values():
        group.sort(key=lambda spot: spot.x)
        centres = [spawn_centre(spot, engine) for spot in group]
        for i, (left, origin) in enumerate(zip(group, centres)):
            for right, other in zip(group[i + 1:], centres[i + 1:]):
                yield left, right, other[0] - origin[0], origin, other


def with_gap_spawns(
    game_map: GeneratedMap,
    min_gap: float,
    max_gap: float,
    require_sight: bool,
) -> GeneratedMap | None:
    """The same cave, with team A left of team B and the centres `min_gap` to `max_gap` apart.

    Team A on the left means both agents see the enemy to their right: team B's view is mirrored.
    A sight rung keeps a pair that can already shoot. A travel rung keeps a pair beyond rifle range.
    """
    rifle = EngineConfig().rifle.range
    grid = game_map.grid
    for left, right, gap, origin, other in _same_floor_pairs(game_map):
        if gap < min_gap:
            continue
        if gap > max_gap:
            continue
        visible = line_of_sight(grid, origin, other)
        in_range = gap < rifle
        if require_sight:
            if visible and in_range:
                return replace(game_map, spawns=Spawns((left,), (right,)))
            continue
        if not in_range:
            return replace(game_map, spawns=Spawns((left,), (right,)))
    return None


def maps_for_rung(maps: list[GeneratedMap], rung: SpawnRung, limit: int) -> list[GeneratedMap]:
    """Up to `limit` maps for this rung. Official spawns are copied through unchanged."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if rung.official:
        if len(maps) < limit:
            raise ValueError(f"need {limit} maps for the official rung, found {len(maps)}")
        return list(maps[:limit])
    placed = []
    for game_map in maps:
        moved = with_gap_spawns(game_map, rung.min_gap, rung.max_gap, rung.require_sight)
        if moved is not None:
            placed.append(moved)
            if len(placed) >= limit:
                break
    if len(placed) < limit:
        raise ValueError(f"rung {rung.name} found {len(placed)} maps, needed {limit}")
    return placed
