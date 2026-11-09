from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class NoiseConfig:
    # Size of one gradient cell in tiles for the first octave; sets the scale of the largest caves.
    cell_w: float = 16.0
    cell_h: float = 10.0
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    fade: Literal["quintic", "linear"] = "quintic"

    def __post_init__(self):
        if self.octaves < 1:
            raise ValueError("octaves must be >= 1")
        if not 0 < self.persistence <= 1:
            raise ValueError("persistence must be in (0, 1]")
        if self.lacunarity <= 1:
            raise ValueError("lacunarity must be > 1")
        smallest = self.lacunarity ** (self.octaves - 1)
        if min(self.cell_w, self.cell_h) / smallest < 1:
            raise ValueError("highest octave would have cells smaller than one tile")


@dataclass(frozen=True)
class TerrainConfig:
    # Per-seed values are drawn uniformly from these (low, high) ranges.
    rock_fraction: tuple[float, float] = (0.40, 0.55)
    # Shift added to the noise that grows linearly from -bias/2 at the top to +bias/2 at the bottom.
    height_bias: tuple[float, float] = (0.0, 0.3)
    # Left/right mirror symmetry for fairness; asymmetric maps are reserved for OOD evaluation.
    mirror: bool = True
    # Cellular-automata smoothing: a tile becomes rock with >= smooth_rock_min rock neighbours (of 8),
    # air with <= smooth_air_max, and is unchanged in between.
    smooth_passes: int = 3
    smooth_rock_min: int = 5
    smooth_air_max: int = 3

    def __post_init__(self):
        lo, hi = self.rock_fraction
        if not 0 < lo <= hi < 1:
            raise ValueError("rock_fraction must satisfy 0 < low <= high < 1")
        lo, hi = self.height_bias
        if not 0 <= lo <= hi:
            raise ValueError("height_bias must satisfy 0 <= low <= high")
        if self.smooth_passes < 0:
            raise ValueError("smooth_passes must be >= 0")
        if not 0 <= self.smooth_air_max < self.smooth_rock_min <= 8:
            raise ValueError("smoothing needs 0 <= smooth_air_max < smooth_rock_min <= 8")


@dataclass(frozen=True)
class PitConfig:
    # Pits per map, drawn uniformly from this inclusive range: 1 = centred, 2 = mirrored pair.
    # Training starts at (0, 0) and is widened to (0, 2) as a curriculum step.
    count: tuple[int, int] = (0, 2)
    width: tuple[int, int] = (3, 6)
    # Deepest shaft carved upward from the bottom edge to reach air.
    max_depth: int = 10
    # An agent dies once its whole hitbox is more than this many tiles below the bottom edge.
    kill_margin: float = 3.0
    # Random positions tried before giving up (validation then retries the whole map).
    tries: int = 40
    # Platforms may not use columns within this many tiles of a pit, so none can cover it.
    platform_margin: int = 2

    def __post_init__(self):
        lo, hi = self.count
        if not 0 <= lo <= hi <= 2:
            raise ValueError("pit count must satisfy 0 <= low <= high <= 2")
        lo, hi = self.width
        if not 2 <= lo <= hi:
            raise ValueError("pit width must satisfy 2 <= low <= high (agent clearance is 2 wide)")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.kill_margin < 0:
            raise ValueError("kill_margin must be >= 0")
        if self.tries < 1 or self.platform_margin < 0:
            raise ValueError("tries must be >= 1 and platform_margin >= 0")


@dataclass(frozen=True)
class RegionConfig:
    # Box that must fit entirely in air for a position to be usable: agent hitbox 0.8 x 1.6 plus margin.
    clearance_w: int = 2
    clearance_h: int = 3
    # Regions with fewer usable positions than this are filled with rock instead of joined.
    min_region: int = 60
    # Cost of digging one rock tile relative to moving one tile through air.
    rock_cost: float = 12.0
    # Multiplier on digging vertically right after a vertical step. Straight dug shafts taller
    # than the climb budget are one-way traps; this makes tunnels climb as staircases instead.
    vertical_run_cost: float = 3.0
    # Required number of separate routes across the centre line, unless one opening is at
    # least wide_crossing rows tall (too wide for one agent to block).
    min_crossings: int = 2
    wide_crossing: int = 12
    # New centre-line crossings must be at least this many rows away from existing ones.
    crossing_gap: int = 6
    # Safety cap on tunnels dug per map.
    max_tunnels: int = 16

    def __post_init__(self):
        if self.clearance_w < 1 or self.clearance_h < 1:
            raise ValueError("clearance must be at least 1 x 1")
        if self.min_region < 1:
            raise ValueError("min_region must be >= 1")
        if self.rock_cost < 0:
            raise ValueError("rock_cost must be >= 0")
        if self.vertical_run_cost < 1:
            raise ValueError("vertical_run_cost must be >= 1")
        if self.min_crossings < 1 or self.crossing_gap < 0 or self.max_tunnels < 1 or self.wide_crossing < 1:
            raise ValueError("min_crossings, wide_crossing and max_tunnels must be >= 1, crossing_gap >= 0")


@dataclass(frozen=True)
class PlatformConfig:
    width: tuple[int, int] = (4, 8)
    # Air required above (standing room) and below (room to pass under) a platform.
    clear_above: int = 3
    clear_below: int = 3
    # One cover placement per this many air tiles, up to max_cover placements (each mirrored).
    cover_area: int = 800
    max_cover: int = 4
    # Safety caps: ledge rounds per map, position tries per platform, and reachability
    # checks per ledge (each check costs ~30 ms).
    max_needed: int = 12
    tries: int = 60
    max_evaluations: int = 8

    def __post_init__(self):
        lo, hi = self.width
        if not 2 <= lo <= hi:
            raise ValueError("platform width must satisfy 2 <= low <= high")
        if self.clear_above < 1 or self.clear_below < 1:
            raise ValueError("platform clearances must be >= 1")
        if (self.cover_area < 1 or self.max_cover < 0 or self.max_needed < 0 or self.tries < 1
                or self.max_evaluations < 1):
            raise ValueError("invalid platform counts")


@dataclass(frozen=True)
class SpawnConfig:
    # Spawn points per team; team B's are the mirror images of team A's.
    per_team: int = 6
    # Minimum distance between two spawns of the same team, in tiles (between anchors).
    spacing: float = 6.0
    # Spawns may not use columns within this many tiles of a pit.
    pit_gap: int = 6
    # Minimum gap between a spawn's box and the centre line, in tiles.
    centre_gap: int = 16

    def __post_init__(self):
        if self.per_team < 1:
            raise ValueError("per_team must be >= 1")
        if self.spacing < 0 or self.pit_gap < 0 or self.centre_gap < 0:
            raise ValueError("spacing, pit_gap and centre_gap must be >= 0")


@dataclass(frozen=True)
class MapConfig:
    width: int = 96
    height: int = 48
    # Solid rock frame on all four sides; pits later cut through the bottom edge.
    border: int = 1
    # Contract with the physics engine: no required climb is taller than this many tiles,
    # and the engine must be able to climb at least this far on one full tank.
    climb_budget: int = 12
    # A map is acceptable when it has no traps (every floor reachable from the main area can
    # get back) and at most this fraction of floor is unreachable. Unreachable floor is never
    # visited, and spawns are only placed on the main area.
    max_stranded: float = 0.02
    # Air no reachable flight can occupy is filled with rock when it is at most this fraction
    # of the map's air; larger unreachable caverns are left for validation to reject.
    max_unreachable_fill: float = 0.03
    # Attempts per seed before generate_valid_map gives up and raises.
    max_attempts: int = 30
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    platforms: PlatformConfig = field(default_factory=PlatformConfig)
    pits: PitConfig = field(default_factory=PitConfig)
    spawns: SpawnConfig = field(default_factory=SpawnConfig)

    def __post_init__(self):
        if self.border < 1:
            raise ValueError("border must be >= 1")
        if self.climb_budget < 1:
            raise ValueError("climb_budget must be >= 1")
        if not 0 <= self.max_stranded < 1:
            raise ValueError("max_stranded must be in [0, 1)")
        if not 0 <= self.max_unreachable_fill < 1:
            raise ValueError("max_unreachable_fill must be in [0, 1)")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if min(self.width, self.height) <= 2 * self.border + 2:
            raise ValueError("map is too small for its border")
