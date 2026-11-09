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


@dataclass(frozen=True)
class MapConfig:
    width: int = 96
    height: int = 48
    # Solid rock frame on all four sides; pits later cut through the bottom edge.
    border: int = 1
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    pits: PitConfig = field(default_factory=PitConfig)

    def __post_init__(self):
        if self.border < 1:
            raise ValueError("border must be >= 1")
        if min(self.width, self.height) <= 2 * self.border + 2:
            raise ValueError("map is too small for its border")
