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
class MapConfig:
    width: int = 96
    height: int = 48
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
