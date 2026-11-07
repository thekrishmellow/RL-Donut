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
class MapConfig:
    width: int = 96
    height: int = 48
    noise: NoiseConfig = field(default_factory=NoiseConfig)
