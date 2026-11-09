"""Master seed -> independent, named random streams for each map component.

Each component gets its own stream keyed by a fixed ``Stream`` id, so adding a
new component never changes the output of existing ones for the same seed.
"""

from enum import IntEnum

import numpy as np

# Bump whenever any generation algorithm changes output for an existing seed.
GENERATOR_VERSION = 1


class Stream(IntEnum):
    # Values are part of the seed contract: never renumber, only append.
    TERRAIN = 0
    PLATFORMS = 1
    CORRIDORS = 2
    SPAWNS = 3
    DOMAIN = 4
    PARAMS = 5
    PITS = 6


def make_rng(master_seed: int, stream: Stream, attempt: int = 0) -> np.random.Generator:
    """Return the generator for one component of one map.

    ``attempt`` selects a fresh stream when a map fails validation and must be
    regenerated, instead of deriving a new seed arithmetically.
    """
    if not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError(f"master_seed must be a non-negative int, got {master_seed!r}")
    if attempt < 0:
        raise ValueError(f"attempt must be non-negative, got {attempt}")
    seq = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(int(stream), int(attempt)))
    return np.random.Generator(np.random.PCG64(seq))
