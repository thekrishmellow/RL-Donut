"""The agent's action: three discrete choices and one continuous aim angle.

| Part | Kind | Values |
|---|---|---|
| move | discrete, 3 | 0 = left, 1 = none, 2 = right |
| jetpack | discrete, 2 | 0 = off, 1 = on |
| shoot | discrete, 2 | 0 = no, 1 = yes (hold the trigger) |
| aim | continuous, 1 | angle in radians, any real number: 0 = right, pi/2 = down |

A policy samples the discrete parts from categorical heads and the aim from a Gaussian head.
The aim needs no clipping or wrapping: the engine only uses its cosine and sine.
"""

from dataclasses import dataclass

import numpy as np

DISCRETE_NVEC = (3, 2, 2)
CONTINUOUS_DIM = 1


@dataclass(frozen=True)
class Controls:
    """Engine controls for every agent, each (M, A)."""
    move: np.ndarray        # -1, 0 or 1
    jet: np.ndarray         # bool
    aim: np.ndarray         # radians
    trigger: np.ndarray     # bool


def decode(discrete: np.ndarray, continuous: np.ndarray) -> Controls:
    """Policy actions -> engine controls. `discrete` is (M, A, 3) int, `continuous` (M, A, 1) float."""
    discrete, continuous = np.asarray(discrete), np.asarray(continuous, dtype=np.float64)
    if discrete.shape[-1] != len(DISCRETE_NVEC) or continuous.shape[-1] != CONTINUOUS_DIM:
        raise ValueError(f"actions must end in {len(DISCRETE_NVEC)} discrete and {CONTINUOUS_DIM} continuous parts")
    if discrete.shape[:-1] != continuous.shape[:-1]:
        raise ValueError("discrete and continuous actions must cover the same agents")
    if ((discrete < 0) | (discrete >= np.array(DISCRETE_NVEC))).any():
        raise ValueError(f"discrete actions must be below {DISCRETE_NVEC}")
    if not np.isfinite(continuous).all():
        raise ValueError("aim angles must be finite")
    return Controls(move=discrete[..., 0].astype(np.int64) - 1, jet=discrete[..., 1] == 1,
                    aim=continuous[..., 0], trigger=discrete[..., 2] == 1)


def encode(controls: Controls) -> tuple[np.ndarray, np.ndarray]:
    """Engine controls -> policy actions (for scripted bots that think in controls)."""
    discrete = np.stack([np.asarray(controls.move) + 1, np.asarray(controls.jet, dtype=np.int64),
                         np.asarray(controls.trigger, dtype=np.int64)], axis=-1).astype(np.int64)
    return discrete, np.asarray(controls.aim, dtype=np.float64)[..., None]
