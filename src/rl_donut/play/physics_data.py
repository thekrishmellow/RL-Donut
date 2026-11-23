"""Turn a played match into training rows the physics will accept.

The tile window is already in the agent's own frame. A move or a jet that points
into a neighbouring rock tile is the control that walks into a building. Those
rows are not lessons. When an enemy is actually seen, the shot that connects is
the angle of their offset, and a human aim close to that line is kept.
"""

import numpy as np

from rl_donut.env.observe import CROP_H, CROP_W, FEATURE_NAMES
from rl_donut.rl.clone import Demonstrations


_SEEN = FEATURE_NAMES.index("enemy0_seen_by_me")
_DX = FEATURE_NAMES.index("enemy0_dx")
_DY = FEATURE_NAMES.index("enemy0_dy")
_CROP = len(FEATURE_NAMES)
_ROW = CROP_H // 2
_COL = CROP_W // 2


def tile_window(obs: np.ndarray) -> np.ndarray:
    """(…, 13, 21) rock/air window. Rock is 1, air is 0, below the map is -1."""
    obs = np.asarray(obs, dtype=np.float32)
    return obs[..., _CROP:].reshape(obs.shape[:-1] + (CROP_H, CROP_W))


def presses_rock(obs: np.ndarray, discrete: np.ndarray) -> np.ndarray:
    """True when the move or the jet points into rock beside the body."""
    window = tile_window(obs)
    discrete = np.asarray(discrete)
    rock = window > 0.5
    body = rock[..., _ROW - 1:_ROW + 2, :]
    into_left = (discrete[..., 0] == 0) & body[..., _COL - 1].any(axis=-1)
    into_right = (discrete[..., 0] == 2) & body[..., _COL + 1].any(axis=-1)
    into_ceiling = (discrete[..., 1] == 1) & rock[..., _ROW - 1, _COL]
    return into_left | into_right | into_ceiling


def seen_aim_target(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aim in the agent's own frame toward an enemy they can see, and that mask.

    dx and dy are already in that frame. The scale cancels in the angle.
    """
    obs = np.asarray(obs, dtype=np.float32)
    seen = obs[..., _SEEN] > 0.5
    target = np.arctan2(obs[..., _DY], obs[..., _DX]).astype(np.float32)
    target = np.where(seen, target, 0.0).astype(np.float32)
    return target, seen


def air_legal(demos: Demonstrations) -> np.ndarray:
    """Rows whose controls do not push the body into rock."""
    if demos.obs.shape[-1] < _CROP + CROP_H * CROP_W:
        return np.ones(len(demos), dtype=bool)
    return ~presses_rock(demos.obs, demos.discrete)


def corrected_aim(obs: np.ndarray, continuous: np.ndarray, tolerance: float = 0.35) -> np.ndarray:
    """Keep a human aim that already lies near the visible body. Otherwise use that line."""
    continuous = np.array(continuous, dtype=np.float32, copy=True)
    target, seen = seen_aim_target(obs)
    if not np.any(seen):
        return continuous
    human = continuous[seen, 0]
    error = np.arctan2(np.sin(human - target[seen]), np.cos(human - target[seen]))
    continuous[seen, 0] = np.where(np.abs(error) < tolerance, human, target[seen]).astype(np.float32)
    return continuous


def legal_demonstrations(demos: Demonstrations) -> Demonstrations:
    """The subset of a match file the engine can carry out, with aims that can hit."""
    mask = air_legal(demos)
    kept = demos.select(mask)
    if len(kept) == 0:
        return kept
    return Demonstrations(
        kept.obs,
        kept.discrete,
        corrected_aim(kept.obs, kept.continuous),
        kept.match_id,
    )


def hold_in_air(obs: np.ndarray, discrete: np.ndarray, continuous: np.ndarray):
    """Turn off a move or a jet that the tile window shows running into rock.

    The engine would stop the body on that tile anyway. Dropping the control
    here means the policy is not rewarded for pressing into a building.
    """
    discrete = np.array(discrete, copy=True)
    continuous = np.array(continuous, dtype=np.float32, copy=True)
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] < _CROP + CROP_H * CROP_W:
        return discrete, continuous
    blocked = presses_rock(obs, discrete)
    discrete[blocked, 0] = 1
    discrete[blocked, 1] = 0
    return discrete, continuous


def aim_error_degrees(predicted: np.ndarray, obs: np.ndarray) -> float:
    """Median absolute aim error, in degrees, on rows where an enemy is seen."""
    target, seen = seen_aim_target(obs)
    if not np.any(seen):
        return float("nan")
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    error = np.arctan2(np.sin(predicted[seen] - target[seen]), np.cos(predicted[seen] - target[seen]))
    return float(np.degrees(np.median(np.abs(error))))
