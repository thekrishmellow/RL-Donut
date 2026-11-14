"""Behavioral style policies and asymmetric reward shaping (Phase 7 Step 7.1).

Defines specialized combat archetypes to prevent strategy cycling in self-play:
- AGGRESSIVE: 2x damage reward, aggressive pursuit
- DEFENSIVE: 2x damage taken penalty, cover/retreat prioritization
- SNIPER: Long-range hit and kill bonuses (> 15 tiles)
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any
import numpy as np

from rl_donut.rl.rewards import RewardWeights
from rl_donut.rl.stand_in import Events


class CombatStyle(str, Enum):
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    SNIPER = "sniper"


@dataclass(frozen=True)
class StyleRewardConfig:
    weights: RewardWeights
    long_range_bonus: float = 0.0        # Extra reward per hit beyond 15 tiles
    min_sniper_range: float = 15.0


STYLE_CONFIGS: dict[CombatStyle, StyleRewardConfig] = {
    CombatStyle.BALANCED: StyleRewardConfig(
        weights=RewardWeights(),
        long_range_bonus=0.0,
    ),
    CombatStyle.AGGRESSIVE: StyleRewardConfig(
        weights=RewardWeights(
            kill=0.5,
            death=-0.2,
            damage_dealt=0.006,          # 2x standard damage reward
            damage_taken=-0.002,
            step_penalty=0.0010,         # 2x step penalty to force engagement
        ),
        long_range_bonus=0.0,
    ),
    CombatStyle.DEFENSIVE: StyleRewardConfig(
        weights=RewardWeights(
            kill=0.2,
            death=-0.5,                  # 1.6x penalty for death
            damage_dealt=0.002,
            damage_taken=-0.006,         # 2x penalty for taking damage
            step_penalty=0.0002,         # Low step penalty to allow patience
        ),
        long_range_bonus=0.0,
    ),
    CombatStyle.SNIPER: StyleRewardConfig(
        weights=RewardWeights(
            kill=0.4,
            death=-0.3,
            damage_dealt=0.003,
            damage_taken=-0.003,
            step_penalty=0.0004,
        ),
        long_range_bonus=0.004,          # Significant bonus for ranged hits
        min_sniper_range=15.0,
    ),
}


def long_hits(events: Any, env: Any, min_range: float) -> np.ndarray:
    """(M, A) count of this step's hits landed from at least `min_range` tiles.

    Distance runs from the shooter's position after the step to shot.hit.
    That can be off by up to 0.4 tiles from where the shot left, which is fine against
    a 15-tile threshold. Matches ending on this step (events.done[m] == True) are skipped
    because env.world already holds the next match's spawns.
    """
    M, A = events.damage_dealt.shape
    counts = np.zeros((M, A), dtype=np.float32)
    if not hasattr(events, "shot_log") or not events.shot_log or env is None:
        return counts

    w = getattr(env, "world", None)
    if w is None:
        return counts

    for _, shot in events.shot_log:
        if shot.target < 0 or shot.hit is None:
            continue
        m = shot.match
        if events.done[m]:
            continue
        s = shot.shooter
        sx, sy = w.x[m, s], w.y[m, s]
        hx, hy = shot.hit
        dist = math.hypot(hx - sx, hy - sy)
        if dist >= min_range:
            counts[m, s] += 1.0

    return counts


def compute_styled_reward(
    events: Events,
    team: np.ndarray,
    style: CombatStyle = CombatStyle.BALANCED,
    alpha: float = 1.0,
    long_hit_counts: np.ndarray | None = None,
    hit_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-agent reward tailored to a specific combat behavioral archetype."""
    cfg = STYLE_CONFIGS[style]
    M, A = events.damage_dealt.shape

    # 1. Base outcome reward
    sparse = np.zeros((M, A), dtype=np.float32)
    for a in range(A):
        t = team[a]
        team_outcome = events.outcome[:, t]
        sparse[:, a] = np.where(
            team_outcome > 0,
            cfg.weights.win,
            np.where(team_outcome < 0, cfg.weights.loss, 0.0),
        )

    # 2. Dense shaping
    w = cfg.weights
    shaping = (
        w.kill * events.kills
        + w.death * events.deaths
        + w.damage_dealt * events.damage_dealt
        + w.damage_taken * events.damage_taken
        - w.step_penalty
    )

    if w.fall_penalty != 0.0:
        shaping += w.fall_penalty * events.fell.astype(np.float32)

    # 3. Sniper long-range bonus
    if cfg.long_range_bonus > 0.0:
        if long_hit_counts is not None:
            shaping += cfg.long_range_bonus * long_hit_counts.astype(np.float32)
        elif hit_distances is not None:
            long_range_hits = (hit_distances >= cfg.min_sniper_range) & (events.hits > 0)
            shaping += cfg.long_range_bonus * long_range_hits.astype(np.float32)

    return (sparse + alpha * shaping).astype(np.float32)


def styled_reward_fn(style: CombatStyle):
    """Reward function adapter for Runner using a given combat style."""
    cfg = STYLE_CONFIGS[style]

    def reward(events, team, *, alpha, weights, env):
        counts = long_hits(events, env, cfg.min_sniper_range) if cfg.long_range_bonus else None
        return compute_styled_reward(events, team, style=style, alpha=alpha, long_hit_counts=counts)

    return reward

