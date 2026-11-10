"""Behavioral style policies and asymmetric reward shaping (Phase 7 Step 7.1).

Defines specialized combat archetypes to prevent strategy cycling in self-play:
- AGGRESSIVE: 2x damage reward, aggressive pursuit
- DEFENSIVE: 2x damage taken penalty, cover/retreat prioritization
- SNIPER: Long-range hit and kill bonuses (> 15 tiles)
"""

from dataclasses import dataclass
from enum import Enum
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


def compute_styled_reward(
    events: Events,
    team: np.ndarray,
    style: CombatStyle = CombatStyle.BALANCED,
    alpha: float = 1.0,
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
    if cfg.long_range_bonus > 0.0 and hit_distances is not None:
        long_range_hits = (hit_distances >= cfg.min_sniper_range) & (events.hits > 0)
        shaping += cfg.long_range_bonus * long_range_hits.astype(np.float32)

    return (sparse + alpha * shaping).astype(np.float32)

