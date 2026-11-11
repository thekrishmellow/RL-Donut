"""Reward function and shaping for RL-Donut agents (Phase 5 Step 5.3).

Transforms raw game Events into scalar rewards for PPO training:
    reward = sparse_outcome + alpha * shaping

Sparse reward rewards winning the match; shaping terms provide dense feedback for
damage and kills to accelerate early training. Alpha decays as the agent learns.
"""

from dataclasses import dataclass
import numpy as np

from rl_donut.rl.stand_in import Events


@dataclass(frozen=True)
class RewardWeights:
    # Sparse match end rewards (per team outcome: +1, -1, 0)
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0

    # Dense shaping rewards (scaled by alpha)
    kill: float = 0.3
    death: float = -0.3
    damage_dealt: float = 0.003      # 100 dmg = +0.30
    damage_taken: float = -0.003     # 100 dmg = -0.30
    step_penalty: float = 0.0005     # 900 steps = -0.45 total
    fall_penalty: float = -0.3       # suicide / pit fall penalty


def anneal_alpha(
    current_step: int,
    total_steps: int,
    initial_alpha: float = 1.0,
    final_alpha: float = 0.1,
    anneal_fraction: float = 0.6,
) -> float:
    """Linearly decay shaping weight alpha over anneal_fraction of training."""
    if total_steps <= 0:
        return float(final_alpha)
    anneal_steps = max(1, int(total_steps * anneal_fraction))
    progress = min(1.0, max(0.0, current_step / anneal_steps))
    return float(initial_alpha + progress * (final_alpha - initial_alpha))


def compute_reward(
    events: Events,
    team: np.ndarray,
    alpha: float = 1.0,
    weights: RewardWeights | None = None,
) -> np.ndarray:
    """Compute per-agent reward of shape (M, A) float32 from Events."""
    w = weights or RewardWeights()
    M, A = events.damage_dealt.shape

    # 1. Sparse outcome reward on match completion
    # outcome has shape (M, 2) for [team_A, team_B]
    # Map team outcome to each agent slot
    sparse = np.zeros((M, A), dtype=np.float32)
    for a in range(A):
        t = team[a]
        team_outcome = events.outcome[:, t]
        sparse[:, a] = np.where(
            team_outcome > 0,
            w.win,
            np.where(team_outcome < 0, w.loss, 0.0),
        )

    # 2. Dense shaping rewards
    shaping = (
        w.kill * events.kills
        + w.death * events.deaths
        + w.damage_dealt * events.damage_dealt
        + w.damage_taken * events.damage_taken
        - w.step_penalty
    )

    # Pit fall suicide penalty
    if w.fall_penalty != 0.0:
        shaping += w.fall_penalty * events.fell.astype(np.float32)

    total_reward = (sparse + alpha * shaping).astype(np.float32)
    return total_reward

