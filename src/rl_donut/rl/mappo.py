"""Multi-Agent PPO (MAPPO) utilities and 2v2 team rewards (Phase 6 Step 6.5)."""

import numpy as np
import torch
import torch.nn as nn

from rl_donut.rl.model import layer_init


class CentralizedCritic(nn.Module):
    """Centralized value network for 2v2 MAPPO taking global environment state."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def blend_team_rewards(
    rewards: np.ndarray,
    team: np.ndarray,
    team_weight: float = 0.5,
) -> np.ndarray:
    """Blend individual agent rewards with team average: (1 - w) * own + w * team_avg."""
    M, A = rewards.shape
    blended = np.zeros_like(rewards, dtype=np.float32)

    for m in range(M):
        for t in np.unique(team):
            mask = (team == t)
            team_avg = rewards[m, mask].mean()
            blended[m, mask] = (1.0 - team_weight) * rewards[m, mask] + team_weight * team_avg

    return blended

