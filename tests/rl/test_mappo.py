import numpy as np
import torch
from rl_donut.rl.mappo import CentralizedCritic, blend_team_rewards


def test_centralized_critic_forward():
    state_dim = 64
    critic = CentralizedCritic(state_dim=state_dim, hidden_dim=32)
    s = torch.randn(4, state_dim)
    val = critic(s)
    assert val.shape == (4, 1)


def test_blend_team_rewards():
    # 2 matches, 4 agents (team 0: agents 0,1; team 1: agents 2,3)
    team = np.array([0, 0, 1, 1])
    rewards = np.array([
        [1.0, 0.0, 0.5, 0.5],
        [2.0, 0.0, -1.0, 1.0],
    ], dtype=np.float32)

    # 50% team blend: agent 0 has own=1.0, team_avg=0.5 -> 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    blended = blend_team_rewards(rewards, team, team_weight=0.5)
    assert blended[0, 0] == 0.75
    assert blended[0, 1] == 0.25
    assert blended[0, 2] == 0.50
    assert blended[0, 3] == 0.50

