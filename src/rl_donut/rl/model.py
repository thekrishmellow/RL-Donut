"""Actor-Critic neural network architecture for RL-Donut MultiDiscrete action space."""

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared or separate MLP network with multiple categorical heads for MultiDiscrete actions."""

    def __init__(
        self,
        obs_dim: int,
        action_nvec: Sequence[int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_nvec = tuple(action_nvec)

        # Critic network: obs -> scalar value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor backbone: obs -> latent features
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Multi-head action outputs: one categorical linear head per action dimension
        self.action_heads = nn.ModuleList([
            layer_init(nn.Linear(hidden_dim, n), std=0.01) for n in self.action_nvec
        ])

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s), returns shape (batch_size, 1)."""
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or evaluate given action) and compute log-prob, entropy and value.

        Args:
            x: Observation tensor of shape (B, obs_dim)
            action: Optional action tensor of shape (B, len(action_nvec))

        Returns:
            actions: (B, len(action_nvec)) int64
            log_prob: (B,) float32 (summed over action heads)
            entropy: (B,) float32 (summed over action heads)
            value: (B, 1) float32
        """
        latent = self.actor_backbone(x)
        dists = [Categorical(logits=head(latent)) for head in self.action_heads]

        if action is None:
            # Sample each action head
            sampled_actions = [dist.sample() for dist in dists]
            action = torch.stack(sampled_actions, dim=-1)

        # Log prob is sum of log probs across independent action heads
        log_prob = torch.stack(
            [dist.log_prob(action[..., i]) for i, dist in enumerate(dists)],
            dim=-1,
        ).sum(dim=-1)

        # Total entropy is sum of head entropies
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(dim=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value
