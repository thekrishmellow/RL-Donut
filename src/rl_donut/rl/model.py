"""Actor-critic with categorical heads for the discrete actions and a Gaussian aim head."""

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Categorical heads for move, jet, and shoot, plus a Gaussian head for the aim angle.

    `cont_dim=0` keeps the older discrete-only return, used by the bandit test.
    The match environment uses `cont_dim=1`. The aim sample is not clipped or wrapped.
    """

    def __init__(
        self,
        obs_dim: int,
        action_nvec: Sequence[int],
        hidden_dim: int = 256,
        cont_dim: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_nvec = tuple(action_nvec)
        self.cont_dim = cont_dim

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
        # The mean follows the observation. The spread is one learned number, shared across states.
        if cont_dim:
            self.aim_mean = layer_init(nn.Linear(hidden_dim, cont_dim), std=0.01)
            self.aim_log_std = nn.Parameter(torch.zeros(cont_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s), returns shape (batch_size, 1)."""
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        discrete: torch.Tensor | None = None,
        continuous: torch.Tensor | None = None,
    ):
        """Sample or score actions.

        With an aim head, returns discrete, continuous, log_prob, entropy, value.
        Without one, returns discrete, log_prob, entropy, value.
        """
        latent = self.actor_backbone(x)
        dists = [Categorical(logits=head(latent)) for head in self.action_heads]
        if discrete is None:
            discrete = torch.stack([dist.sample() for dist in dists], dim=-1)

        log_prob = torch.stack(
            [dist.log_prob(discrete[..., i]) for i, dist in enumerate(dists)], dim=-1
        ).sum(dim=-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(dim=-1)
        value = self.critic(x)
        if not self.cont_dim:
            return discrete, log_prob, entropy, value

        mean = self.aim_mean(latent)
        aim = Normal(mean, self.aim_log_std.exp().expand_as(mean))
        if continuous is None:
            continuous = aim.sample()
        log_prob = log_prob + aim.log_prob(continuous).sum(dim=-1)
        entropy = entropy + aim.entropy().sum(dim=-1)
        return discrete, continuous, log_prob, entropy, value

