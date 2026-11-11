"""Proximal Policy Optimization (PPO) training core for RL-Donut.

CleanRL-style implementation supporting MultiDiscrete actions and batched environments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_donut.env.policy import PolicyContext
from rl_donut.rl.model import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    clip_vloss: bool = True
    normalize_advantages: bool = True


def compute_gae(
    rewards: np.ndarray,      # (steps, n_envs)
    values: np.ndarray,       # (steps, n_envs)
    dones: np.ndarray,        # (steps, n_envs)
    next_value: np.ndarray,   # (n_envs,)
    next_done: np.ndarray,    # (n_envs,)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
    steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0

    for t in reversed(range(steps)):
        if t == steps - 1:
            nextnonterminal = 1.0 - next_done.astype(np.float32)
            nextvalues = next_value
        else:
            # The done flag belongs to the transition that produced it, so a match
            # ending at t cuts the advantage chain at t.
            nextnonterminal = 1.0 - dones[t].astype(np.float32)
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


class LearnedPolicy:
    """Wrapper implementing the Policy protocol from a trained ActorCritic model."""

    def __init__(self, model: ActorCritic, name: str = "ppo_agent", device: torch.device | None = None):
        self.name = name
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device)
            out = self.model.get_action_and_value(obs_tensor)
        if self.model.cont_dim:
            discrete, continuous, _, _, _ = out
            return discrete.cpu().numpy().astype(np.int64), continuous.cpu().numpy().astype(np.float32)
        actions, _, _, _ = out
        return actions.cpu().numpy()


class PPOTrainer:
    """PPO trainer managing optimization epochs, advantage normalization, and clipping."""

    def __init__(
        self,
        model: ActorCritic,
        cfg: PPOConfig | None = None,
        device: torch.device | None = None,
    ):
        self.cfg = cfg or PPOConfig()
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)

    def update(
        self,
        obs: np.ndarray,          # (steps, n_envs, obs_dim)
        actions: np.ndarray,      # (steps, n_envs, action_dim)
        log_probs: np.ndarray,    # (steps, n_envs)
        returns: np.ndarray,      # (steps, n_envs)
        advantages: np.ndarray,   # (steps, n_envs)
        values: np.ndarray,       # (steps, n_envs)
        continuous: np.ndarray | None = None,  # (steps, n_envs, cont_dim)
    ) -> dict[str, float]:
        """Perform PPO mini-batch updates over the collected rollout buffer."""
        self.model.train()

        # Flatten rollout dimensions
        b_obs = torch.from_numpy(obs.reshape(-1, self.model.obs_dim)).float().to(self.device)
        b_actions = torch.from_numpy(actions.reshape(-1, len(self.model.action_nvec))).long().to(self.device)
        b_cont = None
        if self.model.cont_dim:
            if continuous is None:
                raise ValueError("continuous actions are required when the model has an aim head")
            b_cont = torch.from_numpy(continuous.reshape(-1, self.model.cont_dim)).float().to(self.device)
        b_log_probs = torch.from_numpy(log_probs.reshape(-1)).float().to(self.device)
        b_returns = torch.from_numpy(returns.reshape(-1)).float().to(self.device)
        b_advantages = torch.from_numpy(advantages.reshape(-1)).float().to(self.device)
        b_values = torch.from_numpy(values.reshape(-1)).float().to(self.device)

        batch_size = b_obs.shape[0]
        minibatch_size = max(1, batch_size // self.cfg.n_minibatches)
        indices = np.arange(batch_size)

        clip_fracs = []
        pg_losses = []
        v_losses = []
        ent_losses = []
        approx_kls = []

        for epoch in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_advantages = b_advantages[mb_idx]
                if self.cfg.normalize_advantages and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                if b_cont is None:
                    _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx]
                    )
                else:
                    _, _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx], b_cont[mb_idx]
                    )
                log_ratio = new_log_prob - b_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                    clip_fracs.append(clip_frac.item())

                # Clipped policy surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (with optional clipping)
                new_value = new_value.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        return {
            "loss_policy": float(np.mean(pg_losses)),
            "loss_value": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": float(explained_var),
        }

    def save_checkpoint(self, path: Path | str, extra_meta: dict[str, Any] | None = None) -> None:
        """Save model weights, optimizer, and metadata to path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "obs_dim": self.model.obs_dim,
            "action_nvec": self.model.action_nvec,
            "cont_dim": self.model.cont_dim,
            "extra_meta": extra_meta or {},
        }
        torch.save(ckpt, p)

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """Load model weights and optimizer from path."""
        p = Path(path)
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt.get("extra_meta", {})

