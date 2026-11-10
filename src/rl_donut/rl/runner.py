"""Rollout runner and training loop coordinating learner and opponents."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import PPOTrainer, compute_gae
from rl_donut.rl.rewards import RewardWeights, anneal_alpha, compute_reward
from rl_donut.rl.stand_in import Policy, PolicyContext, StandInEnv


@dataclass
class RolloutMetrics:
    loss_policy: float
    loss_value: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    mean_return: float
    mean_length: float
    alpha: float
    completed_episodes: int


class Runner:
    """Manages the rollout collection and policy updates for PPO vs an opponent."""

    def __init__(
        self,
        env: StandInEnv,
        learner: ActorCritic,
        opponent: Policy,
        trainer: PPOTrainer,
        n_steps: int = 128,
        weights: RewardWeights | None = None,
        total_training_steps: int = 100_000,
        device: torch.device | None = None,
    ):
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.trainer = trainer
        self.n_steps = n_steps
        self.weights = weights or RewardWeights()
        self.total_training_steps = total_training_steps
        self.device = device or torch.device("cpu")

        self.global_step = 0
        self.rng = np.random.default_rng(env.seed)

        # Precompute learner and opponent slots per match (alternating sides)
        # Match m: learner is team (m % 2), opponent is team 1 - (m % 2)
        M, A = env.n_matches, env.n_agents
        self.team_size = env.team_size
        self.learner_slots = []
        self.opponent_slots = []

        for m in range(M):
            learner_team = m % 2
            opp_team = 1 - learner_team
            l_agents = [a for a in range(A) if env.team[a] == learner_team]
            o_agents = [a for a in range(A) if env.team[a] == opp_team]
            for a in l_agents:
                self.learner_slots.append((m, a))
            for a in o_agents:
                self.opponent_slots.append((m, a))

        self.learner_slots = np.array(self.learner_slots)    # (M * team_size, 2)
        self.opponent_slots = np.array(self.opponent_slots)  # (M * team_size, 2)
        self.n_learner_envs = len(self.learner_slots)

        self.last_obs = env.reset()
        self.episode_returns = np.zeros(self.n_learner_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.n_learner_envs, dtype=np.int32)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []

    def _extract_slots(self, arr: np.ndarray, slots: np.ndarray) -> np.ndarray:
        return arr[slots[:, 0], slots[:, 1]]

    def step_rollout(self) -> RolloutMetrics:
        """Collect one rollout buffer of n_steps, compute GAE and perform PPO update."""
        steps = self.n_steps
        N_L = self.n_learner_envs
        obs_dim = self.env.obs_dim
        action_dim = len(self.env.action_nvec)

        buf_obs = np.zeros((steps, N_L, obs_dim), dtype=np.float32)
        buf_actions = np.zeros((steps, N_L, action_dim), dtype=np.int64)
        buf_log_probs = np.zeros((steps, N_L), dtype=np.float32)
        buf_rewards = np.zeros((steps, N_L), dtype=np.float32)
        buf_dones = np.zeros((steps, N_L), dtype=bool)
        buf_values = np.zeros((steps, N_L), dtype=np.float32)

        cur_obs = self.last_obs

        for t in range(steps):
            self.global_step += N_L
            alpha = anneal_alpha(self.global_step, self.total_training_steps)

            # 1. Split observations
            l_obs = self._extract_slots(cur_obs, self.learner_slots)
            o_obs = self._extract_slots(cur_obs, self.opponent_slots)

            # 2. Get actions from learner model
            with torch.no_grad():
                l_obs_t = torch.from_numpy(l_obs).float().to(self.device)
                l_actions_t, l_log_probs_t, _, l_values_t = self.learner.get_action_and_value(l_obs_t)

            l_actions = l_actions_t.cpu().numpy()
            l_log_probs = l_log_probs_t.cpu().numpy()
            l_values = l_values_t.cpu().numpy().squeeze(-1)

            # 3. Get actions from opponent policy
            ctx = PolicyContext(env=self.env, slots=self.opponent_slots, rng=self.rng)
            o_actions = self.opponent.act(o_obs, ctx)

            # 4. Assemble full environment action tensor (M, A, action_dim)
            full_actions = np.zeros((self.env.n_matches, self.env.n_agents, action_dim), dtype=np.int64)
            full_actions[self.learner_slots[:, 0], self.learner_slots[:, 1]] = l_actions
            full_actions[self.opponent_slots[:, 0], self.opponent_slots[:, 1]] = o_actions

            # 5. Advance environment
            res = self.env.step(full_actions)

            # 6. Compute rewards
            full_rewards = compute_reward(res.events, self.env.team, alpha=alpha, weights=self.weights)
            l_rewards = self._extract_slots(full_rewards, self.learner_slots)

            # 7. Check terminations (match ends)
            l_dones = res.done[self.learner_slots[:, 0]]

            # Store in buffer
            buf_obs[t] = l_obs
            buf_actions[t] = l_actions
            buf_log_probs[t] = l_log_probs
            buf_rewards[t] = l_rewards
            buf_dones[t] = l_dones
            buf_values[t] = l_values

            self.episode_returns += l_rewards
            self.episode_lengths += 1

            if l_dones.any():
                for idx in np.nonzero(l_dones)[0]:
                    self.completed_returns.append(float(self.episode_returns[idx]))
                    self.completed_lengths.append(int(self.episode_lengths[idx]))
                    self.episode_returns[idx] = 0.0
                    self.episode_lengths[idx] = 0

            cur_obs = res.obs

        self.last_obs = cur_obs

        # Compute next value for GAE
        with torch.no_grad():
            next_l_obs = self._extract_slots(cur_obs, self.learner_slots)
            next_l_obs_t = torch.from_numpy(next_l_obs).float().to(self.device)
            next_values = self.learner.get_value(next_l_obs_t).cpu().numpy().squeeze(-1)

        next_dones = res.done[self.learner_slots[:, 0]]
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, next_values, next_dones,
            gamma=self.trainer.cfg.gamma, gae_lambda=self.trainer.cfg.gae_lambda
        )

        # Perform PPO update
        update_metrics = self.trainer.update(
            buf_obs, buf_actions, buf_log_probs, returns, advantages, buf_values
        )

        mean_ret = float(np.mean(self.completed_returns[-50:])) if self.completed_returns else 0.0
        mean_len = float(np.mean(self.completed_lengths[-50:])) if self.completed_lengths else 0.0

        return RolloutMetrics(
            loss_policy=update_metrics["loss_policy"],
            loss_value=update_metrics["loss_value"],
            entropy=update_metrics["entropy"],
            approx_kl=update_metrics["approx_kl"],
            clip_fraction=update_metrics["clip_fraction"],
            explained_variance=update_metrics["explained_variance"],
            mean_return=mean_ret,
            mean_length=mean_len,
            alpha=float(alpha),
            completed_episodes=len(self.completed_returns),
        )
