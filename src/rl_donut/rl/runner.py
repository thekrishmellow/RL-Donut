"""Rollout runner and training loop coordinating learner and opponents."""

from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch

from rl_donut.env.match_env import MatchEnv
from rl_donut.env.policy import Policy, PolicyContext
from rl_donut.rl.circulate import MapRoamer, circulate
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import PPOTrainer, compute_gae
from rl_donut.rl.rewards import RewardWeights, anneal_alpha, compute_reward


RewardFn = Callable[..., np.ndarray]


def default_reward(events, team, *, alpha, weights, env):
    return compute_reward(events, team, alpha=alpha, weights=weights)


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
    """Collects rollouts on MatchEnv and updates the learner with PPO."""

    def __init__(
        self,
        env: MatchEnv,
        learner: ActorCritic,
        opponent: Policy,
        trainer: PPOTrainer,
        n_steps: int = 128,
        weights: RewardWeights | None = None,
        total_training_steps: int = 100_000,
        reward_fn: RewardFn | None = None,
        device: torch.device | None = None,
    ):
        if learner.cont_dim != env.action_cont_dim:
            raise ValueError("learner aim head and the environment's continuous action size disagree")
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.trainer = trainer
        self.n_steps = n_steps
        self.weights = weights or RewardWeights()
        self.total_training_steps = total_training_steps
        self.reward_fn = reward_fn or default_reward
        self.device = device or torch.device("cpu")
        self.completed_outcomes: list[int] = []

        self.global_step = 0
        self.rng = np.random.default_rng(getattr(env, "seed", env.world.rng_seed))

        # Match m: the learner takes team (m % 2), so both sides are trained equally.
        M, A = env.n_matches, env.n_agents
        self.team_size = env.n_agents // 2
        self.learner_slots = []
        self.opponent_slots = []

        for m in range(M):
            learner_team = m % 2
            opp_team = 1 - learner_team
            for a in range(A):
                if env.team[a] == learner_team:
                    self.learner_slots.append((m, a))
                elif env.team[a] == opp_team:
                    self.opponent_slots.append((m, a))

        self.learner_slots = np.array(self.learner_slots)
        self.opponent_slots = np.array(self.opponent_slots)
        self.n_learner_envs = len(self.learner_slots)

        self.last_obs = env.reset()
        self.episode_returns = np.zeros(self.n_learner_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.n_learner_envs, dtype=np.int32)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        self.roamer = MapRoamer()

    def pop_outcomes(self) -> list[int]:
        out, self.completed_outcomes = self.completed_outcomes, []
        return out

    def _extract_slots(self, arr: np.ndarray, slots: np.ndarray) -> np.ndarray:
        return arr[slots[:, 0], slots[:, 1]]

    def _value(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device)
            return self.learner.get_value(tensor).cpu().numpy().squeeze(-1)

    def step_rollout(self) -> RolloutMetrics:
        """Collect one rollout of n_steps, bootstrap truncated matches, and update."""
        steps = self.n_steps
        N_L = self.n_learner_envs
        obs_dim = self.env.obs_dim
        action_dim = len(self.env.action_nvec)
        cont_dim = self.env.action_cont_dim
        gamma = self.trainer.cfg.gamma

        buf_obs = np.zeros((steps, N_L, obs_dim), dtype=np.float32)
        buf_actions = np.zeros((steps, N_L, action_dim), dtype=np.int64)
        buf_cont = np.zeros((steps, N_L, cont_dim), dtype=np.float32)
        buf_log_probs = np.zeros((steps, N_L), dtype=np.float32)
        buf_rewards = np.zeros((steps, N_L), dtype=np.float32)
        buf_dones = np.zeros((steps, N_L), dtype=bool)
        buf_values = np.zeros((steps, N_L), dtype=np.float32)

        cur_obs = self.last_obs

        for t in range(steps):
            self.global_step += N_L
            alpha = anneal_alpha(self.global_step, self.total_training_steps)

            l_obs = self._extract_slots(cur_obs, self.learner_slots)
            o_obs = self._extract_slots(cur_obs, self.opponent_slots)

            with torch.no_grad():
                l_obs_t = torch.from_numpy(l_obs).float().to(self.device)
                l_disc_t, l_cont_t, l_log_probs_t, _, l_values_t = self.learner.get_action_and_value(l_obs_t)

            l_actions = l_disc_t.cpu().numpy()
            l_cont = l_cont_t.cpu().numpy().astype(np.float32)
            l_values = l_values_t.cpu().numpy().squeeze(-1)
            learner_ctx = PolicyContext(self.env, self.learner_slots, self.rng)
            l_actions, l_cont = circulate(l_obs, l_actions, l_cont, learner_ctx, self.roamer)
            # Score the action that will actually be played, including a patrol step.
            with torch.no_grad():
                _, _, rescored, _, _ = self.learner.get_action_and_value(
                    l_obs_t,
                    torch.as_tensor(l_actions, dtype=torch.long, device=self.device),
                    torch.as_tensor(l_cont, dtype=torch.float32, device=self.device),
                )
            l_log_probs = rescored.cpu().numpy()

            ctx = PolicyContext(env=self.env, slots=self.opponent_slots, rng=self.rng)
            o_actions, o_cont = self.opponent.act(o_obs, ctx)

            full_discrete = np.zeros((self.env.n_matches, self.env.n_agents, action_dim), dtype=np.int64)
            full_cont = np.zeros((self.env.n_matches, self.env.n_agents, cont_dim), dtype=np.float32)
            full_discrete[self.learner_slots[:, 0], self.learner_slots[:, 1]] = l_actions
            full_cont[self.learner_slots[:, 0], self.learner_slots[:, 1]] = l_cont
            full_discrete[self.opponent_slots[:, 0], self.opponent_slots[:, 1]] = o_actions
            full_cont[self.opponent_slots[:, 0], self.opponent_slots[:, 1]] = o_cont

            res = self.env.step(full_discrete, full_cont)

            full_rewards = self.reward_fn(res.events, self.env.team, alpha=alpha, weights=self.weights, env=self.env)
            l_rewards = self._extract_slots(full_rewards, self.learner_slots)
            l_dones = res.done[self.learner_slots[:, 0]]

            # The timer truncates the match. Fold V(final_obs) into the reward and
            # mark the step done so the advantage chain stops without dropping that value.
            stored = l_rewards
            if l_dones.any():
                v_final = self._value(self._extract_slots(res.final_obs, self.learner_slots))
                stored = l_rewards + gamma * v_final * l_dones.astype(np.float32)

            buf_obs[t] = l_obs
            buf_actions[t] = l_actions
            buf_cont[t] = l_cont
            buf_log_probs[t] = l_log_probs
            buf_rewards[t] = stored
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
                    m, a = self.learner_slots[idx]
                    self.completed_outcomes.append(int(res.events.outcome[m, self.env.team[a]]))

            cur_obs = res.obs

        self.last_obs = cur_obs
        next_values = self._value(self._extract_slots(cur_obs, self.learner_slots))
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, next_values, buf_dones[-1],
            gamma=gamma, gae_lambda=self.trainer.cfg.gae_lambda,
        )

        update_metrics = self.trainer.update(
            buf_obs, buf_actions, buf_log_probs, returns, advantages, buf_values,
            continuous=buf_cont,
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
