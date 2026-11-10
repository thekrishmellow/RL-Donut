"""Reinforcement learning: PPO, rewards, self-play, opponent pool and Elo."""

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.checkpoints import CheckpointManager
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer, compute_gae
from rl_donut.rl.rewards import RewardWeights, anneal_alpha, compute_reward
from rl_donut.rl.runner import RolloutMetrics, Runner
from rl_donut.rl.stand_in import Events, Policy, PolicyContext, StandInEnv, StepResult
from rl_donut.rl.watchlist import WatchlistReport, WatchlistThresholds, WatchlistTracker

__all__ = [
    "ActorCritic",
    "CheckpointManager",
    "Events",
    "LearnedPolicy",
    "PPOConfig",
    "PPOTrainer",
    "Policy",
    "PolicyContext",
    "RandomBot",
    "RewardWeights",
    "RolloutMetrics",
    "Runner",
    "StandInEnv",
    "StaticTargetBot",
    "StepResult",
    "WatchlistReport",
    "WatchlistThresholds",
    "WatchlistTracker",
    "anneal_alpha",
    "compute_gae",
    "compute_reward",
]
