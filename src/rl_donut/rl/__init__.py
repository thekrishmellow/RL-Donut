"""Reinforcement learning: PPO, rewards, self-play, opponent pool and Elo."""

from rl_donut.rl.adaptive import AdaptiveConfig, AdaptiveMatchmaker
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.checkpoints import CheckpointManager
from rl_donut.rl.curriculum import CurriculumManager, CurriculumStage
from rl_donut.rl.elo import EloReport, MatchResult, compute_elo_ratings, run_round_robin
from rl_donut.rl.evaluate import EvalReport, evaluate_policy, wilson_score_interval
from rl_donut.rl.generalization import (
    GeneralizationReport,
    build_ood_maps,
    compute_generalization_gap,
    evaluate_generalization,
    get_ood_engine_configs,
)
from rl_donut.rl.mappo import CentralizedCritic, blend_team_rewards
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.pool import OpponentPool, PoolMember
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer, compute_gae
from rl_donut.rl.rewards import RewardWeights, anneal_alpha, compute_reward
from rl_donut.rl.runner import RolloutMetrics, Runner
from rl_donut.rl.stand_in import Events, Policy, PolicyContext, StandInEnv, StepResult
from rl_donut.rl.styles import CombatStyle, StyleRewardConfig, compute_styled_reward
from rl_donut.rl.watchlist import WatchlistReport, WatchlistThresholds, WatchlistTracker

__all__ = [
    "ActorCritic",
    "AdaptiveConfig",
    "AdaptiveMatchmaker",
    "CentralizedCritic",
    "CheckpointManager",
    "CombatStyle",
    "CurriculumManager",
    "CurriculumStage",
    "EloReport",
    "Events",
    "GeneralizationReport",
    "LearnedPolicy",
    "MatchResult",
    "OpponentPool",
    "PPOConfig",
    "PPOTrainer",
    "Policy",
    "PolicyContext",
    "PoolMember",
    "RandomBot",
    "RewardWeights",
    "RolloutMetrics",
    "Runner",
    "StandInEnv",
    "StaticTargetBot",
    "StepResult",
    "StyleRewardConfig",
    "WatchlistReport",
    "WatchlistThresholds",
    "WatchlistTracker",
    "anneal_alpha",
    "blend_team_rewards",
    "build_ood_maps",
    "compute_elo_ratings",
    "compute_gae",
    "compute_generalization_gap",
    "compute_reward",
    "compute_styled_reward",
    "evaluate_generalization",
    "evaluate_policy",
    "get_ood_engine_configs",
    "run_round_robin",
    "wilson_score_interval",
]
