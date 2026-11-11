from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner


def test_runner_executes_rollout_and_updates():
    """Runner plays MatchEnv, including one match ending inside the rollout."""
    env = MatchEnv(
        2, team_size=1, split="val", seed=101,
        cfg=EngineConfig(team_size=1, match=MatchConfig(duration=1.0)),
    )
    model = ActorCritic(env.obs_dim, env.action_nvec, hidden_dim=32, cont_dim=env.action_cont_dim)
    trainer = PPOTrainer(model, PPOConfig(n_epochs=2, n_minibatches=1))
    opponent = StaticTargetBot()

    runner = Runner(
        env=env,
        learner=model,
        opponent=opponent,
        trainer=trainer,
        n_steps=16,
        total_training_steps=1000,
    )

    metrics1 = runner.step_rollout()
    assert metrics1.loss_policy is not None
    assert metrics1.loss_value is not None
    assert metrics1.entropy is not None
    assert runner.global_step == 16 * 2  # 16 steps * 2 matches

    # Run a second update against RandomBot
    runner.opponent = RandomBot()
    metrics2 = runner.step_rollout()
    assert metrics2.loss_policy is not None
    assert runner.global_step == 32 * 2

