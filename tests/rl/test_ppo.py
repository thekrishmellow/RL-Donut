from pathlib import Path
import numpy as np
import pytest
import torch

from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import PPOConfig, PPOTrainer, compute_gae, LearnedPolicy


def test_actor_critic_shapes():
    obs_dim = 32
    action_nvec = (3, 2, 2)
    model = ActorCritic(obs_dim, action_nvec, hidden_dim=64, cont_dim=1)

    batch_size = 8
    x = torch.randn(batch_size, obs_dim)

    discrete, continuous, log_prob, entropy, value = model.get_action_and_value(x)
    assert discrete.shape == (batch_size, 3)
    assert continuous.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
    assert torch.isfinite(continuous).all()

    for i, n in enumerate(action_nvec):
        assert (discrete[:, i] >= 0).all() and (discrete[:, i] < n).all()

    given_d = torch.zeros((batch_size, 3), dtype=torch.long)
    given_c = torch.zeros((batch_size, 1))
    d2, c2, lp2, _, _ = model.get_action_and_value(x, given_d, given_c)
    assert torch.equal(d2, given_d) and torch.equal(c2, given_c)
    assert lp2.shape == (batch_size,)

    discrete_only = ActorCritic(obs_dim, (3, 2, 16, 2), hidden_dim=64, cont_dim=0)
    actions, lp, ent, val = discrete_only.get_action_and_value(x)
    assert actions.shape == (batch_size, 4)
    assert lp.shape == (batch_size,) and ent.shape == (batch_size,) and val.shape == (batch_size, 1)


def test_compute_gae_known_values():
    steps, envs = 3, 1
    rewards = np.array([[1.0], [0.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
    dones = np.zeros((steps, envs), dtype=bool)
    next_value = np.array([0.0], dtype=np.float32)
    next_done = np.array([True], dtype=bool)

    gamma = 0.99
    gae_lambda = 0.95

    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda)
    assert adv.shape == (steps, envs)
    assert ret.shape == (steps, envs)
    np.testing.assert_allclose(ret, adv + values)


def test_ppo_learns_toy_bandit():
    """PPO must learn to pick arm 2 in a 4-armed bandit in a few updates."""
    torch.manual_seed(42)
    np.random.seed(42)

    obs_dim = 4
    action_nvec = (4,)
    model = ActorCritic(obs_dim, action_nvec, hidden_dim=32, cont_dim=0)
    cfg = PPOConfig(lr=0.02, n_epochs=4, n_minibatches=1)
    trainer = PPOTrainer(model, cfg)

    # Optimal arm is index 2
    steps, n_envs = 32, 4
    for _ in range(8):
        obs = np.ones((steps, n_envs, obs_dim), dtype=np.float32)
        with torch.no_grad():
            b_obs = torch.from_numpy(obs.reshape(-1, obs_dim))
            actions_t, log_prob_t, _, values_t = model.get_action_and_value(b_obs)

        actions = actions_t.numpy().reshape(steps, n_envs, 1)
        log_prob = log_prob_t.numpy().reshape(steps, n_envs)
        values = values_t.numpy().reshape(steps, n_envs)

        # Reward = 1.0 if arm 2 chosen, 0 otherwise
        rewards = (actions[..., 0] == 2).astype(np.float32)
        dones = np.zeros((steps, n_envs), dtype=bool)

        adv, ret = compute_gae(
            rewards, values, dones, np.zeros(n_envs, dtype=np.float32),
            np.zeros(n_envs, dtype=bool)
        )
        trainer.update(obs, actions, log_prob, ret, adv, values)

    # After 8 updates, policy should strongly favor arm 2
    test_obs = torch.ones((10, obs_dim), dtype=torch.float32)
    test_actions, _, _, _ = model.get_action_and_value(test_obs)
    arm_choices = test_actions[:, 0].numpy()
    assert (arm_choices == 2).sum() >= 7, f"Expected arm 2 chosen frequently, got {arm_choices}"


def test_checkpoint_round_trip(tmp_path: Path):
    obs_dim = 16
    action_nvec = (3, 2)
    model1 = ActorCritic(obs_dim, action_nvec, hidden_dim=32)
    trainer1 = PPOTrainer(model1)

    ckpt_path = tmp_path / "model.pt"
    trainer1.save_checkpoint(ckpt_path, extra_meta={"update": 42})

    model2 = ActorCritic(obs_dim, action_nvec, hidden_dim=32)
    trainer2 = PPOTrainer(model2)
    meta = trainer2.load_checkpoint(ckpt_path)

    assert meta["update"] == 42
    # Check that model weights match exactly
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_learned_policy_protocol():
    obs_dim = 8
    action_nvec = (3, 2, 2)
    model = ActorCritic(obs_dim, action_nvec, hidden_dim=32, cont_dim=1)
    policy = LearnedPolicy(model, name="agent_v1")

    obs = np.zeros((4, obs_dim), dtype=np.float32)
    discrete, continuous = policy.act(obs)
    assert discrete.shape == (4, 3)
    assert continuous.shape == (4, 1)

