import math

import numpy as np
import torch

from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.mapgen import load_split
from rl_donut.env.policy import PolicyContext, team_slots
from rl_donut.rl.clone import (
    Demonstrations,
    action_log_prob,
    collect_rusher_demos,
    fit_clone,
    mode_actions,
    opening_weights,
    split_demonstrations,
    true_enemy_offset,
    wrap_angle,
)
from rl_donut.rl.evaluate import legal_duration
from rl_donut.rl.model import ActorCritic


def test_wrap_angle_picks_the_near_branch():
    mean = torch.tensor([[-3.0]])
    wrapped = wrap_angle(mean + 2 * math.pi + 0.2, mean)
    torch.testing.assert_close(wrapped, mean + 0.2, atol=1e-5, rtol=0)

    # pi and -pi are the same direction. The loss must treat them as neighbours.
    model = ActorCritic(4, (3, 2, 2), hidden_dim=16, cont_dim=1)
    with torch.no_grad():
        model.aim_mean.weight.zero_()
        model.aim_mean.bias.copy_(torch.tensor([-math.pi + 0.05]))
        model.aim_log_std.fill_(math.log(0.5))
    obs = torch.zeros(1, 4)
    discrete = torch.zeros(1, 3, dtype=torch.long)
    near = action_log_prob(model, obs, discrete, torch.tensor([[-math.pi + 0.05]]))
    around = action_log_prob(model, obs, discrete, torch.tensor([[math.pi + 0.05]]))
    torch.testing.assert_close(near, around, atol=1e-4, rtol=0)


def test_clone_copies_a_fixed_action():
    torch.manual_seed(0)
    model = ActorCritic(4, (3, 2, 2), hidden_dim=32, cont_dim=1)
    obs = np.zeros((128, 4), dtype=np.float32)
    obs[:, 0] = 1.0
    discrete = np.tile(np.array([[2, 1, 1]], dtype=np.int64), (128, 1))
    continuous = np.full((128, 1), 0.4, dtype=np.float32)
    match_id = np.zeros(128, dtype=np.int64)
    demos = Demonstrations(obs, discrete, continuous, match_id)
    with torch.no_grad():
        before = float(action_log_prob(model, obs, discrete, continuous).mean())
    fit_clone(model, demos, epochs=25, batch_size=32, lr=1e-3, seed=0)
    with torch.no_grad():
        after = float(action_log_prob(model, obs[:1], discrete[:1], continuous[:1]).mean())
    assert after > before + 0.5
    chosen, aim = mode_actions(model, obs[:1])
    np.testing.assert_array_equal(chosen, discrete[:1])
    assert abs(float(aim[0, 0]) - 0.4) < 0.15


def test_opening_weight_makes_departure_the_likely_action():
    """The same observation is mostly 'stand still', except the opening of each match."""
    torch.manual_seed(0)
    steps, matches = 50, 8
    match_id = np.tile(np.arange(matches), steps)
    # Rows are step-major: step 0 holds every match, then step 1, and so on.
    leave = np.repeat(np.arange(steps) < 5, matches)
    discrete = np.zeros((steps * matches, 3), dtype=np.int64)
    discrete[:, 0] = np.where(leave, 2, 1)
    demos = Demonstrations(
        np.ones((len(discrete), 4), dtype=np.float32),
        discrete,
        np.zeros((len(discrete), 1), dtype=np.float32),
        match_id,
    )
    plain = ActorCritic(4, (3, 2, 2), hidden_dim=32, cont_dim=1)
    fit_clone(plain, demos, epochs=40, batch_size=64, lr=1e-2, seed=0)
    weighted_model = ActorCritic(4, (3, 2, 2), hidden_dim=32, cont_dim=1)
    fit_clone(
        weighted_model, demos, epochs=40, batch_size=64, lr=1e-2, seed=0,
        sample_weight=opening_weights(demos.match_id, opening_steps=5, opening_weight=30),
    )
    obs = np.ones((1, 4), dtype=np.float32)
    plain_move = int(mode_actions(plain, obs)[0][0, 0])
    weighted_move = int(mode_actions(weighted_model, obs)[0][0, 0])
    assert plain_move == 1
    assert weighted_move == 2


def test_opening_weights_follow_each_match():
    match_id = np.array([0, 1, 0, 1, 0, 1])
    weights = opening_weights(match_id, opening_steps=2, opening_weight=10)
    np.testing.assert_array_equal(weights, [10, 10, 10, 10, 1, 1])


def test_split_keeps_a_match_together():
    match_id = np.repeat(np.arange(10), 3)
    n = len(match_id)
    demos = Demonstrations(np.zeros((n, 2)), np.zeros((n, 3), dtype=np.int64), np.zeros((n, 1)), match_id)
    train, val = split_demonstrations(demos, val_fraction=0.2, seed=1)
    assert set(train.match_id).isdisjoint(set(val.match_id))
    assert len(train) + len(val) == n
    assert len(np.unique(val.match_id)) == 2


def test_recorded_demos_use_the_fair_observation():
    maps = load_split("val")[:2]
    seconds = legal_duration(0.2)
    demos = collect_rusher_demos(2, split="val", seed=3, seconds=seconds, batch_size=2, maps=maps)
    env = MatchEnv(
        1, team_size=1, seed=0,
        cfg=EngineConfig(team_size=1, match=MatchConfig(duration=seconds)),
        maps=maps[:1],
    )
    assert demos.obs.shape[1] == env.obs_dim == 330
    assert demos.discrete.shape == (len(demos), 3)
    assert demos.continuous.shape == (len(demos), 1)
    assert len(np.unique(demos.match_id)) == 2
    assert np.isfinite(demos.obs).all() and np.isfinite(demos.continuous).all()
    assert (demos.discrete[:, 0] < 3).all() and (demos.discrete[:, 1] < 2).all() and (demos.discrete[:, 2] < 2).all()
    # One decision step per match at this length: 0.2 s is 6 ticks, and a decision is 2 ticks.
    assert len(demos) == 2 * 3
    assert np.all(np.bincount(demos.match_id) == 3)


def test_enemy_offset_points_at_the_living_enemy():
    maps = load_split("val")[:1]
    seconds = legal_duration(0.2)
    env = MatchEnv(
        1, team_size=1, seed=0,
        cfg=EngineConfig(team_size=1, match=MatchConfig(duration=seconds)),
        maps=maps,
    )
    env.reset()
    ctx = PolicyContext(env, team_slots(env, 0), np.random.default_rng(0))
    offset = true_enemy_offset(ctx)
    w = env.world
    assert offset.shape == (1, 2)
    np.testing.assert_allclose(offset[0, 0], (w.x[0, 1] - w.x[0, 0]) / 30.0, atol=1e-5)
    np.testing.assert_allclose(offset[0, 1], (w.y[0, 1] - w.y[0, 0]) / 30.0, atol=1e-5)
    recorded = collect_rusher_demos(1, seed=0, seconds=seconds, batch_size=1, maps=maps, with_enemy_offset=True)
    assert recorded.obs.shape[1] == 332
    assert np.any(recorded.obs[:, -2:] != 0)
