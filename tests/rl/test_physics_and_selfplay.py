"""Human rows that enter rock are dropped, and the duel opponent is a copy of the policy."""

import numpy as np
import torch

from rl_donut.env.observe import CROP_H, CROP_W, FEATURE_NAMES, OBS_DIM
from rl_donut.play.physics_data import air_legal, presses_rock, seen_aim_target
from rl_donut.play.selfplay import SelfOpponent, aim_and_fire
from rl_donut.rl.clone import Demonstrations
from rl_donut.rl.model import ActorCritic


def _obs(n=4):
    return np.zeros((n, OBS_DIM), dtype=np.float32)


def _paint(obs, row, col, value=1.0):
    start = len(FEATURE_NAMES)
    obs[..., start + row * CROP_W + col] = value


def test_a_step_into_the_next_rock_is_not_a_lesson():
    obs = _obs(3)
    centre = CROP_H // 2
    _paint(obs, centre, CROP_W // 2 + 1)
    discrete = np.array([[2, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int64)
    _paint(obs[2:3], centre - 1, CROP_W // 2)
    blocked = presses_rock(obs, discrete)
    assert blocked.tolist() == [True, False, True]
    demos = Demonstrations(obs, discrete, np.zeros((3, 1), np.float32), np.arange(3))
    assert air_legal(demos).tolist() == [False, True, False]


def test_seen_enemy_offset_is_the_aim():
    obs = _obs(2)
    seen = FEATURE_NAMES.index("enemy0_seen_by_me")
    dx = FEATURE_NAMES.index("enemy0_dx")
    dy = FEATURE_NAMES.index("enemy0_dy")
    obs[0, seen] = 1
    obs[0, dx] = 1
    obs[0, dy] = 0
    obs[1, dx] = 1
    target, mask = seen_aim_target(obs)
    assert mask.tolist() == [True, False]
    assert abs(float(target[0])) < 1e-5


def test_aim_loss_turns_the_gun_toward_a_seen_enemy():
    torch.manual_seed(0)
    model = ActorCritic(OBS_DIM, (3, 2, 2), hidden_dim=32, cont_dim=1)
    with torch.no_grad():
        model.aim_mean.bias.fill_(1.5)
    obs = _obs(32)
    seen = FEATURE_NAMES.index("enemy0_seen_by_me")
    dx = FEATURE_NAMES.index("enemy0_dx")
    obs[:, seen] = 1
    obs[:, dx] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def error():
        with torch.no_grad():
            mean = model.aim_mean(model.actor_backbone(torch.from_numpy(obs)))
        return float(mean.abs().mean())

    before = error()
    for _ in range(8):
        aim_and_fire(model, optimizer, obs, coef=1.0)
    assert error() < before - 0.2


def test_a_seen_enemy_is_aimed_at_and_a_wall_step_is_cancelled():
    from rl_donut.play.physics_data import engage
    obs = _obs(1)
    seen = FEATURE_NAMES.index("enemy0_seen_by_me")
    dx = FEATURE_NAMES.index("enemy0_dx")
    dy = FEATURE_NAMES.index("enemy0_dy")
    obs[0, seen] = 1
    obs[0, dx] = 0
    obs[0, dy] = 1
    _paint(obs, CROP_H // 2, CROP_W // 2 + 1)
    discrete = np.array([[2, 0, 0]], dtype=np.int64)
    continuous = np.zeros((1, 1), np.float32)
    discrete, continuous = engage(obs, discrete, continuous)
    assert int(discrete[0, 0]) == 1
    assert int(discrete[0, 2]) == 1
    assert abs(float(continuous[0, 0]) - np.pi / 2) < 1e-5


def test_the_opponent_is_a_copy_that_can_act():
    torch.manual_seed(0)
    model = ActorCritic(8, (3, 2, 2), hidden_dim=16, cont_dim=1)
    opponent = SelfOpponent(model)
    assert opponent.name == "self"
    with torch.no_grad():
        model.actor_backbone[0].bias.add_(1)
    opponent.pull(model)
    torch.testing.assert_close(
        opponent.model.actor_backbone[0].bias,
        model.actor_backbone[0].bias,
    )
    discrete, continuous = opponent.act(np.zeros((2, 8), np.float32), None)
    assert discrete.shape == (2, 3)
    assert continuous.shape == (2, 1)
