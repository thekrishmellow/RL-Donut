import math

import numpy as np
import torch

from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.env.observe import FEATURE_NAMES
from rl_donut.env.policy import PolicyContext, team_slots
from rl_donut.rl.circulate import MapRoamer, circulate, enemy_in_view
from rl_donut.rl.model import AIM_STD_INIT_DEG, AIM_STD_MAX_DEG, ActorCritic


def _obs(n, seen=0.0, remembered=0.0):
    obs = np.zeros((n, len(FEATURE_NAMES) + 21 * 13), dtype=np.float32)
    seen_at = FEATURE_NAMES.index("enemy0_seen_by_me")
    remembered_at = FEATURE_NAMES.index("enemy0_remembered")
    obs[:, seen_at] = seen
    obs[:, remembered_at] = remembered
    return obs


def test_aim_spread_stays_the_size_of_a_body():
    model = ActorCritic(4, (3, 2, 2), hidden_dim=16, cont_dim=1)
    fresh = float(model.aim_std().detach())
    assert abs(fresh - math.radians(AIM_STD_INIT_DEG)) < 1e-5
    with torch.no_grad():
        model.aim_log_std.fill_(5.0)
    assert abs(float(model.aim_std().detach()) - math.radians(AIM_STD_MAX_DEG)) < 1e-5

    x = torch.zeros(4, 4)
    given_d = torch.zeros((4, 3), dtype=torch.long)
    given_c = torch.zeros((4, 1))
    with torch.no_grad():
        model.aim_log_std.fill_(math.log(math.radians(2)))
        narrow = model.get_action_and_value(x, given_d, given_c)[3]
        model.aim_log_std.fill_(math.log(math.radians(6)))
        wide = model.get_action_and_value(x, given_d, given_c)[3]
    torch.testing.assert_close(narrow, wide)


def test_hidden_agent_flies_to_another_floor():
    """Official spawns hide the teams. The roamer has to leave that spot."""
    env = MatchEnv(
        1, team_size=1, split="val", seed=0,
        cfg=EngineConfig(team_size=1, match=MatchConfig(duration=8.0)),
    )
    obs = env.reset()
    slots = team_slots(env, 0)
    ctx = PolicyContext(env, slots, np.random.default_rng(0))
    row = obs[slots[:, 0], slots[:, 1]]
    assert not enemy_in_view(row).any()
    start = float(env.world.x[0, 0]), float(env.world.y[0, 0])
    roamer = MapRoamer()
    discrete = np.ones((1, 3), dtype=np.int64)
    continuous = np.zeros((1, 1), dtype=np.float32)
    moved = 0.0
    for _ in range(120):
        action_d, action_c = roamer.apply(row, discrete, continuous, ctx)
        assert int(action_d[0, 2]) == 0
        full_d = np.zeros((1, 2, 3), dtype=np.int64)
        full_c = np.zeros((1, 2, 1), dtype=np.float32)
        full_d[0, 0] = action_d[0]
        full_c[0, 0] = action_c[0]
        result = env.step(full_d, full_c)
        row = result.obs[slots[:, 0], slots[:, 1]]
        if enemy_in_view(row).any():
            break
        moved = max(moved, math.hypot(env.world.x[0, 0] - start[0], env.world.y[0, 0] - start[1]))
        if result.done.any():
            break
    assert moved > 4.0


def test_a_seen_enemy_keeps_the_policy_action():
    obs = _obs(2, seen=1.0)
    assert enemy_in_view(obs).all()
    discrete = np.tile(np.array([[2, 0, 1]], dtype=np.int64), (2, 1))
    continuous = np.ones((2, 1), dtype=np.float32)
    out, cont = circulate(obs, discrete, continuous, None)
    np.testing.assert_array_equal(out, discrete)
    np.testing.assert_array_equal(cont, continuous)

    remembered = _obs(1, remembered=1.0)
    assert enemy_in_view(remembered).all()
