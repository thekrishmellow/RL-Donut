import numpy as np
import pytest

from rl_donut.env import MatchEnv
from rl_donut.env.bots import RusherBot, TurretBot
from rl_donut.env.perception import Reaction, controls_to_actions, look
from rl_donut.env.policy import PolicyContext
from rl_donut.env.visibility import visibility
from rl_donut.mapgen import load_split


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:8]


def placed(maps):
    """One match with both agents in open air and a rock column available to close."""
    env = MatchEnv(1, maps=maps[:1], seed=0)
    env.reset()
    grid = np.zeros_like(env.world.grids[0])
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    grid[30:, :] = True
    env.world.grids[0] = grid
    env.world.x[0] = [16.0, 36.0]
    env.world.y[0] = [28.0, 28.0]
    env.world.alive[:] = True
    env.vis = visibility(env.world)
    return env


def test_look_sees_across_open_air_and_not_through_rock(maps):
    env = placed(maps)
    ctx = PolicyContext(env, np.array([[0, 0]]), np.random.default_rng(0))
    seen = look(ctx)
    assert seen.enemy[0] == 1 and seen.in_range[0] and seen.distance[0] == pytest.approx(20.0)
    env.world.grids[0][10:30, 26] = True
    env.vis = visibility(env.world)
    assert look(ctx).enemy[0] == -1


def test_team_b_aim_is_mirrored_into_its_own_frame(maps):
    env = placed(maps)
    ctx = PolicyContext(env, np.array([[0, 1]]), np.random.default_rng(0))
    discrete, continuous = controls_to_actions(ctx, [0], [False], [np.pi], [False])
    assert discrete[0, 0] == 1 and continuous[0, 0] == pytest.approx(0.0)


def test_reaction_restarts_on_a_new_match(maps):
    env = placed(maps)
    ctx = PolicyContext(env, np.array([[0, 0]]), np.random.default_rng(0))
    memory = Reaction()
    seen = look(ctx)
    assert memory.seen_for(ctx, seen)[0] == 1
    assert memory.seen_for(ctx, seen)[0] == 2
    env.world.episode[0] += 1
    assert memory.seen_for(ctx, seen)[0] == 1


def test_turret_shoots_a_visible_enemy_and_never_moves(maps):
    env = placed(maps)
    ctx = PolicyContext(env, np.array([[0, 0]]), np.random.default_rng(0))
    bot = TurretBot(aim_sigma_deg=0, reaction=1)
    discrete, _ = bot.act(np.zeros((1, 1)), ctx)
    assert discrete[0].tolist() == [1, 0, 1]
    env.world.grids[0][10:30, 26] = True
    env.vis = visibility(env.world)
    discrete, _ = bot.act(np.zeros((1, 1)), ctx)
    assert discrete[0, 2] == 0


def test_turret_waits_out_its_reaction(maps):
    env = placed(maps)
    ctx = PolicyContext(env, np.array([[0, 0]]), np.random.default_rng(0))
    bot = TurretBot(aim_sigma_deg=0, reaction=2)
    assert bot.act(np.zeros((1, 1)), ctx)[0][0, 2] == 0
    assert bot.act(np.zeros((1, 1)), ctx)[0][0, 2] == 1


def test_rusher_walks_toward_an_enemy_on_its_floor(maps):
    env = placed(maps)
    env.world.x[0, 1] = 50.0
    env.world.y[0] = 30 - env.world.cfg.hitbox_h / 2
    ctx = PolicyContext(env, np.array([[0, 0]]), np.random.default_rng(0))
    bot = RusherBot(aim_sigma_deg=0, reaction=1)
    discrete, _ = bot.act(np.zeros((1, 1)), ctx)
    assert discrete[0, 0] == 2          # right, toward the enemy at x = 50
