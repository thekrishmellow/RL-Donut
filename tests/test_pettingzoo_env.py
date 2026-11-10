import subprocess
import sys
import warnings

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

from rl_donut.env import EngineConfig, MatchConfig, MatchEnv
from rl_donut.env.pettingzoo_env import RLDonutParallelEnv
from rl_donut.mapgen import load_split

SECONDS = 4.0
STEPS = 60       # decision steps in a 4 s match


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def short(team_size):
    return EngineConfig(team_size=team_size, match=MatchConfig(duration=SECONDS))


def make(maps, team_size=1, **kwargs):
    return RLDonutParallelEnv(team_size, cfg=short(team_size), maps=maps, **kwargs)


def sampled(env):
    return {agent: env.action_space(agent).sample() for agent in env.agents}


@pytest.mark.parametrize("team_size", [1, 2])
def test_official_parallel_api_test_passes_without_warnings(maps, team_size):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parallel_api_test(make(maps, team_size), num_cycles=3 * STEPS)


def test_official_seed_test_passes(maps):
    parallel_seed_test(lambda: make(maps, 2), num_cycles=STEPS)


def test_agents_spaces_and_state(maps):
    env = make(maps, 2)
    obs, infos = env.reset(seed=0)
    assert env.possible_agents == ["a_0", "a_1", "b_0", "b_1"] == list(obs) == list(infos)
    for agent in env.agents:
        assert env.observation_space(agent).contains(obs[agent])
        assert env.action_space(agent).contains(env.action_space(agent).sample())
    assert env.state_space.contains(env.state())
    np.testing.assert_array_equal(env.state(), env.env.get_state()[0, 0])


def test_same_observations_and_rewards_as_match_env(maps):
    env = make(maps, 2)
    obs, _ = env.reset(seed=5)
    twin = MatchEnv(1, 2, seed=5, cfg=short(2), maps=maps)
    np.testing.assert_array_equal(np.stack(list(obs.values())), twin.reset()[0])
    rng = np.random.default_rng(0)
    for _ in range(STEPS - 1):
        discrete = rng.integers(0, [3, 2, 2], size=(4, 3))
        aim = rng.uniform(-np.pi, np.pi, size=(4, 1))
        actions = {agent: {"discrete": discrete[i], "aim": aim[i]} for i, agent in enumerate(env.agents)}
        obs, rewards, *_ = env.step(actions)
        r = twin.step(discrete[None], aim[None])
        np.testing.assert_array_equal(np.stack(list(obs.values())), r.obs[0])
        assert list(rewards.values()) == (r.events.kills - r.events.deaths)[0].tolist()


def test_match_end_truncates_everyone_and_reports_each_side(maps):
    env = make(maps, 2)
    env.reset(seed=1)
    for step in range(STEPS):
        obs, rewards, terminated, truncated, infos = env.step(sampled(env))
        assert not any(terminated.values())
        if step < STEPS - 1:
            assert not any(truncated.values()) and env.agents
    assert all(truncated.values()) and env.agents == []
    a, b = infos["a_0"], infos["b_0"]
    assert a["outcome"] == -b["outcome"] and a["score"] == b["score"][::-1]
    assert a["map_seed"] == b["map_seed"] and set(a["stats"]) >= {"kills", "deaths", "hover_s"}
    with pytest.raises(RuntimeError):
        env.step(sampled(env))


def test_reset_without_seed_deals_maps_in_deck_order(maps):
    env, twin = make(maps), MatchEnv(1, 1, seed=2, cfg=short(1), maps=maps)
    env.reset(seed=2)
    twin.reset()
    played = []
    for _ in range(3):
        assert env.env.world.seeds[0] == twin.world.seeds[0]
        for _ in range(STEPS):
            *_, infos = env.step(sampled(env))
            twin.step(np.zeros((1, 2, 3), int), np.zeros((1, 2, 1)))
        played.append(infos["a_0"]["map_seed"])
        env.reset()
    assert len(set(played)) == 3


def test_custom_reward_fn(maps):
    seen = []

    def reward_fn(events):
        seen.append(events.kills.shape)
        return np.array([[1.0, -2.0]])

    env = make(maps, reward_fn=reward_fn)
    env.reset(seed=0)
    _, rewards, *_ = env.step(sampled(env))
    assert rewards == {"a_0": 1.0, "b_0": -2.0} and seen == [(1, 2)]


def test_bad_actions_rejected(maps):
    env = make(maps)
    with pytest.raises(RuntimeError):
        env.step({})
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({"a_0": env.action_space("a_0").sample()})
    with pytest.raises(ValueError):
        env.step({"a_0": {"discrete": [3, 0, 0], "aim": [0.0]}, "b_0": {"discrete": [1, 0, 0], "aim": [0.0]}})


def test_engine_does_not_import_pettingzoo():
    code = "import sys, rl_donut.env; assert 'pettingzoo' not in sys.modules and 'gymnasium' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
