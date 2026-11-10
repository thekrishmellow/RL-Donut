import numpy as np
import pytest

from rl_donut.env import EngineConfig, MatchConfig, MatchEnv
from rl_donut.env.bots import RandomBot, StaticTargetBot
from rl_donut.env.policy import PolicyContext, play_matches, team_slots
from rl_donut.mapgen import load_split

SECONDS = 10.0


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def env(maps, n=8, team_size=1, seed=0):
    cfg = EngineConfig(team_size=team_size, match=MatchConfig(duration=SECONDS))
    return MatchEnv(n, team_size, seed=seed, cfg=cfg, maps=maps)


def test_team_slots_are_match_major(maps):
    e = env(maps, n=3, team_size=2)
    assert team_slots(e, 1).tolist() == [[0, 2], [0, 3], [1, 2], [1, 3], [2, 2], [2, 3]]


@pytest.mark.parametrize("team_size", [1, 2])
def test_play_matches_returns_one_info_per_match(maps, team_size):
    infos = play_matches(env(maps, team_size=team_size), RandomBot(), RandomBot())
    assert len(infos) == 8
    for info in infos:
        assert info["outcome"].tolist() in ([1, -1], [-1, 1], [0, 0])
        assert info["agents"]["kills"].shape == (2 * team_size,)


def test_static_target_never_scores_or_leaves_the_ground(maps):
    for info in play_matches(env(maps), RandomBot(), StaticTargetBot()):
        a = info["agents"]
        assert a["kills"][1] == 0 and a["shots"][1] == 0 and a["airborne_s"][1] == 0
        assert info["outcome"][0] >= 0


def test_static_target_does_not_move(maps):
    e = env(maps)
    e.reset()
    x0 = e.world.x[:, 1].copy()
    ctx = PolicyContext(e, team_slots(e, 1), np.random.default_rng(0))
    bot, rand = StaticTargetBot(), RandomBot()
    rctx = PolicyContext(e, team_slots(e, 0), np.random.default_rng(1))
    for _ in range(100):
        obs = e.observer.observe(e.vis)
        d = np.zeros((8, 2, 3), dtype=np.int64)
        c = np.zeros((8, 2, 1))
        d[:, 0], c[:, 0] = rand.act(obs[:, 0], rctx)
        d[:, 1], c[:, 1] = bot.act(obs[:, 1], ctx)
        r = e.step(d, c)
        alive = e.world.alive[:, 1] & ~r.events.deaths[:, 1].astype(bool)
        np.testing.assert_array_equal(e.world.x[alive, 1], x0[alive])


def test_same_seed_same_results(maps):
    def run(seed):
        infos = play_matches(env(maps, seed=3), RandomBot(), RandomBot(), seed=seed)
        return [(i["map_seed"], i["agents"]["shots"].tolist(), i["agents"]["damage_dealt"].tolist()) for i in infos]
    assert run(0) == run(0) != run(1)


def test_wrong_action_shape_rejected(maps):
    class Broken:
        name = "broken"

        def act(self, obs, ctx):
            return np.zeros((len(obs), 2), dtype=np.int64), np.zeros((len(obs), 1))

    with pytest.raises(ValueError, match="broken"):
        play_matches(env(maps), Broken(), RandomBot())
