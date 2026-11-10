import math

import numpy as np
import pytest

from rl_donut.env import DISCRETE_NVEC, OBS_DIM
from rl_donut.env.match_env import STATE_DIM, STATS, MapDeck, MatchEnv
from rl_donut.env.observe import FEATURE_NAMES
from rl_donut.mapgen import load_split

STEPS = 900      # decision steps in a 60 s match


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def random_actions(env, rng):
    shape = (env.n_matches, env.n_agents)
    discrete = np.stack([rng.integers(0, n, size=shape) for n in DISCRETE_NVEC], axis=-1)
    return discrete, rng.uniform(-math.pi, math.pi, size=shape + (1,))


def aggressive_actions(env, rng):
    """Move towards the enemy side, jet often, aim roughly at the first enemy, fire a lot."""
    w = env.world
    shape = (env.n_matches, env.n_agents)
    discrete = np.stack([np.where(rng.random(shape) < 0.75, 2, rng.integers(0, 3, shape)),
                         rng.random(shape) < 0.45, rng.random(shape) < 0.6], axis=-1).astype(np.int64)
    enemy = (np.arange(env.n_agents) + env.world.cfg.team_size) % env.n_agents
    angle = np.arctan2(w.y[:, enemy] - w.y, w.x[:, enemy] - w.x)
    angle = np.where(env.mirror, np.pi - angle, angle) + rng.normal(0, 0.05, shape)   # into each agent's frame
    return discrete, angle[..., None]


def test_contract_attributes(maps):
    env = MatchEnv(4, team_size=2, maps=maps)
    assert env.obs_dim == OBS_DIM == 330 and env.action_nvec == (3, 2, 2) and env.action_cont_dim == 1
    assert env.state_dim == STATE_DIM == 50 and env.spec_version == "0.3"
    assert env.team.tolist() == [0, 0, 1, 1] and env.n_agents == 4


def test_step_before_reset_fails(maps):
    env = MatchEnv(2, maps=maps)
    with pytest.raises(RuntimeError):
        env.step(*random_actions(env, np.random.default_rng(0)))


def test_shapes(maps):
    env = MatchEnv(3, team_size=2, maps=maps)
    obs = env.reset()
    assert obs.shape == (3, 4, OBS_DIM) and obs.dtype == np.float32
    r = env.step(*random_actions(env, np.random.default_rng(0)))
    assert r.obs.shape == r.final_obs.shape == (3, 4, OBS_DIM)
    assert r.done.shape == (3,) and r.events.kills.shape == (3, 4) and len(r.info) == 3
    assert env.get_state().shape == (3, 2, STATE_DIM)


def test_deck_deals_every_map_once_per_pass(maps):
    deck = MapDeck(maps[:7], np.random.default_rng(0))
    first, second = deck.draw(7), deck.draw(7)
    assert sorted(m.seed for m in first) == sorted(m.seed for m in second) == sorted(m.seed for m in maps[:7])
    assert [m.seed for m in first] != [m.seed for m in second]


def test_map_filter(maps):
    env = MatchEnv(8, maps=maps, map_filter=lambda m: not m.pits)
    assert all(not m.pits for m in env.deck.maps) and len(env.deck.maps) < len(maps)


def test_full_match_auto_reset_and_stats(maps):
    env = MatchEnv(8, team_size=2, maps=maps, seed=3)
    rng = np.random.default_rng(0)
    env.reset()
    seeds = env.world.seeds.copy()
    totals = {k: np.zeros((8, 4)) for k in ("kills", "deaths", "shots", "hits", "damage_dealt")}
    for step in range(STEPS):
        r = env.step(*aggressive_actions(env, rng))
        for k in totals:
            totals[k] += getattr(r.events, k)
        if step < STEPS - 1:
            assert not r.done.any() and r.info == [{}] * 8
    assert r.done.all()
    assert (env.world.tick == 0).all() and (env.world.score == 0).all()
    assert (env.world.seeds != seeds).any()                              # next maps from the deck
    time_left = FEATURE_NAMES.index("time_left")
    assert np.allclose(r.final_obs[..., time_left], 0.0) and np.allclose(r.obs[..., time_left], 1.0)
    for m, info in enumerate(r.info):
        a = info["agents"]
        assert set(a) == set(STATS) and info["map_seed"] == seeds[m]
        for k in totals:
            np.testing.assert_allclose(a[k], totals[k][m])
        assert (a["spawn_kills"] <= a["kills"]).all() and (a["falls"] <= a["deaths"]).all()
        assert (0 <= a["hover_s"]).all() and (a["hover_s"] <= a["airborne_s"]).all() and (a["airborne_s"] <= 60).all()
        assert info["score"].tolist() == [a["kills"][:2].sum(), a["kills"][2:].sum()]
    assert sum(info["agents"]["hits"].sum() for info in r.info) > 0
    fired = [d for info in r.info for d in info["agents"]["fire_distance"] if not np.isnan(d)]
    assert fired and all(0 < d < 110 for d in fired)


def test_same_seed_same_run_and_different_seed_differs(maps):
    def run(seed):
        env = MatchEnv(4, team_size=1, maps=maps, seed=seed)
        rng = np.random.default_rng(0)
        out = [env.reset()]
        for _ in range(150):
            r = env.step(*aggressive_actions(env, rng))
            out.append(r.obs)
        return np.stack(out), env.world.seeds.copy()
    (a, sa), (b, sb), (c, sc) = run(1), run(1), run(2)
    np.testing.assert_array_equal(a, b)
    assert (sa == sb).all() and not ((sa == sc).all() and np.array_equal(a, c))


def test_same_action_on_both_sides_keeps_the_match_mirrored(maps):
    """Maps are mirror images and team B acts in a mirrored frame, so giving both sides the same
    actions (no shooting) keeps every observation equal to its twin's."""
    m = next(m for m in maps if (m.grid == m.grid[:, ::-1]).all())
    env = MatchEnv(1, team_size=2, maps=[m])
    obs = env.reset()
    rng = np.random.default_rng(0)
    swap = [2, 3, 0, 1]
    for _ in range(100):
        d, c = random_actions(env, rng)
        d[..., 2] = 0
        d[:, 2:], c[:, 2:] = d[:, :2], c[:, :2]
        obs = env.step(d, c).obs
        np.testing.assert_allclose(obs, obs[:, swap], atol=1e-6)
    state = env.get_state()
    np.testing.assert_allclose(state[:, 0], state[:, 1], atol=1e-6)


def test_state_sees_through_walls_and_orders_own_team_first(maps):
    env = MatchEnv(2, team_size=1, maps=maps)
    env.reset()
    s = env.get_state()
    w = env.world
    W = w.grids.shape[2]
    assert s[0, 0, 0] == s[0, 0, 24] == 1 and s[0, 0, 12] == s[0, 0, 36] == 0     # 1v1: second slots empty
    assert s[0, 0, 2] == pytest.approx(w.x[0, 0] / W) and s[0, 0, 26] == pytest.approx(w.x[0, 1] / W)
    assert s[0, 1, 2] == pytest.approx((W - w.x[0, 1]) / W)                          # team B: itself first, mirrored
