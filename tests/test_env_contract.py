"""Shared contract test: any environment the trainer uses must pass it (handoff guide, section 4).

Owned jointly by Team A and Team B. Each entry of `FACTORIES` builds an environment as
`factory(n_matches, team_size, seed, match_seconds)`; `match_seconds` shortens matches so
auto-reset can be tested quickly. To check another environment (for example a stand-in),
add a factory for it.

`SPECS` records what each `spec_version` promises. Changing observations, actions, state or
events means bumping `spec_version`, adding its entry here, and saying so to the other team.
"""

import hashlib
import math

import numpy as np
import pytest

from rl_donut.env import FEATURE_NAMES, EngineConfig, MatchEnv
from rl_donut.env.config import MatchConfig
from rl_donut.env.match_env import STATS

SPECS = {
    "0.3": dict(obs_dim=330, action_nvec=(3, 2, 2), action_cont_dim=1, state_dim=50,
                features="e5b3b17503ad7b5f", stats=STATS),
}
EVENT_KINDS = {"damage_dealt": "f", "damage_taken": "f", "kills": "i", "deaths": "i", "fell": "b",
               "shots": "i", "hits": "i"}


def match_env(n_matches, team_size, seed, match_seconds=None):
    match = MatchConfig(duration=match_seconds) if match_seconds else MatchConfig()
    return MatchEnv(n_matches, team_size=team_size, split="val", seed=seed,
                    cfg=EngineConfig(team_size=team_size, match=match))


FACTORIES = {"MatchEnv": match_env}


@pytest.fixture(params=[(name, ts) for name in FACTORIES for ts in (1, 2)], ids=lambda p: f"{p[0]}-{p[1]}v{p[1]}")
def make(request):
    name, team_size = request.param
    return lambda n_matches=4, seed=0, match_seconds=None: FACTORIES[name](n_matches, team_size, seed, match_seconds)


def random_actions(env, rng, fire=0.5):
    shape = (env.n_matches, env.n_agents)
    discrete = np.stack([rng.integers(0, n, size=shape) for n in env.action_nvec], axis=-1).astype(np.int64)
    discrete[..., -1] = rng.random(shape) < fire
    continuous = rng.uniform(-math.pi, math.pi, size=shape + (env.action_cont_dim,)).astype(np.float32)
    return discrete, continuous


def play(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    return [env.step(*random_actions(env, rng)) for _ in range(steps)]


# --- attributes and the spec ---

def test_attributes_match_the_spec(make):
    env = make()
    assert isinstance(env.spec_version, str) and env.spec_version in SPECS, "unknown spec_version: add it to SPECS"
    spec = SPECS[env.spec_version]
    assert env.obs_dim == spec["obs_dim"] and tuple(env.action_nvec) == spec["action_nvec"]
    assert env.action_cont_dim == spec["action_cont_dim"] and env.state_dim == spec["state_dim"]
    assert env.n_matches == 4 and env.team.shape == (env.n_agents,)
    team_size = env.n_agents // 2
    assert env.team.tolist() == [0] * team_size + [1] * team_size


def test_observation_layout_is_pinned_by_the_spec(make):
    env = make()
    spec = SPECS[env.spec_version]
    assert hashlib.sha256(",".join(FEATURE_NAMES).encode()).hexdigest()[:16] == spec["features"], \
        "observation features changed: bump spec_version"
    assert tuple(STATS) == spec["stats"]


# --- shapes and kinds ---

def test_reset(make):
    env = make()
    obs = env.reset()
    assert obs.shape == (env.n_matches, env.n_agents, env.obs_dim) and obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_step_shapes_and_kinds(make):
    env = make()
    env.reset()
    r = env.step(*random_actions(env, np.random.default_rng(0)))
    M, A = env.n_matches, env.n_agents
    assert r.obs.shape == r.final_obs.shape == (M, A, env.obs_dim)
    assert r.obs.dtype == r.final_obs.dtype == np.float32
    assert r.done.shape == (M,) and r.done.dtype == bool
    assert isinstance(r.info, list) and len(r.info) == M
    for name, kind in EVENT_KINDS.items():
        value = getattr(r.events, name)
        assert value.shape == (M, A), name
        assert value.dtype.kind == kind or (kind == "i" and value.dtype.kind == "u"), name
    assert r.events.outcome.shape == (M, 2) and r.events.outcome.dtype.kind == "i"


def test_state(make):
    env = make()
    env.reset()
    state = env.get_state()
    assert state.shape == (env.n_matches, 2, env.state_dim) and state.dtype == np.float32
    assert np.isfinite(state).all()


@pytest.mark.parametrize("bad", ["move", "shape", "aim"])
def test_invalid_actions_are_rejected(make, bad):
    env = make()
    env.reset()
    discrete, continuous = random_actions(env, np.random.default_rng(0))
    if bad == "move":
        discrete[0, 0, 0] = env.action_nvec[0]
    elif bad == "shape":
        discrete = discrete[..., :-1]
    else:
        continuous[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        env.step(discrete, continuous)


# --- behaviour ---

def test_same_seed_same_run_different_seed_differs(make):
    def run(seed):
        env = make(seed=seed)
        out = [env.reset()]
        for r in play(env, 40):
            out += [r.obs, r.events.damage_dealt.astype(np.float32)[..., None].repeat(env.obs_dim, -1)]
        return np.stack(out)
    a, b, c = run(7), run(7), run(8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_auto_reset_final_obs_and_info(make):
    env = make(match_seconds=0.4)                      # 6 decision steps
    first = env.reset()
    results = play(env, 6)
    for r in results[:-1]:
        assert not r.done.any() and all(info == {} for info in r.info)
        assert (r.events.outcome == 0).all()
    end = results[-1]
    assert end.done.all()                              # matches start together, so they end together
    assert np.isfinite(end.final_obs).all()
    assert not np.array_equal(end.final_obs, end.obs)  # obs is the next match's first observation
    np.testing.assert_array_equal(end.obs.shape, first.shape)
    for m, info in enumerate(end.info):
        assert {"map_seed", "score", "outcome", "agents"} <= set(info)
        assert set(info["agents"]) == set(SPECS[env.spec_version]["stats"])
        assert all(np.shape(v) == (env.n_agents,) for v in info["agents"].values())
        assert info["outcome"].tolist() in ([1, -1], [-1, 1], [0, 0])
        np.testing.assert_array_equal(info["outcome"], end.events.outcome[m])


def test_events_add_up_over_a_match(make):
    env = make(n_matches=6, match_seconds=6.0)         # 90 decision steps
    env.reset()
    results = play(env, 90)
    kills = sum(r.events.kills for r in results)
    deaths = sum(r.events.deaths for r in results)
    for r in results:
        ev = r.events
        assert (ev.hits <= ev.shots).all() and (ev.damage_dealt >= 0).all() and (ev.damage_taken >= 0).all()
        assert set(np.unique(ev.deaths)) <= {0, 1} and (ev.deaths[ev.fell] == 1).all()
        np.testing.assert_allclose(ev.damage_dealt.sum(axis=1), ev.damage_taken.sum(axis=1))   # no friendly fire
    assert (kills.sum(axis=1) <= deaths.sum(axis=1)).all()                                     # every kill is a death
    team_size = env.n_agents // 2
    for m, info in enumerate(results[-1].info):
        assert info["score"].tolist() == [kills[m, :team_size].sum(), kills[m, team_size:].sum()]
        np.testing.assert_array_equal(info["agents"]["kills"], kills[m])
