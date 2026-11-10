"""Shared contract test for environments conforming to docs/handoff_phase5_6.md §4.

Owned jointly by Team A and Team B.
Must pass for StandInEnv (Team B) and MatchEnv (Team A).
"""

import numpy as np
import pytest

from rl_donut.rl.stand_in import StandInEnv, Events, StepResult


@pytest.fixture(params=[1, 2])
def env(request):
    """Factory fixture testing both 1v1 (team_size=1) and 2v2 (team_size=2)."""
    return StandInEnv(n_matches=4, team_size=request.param, split="val", seed=42)


def test_spec_version_present(env):
    assert hasattr(env, "spec_version")
    assert isinstance(env.spec_version, str)
    assert len(env.spec_version) > 0


def test_reset_shape_and_dtype(env):
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (env.n_matches, env.n_agents, env.obs_dim)


def test_step_shapes_and_dtypes(env):
    env.reset()
    actions = np.zeros((env.n_matches, env.n_agents, len(env.action_nvec)), dtype=np.int64)
    # actions: move=1 (stay), jet=0, aim=0, shoot=0
    actions[..., 0] = 1

    res = env.step(actions)
    assert isinstance(res, StepResult)
    assert res.obs.shape == (env.n_matches, env.n_agents, env.obs_dim)
    assert res.obs.dtype == np.float32

    assert res.done.shape == (env.n_matches,)
    assert res.done.dtype == bool

    assert res.final_obs.shape == (env.n_matches, env.n_agents, env.obs_dim)
    assert res.final_obs.dtype == np.float32

    assert isinstance(res.info, list)
    assert len(res.info) == env.n_matches

    # Events checks
    ev = res.events
    assert isinstance(ev, Events)
    assert ev.damage_dealt.shape == (env.n_matches, env.n_agents)
    assert ev.damage_dealt.dtype == np.float32
    assert ev.damage_taken.shape == (env.n_matches, env.n_agents)
    assert ev.damage_taken.dtype == np.float32
    assert ev.kills.shape == (env.n_matches, env.n_agents)
    assert ev.kills.dtype == np.int64
    assert ev.deaths.shape == (env.n_matches, env.n_agents)
    assert ev.deaths.dtype == np.int64
    assert ev.fell.shape == (env.n_matches, env.n_agents)
    assert ev.fell.dtype == bool
    assert ev.shots.shape == (env.n_matches, env.n_agents)
    assert ev.shots.dtype == np.int64
    assert ev.hits.shape == (env.n_matches, env.n_agents)
    assert ev.hits.dtype == np.int64
    assert ev.outcome.shape == (env.n_matches, 2)
    assert ev.outcome.dtype == np.int64


def test_outcome_only_when_done(env):
    env.reset()
    actions = np.ones((env.n_matches, env.n_agents, len(env.action_nvec)), dtype=np.int64)
    res = env.step(actions)
    not_done = ~res.done
    if not_done.any():
        assert (res.events.outcome[not_done] == 0).all(), "outcome must be 0 for matches that are not done"


def test_seeding_reproducibility():
    """Two environments with identical seeds must produce identical initial observations and transitions."""
    env1 = StandInEnv(n_matches=2, team_size=1, split="val", seed=123)
    env2 = StandInEnv(n_matches=2, team_size=1, split="val", seed=123)

    obs1 = env1.reset()
    obs2 = env2.reset()
    np.testing.assert_allclose(obs1, obs2)

    actions = np.zeros((2, 2, 4), dtype=np.int64)
    actions[..., 0] = 2  # move right
    actions[..., 1] = 1  # jetpack on

    res1 = env1.step(actions)
    res2 = env2.step(actions)

    np.testing.assert_allclose(res1.obs, res2.obs)
    np.testing.assert_allclose(res1.events.damage_dealt, res2.events.damage_dealt)


def test_auto_reset_at_match_end():
    """A short match must auto-reset on completion while preserving final_obs."""
    env = StandInEnv(n_matches=2, team_size=1, split="val", seed=99, match_duration_seconds=0.2)
    env.reset()

    actions = np.zeros((2, 2, 4), dtype=np.int64)
    actions[..., 0] = 1  # stay

    done_seen = False
    for _ in range(10):
        res = env.step(actions)
        if res.done.any():
            done_seen = True
            # final_obs must have valid float32 values
            assert not np.isnan(res.final_obs).any()
            # next obs has reset match time
            assert res.obs.shape == res.final_obs.shape
            break

    assert done_seen, "match should have ended within the time limit"


def test_events_consistency_over_combat():
    """Hits cannot exceed shots, and damage dealt is non-negative."""
    env = StandInEnv(n_matches=2, team_size=1, split="val", seed=42)
    env.reset()

    # Shoot with aim 0
    actions = np.zeros((2, 2, 4), dtype=np.int64)
    actions[..., 0] = 1  # stay
    actions[..., 3] = 1  # trigger pulled

    for _ in range(20):
        res = env.step(actions)
        ev = res.events
        assert (ev.hits <= ev.shots).all()
        assert (ev.damage_dealt >= 0.0).all()
        assert (ev.damage_taken >= 0.0).all()
