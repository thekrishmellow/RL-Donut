import numpy as np
import pytest

from rl_donut.rl.rewards import RewardWeights, anneal_alpha, compute_reward
from rl_donut.rl.stand_in import Events


def make_dummy_events(M=2, A=2):
    return Events(
        damage_dealt=np.zeros((M, A), dtype=np.float32),
        damage_taken=np.zeros((M, A), dtype=np.float32),
        kills=np.zeros((M, A), dtype=np.int64),
        deaths=np.zeros((M, A), dtype=np.int64),
        fell=np.zeros((M, A), dtype=bool),
        shots=np.zeros((M, A), dtype=np.int64),
        hits=np.zeros((M, A), dtype=np.int64),
        outcome=np.zeros((M, 2), dtype=np.int64),
    )


def test_anneal_alpha():
    total = 1000
    assert anneal_alpha(0, total, 1.0, 0.1, 0.6) == 1.0
    assert anneal_alpha(300, total, 1.0, 0.1, 0.6) == pytest.approx(0.55)
    assert anneal_alpha(600, total, 1.0, 0.1, 0.6) == pytest.approx(0.1)
    assert anneal_alpha(900, total, 1.0, 0.1, 0.6) == pytest.approx(0.1)


def test_kill_and_damage_rewards():
    ev = make_dummy_events(1, 2)
    team = np.array([0, 1])

    # Agent 0 dealt 100 dmg and 1 kill; agent 1 took 100 dmg and 1 death
    ev.damage_dealt[0, 0] = 100.0
    ev.kills[0, 0] = 1
    ev.damage_taken[0, 1] = 100.0
    ev.deaths[0, 1] = 1

    r = compute_reward(ev, team, alpha=1.0)
    # Agent 0: 0.3 (kill) + 0.3 (100 dmg) - 0.0005 (step) = 0.5995
    assert r[0, 0] == pytest.approx(0.5995, abs=1e-4)
    # Agent 1: -0.3 (death) - 0.3 (dmg taken) - 0.0005 (step) = -0.6005
    assert r[0, 1] == pytest.approx(-0.6005, abs=1e-4)


def test_sparse_win_loss_reward():
    ev = make_dummy_events(1, 2)
    team = np.array([0, 1])

    # Team 0 won, Team 1 lost
    ev.outcome[0] = [1, -1]
    r = compute_reward(ev, team, alpha=0.0)

    assert r[0, 0] == 1.0
    assert r[0, 1] == -1.0


def test_pit_fall_penalty():
    ev = make_dummy_events(1, 2)
    team = np.array([0, 1])

    ev.fell[0, 0] = True
    ev.deaths[0, 0] = 1

    r = compute_reward(ev, team, alpha=1.0)
    # Agent 0 has death penalty (-0.3) + fall penalty (-0.3) - step penalty
    assert r[0, 0] == pytest.approx(-0.6005, abs=1e-4)
