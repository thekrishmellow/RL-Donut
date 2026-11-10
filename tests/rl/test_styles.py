import numpy as np
import pytest

from rl_donut.rl.stand_in import Events
from rl_donut.rl.styles import CombatStyle, compute_styled_reward


def test_styled_rewards_aggressive():
    ev = Events(
        damage_dealt=np.array([[100.0, 0.0]], dtype=np.float32),
        damage_taken=np.array([[0.0, 100.0]], dtype=np.float32),
        kills=np.array([[1, 0]], dtype=np.int64),
        deaths=np.array([[0, 1]], dtype=np.int64),
        fell=np.zeros((1, 2), dtype=bool),
        shots=np.zeros((1, 2), dtype=np.int64),
        hits=np.zeros((1, 2), dtype=np.int64),
        outcome=np.zeros((1, 2), dtype=np.int64),
    )
    team = np.array([0, 1])

    # Aggressive style: 2x damage dealt (0.006 * 100 = 0.6) + kill (0.5) - step (0.001) = 1.099
    r_agg = compute_styled_reward(ev, team, style=CombatStyle.AGGRESSIVE, alpha=1.0)
    assert r_agg[0, 0] == pytest.approx(1.099, abs=1e-3)


def test_styled_rewards_defensive():
    ev = Events(
        damage_dealt=np.array([[0.0, 100.0]], dtype=np.float32),
        damage_taken=np.array([[100.0, 0.0]], dtype=np.float32),
        kills=np.zeros((1, 2), dtype=np.int64),
        deaths=np.array([[1, 0]], dtype=np.int64),
        fell=np.zeros((1, 2), dtype=bool),
        shots=np.zeros((1, 2), dtype=np.int64),
        hits=np.zeros((1, 2), dtype=np.int64),
        outcome=np.zeros((1, 2), dtype=np.int64),
    )
    team = np.array([0, 1])

    # Defensive style: 2x damage taken penalty (-0.006 * 100 = -0.6) + death (-0.5) - step (0.0002) = -1.1002
    r_def = compute_styled_reward(ev, team, style=CombatStyle.DEFENSIVE, alpha=1.0)
    assert r_def[0, 0] == pytest.approx(-1.1002, abs=1e-3)


def test_styled_rewards_sniper_range_bonus():
    ev = Events(
        damage_dealt=np.array([[15.0, 0.0]], dtype=np.float32),
        damage_taken=np.array([[0.0, 15.0]], dtype=np.float32),
        kills=np.zeros((1, 2), dtype=np.int64),
        deaths=np.zeros((1, 2), dtype=np.int64),
        fell=np.zeros((1, 2), dtype=bool),
        shots=np.array([[1, 0]], dtype=np.int64),
        hits=np.array([[1, 0]], dtype=np.int64),
        outcome=np.zeros((1, 2), dtype=np.int64),
    )
    team = np.array([0, 1])

    # Hit distance 20 tiles (> 15) triggers sniper bonus
    hit_dists = np.array([[20.0, 0.0]], dtype=np.float32)
    r_sniper = compute_styled_reward(ev, team, style=CombatStyle.SNIPER, alpha=1.0, hit_distances=hit_dists)
    # 0.003 * 15 (0.045) + 0.004 (sniper bonus) - 0.0004 (step) = 0.0486
    assert r_sniper[0, 0] == pytest.approx(0.0486, abs=1e-4)

