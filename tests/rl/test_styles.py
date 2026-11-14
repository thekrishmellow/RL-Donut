import numpy as np
import pytest

from types import SimpleNamespace
from rl_donut.env.rifle import Shot
from rl_donut.rl.stand_in import Events
from rl_donut.rl.styles import CombatStyle, compute_styled_reward, long_hits, styled_reward_fn


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


def test_styled_reward_bonus_size():
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

    r_0 = compute_styled_reward(ev, team, style=CombatStyle.SNIPER, alpha=1.0, long_hit_counts=np.array([[0, 0]]))
    r_2 = compute_styled_reward(ev, team, style=CombatStyle.SNIPER, alpha=1.0, long_hit_counts=np.array([[2, 0]]))
    # Exactly 2 * 0.004 = 0.008 bonus
    assert r_2[0, 0] - r_0[0, 0] == pytest.approx(0.008, abs=1e-5)


def test_long_hits_filtering_20_vs_10_tiles():
    world = SimpleNamespace(x=np.array([[10.0, 30.0]]), y=np.array([[20.0, 20.0]]))
    env = SimpleNamespace(world=world)

    # 20-tile hit at (30.0, 20.0) from shooter at (10.0, 20.0)
    shot20 = Shot(match=0, shooter=0, angle=0.0, target=1, hit=(30.0, 20.0))
    # 10-tile hit at (20.0, 20.0) from shooter at (10.0, 20.0)
    shot10 = Shot(match=0, shooter=0, angle=0.0, target=1, hit=(20.0, 20.0))

    events = SimpleNamespace(
        damage_dealt=np.zeros((1, 2)),
        done=np.array([False]),
        shot_log=[(0, shot20)],
    )
    assert long_hits(events, env, min_range=15.0)[0, 0] == 1.0

    events.shot_log = [(0, shot10)]
    assert long_hits(events, env, min_range=15.0)[0, 0] == 0.0

