import numpy as np
from rl_donut.rl.adaptive import AdaptiveConfig, AdaptiveMatchmaker
from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.pool import OpponentPool


def test_adaptive_difficulty_increases_on_win_streak():
    pool = OpponentPool()
    pool._add_snapshot(StaticTargetBot("snap_1"), update=10)
    matchmaker = AdaptiveMatchmaker(pool, AdaptiveConfig(target_win_rate_max=0.60))

    initial_bias = matchmaker.difficulty_bias

    # Record 10 consecutive wins
    for _ in range(10):
        matchmaker.record_match_outcome(won=True, lost=False)

    assert matchmaker.current_win_rate == 1.0
    assert matchmaker.difficulty_bias > initial_bias


def test_adaptive_difficulty_decreases_on_loss_streak():
    pool = OpponentPool()
    pool._add_snapshot(StaticTargetBot("snap_1"), update=10)
    matchmaker = AdaptiveMatchmaker(pool, AdaptiveConfig(target_win_rate_min=0.50))

    initial_bias = matchmaker.difficulty_bias

    # Record 10 consecutive losses
    for _ in range(10):
        matchmaker.record_match_outcome(won=False, lost=True)

    assert matchmaker.current_win_rate == 0.0
    assert matchmaker.difficulty_bias < initial_bias


def test_adaptive_opponent_selection():
    pool = OpponentPool()
    pool._add_snapshot(StaticTargetBot("snap_1"), update=10)
    matchmaker = AdaptiveMatchmaker(pool)

    rng = np.random.default_rng(42)
    opp = matchmaker.select_opponent(rng)
    assert opp is not None
    assert isinstance(opp.name, str)

