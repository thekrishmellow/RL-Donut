import numpy as np
from rl_donut.rl.adaptive import AdaptiveConfig, AdaptiveMatchmaker
from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.pool import OpponentPool


def test_adaptive_twenty_wins_and_losses():
    pool = OpponentPool()
    pool._add_snapshot(StaticTargetBot("snap_1"), update=10)
    matchmaker = AdaptiveMatchmaker(pool)

    # Twenty wins raise difficulty bias
    for _ in range(20):
        matchmaker.record_match_outcome(won=True, lost=False)
    assert matchmaker.difficulty_bias > 0.0

    # Twenty losses lower difficulty bias
    matchmaker_loss = AdaptiveMatchmaker(pool)
    for _ in range(20):
        matchmaker_loss.record_match_outcome(won=False, lost=True)
    assert matchmaker_loss.difficulty_bias < 0.0

    # Twenty draws leave difficulty bias at 0.0 (win rate 0.5 is within 0.50 - 0.60)
    matchmaker_draw = AdaptiveMatchmaker(pool)
    for _ in range(20):
        matchmaker_draw.record_match_outcome(won=False, lost=False)
    assert matchmaker_draw.current_win_rate == 0.5
    assert matchmaker_draw.difficulty_bias == 0.0


def test_adaptive_opponent_selection():
    pool = OpponentPool()
    pool._add_snapshot(StaticTargetBot("snap_1"), update=10)
    matchmaker = AdaptiveMatchmaker(pool)

    rng = np.random.default_rng(42)
    opp = matchmaker.select_opponent(rng)
    assert opp is not None
    assert isinstance(opp.name, str)

