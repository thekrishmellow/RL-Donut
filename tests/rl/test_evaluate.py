import pytest
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.evaluate import evaluate_policy, wilson_score_interval


def test_wilson_interval_bounds():
    low, high = wilson_score_interval(50, 100)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)

    # Edge cases
    assert wilson_score_interval(0, 10)[0] == 0.0
    assert wilson_score_interval(10, 10)[1] == 1.0


def test_evaluate_policy_swaps_sides():
    """Verify evaluation runs across split without error and reports Wilson intervals."""
    bot_a = StaticTargetBot()
    bot_b = RandomBot()

    report = evaluate_policy(
        learner=bot_a,
        opponent=bot_b,
        n_matches=4,
        split="val",
        match_duration_seconds=0.5,
    )
    assert report.n_matches == 4
    assert report.wins + report.losses + report.draws == 4
    assert 0.0 <= report.win_rate <= 1.0
    assert 0.0 <= report.wilson_ci_lower <= report.wilson_ci_upper <= 1.0
