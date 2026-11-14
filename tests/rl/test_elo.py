from pathlib import Path
import random
import pytest
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.elo import (
    MatchResult,
    bootstrap_elo_ci,
    compute_elo_ratings,
    compute_expected_score,
    EloReport,
    run_round_robin,
)


def test_expected_score():
    # Equal ratings -> 50%
    assert compute_expected_score(1500, 1500) == pytest.approx(0.5)
    # Higher rating has higher expected score
    assert compute_expected_score(1600, 1400) > 0.75
    assert compute_expected_score(1400, 1600) < 0.25


def test_compute_elo_ratings_ordering():
    # Player A beats Player B 10 times, Player B beats Player C 10 times
    matches = [
        MatchResult("player_a", "player_b", wins_a=10, wins_b=0, draws=0),
        MatchResult("player_b", "player_c", wins_a=10, wins_b=0, draws=0),
    ]
    ratings = compute_elo_ratings(matches, anchor_name="player_c", anchor_rating=1000.0)
    assert ratings["player_a"] > ratings["player_b"] > ratings["player_c"]
    assert ratings["player_c"] == 1000.0


def test_bradley_terry_scaling():
    # A beats B 75% of the time, B beats C 75% of the time
    matches = [
        MatchResult("player_a", "player_b", wins_a=75, wins_b=25, draws=0),
        MatchResult("player_b", "player_c", wins_a=75, wins_b=25, draws=0),
    ]
    ratings = compute_elo_ratings(matches, anchor_name="player_c", anchor_rating=1200.0)
    assert ratings["player_a"] > ratings["player_b"] > ratings["player_c"]
    diff_ab = ratings["player_a"] - ratings["player_b"]
    diff_bc = ratings["player_b"] - ratings["player_c"]
    # 400 * log10(3) ≈ 190.85
    assert diff_ab == pytest.approx(190.85, abs=2.0)
    assert diff_bc == pytest.approx(190.85, abs=2.0)


def test_bradley_terry_order_invariance():
    matches = [
        MatchResult("player_a", "player_b", wins_a=12, wins_b=8, draws=4),
        MatchResult("player_b", "player_c", wins_a=15, wins_b=5, draws=2),
        MatchResult("player_a", "player_c", wins_a=18, wins_b=3, draws=1),
    ]
    r1 = compute_elo_ratings(matches)
    shuffled = matches.copy()
    random.Random(123).shuffle(shuffled)
    r2 = compute_elo_ratings(shuffled)
    assert r1 == r2


def test_bradley_terry_all_draws():
    matches = [
        MatchResult("player_a", "player_b", wins_a=0, wins_b=0, draws=10),
        MatchResult("player_b", "player_c", wins_a=0, wins_b=0, draws=10),
        MatchResult("player_a", "player_c", wins_a=0, wins_b=0, draws=10),
    ]
    ratings = compute_elo_ratings(matches, anchor_name="player_a", anchor_rating=1200.0)
    for p in ["player_a", "player_b", "player_c"]:
        assert ratings[p] == pytest.approx(1200.0, abs=0.1)


def test_bootstrap_elo_ci():
    matches = [
        MatchResult("player_a", "player_b", wins_a=15, wins_b=5, draws=0),
    ]
    ci = bootstrap_elo_ci(matches, anchor_name="player_b", anchor_rating=1200.0, n_bootstraps=50)
    assert "player_a" in ci and "player_b" in ci
    low_a, high_a = ci["player_a"]
    assert low_a <= high_a


def test_elo_report_json(tmp_path: Path):
    report = EloReport(
        ratings={"bot_a": 1300.0, "bot_b": 1100.0},
        matches=[MatchResult("bot_a", "bot_b", wins_a=5, wins_b=1, draws=2)],
        anchor_name="bot_b",
        anchor_rating=1100.0,
        ratings_ci={"bot_a": (1250.0, 1350.0), "bot_b": (1100.0, 1100.0)},
    )
    json_path = tmp_path / "elo.json"
    report.to_json(json_path)
    assert json_path.exists()


def test_run_round_robin_match_seconds():
    policies = [RandomBot(), StaticTargetBot()]
    report = run_round_robin(policies, n_matches_per_pair=2, match_seconds=1.0, bootstrap_ci=False)
    assert len(report.matches) == 1
    assert "random_bot" in report.ratings
    assert "static_target_bot" in report.ratings
