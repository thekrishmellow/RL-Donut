"""Tests for Phase 10 benchmark harness, statistics, and test split protection."""

from pathlib import Path
import pytest
import numpy as np

from rl_donut.env.bots import RusherBot, StaticTargetBot
from rl_donut.mapgen import load_split
from rl_donut.rl.benchmark import (
    MatchRecord,
    difference_interval,
    guard_test_split,
    play_pairing,
    summarise,
)
from rl_donut.rl.evaluate import wilson_score_interval


def test_difference_interval_worked_example():
    # 56/70 vs 48/80 -> d = 0.2, interval ≈ [0.05, 0.33]
    low, high = difference_interval(56, 70, 48, 80)
    assert low == pytest.approx(0.053, abs=0.01)
    assert high == pytest.approx(0.334, abs=0.01)


def test_summarise_records():
    # 3 wins and 1 loss
    dummy_records = [
        MatchRecord(
            map_seed=i,
            learner_team=i % 2,
            outcome=1 if i < 3 else -1,
            score=(1, 0) if i < 3 else (0, 1),
            kills=1 if i < 3 else 0,
            deaths=0 if i < 3 else 1,
            falls=0,
            damage_dealt=100.0 if i < 3 else 10.0,
            damage_taken=10.0 if i < 3 else 100.0,
            shots=10,
            hits=7,
            airborne_s=12.0,
            hover_s=2.0,
            fire_distance=14.0,
            spawn_kills=0,
        )
        for i in range(4)
    ]
    s = summarise(dummy_records)
    assert s["n_matches"] == 4
    assert s["wins"] == 3
    assert s["losses"] == 1
    assert s["draws"] == 0
    assert s["win_rate"] == pytest.approx(0.75)
    ci_low, ci_high = wilson_score_interval(3, 4)
    assert s["ci_lower"] == pytest.approx(ci_low, abs=1e-4)
    assert s["ci_upper"] == pytest.approx(ci_high, abs=1e-4)
    assert s["accuracy"] == pytest.approx(0.7)


def test_play_pairing_static_vs_static():
    val_maps = load_split("val")[:4]
    bot_a = StaticTargetBot()
    bot_b = StaticTargetBot()

    records = play_pairing(
        learner=bot_a,
        opponent=bot_b,
        maps=val_maps,
        n_matches=2,
        seed=123,
        seconds=1.0,
    )
    assert len(records) == 2
    for r in records:
        assert r.outcome == 0  # draw
        assert r.shots == 0
        assert r.damage_dealt == 0.0


def test_play_pairing_odd_matches_raises():
    val_maps = load_split("val")[:4]
    bot = StaticTargetBot()
    with pytest.raises(ValueError, match="n_matches must be even"):
        play_pairing(bot, bot, maps=val_maps, n_matches=3, seed=123, seconds=1.0)


def test_play_pairing_rusher_vs_static():
    val_maps = load_split("val")[:4]
    rusher = RusherBot()
    static = StaticTargetBot()

    # Rusher should win all matches vs static target
    records = play_pairing(
        learner=rusher,
        opponent=static,
        maps=val_maps,
        n_matches=4,
        seed=123,
        seconds=60.0,
    )
    assert len(records) == 4
    for r in records:
        assert r.outcome == 1
        assert r.kills > 0


def test_guard_test_split(tmp_path: Path):
    marker_dir = tmp_path / "final"

    # Non-test split passes without error
    guard_test_split("val", marker_dir=marker_dir)

    # Test split without --final raises RuntimeError
    with pytest.raises(RuntimeError, match="requires the --final flag"):
        guard_test_split("test", final=False, marker_dir=marker_dir)

    # Test split with --final writes marker
    guard_test_split("test", final=True, marker_dir=marker_dir)
    marker_file = marker_dir / "TEST_USED"
    assert marker_file.exists()
    assert "git_commit" in marker_file.read_text()

    # Second call without --force raises RuntimeError
    with pytest.raises(RuntimeError, match="Test split already used"):
        guard_test_split("test", final=True, force=False, marker_dir=marker_dir)

    # Second call with --force succeeds
    guard_test_split("test", final=True, force=True, marker_dir=marker_dir)
    assert "forced: True" in marker_file.read_text()

