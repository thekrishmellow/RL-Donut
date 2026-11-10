"""Tests for Phase 8 generalization evaluation across splits and OOD conditions."""

from pathlib import Path
import pytest
import numpy as np

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.generalization import (
    GeneralizationReport,
    build_ood_maps,
    compute_generalization_gap,
    evaluate_generalization,
    get_ood_engine_configs,
)


def test_get_ood_engine_configs():
    configs = get_ood_engine_configs()
    assert "gravity_low" in configs
    assert "gravity_high" in configs
    assert "thrust_low" in configs
    assert "thrust_high" in configs

    for name, cfg in configs.items():
        assert cfg.physics.thrust > cfg.physics.gravity, f"thrust must exceed gravity in {name}"
        assert cfg.physics.max_fall_speed * cfg.dt < 1.0


def test_build_ood_maps():
    maps = build_ood_maps(start_seed=30_000, count=2)
    assert len(maps) == 2
    assert maps[0].seed == 30_000
    assert maps[1].seed == 30_001
    assert maps[0].grid.ndim == 2
    assert maps[0].grid.shape[0] > 0
    assert maps[0].grid.shape[1] > 0


def test_compute_generalization_gap():
    gap = compute_generalization_gap(train_win_rate=0.80, eval_win_rate=0.65)
    assert gap == pytest.approx(0.15, abs=1e-4)

    gap_neg = compute_generalization_gap(train_win_rate=0.50, eval_win_rate=0.55)
    assert gap_neg == pytest.approx(-0.05, abs=1e-4)


def test_evaluate_generalization_smoke(tmp_path: Path):
    action_nvec = (3, 2, 16, 2)
    bot_a = RandomBot(action_nvec)
    bot_b = StaticTargetBot(action_nvec)

    report = evaluate_generalization(
        learner=bot_a,
        opponent=bot_b,
        n_matches_per_condition=2,
        seed=123,
        match_duration_seconds=3.0,
    )

    assert isinstance(report, GeneralizationReport)
    assert "train" in report.conditions
    assert "val" in report.conditions
    assert "gravity_low" in report.conditions
    assert "gravity_high" in report.conditions
    assert "ood_seeds_30k" in report.conditions

    out_file = tmp_path / "gen_report.json"
    report.save(out_file)
    assert out_file.exists()
