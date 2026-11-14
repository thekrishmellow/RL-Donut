"""Tests for dashboard loaders and run data parsing (Phase 9)."""

import json
from pathlib import Path
import pytest

from rl_donut.runs import list_runs, load_config, load_json, load_metrics, series


def test_dashboard_loader_five_rows(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    metrics_file = run_dir / "metrics.jsonl"
    rows = [
        {"update": i, "mean_return": float(i) * 0.1, "entropy": 1.5 - float(i) * 0.05}
        for i in range(1, 6)
    ]
    with open(metrics_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    loaded = load_metrics(run_dir)
    assert len(loaded) == 5

    upds, rets = series(loaded, "mean_return")
    assert upds == [1, 2, 3, 4, 5]
    assert len(rets) == 5
    assert rets[0] == pytest.approx(0.1)

    upds_ent, ents = series(loaded, "entropy")
    assert upds_ent == [1, 2, 3, 4, 5]
    assert len(ents) == 5


def test_dashboard_loader_skips_broken_lines_and_lists_runs(tmp_path: Path):
    # Valid run
    valid_dir = tmp_path / "valid_run"
    valid_dir.mkdir()
    with open(valid_dir / "config.json", "w") as f:
        json.dump({"run_id": "valid_run", "lr": 3e-4}, f)
    with open(valid_dir / "metrics.jsonl", "w") as f:
        f.write('{"update": 1, "mean_return": 0.5}\n')
        f.write('BROKEN LINE NOT JSON\n')
        f.write('\n')
        f.write('{"update": 2, "mean_return": 0.8}\n')

    # Directory with no metrics.jsonl
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # list_runs only includes valid_dir
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].name == "valid_run"

    # load_metrics skips broken and empty lines
    metrics = load_metrics(valid_dir)
    assert len(metrics) == 2
    assert metrics[0]["update"] == 1
    assert metrics[1]["update"] == 2

    # load_config loads config.json
    cfg = load_config(valid_dir)
    assert cfg.get("lr") == 3e-4

    # load_json returns None for missing file
    assert load_json(valid_dir, "missing.json") is None

