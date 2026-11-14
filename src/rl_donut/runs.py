"""File and run loading utilities for dashboard and offline analysis (Phase 9).

This module contains no Streamlit or PyTorch imports so it can be tested in isolation.
"""

import json
from pathlib import Path
from typing import Any


def list_runs(root: Path | str = Path("runs")) -> list[Path]:
    """List run directories containing metrics.jsonl, sorted newest first by modification time."""
    p = Path(root)
    if not p.exists() or not p.is_dir():
        return []

    runs = []
    for item in p.iterdir():
        if item.is_dir() and (item / "metrics.jsonl").is_file():
            runs.append(item)

    runs.sort(key=lambda d: (d / "metrics.jsonl").stat().st_mtime, reverse=True)
    return runs


def load_config(run_dir: Path | str) -> dict[str, Any]:
    """Load config.json from run directory, or empty dict if not found."""
    p = Path(run_dir) / "config.json"
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_metrics(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load metrics.jsonl from run directory, skipping empty and broken lines."""
    p = Path(run_dir) / "metrics.jsonl"
    if not p.is_file():
        return []

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                continue
    return rows


def load_json(run_dir: Path | str, name: str) -> dict[str, Any] | None:
    """Load an arbitrary JSON report from run directory, or None if missing or broken."""
    p = Path(run_dir) / name
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """Extract (updates, values) where key is present in row, skipping missing or None values."""
    updates: list[int] = []
    values: list[float] = []

    for r in rows:
        if "update" in r and key in r and r[key] is not None:
            try:
                val = float(r[key])
                upd = int(r["update"])
                updates.append(upd)
                values.append(val)
            except (ValueError, TypeError):
                continue

    return updates, values

