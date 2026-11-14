"""Alias module running runs tests from tests/rl/test_dashboard.py."""

from tests.rl.test_dashboard import (
    test_dashboard_loader_five_rows,
    test_dashboard_loader_skips_broken_lines_and_lists_runs,
)

__all__ = [
    "test_dashboard_loader_five_rows",
    "test_dashboard_loader_skips_broken_lines_and_lists_runs",
]

