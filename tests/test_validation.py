from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import (
    AIR, ROCK, MapConfig, MapGenerationError, generate_map, generate_valid_map, validate,
)
from rl_donut.mapgen import validation as V
from rl_donut.mapgen.spawns import Spawns

CFG = MapConfig()


@pytest.fixture(scope="module")
def valid():
    return generate_valid_map(1, CFG)


# --- validate ---

def test_valid_map_has_no_reasons(valid):
    assert validate(valid, CFG) == []


def test_trap_is_rejected(valid):
    assert V.TRAP in validate(replace(valid, trap_free=False), CFG)


def test_too_much_unreachable_floor_is_rejected(valid):
    assert V.STRANDED in validate(replace(valid, stranded_fraction=CFG.max_stranded + 0.01), CFG)


def test_missing_pit_is_rejected(valid):
    assert V.PITS in validate(replace(valid, pits_requested=len(valid.pits) + 1), CFG)


def test_missing_spawns_are_rejected(valid):
    fewer = Spawns(valid.spawns.team_a[:-1], valid.spawns.team_b[:-1])
    assert V.SPAWNS in validate(replace(valid, spawns=fewer), CFG)


def test_asymmetric_grid_is_rejected(valid):
    grid = valid.grid.copy()
    ys, xs = np.nonzero(grid[:, : CFG.width // 2] == AIR)
    grid[ys[0], xs[0]] = ROCK
    assert V.ASYMMETRIC in validate(replace(valid, grid=grid), CFG)


def test_split_map_is_rejected(valid):
    grid = valid.grid.copy()
    grid[:, CFG.width // 2 - 1: CFG.width // 2 + 1] = ROCK       # symmetric wall down the middle
    assert V.REGIONS in validate(replace(valid, grid=grid), CFG)


def test_broken_border_is_rejected(valid):
    grid = valid.grid.copy()
    grid[0, 10] = grid[0, CFG.width - 11] = AIR                  # symmetric hole in the ceiling
    assert validate(replace(valid, grid=grid), CFG) == [V.BORDER]


# --- retries ---

def test_retries_record_why_earlier_attempts_failed():
    m = generate_valid_map(6, CFG)
    assert m.attempt == len(m.rejections) == 4
    assert all(reasons for reasons in m.rejections)
    for attempt, reasons in enumerate(m.rejections):
        assert validate(generate_map(6, CFG, attempt), CFG) == list(reasons)


def test_valid_map_matches_the_plain_attempt():
    m = generate_valid_map(4, CFG)
    assert np.array_equal(m.grid, generate_map(4, CFG, m.attempt).grid)


def test_design_settings_stay_fixed_across_attempts():
    assert generate_map(6, CFG, 0).params == generate_map(6, CFG, 4).params


def test_first_attempt_valid_seed_has_no_rejections(valid):
    assert valid.attempt == 0 and valid.rejections == ()


def test_exhausted_attempts_raise():
    cfg = replace(CFG, max_attempts=2, spawns=replace(CFG.spawns, per_team=50))
    with pytest.raises(MapGenerationError):
        generate_valid_map(1, cfg)


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        MapConfig(max_attempts=0)


def test_golden_retry_history_seed_0():
    m = generate_valid_map(0, CFG)
    assert (m.attempt, m.rejections, m.params.pit_count) == (1, (("pits",),), 1)
