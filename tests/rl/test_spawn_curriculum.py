import numpy as np

from rl_donut.env.config import EngineConfig
from rl_donut.env.state import spawn_centre
from rl_donut.mapgen import load_split
from rl_donut.mapgen.geometry import line_of_sight
from rl_donut.rl.spawn_curriculum import (
    RUNG_LADDER,
    SpawnCurriculum,
    maps_for_rung,
    with_gap_spawns,
)


def test_gate_advances_one_rung_and_stops_on_official():
    ladder = SpawnCurriculum(gate=0.5)
    assert ladder.rung.name == "tiles_8"
    assert ladder.consider(0.49) is False
    assert ladder.rung.name == "tiles_8"
    assert ladder.consider(0.5) is True
    assert ladder.rung.name == "tiles_14"
    ladder.index = len(RUNG_LADDER) - 1
    assert ladder.finished
    assert ladder.consider(1.0) is False
    assert ladder.rung.name == "official"


def _place(min_gap, max_gap, sight, scan=60):
    for game_map in load_split("val")[:scan]:
        placed = with_gap_spawns(game_map, min_gap, max_gap, sight)
        if placed is not None:
            return placed
    raise AssertionError(f"no pair in the first {scan} val maps for gap {min_gap}-{max_gap}")


def _gap(game_map):
    engine = EngineConfig()
    left = spawn_centre(game_map.spawns.team_a[0], engine)
    right = spawn_centre(game_map.spawns.team_b[0], engine)
    return left, right, right[0] - left[0]


def test_sight_rung_is_a_few_tiles_and_already_visible():
    placed = _place(6.0, 12.0, True)
    left, right, gap = _gap(placed)
    assert 6.0 <= gap <= 12.0
    assert line_of_sight(placed.grid, left, right)
    assert gap < EngineConfig().rifle.range


def test_middle_rung_stays_in_sight_and_in_range():
    placed = _place(12.0, 16.0, True)
    left, right, gap = _gap(placed)
    assert 12.0 <= gap <= 16.0
    assert line_of_sight(placed.grid, left, right)
    assert gap < EngineConfig().rifle.range


def test_travel_rung_is_beyond_rifle_range():
    placed = _place(32.0, 48.0, False)
    _, _, gap = _gap(placed)
    assert 32.0 <= gap <= 48.0
    assert gap >= EngineConfig().rifle.range


def test_official_rung_keeps_the_map_spawns():
    source = load_split("val")[:4]
    placed = maps_for_rung(source, RUNG_LADDER[-1], limit=4)
    for original, copy in zip(source, placed):
        assert copy.spawns == original.spawns
        assert np.array_equal(copy.grid, original.grid)
