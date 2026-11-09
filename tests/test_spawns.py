from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import MapConfig, SpawnConfig, generate_map
from rl_donut.mapgen.pits import pit_columns
from rl_donut.mapgen.platforms import reach_of
from rl_donut.mapgen.spawns import Spawn, mirror, place_spawns, sees

CFG = MapConfig()
SC = CFG.spawns
W, H = CFG.regions.clearance_w, CFG.regions.clearance_h


def test_mirror_is_its_own_inverse():
    s = Spawn(10, 7)
    assert mirror(mirror(s, 96, W), 96, W) == s
    assert mirror(s, 96, W) == Spawn(10, 87)


@pytest.fixture(scope="module")
def maps(generated_maps):
    return generated_maps[:40]


def test_teams_are_mirror_images(maps):
    for m in maps:
        assert m.spawns.team_b == tuple(mirror(s, CFG.width, W) for s in m.spawns.team_a)


def test_spawns_stand_on_main_floor(maps):
    for m in maps:
        r = reach_of(m.grid, CFG)
        for s in m.spawns.team_a + m.spawns.team_b:
            seg = r.segments[s.y, s.x]
            assert seg >= 0 and r.main[seg]


def test_no_spawn_sees_an_enemy_spawn(maps):
    for m in maps:
        for a in m.spawns.team_a:
            for b in m.spawns.team_b:
                assert not sees(m.grid, a, b, CFG)


def test_spawns_keep_their_distances(maps):
    for m in maps:
        team = m.spawns.team_a
        near_pit = pit_columns(m.pits, CFG.width, SC.pit_gap)
        for i, s in enumerate(team):
            assert s.x + W <= CFG.width // 2 - SC.centre_gap
            assert not near_pit[s.x: s.x + W].any()
            for t in team[i + 1:]:
                assert np.hypot(s.y - t.y, s.x - t.x) >= SC.spacing


def test_most_maps_get_every_spawn(maps):
    full = [len(m.spawns.team_a) == SC.per_team for m in maps]
    assert np.mean(full) >= 0.8


def test_spawn_count_is_configurable():
    cfg = replace(CFG, spawns=replace(SC, per_team=2))
    assert len(generate_map(3, cfg).spawns.team_a) == 2


def test_impossible_rules_give_no_spawns():
    cfg = replace(CFG, spawns=replace(SC, centre_gap=CFG.width))
    m = generate_map(3, cfg)
    assert m.spawns.team_a == () and m.spawns.team_b == ()


def test_spawns_are_reproducible():
    m = generate_map(9)
    again = place_spawns(m.grid, reach_of(m.grid, CFG), m.pits, 9, CFG)
    assert again == m.spawns


@pytest.mark.parametrize("kwargs", [{"per_team": 0}, {"spacing": -1.0}, {"pit_gap": -1}, {"centre_gap": -1}])
def test_spawn_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SpawnConfig(**kwargs)
