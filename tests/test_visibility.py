import math

import numpy as np
import pytest

from rl_donut.env import EngineConfig, World, match_step
from rl_donut.env.raycast import segments
from rl_donut.env.visibility import target_points, visibility
from rl_donut.mapgen import line_of_sight, load_split

FLOOR = 46


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:64]


def room(maps, positions, team_size=1):
    """One match in an empty room with a 1-tile frame and the floor at row 46."""
    world = World.single(maps[0], EngineConfig(team_size=team_size))
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[FLOOR:, :] = True
    world.grids[0] = grid
    for i, (x, y) in enumerate(positions):
        world.x[0, i], world.y[0, i] = x, y
    return world


def test_open_room_any_distance(maps):
    world = room(maps, [(2.0, 30.0), (93.0, 10.0)])
    v = visibility(world)
    assert v.own[0, 0, 1] and v.own[0, 1, 0]
    assert not v.own[0, 0, 0] and not v.own[0, 1, 1]


def test_standing_agents_see_each_other_along_the_floor(maps):
    y = FLOOR - EngineConfig().hitbox_h / 2
    world = room(maps, [(5.0, y), (40.0, y)])
    assert visibility(world).own[0, 0, 1]


def test_wall_blocks(maps):
    world = room(maps, [(10.0, 30.0), (20.0, 30.0)])
    world.grids[0, :, 15] = True
    assert not visibility(world).own[0].any()


def test_peeking_head_is_seen_though_the_centre_is_hidden(maps):
    world = room(maps, [(10.0, 30.0), (20.0, 30.0)])
    world.grids[0, 30:, 15] = True             # a low wall: the centre line meets it, the head line passes above
    centre_clear, _ = segments(world.grids, [0], [10.0], [30.0], [20.0], [30.0])
    assert not centre_clear[0]
    assert visibility(world).own[0, 0, 1]


def test_dead_agents_neither_see_nor_are_seen(maps):
    world = room(maps, [(10.0, 30.0), (20.0, 30.0)])
    world.alive[0, 1] = False
    v = visibility(world)
    assert not v.own.any() and not v.team.any()


def test_team_sight_in_2v2(maps):
    # team A: agents 0 and 1; team B: 2 and 3. A wall hides agent 0 from both enemies.
    world = room(maps, [(5.0, 30.0), (30.0, 30.0), (40.0, 30.0), (50.0, 30.0)], team_size=2)
    world.grids[0, :, 10] = True
    v = visibility(world)
    assert not v.own[0, 0].any() and v.own[0, 1, 2] and v.own[0, 1, 3]
    assert v.team[0, 0, 2] and v.team[0, 0, 3]                       # told by teammate 1
    assert not v.own[0, 0, 1] and not v.team[0, 0, 1]                # allies aren't in the matrix
    assert not v.team[0, 2, 0] and v.team[0, 2, 1] and v.team[0, 3, 1]
    world.alive[0, 1] = False                                        # a dead teammate reports nothing
    v = visibility(world)
    assert not v.team[0, 0].any()


def brute_force(world):
    """Reference: per pair and point, with the map generator's own line_of_sight."""
    M, A = world.x.shape
    points = target_points(world.cfg)
    own = np.zeros((M, A, A), dtype=bool)
    for m in range(M):
        for i in range(A):
            for j in range(A):
                if world.team[i] != world.team[j] and world.alive[m, i] and world.alive[m, j]:
                    eye = (world.x[m, i], world.y[m, i])
                    own[m, i, j] = any(line_of_sight(world.grids[m].astype(np.uint8), eye,
                                                     (world.x[m, j] + dx, world.y[m, j] + dy)) for dx, dy in points)
    return own


def test_matches_brute_force_during_real_matches(maps):
    world = World.from_maps(maps, EngineConfig(team_size=2))
    rng = np.random.default_rng(0)
    seen = 0
    for step in range(120):
        move = np.where(world.team == 0, 1, -1)[None].repeat(len(maps), 0) * (rng.random(world.x.shape) < 0.7)
        match_step(world, move, rng.random(world.x.shape) < 0.4,
                   rng.uniform(-math.pi, math.pi, world.x.shape), rng.random(world.x.shape) < 0.3)
        if step % 20 == 19:
            v = visibility(world)
            np.testing.assert_array_equal(v.own, brute_force(world))
            seen += v.own.sum()
    assert seen > 50
