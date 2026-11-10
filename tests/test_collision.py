import copy

import numpy as np
import pytest

from rl_donut.env import EngineConfig, World, move, on_ground, overlaps_rock, tile_solid
from rl_donut.env.collision import move_axis
from rl_donut.mapgen import load_split

DT = 1 / 30
CFG = EngineConfig()
HW, HH = CFG.hitbox_w / 2, CFG.hitbox_h / 2


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def room(maps, x, y, vx=0.0, vy=0.0):
    """One match, one agent per team, on an empty 96 x 48 room with a 1-tile frame and a
    floor at row 40. Agent 0 is placed at (x, y); agent 1 is parked and not moved."""
    world = World.single(maps[0])
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[40:, :] = True
    world.grids[0] = grid
    world.x[0, 0], world.y[0, 0], world.vx[0, 0], world.vy[0, 0] = x, y, vx, vy
    world.alive[0, 1] = False
    return world


def steps(world, n):
    for _ in range(n):
        move(world, DT)


# --- outside-the-map rules ---

def test_outside_left_right_and_top_are_solid_and_below_is_air(maps):
    grids = World.single(maps[0]).grids
    m = np.zeros(5, dtype=np.int64)
    rows = np.array([-1, 10, 10, 48, 60])
    cols = np.array([10, -1, 96, 10, 50])
    assert tile_solid(grids, m, rows, cols).tolist() == [True, True, True, False, False]


def test_below_the_map_but_outside_the_sides_is_solid(maps):
    grids = World.single(maps[0]).grids
    assert tile_solid(grids, np.array([0]), np.array([50]), np.array([-1])).tolist() == [True]


# --- exact contacts ---

def test_falling_agent_lands_exactly_on_the_floor(maps):
    world = room(maps, x=10.0, y=30.0, vy=18.0)
    steps(world, 30)
    assert world.y[0, 0] + HH == pytest.approx(40.0, abs=1e-12)
    assert world.vy[0, 0] == 0.0 and on_ground(world)[0, 0]


def test_rising_agent_stops_exactly_at_the_ceiling(maps):
    world = room(maps, x=10.0, y=10.0, vy=-18.0)
    steps(world, 30)
    assert world.y[0, 0] - HH == pytest.approx(1.0, abs=1e-12)
    assert world.vy[0, 0] == 0.0 and not on_ground(world)[0, 0]


def test_wall_stops_only_horizontal_speed(maps):
    world = room(maps, x=90.0, y=20.0, vx=6.0, vy=3.0)
    steps(world, 40)
    assert world.x[0, 0] + HW == pytest.approx(95.0, abs=1e-12)
    assert world.vx[0, 0] == 0.0 and world.vy[0, 0] == 3.0


def test_agent_touching_a_wall_can_slide_along_it_and_leave_it(maps):
    world = room(maps, x=1.0 + HW, y=20.0, vy=-6.0)
    steps(world, 5)
    assert world.vy[0, 0] == -6.0 and world.y[0, 0] < 20.0
    world.vx[0, 0], world.vy[0, 0] = 6.0, 0.0
    steps(world, 1)
    assert world.x[0, 0] == pytest.approx(1.0 + HW + 0.2)


def test_running_along_the_floor_is_not_blocked_by_it(maps):
    world = room(maps, x=10.0, y=40.0 - HH, vx=6.0)
    steps(world, 30)
    assert world.x[0, 0] == pytest.approx(16.0) and on_ground(world)[0, 0]


def test_landing_from_a_snapped_position_stays_on_the_edge(maps):
    """Snapped coordinates like 40 - 0.8 are inexact in floating point; repeated contact must not drift."""
    world = room(maps, x=10.0, y=39.0 - HH, vy=12.0)
    for _ in range(300):
        world.vy[0, 0] = 12.0
        move(world, DT)
    assert world.y[0, 0] + HH == pytest.approx(40.0, abs=1e-12) and on_ground(world)[0, 0]


# --- standing ---

def test_everyone_stands_after_reset(maps):
    world = World.from_maps(maps, EngineConfig(team_size=2))
    assert on_ground(world).all()


def test_hovering_agent_is_not_standing(maps):
    world = room(maps, x=10.0, y=40.0 - HH - 0.01)
    assert not on_ground(world)[0, 0]


def test_dead_agents_neither_move_nor_stand(maps):
    world = World.single(maps[0])
    world.alive[0, 0] = False
    x, y = world.x.copy(), world.y.copy()
    world.vx[:] = 6.0
    world.vy[:] = -6.0
    move(world, DT)
    assert world.x[0, 0] == x[0, 0] and world.y[0, 0] == y[0, 0]
    assert not on_ground(world)[0, 0]


# --- pits ---

def test_agent_falls_through_a_pit_below_the_map(maps):
    m = next(m for m in maps if m.pits)
    pit = m.pits[0]
    world = World.single(m)
    world.x[0, 0], world.y[0, 0] = pit.x0 + pit.width / 2, pit.top - HH - 0.5
    for _ in range(200):
        world.vx[0, 0], world.vy[0, 0] = 0.0, 18.0
        move(world, DT)
    assert world.y[0, 0] - HH > m.grid.shape[0]
    assert not overlaps_rock(world)[0, 0]


# --- safety ---

def test_moving_a_full_tile_in_one_tick_is_refused(maps):
    world = room(maps, x=10.0, y=20.0, vx=30.0)
    with pytest.raises(ValueError):
        move(world, DT)


# --- properties on real maps ---

def random_walk(maps, team_size, ticks, seed):
    world = World.from_maps(maps, EngineConfig(team_size=team_size))
    rng = np.random.default_rng(seed)
    for _ in range(ticks):
        world.vx[:] = rng.choice([-6.0, 0.0, 6.0], world.vx.shape)
        world.vy[:] = np.clip(world.vy + rng.uniform(-6, 6, world.vy.shape), -28.0, 28.0)
        move(world, DT)
        assert not overlaps_rock(world).any()
        world.alive &= world.y < world.kill_y[:, None]
        yield world


@pytest.mark.parametrize("team_size", [1, 2])
def test_random_motion_never_enters_rock(maps, team_size):
    contacts = 0
    for world in random_walk(maps, team_size, ticks=600, seed=team_size):
        contacts += int(on_ground(world).sum())
    assert contacts > 1000, "agents should keep hitting floors, or the test proves nothing"


def test_leading_edge_check_matches_checking_every_tile(maps):
    """A move is blocked exactly when the unblocked target position would overlap rock."""
    checked = blocked = 0
    for world in random_walk(maps, 2, ticks=200, seed=7):
        rng = np.random.default_rng(checked)
        for horizontal in (True, False):
            delta = rng.uniform(-0.95, 0.95, world.x.shape)
            target = copy.deepcopy(world)
            (target.x if horizontal else target.y)[...] += delta
            expect_blocked = overlaps_rock(target) & world.alive
            moved = copy.deepcopy(world)
            move_axis(moved, delta, horizontal, moved.alive)
            pos = moved.x if horizontal else moved.y
            got_blocked = world.alive & (pos != (target.x if horizontal else target.y))
            assert np.array_equal(got_blocked, expect_blocked)
            assert not overlaps_rock(moved).any()
            checked += int(world.alive.sum())
            blocked += int(got_blocked.sum())
    assert checked > 10_000 and blocked > 0.05 * checked
