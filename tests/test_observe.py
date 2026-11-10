import math

import numpy as np
import pytest

from rl_donut.env import TEAM_B, EngineConfig, RifleConfig, World, match_step
from rl_donut.env.actions import decode, encode
from rl_donut.env.match import MatchEvents
from rl_donut.env.observe import (CROP_H, CROP_W, ENEMY_FEATURES, FEATURE_NAMES, OBS_DIM, BELOW_MAP, Observer)
from rl_donut.env.visibility import visibility
from rl_donut.mapgen import load_split

FLOOR = 46
EXACT = RifleConfig(spread_deg=0.0)


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:64]


def col(name):
    return FEATURE_NAMES.index(name)


def room(maps, positions, team_size=1):
    world = World.single(maps[0], EngineConfig(team_size=team_size, rifle=EXACT))
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[FLOOR:, :] = True
    world.grids[0] = grid
    for i, (x, y) in enumerate(positions):
        world.x[0, i], world.y[0, i] = x, y
    return world


def look(world, observer):
    return observer.observe(visibility(world))


def no_events(world):
    """An empty decision step for `Observer.record`."""
    z = np.zeros((0, 0))
    return MatchEvents(*([z] * 9), shot_log=[], death_log=[], respawn_log=[])


def test_layout():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 57
    assert OBS_DIM == 57 + CROP_W * CROP_H == 330


def test_one_v_one_leaves_second_enemy_and_ally_empty(maps):
    world = room(maps, [(10.5, 30.5), (20.5, 30.5)])
    o = look(world, Observer(world))
    assert o.shape == (1, 2, OBS_DIM) and o.dtype == np.float32
    for name in FEATURE_NAMES:
        if name.startswith("enemy1_") or name.startswith("ally_"):
            assert o[0, :, col(name)].tolist() == ([1.0, 1.0] if name == "enemy1_since_seen" else [0.0, 0.0]), name
    assert o[0, 0, col("enemy0_present")] == 1 and o[0, 0, col("enemy0_seen_by_me")] == 1
    assert o[0, 0, col("enemy0_dx")] == pytest.approx(10 / 30)
    assert o[0, 1, col("enemy0_dx")] == pytest.approx(10 / 30)    # mirrored: team B sees A ahead of it too


def test_hidden_enemy_is_remembered_then_forgotten(maps):
    world = room(maps, [(10.5, 30.5), (20.5, 30.5)])
    world.grids[0, :, 15] = True
    observer = Observer(world)
    o = look(world, observer)[0, 0]
    assert [o[col(f"enemy0_{f}")] for f in ("seen_by_me", "seen_by_team", "remembered", "dx", "dy", "since_seen")] \
        == [0, 0, 0, 0, 0, 1]
    world.grids[0, :, 15] = False                       # wall gone: seen, with live details
    world.vx[0, 1], world.health[0, 1] = 3.0, 55.0
    o = look(world, observer)[0, 0]
    assert o[col("enemy0_seen_by_me")] == 1 and o[col("enemy0_remembered")] == 1
    assert o[col("enemy0_health")] == pytest.approx(0.55) and o[col("enemy0_vx")] == pytest.approx(0.5)
    world.grids[0, :, 15] = True                        # hidden again, and it moves away
    world.x[0, 1] = 25.5
    for _ in range(5):
        observer.record(no_events(world))
    o = look(world, observer)[0, 0]
    assert o[col("enemy0_seen_by_me")] == 0 and o[col("enemy0_remembered")] == 1
    assert o[col("enemy0_dx")] == pytest.approx(10 / 30)                 # where it was last seen
    assert o[col("enemy0_vx")] == o[col("enemy0_health")] == o[col("enemy0_aim_cos")] == 0   # no live details
    assert o[col("enemy0_since_seen")] == pytest.approx(10 / 300)
    for _ in range(150):
        observer.record(no_events(world))
    o = look(world, observer)[0, 0]
    assert o[col("enemy0_remembered")] == 0 and o[col("enemy0_dx")] == 0 and o[col("enemy0_since_seen")] == 1


def test_team_sight_gives_live_position_without_own_sight(maps):
    world = room(maps, [(5.5, 30.5), (30.5, 30.5), (40.5, 30.5), (50.5, 30.5)], team_size=2)
    world.grids[0, :, 10] = True
    o = look(world, Observer(world))[0, 0]
    assert o[col("enemy0_seen_by_me")] == 0 and o[col("enemy0_seen_by_team")] == 1
    assert o[col("enemy0_dx")] == pytest.approx(35 / 30)
    assert o[col("ally_present")] == 1 and o[col("ally_dx")] == pytest.approx(25 / 30)


def test_hit_direction_points_back_at_the_shooter(maps):
    world = room(maps, [(10.5, 30.5), (20.5, 30.5)])
    observer = Observer(world)
    held = np.zeros((1, 2), dtype=bool)
    world.vy[...] = 0
    ev = match_step(world, np.zeros((1, 2), dtype=int), held, np.array([[0.0, math.pi]]), np.array([[True, False]]))
    observer.record(ev)
    o = look(world, observer)
    assert ev.hits[0, 0] == 1
    assert o[0, 1, col("since_hit")] == 0
    # The shot travelled right, so it came from the left; team B's view is mirrored, so "from
    # the left" in the world is "from ahead" (+x) for it.
    assert o[0, 1, col("hit_from_cos")] == pytest.approx(1.0) and o[0, 1, col("hit_from_sin")] == pytest.approx(0.0, abs=1e-9)
    assert o[0, 0, col("since_hit")] == 1 and o[0, 0, col("hit_from_cos")] == 0


def test_tile_grid_marks_rock_air_and_below_the_map(maps):
    world = room(maps, [(30.5, 44.5), (60.5, 30.5)])
    world.grids[0, FLOOR:, 28:33] = False               # a pit under agent 0
    observer = Observer(world)
    crop = observer.crop()[0, 0]
    assert crop[CROP_H // 2, CROP_W // 2] == 0          # its own tile is air
    below = crop[CROP_H // 2 + 4:, :]                   # rows 48 and beyond: off the bottom of the map
    assert (below[:, 8:13] == BELOW_MAP).all()          # straight down the pit
    assert (crop[CROP_H // 2 + 2:CROP_H // 2 + 4, :8] == 1).all()   # floor beside the pit
    assert set(np.unique(observer.crop())) <= {BELOW_MAP, 0.0, 1.0}


def mirror_world(world):
    """The same situation reflected left-right, with the teams swapped (agent i <-> i + team_size)."""
    W = world.grids.shape[2]
    twin = World.single(load_split("val")[0], world.cfg)
    twin.grids[0] = world.grids[0][:, ::-1]
    A, ts = world.x.shape[1], world.cfg.team_size
    swap = (np.arange(A) + ts) % A
    for name in ("y", "vy", "fuel", "alive", "ammo", "cooldown", "reload", "health", "respawn", "protection"):
        getattr(twin, name)[0, swap] = getattr(world, name)[0]
    twin.x[0, swap] = W - world.x[0]
    twin.vx[0, swap] = -world.vx[0]
    twin.aim[0, swap] = math.pi - world.aim[0]
    twin.score[0] = world.score[0, ::-1]
    twin.tick[0] = world.tick[0]
    return twin, swap


@pytest.mark.parametrize("team_size", [1, 2])
def test_mirrored_world_gives_identical_observations(maps, team_size):
    rng = np.random.default_rng(team_size)
    seen = 0
    for m in maps[:20]:
        world = World.single(m, EngineConfig(team_size=team_size))
        world.grids[0, 5:9, 20:23] = True               # break the map's own symmetry
        A = world.x.shape[1]
        air = np.argwhere(~world.grids[0, 1:-2, 1:-1]) + 1
        spots = air[rng.choice(len(air), A, replace=False)]
        world.x[0] = spots[:, 1] + rng.uniform(0.1, 0.9, A)
        world.y[0] = spots[:, 0] + rng.uniform(0.1, 0.9, A)
        world.vx[0], world.vy[0] = rng.normal(size=A), rng.normal(size=A)
        world.aim[0] = rng.uniform(-math.pi, math.pi, A)
        world.health[0] = rng.uniform(10, 100, A)
        world.score[0] = rng.integers(0, 5, 2)
        world.tick[0] = 400
        twin, swap = mirror_world(world)
        a, b = look(world, Observer(world)), look(twin, Observer(twin))
        np.testing.assert_allclose(a[0], b[0, swap], atol=1e-5)
        assert not np.allclose(a[0], b[0])               # the twin really is a different situation
        seen += a[0, :, col("enemy0_seen_by_me")].sum()
    assert seen > 0


@pytest.mark.parametrize("team_size", [1, 2])
def test_both_sides_see_mirror_images_at_the_start(maps, team_size):
    """Maps and spawns are mirror images, so at the start each agent's view equals its mirror
    twin's; spawn positions are whole numbers, the case where a naive flip is a column off."""
    world = World.from_maps(maps, EngineConfig(team_size=team_size))
    assert (world.x == np.round(world.x)).any()
    o = look(world, Observer(world))
    swap = (np.arange(2 * team_size) + team_size) % (2 * team_size)
    np.testing.assert_allclose(o, o[:, swap], atol=1e-6)


def test_mirrored_actions_round_trip(maps):
    team = np.array([0, 1])
    mirror = team == TEAM_B
    discrete = np.array([[[2, 1, 0], [2, 1, 0]]])
    continuous = np.array([[[0.3], [0.3]]])
    c = decode(discrete, continuous, mirror)
    assert c.move.tolist() == [[1, -1]]                 # "right" in team B's frame is left in the world
    assert c.aim[0, 1] == pytest.approx(math.pi - 0.3)
    d2, c2 = encode(c, mirror)
    np.testing.assert_array_equal(d2, discrete)
    np.testing.assert_allclose(c2, continuous)


def test_real_matches_stay_finite_and_scaled(maps):
    world = World.from_maps(maps, EngineConfig(team_size=2))
    observer = Observer(world)
    rng = np.random.default_rng(0)
    o = look(world, observer)
    for _ in range(300):
        move = np.where(world.team == 0, 1, -1)[None].repeat(len(maps), 0) * (rng.random(world.x.shape) < 0.7)
        ev = match_step(world, move, rng.random(world.x.shape) < 0.45,
                        rng.uniform(-math.pi, math.pi, world.x.shape), rng.random(world.x.shape) < 0.3)
        observer.record(ev)
        o = look(world, observer)
        assert np.isfinite(o).all() and np.abs(o).max() <= 4.0
    assert o[..., col("enemy0_seen_by_team")].any()
