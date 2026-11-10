import math

import numpy as np
import pytest

from rl_donut.env import TEAM_A, EngineConfig, RifleConfig, World, fire, physics_tick, ray_box
from rl_donut.mapgen import load_split

CFG = EngineConfig()
EXACT = EngineConfig(rifle=RifleConfig(spread_deg=0.0))
EXACT_2V2 = EngineConfig(team_size=2, rifle=RifleConfig(spread_deg=0.0))
R = CFG.rifle
FLOOR = 46


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:100]


def open_room(maps, cfg=EXACT, positions=((10.0, 30.0), (20.0, 30.0))):
    """One match in an empty room with a 1-tile frame; agents placed at `positions`."""
    world = World.single(maps[0], cfg)
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[FLOOR:, :] = True
    world.grids[0] = grid
    for i, (x, y) in enumerate(positions):
        world.x[0, i], world.y[0, i] = x, y
    return world


def shoot(world, shooter, angle):
    aim = world.aim.copy()
    aim[0, shooter] = angle
    trigger = np.zeros_like(world.alive)
    trigger[0, shooter] = True
    world.cooldown[0, shooter] = world.reload[0, shooter] = 0
    world.ammo[0, shooter] = R.magazine
    return fire(world, aim, trigger)[0]


def aim_at(world, i, j):
    return math.atan2(world.y[0, j] - world.y[0, i], world.x[0, j] - world.x[0, i])


# --- settings ---

@pytest.mark.parametrize("kwargs", [dict(damage=0), dict(shot_interval=0), dict(magazine=0),
                                    dict(range=-1), dict(spread_deg=-1), dict(spread_deg=90)])
def test_invalid_rifle_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RifleConfig(**kwargs)


def test_rifle_times_must_be_whole_ticks():
    with pytest.raises(ValueError):
        EngineConfig(rifle=RifleConfig(shot_interval=0.21))
    assert (CFG.shot_interval_ticks, CFG.reload_ticks) == (6, 45)


def test_reset_loads_rifles_and_aims_teams_at_each_other(maps):
    world = World.from_maps(maps[:3], EngineConfig(team_size=2))
    world.ammo[:] = 0
    world.reload[:] = world.cooldown[:] = 7
    world.aim[:] = 1.0
    world.reset([1])
    assert (world.ammo[1] == R.magazine).all() and (world.reload[1] == 0).all() and (world.cooldown[1] == 0).all()
    np.testing.assert_array_equal(world.aim[1], np.where(world.team == TEAM_A, 0.0, np.pi))
    assert (world.ammo[0] == 0).all() and (world.aim[2] == 1.0).all()


# --- the ray ---

def test_ray_box():
    assert ray_box(0, 0, 1, 0, 2, -1, 3, 1) == 2
    assert ray_box(0, 0, -1, 0, 2, -1, 3, 1) is None          # box behind
    assert ray_box(0, 0, 1, 0, 2, 0.5, 3, 1) is None          # passes below
    assert ray_box(2.5, 0, 1, 0, 2, -1, 3, 1) == 0            # starts inside
    d = math.sqrt(0.5)
    assert ray_box(0, 0, d, d, 2, 2, 3, 3) == pytest.approx(2 * math.sqrt(2))


# --- hits ---

def test_hits_enemy_in_open_air(maps):
    world = open_room(maps)
    s = shoot(world, 0, 0.0)
    assert s.target == 1
    assert s.hit == pytest.approx((20.0 - EXACT.hitbox_w / 2, 30.0))


def test_misses_when_aiming_away_or_past(maps):
    world = open_room(maps)
    assert shoot(world, 0, math.pi).target == -1
    # The hitbox is 1.6 tall: 10 tiles away it spans about +-4.6 degrees.
    assert shoot(world, 0, math.radians(4.0)).target == 1
    assert shoot(world, 0, math.radians(5.0)).target == -1


def test_range_limit(maps):
    edge = 10.0 + R.range + EXACT.hitbox_w / 2
    assert shoot(open_room(maps, positions=((10.0, 30.0), (edge - 0.01, 30.0))), 0, 0.0).target == 1
    assert shoot(open_room(maps, positions=((10.0, 30.0), (edge + 0.01, 30.0))), 0, 0.0).target == -1


def test_rock_blocks_the_shot(maps):
    world = open_room(maps)
    world.grids[0, 29:32, 15] = True
    assert shoot(world, 0, 0.0).target == -1
    world.grids[0, 29:32, 15] = False
    world.grids[0, 25:27, 15] = True   # above the line: no effect
    assert shoot(world, 0, 0.0).target == 1


def test_corner_between_diagonal_rocks_blocks(maps):
    """A shot through the exact corner where two rocks touch diagonally is blocked, as in line_of_sight."""
    world = open_room(maps, positions=((10.5, 30.5), (13.5, 33.5)))
    world.grids[0, 31, 12] = world.grids[0, 32, 11] = True
    assert shoot(world, 0, aim_at(world, 0, 1)).target == -1


def test_nearest_enemy_is_hit_and_teammates_are_ignored(maps):
    # 2v2 slots: [A, A, B, B]. A teammate stands in front, then two enemies in a line.
    world = open_room(maps, EXACT_2V2, ((10.0, 30.0), (14.0, 30.0), (24.0, 30.0), (18.0, 30.0)))
    assert shoot(world, 0, 0.0).target == 3
    world.alive[0, 3] = False
    assert shoot(world, 0, 0.0).target == 2


def test_overlapping_enemy_is_hit_at_the_muzzle(maps):
    world = open_room(maps, positions=((10.0, 30.0), (10.2, 30.3)))
    s = shoot(world, 0, math.pi / 2)
    assert s.target == 1 and s.hit == pytest.approx((10.0, 30.0))


# --- timing and ammo ---

def test_fire_rate_magazine_and_reload(maps):
    world = open_room(maps, CFG)
    aim, trigger = world.aim.copy(), np.zeros_like(world.alive)
    trigger[0, 0] = True
    fired = [t for t in range(400) if fire(world, aim, trigger)]
    interval, reload = CFG.shot_interval_ticks, CFG.reload_ticks
    first = [k * interval for k in range(R.magazine)]
    resume = first[-1] + reload
    assert fired[:R.magazine] == first
    assert fired[R.magazine] == resume                 # 1.5 s after the last round
    assert fired[R.magazine + 1] == resume + interval
    assert world.ammo[0, 1] == R.magazine              # the other rifle never fired


def test_releasing_the_trigger_keeps_the_cooldown(maps):
    world = open_room(maps, CFG)
    aim, on, off = world.aim.copy(), np.zeros_like(world.alive), np.zeros_like(world.alive)
    on[0, 0] = True
    assert fire(world, aim, on)
    for _ in range(3):
        assert not fire(world, aim, off)
    assert not fire(world, aim, on) and not fire(world, aim, on)
    assert fire(world, aim, on)                        # 6 ticks after the first shot


def test_dead_agents_neither_fire_nor_turn(maps):
    world = open_room(maps, CFG)
    world.alive[0, 0] = False
    shots = fire(world, np.full((1, 2), 1.0), np.ones((1, 2), dtype=bool))
    assert [s.shooter for s in shots] == [1]
    assert world.aim[0, 0] == 0.0 and world.aim[0, 1] == 1.0
    assert shots[0].target == -1                       # a dead enemy can't be hit


# --- spread and determinism ---

def test_spread_stays_within_bounds_and_is_uniform(maps):
    world = open_room(maps, CFG)
    offsets = np.array([shoot(world, 0, 0.0).angle for _ in range(4000)])
    limit = math.radians(R.spread_deg)
    assert np.abs(offsets).max() <= limit
    assert abs(offsets.mean()) < 0.05 * limit
    assert np.histogram(offsets, bins=4, range=(-limit, limit))[0].min() > 900


def test_spread_is_seeded_per_match(maps):
    def angles(world, match):
        aim, trigger = world.aim.copy(), np.ones_like(world.alive)
        out = []
        for _ in range(60):
            out += [s.angle for s in fire(world, aim, trigger) if s.match == match]
        return out

    alone = angles(World.single(maps[1], CFG), 0)
    batched = angles(World.from_maps(maps[:3], CFG), 1)
    assert alone == batched                            # the batch around a match doesn't matter
    assert alone != angles(World.single(maps[1], CFG, rng_seed=1), 0)
    world = World.single(maps[1], CFG)
    angles(world, 0)
    world.reset()
    assert angles(world, 0) != alone                   # a new episode draws new spread


# --- contracts ---

def test_agrees_with_dense_sampling(maps):
    """Hits match an independent check: sample the segment to the entry point every 0.002 tiles
    and look for rock. Shots passing within 0.01 tiles of a tile corner are skipped, since
    sampling can't settle those."""
    rng = np.random.default_rng(0)
    checked = hits = 0
    for m in maps[:40]:
        world = World.single(m, EXACT)
        open_spots = np.argwhere(~world.grids[0, 1:-1, 1:-1]) + 1
        for _ in range(80):
            r0, c0 = open_spots[rng.integers(len(open_spots))]
            near = open_spots[np.abs(open_spots - (r0, c0)).max(axis=1) <= 12]
            r1, c1 = near[rng.integers(len(near))]
            world.x[0] = [c0 + rng.uniform(0.4, 0.6), c1 + rng.uniform(0.4, 0.6)]
            world.y[0] = [r0 + rng.uniform(0.2, 0.8), r1 + rng.uniform(0.2, 0.8)]
            angle = aim_at(world, 0, 1) + rng.uniform(-0.05, 0.05)
            s = shoot(world, 0, angle)
            hw, hh = EXACT.hitbox_w / 2, EXACT.hitbox_h / 2
            t = ray_box(world.x[0, 0], world.y[0, 0], math.cos(angle), math.sin(angle),
                        world.x[0, 1] - hw, world.y[0, 1] - hh, world.x[0, 1] + hw, world.y[0, 1] + hh)
            if t is None or t > R.range:
                assert s.target == -1
                continue
            ts = np.arange(0, t, 0.002)
            px, py = world.x[0, 0] + ts * math.cos(angle), world.y[0, 0] + ts * math.sin(angle)
            if (np.hypot(px - np.round(px), py - np.round(py)) < 0.01).any():
                continue
            rows, cols = np.floor(py).astype(int), np.floor(px).astype(int)
            clear = not world.grids[0, rows, cols].any()
            assert (s.target == 1) == clear, (m.seed, world.x[0], world.y[0], angle)
            checked += 1
            hits += clear
    assert checked > 1500 and 100 < hits < checked - 100


def test_spawns_cannot_shoot_enemy_spawns(maps):
    """M40: spawns were placed out of each other's sight, so an agent on a team A spawn can't hit
    one on any team B spawn, aiming at its centre or near any corner of its hitbox."""
    cfg = EXACT
    hw, hh = cfg.hitbox_w / 2 - 0.01, cfg.hitbox_h / 2 - 0.01
    for m in maps:
        world = World.single(m, cfg)
        for a in world.spawns[0, 0]:
            for b in world.spawns[0, 1]:
                for ox, oy in ((0, 0), (-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh)):
                    world.x[0], world.y[0] = [a[0], b[0]], [a[1], b[1]]
                    angle = math.atan2(b[1] + oy - a[1], b[0] + ox - a[0])
                    assert shoot(world, 0, angle).target == -1, (m.seed, a, b)


def test_shooting_on_real_maps_with_physics(maps):
    """Random movement with 2 agents per team on 20 val maps, team B starting on team A's spawns
    and each agent aiming roughly at its partner enemy: hits happen, only enemies are hit, and
    every hit point is on the target's hitbox, within range, with a clear line to it."""
    from rl_donut.mapgen import line_of_sight

    cfg = EngineConfig(team_size=2)
    world = World.from_maps(maps[:20], cfg)
    world.x[:, 2:], world.y[:, 2:] = world.x[:, :2], world.y[:, :2]
    rng = np.random.default_rng(1)
    enemy = (np.arange(cfg.n_agents) + cfg.team_size) % cfg.n_agents
    hw, hh = cfg.hitbox_w / 2, cfg.hitbox_h / 2
    shots = hits = 0
    for tick in range(900):
        if tick % 2 == 0:
            move = rng.integers(-1, 2, size=world.alive.shape)
            jet = rng.random(world.alive.shape) < 0.6
            trigger = rng.random(world.alive.shape) < 0.5
        aim = np.arctan2(world.y[:, enemy] - world.y, world.x[:, enemy] - world.x)
        physics_tick(world, move, jet)
        for s in fire(world, aim + rng.uniform(-0.1, 0.1, size=aim.shape), trigger):
            shots += 1
            if s.target < 0:
                continue
            hits += 1
            m, i, j = s.match, s.shooter, s.target
            assert world.team[j] != world.team[i]
            hx, hy = s.hit
            assert abs(hx - world.x[m, j]) <= hw + 1e-9 and abs(hy - world.y[m, j]) <= hh + 1e-9
            assert math.hypot(hx - world.x[m, i], hy - world.y[m, i]) <= R.range + 1e-9
            assert line_of_sight(world.grids[m], (world.x[m, i], world.y[m, i]), s.hit)
    assert shots > 0.4 * 20 * 4 * 900 / cfg.shot_interval_ticks
    assert hits > 50
