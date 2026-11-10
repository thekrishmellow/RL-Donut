import math

import numpy as np
import pytest

from rl_donut.env import (EngineConfig, LifeConfig, RifleConfig, Shot, World, fire, life_tick, overlaps_rock,
                          physics_tick, pick_spawn)
from rl_donut.mapgen import line_of_sight, load_split

CFG = EngineConfig(rifle=RifleConfig(spread_deg=0.0))
CFG_2V2 = EngineConfig(team_size=2, rifle=RifleConfig(spread_deg=0.0))
L = CFG.life
DAMAGE = CFG.rifle.damage
RESPAWN, PROTECTION, DELAY = CFG.ticks(L.respawn_time), CFG.ticks(L.protection_time), CFG.ticks(L.regen_delay)


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:60]


def open_room(maps, cfg=CFG, positions=((10.0, 30.0), (20.0, 30.0))):
    world = World.single(maps[0], cfg)
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[46:, :] = True
    world.grids[0] = grid
    for i, (x, y) in enumerate(positions):
        world.x[0, i], world.y[0, i] = x, y
    return world


def hit(shooter, target, match=0):
    return Shot(match, shooter, 0.0, target, (0.0, 0.0))


def quiet(world, ticks):
    for _ in range(ticks):
        life_tick(world, [])


# --- settings ---

@pytest.mark.parametrize("kwargs", [dict(max_health=0), dict(respawn_time=0), dict(regen_rate=-1)])
def test_invalid_life_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LifeConfig(**kwargs)


def test_life_times_must_be_whole_ticks():
    with pytest.raises(ValueError):
        EngineConfig(life=LifeConfig(respawn_time=2.01))
    assert (RESPAWN, PROTECTION, DELAY) == (60, 30, 90)


def test_reset_restores_health(maps):
    world = World.from_maps(maps[:2], CFG)
    world.health[:] = 5.0
    world.alive[:] = False
    world.respawn[:] = world.protection[:] = 9
    world.last_damager[:] = 1
    world.reset([0])
    assert (world.health[0] == L.max_health).all() and world.alive[0].all()
    assert (world.respawn[0] == 0).all() and (world.protection[0] == 0).all() and (world.last_damager[0] == -1).all()
    assert (world.health[1] == 5.0).all()


# --- damage and death ---

def test_seven_hits_kill_and_events_add_up(maps):
    world = open_room(maps)
    for k in range(6):
        ev = life_tick(world, [hit(0, 1)])
        assert ev.dealt[0].tolist() == [DAMAGE, 0] and ev.taken[0].tolist() == [0, DAMAGE]
        assert not ev.deaths
    assert world.health[0, 1] == L.max_health - 6 * DAMAGE
    ev = life_tick(world, [hit(0, 1)])
    assert [(d.agent, d.killer, d.fell) for d in ev.deaths] == [(1, 0, False)]
    assert not world.alive[0, 1] and world.respawn[0, 1] == RESPAWN


def test_hits_on_the_same_tick_are_simultaneous(maps):
    world = open_room(maps)
    world.health[0] = DAMAGE
    ev = life_tick(world, [hit(0, 1), hit(1, 0)])
    assert sorted((d.agent, d.killer) for d in ev.deaths) == [(0, 1), (1, 0)]
    assert not world.alive[0].any()


def test_killer_is_the_top_damage_dealer_with_fair_ties(maps):
    world = open_room(maps, CFG_2V2, ((10.0, 30.0), (12.0, 30.0), (20.0, 30.0), (22.0, 30.0)))
    world.health[0, 2] = DAMAGE
    ev = life_tick(world, [hit(0, 2), hit(1, 2), hit(1, 2)])
    assert ev.deaths[0].killer == 1
    killers = []
    for seed in range(40):
        world = open_room(maps, CFG_2V2, ((10.0, 30.0), (12.0, 30.0), (20.0, 30.0), (22.0, 30.0)))
        world.rngs[0] = np.random.default_rng(seed)
        world.health[0, 2] = DAMAGE
        killers.append(life_tick(world, [hit(0, 2), hit(1, 2)]).deaths[0].killer)
    assert set(killers) == {0, 1}


def test_falling_below_the_kill_line_kills_without_a_killer(maps):
    world = open_room(maps)
    life_tick(world, [hit(0, 1)])
    quiet(world, 10)
    world.y[0, 1] = world.kill_y[0] + 0.1
    ev = life_tick(world, [])
    d = ev.deaths[0]
    assert (d.agent, d.killer, d.fell, d.last_damager, d.ticks_since_damage) == (1, -1, True, 0, 11)


# --- regeneration ---

def test_regeneration_waits_then_heals_to_full(maps):
    world = open_room(maps)
    world.health[0, 1] = 50.0
    life_tick(world, [hit(0, 1)])
    quiet(world, DELAY - 1)
    assert world.health[0, 1] == 35.0
    life_tick(world, [])
    assert world.health[0, 1] == pytest.approx(35.0 + L.regen_rate * CFG.dt)
    quiet(world, CFG.tick_rate * 7)
    assert world.health[0, 1] == L.max_health


def test_damage_restarts_the_regeneration_wait(maps):
    world = open_room(maps)
    life_tick(world, [hit(0, 1)])
    quiet(world, DELAY + 10)
    healed = world.health[0, 1]
    life_tick(world, [hit(0, 1)])
    quiet(world, DELAY - 1)
    assert world.health[0, 1] == healed - DAMAGE


# --- respawn and protection ---

def test_respawn_after_two_seconds_with_a_fresh_start(maps):
    world = open_room(maps)
    world.health[0, 1] = DAMAGE
    world.fuel[0, 1], world.ammo[0, 1], world.reload[0, 1] = 3.0, 0, 20
    life_tick(world, [hit(0, 1)])
    for _ in range(RESPAWN - 1):
        assert not life_tick(world, []).respawns
        assert not world.alive[0, 1]
    ev = life_tick(world, [])
    assert ev.respawns == [(0, 1)]
    a = world.agent(0, 1)
    assert a.alive and a.health == L.max_health and a.fuel == CFG.physics.fuel_capacity
    assert (a.ammo, a.reload, a.cooldown, a.vx, a.vy, a.protection) == (CFG.rifle.magazine, 0, 0, 0.0, 0.0, PROTECTION)
    assert (a.x, a.y) in [tuple(s) for s in world.spawns[0, 1]]
    assert a.aim == (0.0 if a.x < 48 else math.pi)


def test_protection_blocks_damage_for_one_second(maps):
    world = open_room(maps)
    world.protection[0, 1] = PROTECTION
    for _ in range(PROTECTION):
        assert life_tick(world, [hit(0, 1)]).taken[0, 1] == 0
    assert life_tick(world, [hit(0, 1)]).taken[0, 1] == DAMAGE


def test_firing_ends_protection(maps):
    world = open_room(maps)
    world.protection[0, 1] = PROTECTION
    ev = life_tick(world, [hit(1, 0), hit(0, 1)])
    assert ev.taken[0, 1] == DAMAGE and world.protection[0, 1] == 0


def test_dead_agents_stay_frozen_and_out_of_reach(maps):
    world = open_room(maps)
    world.health[0, 1] = DAMAGE
    life_tick(world, [hit(0, 1)])
    x, y, health = world.x[0, 1], world.y[0, 1], world.health[0, 1]
    for _ in range(20):
        physics_tick(world, np.ones((1, 2), dtype=np.int64), np.ones((1, 2), dtype=bool))
        shots = fire(world, np.zeros((1, 2)), np.ones((1, 2), dtype=bool))
        assert all(s.target == -1 for s in shots)
        life_tick(world, shots)
    assert (world.x[0, 1], world.y[0, 1], world.health[0, 1]) == (x, y, health)


def test_agents_out_of_play_never_respawn(maps):
    world = open_room(maps)
    world.alive[0, 1] = False
    quiet(world, 200)
    assert not world.alive[0, 1]


def test_respawn_spawn_is_hidden_from_living_enemies(maps):
    """On 60 val maps, with enemies at random open positions: the chosen spawn is out of
    every living enemy's sight whenever such a spawn exists, and otherwise the farthest."""
    rng = np.random.default_rng(0)
    safe_cases = fallback_cases = 0
    for m in maps:
        world = World.single(m, CFG_2V2)
        open_spots = np.argwhere(~world.grids[0, 1:-1, 1:-1]) + 1
        for _ in range(10):
            for j in (2, 3):
                r, c = open_spots[rng.integers(len(open_spots))]
                world.x[0, j], world.y[0, j] = c + 0.5, r + 0.5
            world.alive[0, 3] = rng.random() < 0.5
            eyes = [(world.x[0, j], world.y[0, j]) for j in (2, 3) if world.alive[0, j]]
            spawns = [tuple(s) for s in world.spawns[0, 0]]
            seen = [any(line_of_sight(world.grids[0], e, s) for e in eyes) for s in spawns]
            chosen = spawns.index(pick_spawn(world, 0, 0))
            if not all(seen):
                assert not seen[chosen]
                safe_cases += 1
            else:
                dist = [min(math.hypot(s[0] - e[0], s[1] - e[1]) for e in eyes) for s in spawns]
                assert chosen == int(np.argmax(dist))
                fallback_cases += 1
    assert safe_cases > 500


def test_respawn_spawn_is_random_among_safe_ones(maps):
    world = World.single(maps[0], CFG)
    world.alive[0, 1] = False
    picks = {pick_spawn(world, 0, 0) for _ in range(200)}
    assert len(picks) == len(world.spawns[0, 0])


def test_fallback_when_every_spawn_is_seen(maps):
    world = open_room(maps)
    world.grids[0, 1:-1, 1:-1] = False
    world.x[0, 1], world.y[0, 1] = 5.0, 30.0
    spawns = [tuple(s) for s in world.spawns[0, 0]]
    far = max(spawns, key=lambda s: math.hypot(s[0] - 5.0, s[1] - 30.0))
    assert pick_spawn(world, 0, 0) == far


# --- the whole loop ---

def test_random_fights_on_real_maps(maps):
    """2v2 on 20 val maps for 60 s with physics, rifle and life together, enemies starting on
    each other's spawns: kills, falls and respawns happen; health stays in range; living agents
    never enter rock; nobody stays dead longer than the respawn time; events add up."""
    cfg = EngineConfig(team_size=2)
    world = World.from_maps(maps[:20], cfg)
    world.x[:, 2:], world.y[:, 2:] = world.x[:, :2], world.y[:, :2]
    rng = np.random.default_rng(2)
    enemy = (np.arange(4) + 2) % 4
    dead_for = np.zeros_like(world.respawn)
    kills = falls = respawns = 0
    dealt_total = taken_total = 0.0
    for tick in range(60 * cfg.tick_rate):
        if tick % 2 == 0:
            move = rng.integers(-1, 2, size=world.alive.shape)
            jet = rng.random(world.alive.shape) < 0.6
            trigger = rng.random(world.alive.shape) < 0.7
        aim = np.arctan2(world.y[:, enemy] - world.y, world.x[:, enemy] - world.x)
        physics_tick(world, move, jet)
        ev = life_tick(world, fire(world, aim + rng.uniform(-0.1, 0.1, size=aim.shape), trigger))
        kills += sum(d.killer >= 0 for d in ev.deaths)
        falls += sum(d.fell for d in ev.deaths)
        respawns += len(ev.respawns)
        dealt_total += ev.dealt.sum()
        taken_total += ev.taken.sum()
        for d in ev.deaths:
            assert d.killer < 0 or world.team[d.killer] != world.team[d.agent]
        assert (world.health[world.alive] > 0).all() and (world.health <= cfg.life.max_health).all()
        assert not overlaps_rock(world).any()
        dead_for = np.where(world.alive, 0, dead_for + 1)
        assert dead_for.max() <= RESPAWN
    assert kills > 20 and respawns > 20
    assert dealt_total == taken_total


def test_same_inputs_same_match(maps):
    def play(world):
        rng = np.random.default_rng(5)
        log = []
        for tick in range(600):
            move = rng.integers(-1, 2, size=world.alive.shape)
            aim = rng.uniform(-math.pi, math.pi, size=world.alive.shape)
            physics_tick(world, move, rng.random(world.alive.shape) < 0.6)
            ev = life_tick(world, fire(world, aim, rng.random(world.alive.shape) < 0.7))
            log.append((ev.deaths, ev.respawns, world.x.copy(), world.health.copy()))
        return log

    a = play(World.from_maps(maps[:5], EngineConfig(team_size=2)))
    b = play(World.from_maps(maps[:5], EngineConfig(team_size=2)))
    for (d1, r1, x1, h1), (d2, r2, x2, h2) in zip(a, b):
        assert d1 == d2 and r1 == r2
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(h1, h2)
