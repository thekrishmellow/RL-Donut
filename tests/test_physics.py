import numpy as np
import pytest

from rl_donut.env import EngineConfig, PhysicsConfig, World, on_ground, overlaps_rock, physics_tick
from rl_donut.env.config import MAX_SCALE, MIN_SCALE
from rl_donut.mapgen import load_split

CFG = EngineConfig()
P = CFG.physics
FLOOR = 46
DECISION_TICKS = 2


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def open_room(maps, cfg=CFG):
    """One match on an empty room: 1-tile frame, floor at row 46. Agent 0 stands at x = 10;
    agent 1 is dead so it stays out of the way."""
    world = World.single(maps[0], cfg)
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[FLOOR:, :] = True
    world.grids[0] = grid
    world.x[0, 0], world.y[0, 0] = 10.0, FLOOR - cfg.hitbox_h / 2
    world.vx[:] = world.vy[:] = 0.0
    world.alive[0, 1] = False
    return world


def controls(move=0, jet=False):
    return np.array([[move, 0]]), np.array([[jet, False]])


def climb_height(maps, cfg=CFG):
    """Full thrust from standing until the tank is empty, then coast to the top."""
    world = open_room(maps, cfg)
    start = world.y[0, 0]
    top = start
    while world.fuel[0, 0] > 0 or world.vy[0, 0] < 0:
        physics_tick(world, *controls(jet=True))
        top = min(top, world.y[0, 0])
    return start - top


def level_flight(maps, cfg=CFG):
    """Fly right at constant height on one tank: thrust whenever at or below the start
    height or sinking, decided every 2 ticks like an agent. Returns the distance covered and
    the largest height error."""
    world = open_room(maps, cfg)
    world.y[0, 0] -= 10.0
    start_x, start_y = world.x[0, 0], world.y[0, 0]
    error, tick, jet = 0.0, 0, False
    while world.fuel[0, 0] > 0:
        if tick % DECISION_TICKS == 0:
            jet = world.y[0, 0] >= start_y or world.vy[0, 0] > 0
        physics_tick(world, *controls(move=1, jet=jet))
        error = max(error, abs(world.y[0, 0] - start_y))
        tick += 1
    return world.x[0, 0] - start_x, error


# --- contracts with the map generator ---

def test_climb_contract(maps):
    """M26: one full tank must climb at least the generator's climb budget."""
    height = climb_height(maps)
    assert height >= CFG.map.climb_budget
    assert height == pytest.approx(35.7, abs=0.2)


def test_level_flight_contract(maps):
    """The generator charges 1 fuel unit per sideways tile from a tank of 2 x climb_budget;
    level flight must beat that with 20% to spare."""
    distance, error = level_flight(maps)
    assert distance >= 1.2 * 2 * CFG.map.climb_budget
    assert distance == pytest.approx(30.0, abs=0.3)
    assert error < 0.5


def test_draft_drain_rate_would_leave_no_margin(maps):
    """Why fuel_drain is 32 and not 40: at 40, level flight barely beats 24 tiles."""
    cfg = EngineConfig(physics=PhysicsConfig(fuel_drain=40.0))
    distance, _ = level_flight(maps, cfg)
    assert 2 * CFG.map.climb_budget <= distance < 1.2 * 2 * CFG.map.climb_budget


def scaled(k):
    return EngineConfig(physics=PhysicsConfig(scale_range=(k, k)))


@pytest.mark.parametrize("k", [MIN_SCALE, 0.85, 1.15, MAX_SCALE])
def test_generator_contracts_hold_across_the_physics_scale_range(maps, k):
    """Scaling gravity and thrust together keeps the thrust duty cycle of level flight, so
    the tank lasts as long; climbing stays far above the climb budget."""
    assert climb_height(maps, scaled(k)) >= CFG.map.climb_budget
    distance, error = level_flight(maps, scaled(k))
    assert distance >= 1.2 * 2 * CFG.map.climb_budget
    assert error < 0.5


def test_physics_scale_multiplies_gravity_and_thrust(maps):
    for jet, accel in ((False, P.gravity), (True, P.gravity - P.thrust)):
        world = open_room(maps, scaled(1.2))
        world.y[0, 0] = 20.0
        physics_tick(world, *controls(jet=jet))
        assert world.vy[0, 0] == pytest.approx(1.2 * accel * CFG.dt)


def test_physics_scale_is_drawn_per_match_and_episode(maps):
    cfg = EngineConfig(physics=PhysicsConfig(scale_range=(0.85, 1.15)))
    world = World.from_maps(maps, cfg)
    first = world.physics_scale.copy()
    assert ((first >= 0.85) & (first <= 1.15)).all() and len(np.unique(first)) == len(maps)
    world.reset()
    assert not np.array_equal(world.physics_scale, first)
    assert np.array_equal(World.from_maps(maps, cfg).physics_scale, first)
    assert (World.from_maps(maps).physics_scale == 1.0).all()


def test_physics_scale_leaves_rifle_spread_unchanged(maps):
    plain = World.from_maps(maps[:5])
    varied = World.from_maps(maps[:5], EngineConfig(physics=PhysicsConfig(scale_range=(0.85, 1.15))))
    for a, b in zip(plain.rngs, varied.rngs):
        assert a.random() == b.random()


@pytest.mark.parametrize("bad", [(1.1, 0.9), (0.4, 1.0), (1.0, 1.6)])
def test_invalid_scale_range_rejected(bad):
    with pytest.raises(ValueError):
        PhysicsConfig(scale_range=bad)


def test_speed_limits_must_stay_under_a_tile_per_tick():
    with pytest.raises(ValueError):
        EngineConfig(physics=PhysicsConfig(max_rise_speed=30.0))
    with pytest.raises(ValueError):
        EngineConfig(physics=PhysicsConfig(max_fall_speed=31.0))
    with pytest.raises(ValueError):
        EngineConfig(tick_rate=10)


@pytest.mark.parametrize("bad", [{"thrust": 20.0}, {"gravity": 0.0}, {"fuel_drain": 0.0}])
def test_invalid_physics_rejected(bad):
    with pytest.raises(ValueError):
        PhysicsConfig(**bad)


# --- speeds ---

def test_rising_speed_is_capped(maps):
    world = open_room(maps)
    for _ in range(60):
        physics_tick(world, *controls(jet=True))
        assert world.vy[0, 0] >= -P.max_rise_speed
    assert world.vy[0, 0] == -P.max_rise_speed


def test_falling_speed_is_capped(maps):
    world = open_room(maps)
    world.y[0, 0] = 2.0
    fastest = 0.0
    for _ in range(40):
        physics_tick(world, *controls())
        fastest = max(fastest, world.vy[0, 0])
    assert fastest == P.max_fall_speed


def test_horizontal_speed_is_set_directly_on_ground_and_in_air(maps):
    world = open_room(maps)
    physics_tick(world, *controls(move=1))
    assert world.vx[0, 0] == P.run_speed
    world.y[0, 0] = 20.0
    physics_tick(world, *controls(move=-1))
    assert world.vx[0, 0] == -P.run_speed
    physics_tick(world, *controls(move=0))
    assert world.vx[0, 0] == 0.0


def test_invalid_move_rejected(maps):
    world = open_room(maps)
    with pytest.raises(ValueError):
        physics_tick(world, np.array([[2, 0]]), np.array([[False, False]]))


# --- fuel ---

def test_fuel_drains_at_the_set_rate(maps):
    world = open_room(maps)
    for _ in range(30):
        physics_tick(world, *controls(jet=True))
    assert world.fuel[0, 0] == pytest.approx(P.fuel_capacity - P.fuel_drain)


def test_empty_tank_gives_no_thrust(maps):
    world = open_room(maps)
    world.fuel[0, 0] = 0.0
    physics_tick(world, *controls(jet=True))
    assert on_ground(world)[0, 0] and world.vy[0, 0] == 0.0


def test_fuel_refills_only_while_standing(maps):
    world = open_room(maps)
    world.fuel[0, 0] = 0.0
    world.y[0, 0] = 10.0
    physics_tick(world, *controls())
    assert world.fuel[0, 0] == 0.0
    world.y[0, 0] = FLOOR - CFG.hitbox_h / 2
    refill_ticks = round(P.fuel_capacity / P.fuel_refill * CFG.tick_rate)
    for _ in range(refill_ticks):
        physics_tick(world, *controls())
    assert world.fuel[0, 0] == pytest.approx(P.fuel_capacity)
    physics_tick(world, *controls())
    assert world.fuel[0, 0] == P.fuel_capacity


def test_standing_still_keeps_an_agent_on_the_floor(maps):
    world = open_room(maps)
    y = world.y[0, 0]
    for _ in range(100):
        physics_tick(world, *controls())
    assert world.y[0, 0] == y and world.vy[0, 0] == 0.0 and on_ground(world)[0, 0]


def test_dead_agents_ignore_controls(maps):
    world = open_room(maps)
    x, y, fuel = world.x[0, 1], world.y[0, 1], world.fuel[0, 1]
    physics_tick(world, np.array([[0, 1]]), np.array([[False, True]]))
    assert (world.x[0, 1], world.y[0, 1], world.fuel[0, 1]) == (x, y, fuel)


# --- whole maps ---

def random_controls(rng, shape):
    return rng.integers(-1, 2, shape), rng.random(shape) < 0.6


def test_random_play_on_real_maps_stays_in_air_and_under_a_tile_per_tick(maps):
    world = World.from_maps(maps, EngineConfig(team_size=2))
    rng = np.random.default_rng(0)
    for tick in range(900):
        if tick % DECISION_TICKS == 0:
            move_dir, jet = random_controls(rng, world.x.shape)
        physics_tick(world, move_dir, jet)
        world.alive &= world.y < world.kill_y[:, None]
        assert not overlaps_rock(world).any()
        assert (np.abs(world.vx) * CFG.dt < 1).all() and (np.abs(world.vy) * CFG.dt < 1).all()
        assert ((world.fuel >= 0) & (world.fuel <= P.fuel_capacity)).all()


def test_same_controls_give_the_same_trajectory(maps):
    def run():
        world = World.from_maps(maps[:10], EngineConfig(team_size=2))
        rng = np.random.default_rng(3)
        for tick in range(300):
            if tick % DECISION_TICKS == 0:
                move_dir, jet = random_controls(rng, world.x.shape)
            physics_tick(world, move_dir, jet)
        return world.x.copy(), world.y.copy(), world.fuel.copy()

    for a, b in zip(run(), run()):
        assert np.array_equal(a, b)
