import numpy as np
import pytest

from rl_donut.env import TEAM_A, TEAM_B, EngineConfig, World, spawn_centre
from rl_donut.mapgen import MapConfig, Spawn, load_split
from rl_donut.mapgen.config import RegionConfig


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:50]


def hitbox_tiles(cfg, x, y):
    """Rows and columns of every tile an agent's hitbox overlaps (edges touching don't count)."""
    eps = 1e-9
    cols = range(int(np.floor(x - cfg.hitbox_w / 2 + eps)), int(np.floor(x + cfg.hitbox_w / 2 - eps)) + 1)
    rows = range(int(np.floor(y - cfg.hitbox_h / 2 + eps)), int(np.floor(y + cfg.hitbox_h / 2 - eps)) + 1)
    return rows, cols


# --- config ---

def test_config_rejects_hitbox_larger_than_clearance():
    with pytest.raises(ValueError):
        EngineConfig(hitbox_w=2.5)
    with pytest.raises(ValueError):
        EngineConfig(hitbox_h=3.2)


def test_config_rejects_more_agents_than_spawns():
    with pytest.raises(ValueError):
        EngineConfig(team_size=7)


@pytest.mark.parametrize("bad", [{"team_size": 0}, {"tick_rate": 0}, {"hitbox_w": 0.0}])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        EngineConfig(**bad)


# --- coordinates ---

def test_spawn_centre_puts_feet_on_the_box_floor():
    cfg = EngineConfig()
    x, y = spawn_centre(Spawn(y=10, x=20), cfg)
    assert (x, y) == (21.0, 12.2)
    assert y + cfg.hitbox_h / 2 == 13.0


# --- world construction ---

@pytest.mark.parametrize("team_size", [1, 2])
def test_agents_start_standing_in_air(maps, team_size):
    cfg = EngineConfig(team_size=team_size)
    world = World.from_maps(maps, cfg)
    for m in range(world.n_matches):
        for a in range(cfg.n_agents):
            x, y = world.x[m, a], world.y[m, a]
            rows, cols = hitbox_tiles(cfg, x, y)
            assert not world.grids[m][np.ix_(rows, cols)].any(), "hitbox overlaps rock"
            feet = y + cfg.hitbox_h / 2
            assert feet == int(feet), "feet must rest exactly on a tile edge"
            assert world.grids[m, int(feet), list(cols)].any(), "nothing to stand on"


@pytest.mark.parametrize("team_size", [1, 2])
def test_start_positions_are_mirror_images(maps, team_size):
    cfg = EngineConfig(team_size=team_size)
    world = World.from_maps(maps, cfg)
    a, b = world.team == TEAM_A, world.team == TEAM_B
    assert np.allclose(world.x[:, b], cfg.map.width - world.x[:, a])
    assert np.array_equal(world.y[:, b], world.y[:, a])


def test_teams_are_ordered_a_then_b():
    assert EngineConfig(team_size=2).n_agents == 4
    world_team = World.from_maps(load_split("val")[:1], EngineConfig(team_size=2)).team
    assert world_team.tolist() == [TEAM_A, TEAM_A, TEAM_B, TEAM_B]


def test_batch_equals_single_matches(maps):
    batch = World.from_maps(maps[:5])
    for i, m in enumerate(maps[:5]):
        single = World.single(m)
        assert np.array_equal(batch.x[i], single.x[0]) and np.array_equal(batch.y[i], single.y[0])
        assert np.array_equal(batch.grids[i], single.grids[0])
        assert batch.seeds[i] == m.seed and batch.kill_y[i] == m.kill_y


def test_state_arrays_have_documented_shapes_and_types(maps):
    world = World.from_maps(maps[:3], EngineConfig(team_size=2))
    for name in ("x", "y", "vx", "vy", "fuel", "aim"):
        arr = getattr(world, name)
        assert arr.shape == (3, 4) and arr.dtype == np.float64
    for name in ("ammo", "cooldown", "reload"):
        arr = getattr(world, name)
        assert arr.shape == (3, 4) and arr.dtype == np.int64
    assert world.alive.shape == (3, 4) and world.alive.dtype == bool
    assert world.episode.tolist() == [0, 0, 0] and len(world.rngs) == 3
    assert world.grids.shape == (3, 48, 96) and world.grids.dtype == bool
    assert world.spawns.shape == (3, 2, 6, 2)


def test_reset_restores_only_the_chosen_matches(maps):
    world = World.from_maps(maps[:3])
    start_x = world.x.copy()
    world.x += 1.0
    world.vy[:] = 5.0
    world.alive[:] = False
    world.tick[:] = 99
    world.reset(np.array([1]))
    assert np.array_equal(world.x[1], start_x[1]) and (world.vy[1] == 0).all()
    assert world.alive[1].all() and world.tick[1] == 0
    assert np.array_equal(world.x[[0, 2]], start_x[[0, 2]] + 1.0) and world.tick[0] == 99


def test_agent_view_reads_one_agent(maps):
    world = World.single(maps[0])
    view = world.agent(0, 1)
    assert view.team == TEAM_B and view.alive
    assert (view.x, view.y) == (world.x[0, 1], world.y[0, 1])
    assert isinstance(view.x, float)


def test_maps_of_different_sizes_cannot_share_a_batch(maps):
    with pytest.raises(ValueError):
        World.from_maps([maps[0]], EngineConfig(map=MapConfig(width=80)))
    with pytest.raises(ValueError):
        World.from_maps([])


def test_engine_hitbox_fits_generator_clearance_with_margin():
    """The generator guarantees 2 x 3 tiles of air; the hitbox must keep a margin inside it."""
    cfg, clearance = EngineConfig(), RegionConfig()
    assert cfg.hitbox_w < clearance.clearance_w and cfg.hitbox_h < clearance.clearance_h
