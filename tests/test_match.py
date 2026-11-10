import math

import numpy as np
import pytest

from rl_donut.env import EngineConfig, MatchConfig, RifleConfig, World, match_step
from rl_donut.mapgen import load_split

CFG = EngineConfig(rifle=RifleConfig(spread_deg=0.0))
STEPS = CFG.ticks(CFG.match.duration) // CFG.match.decision_ticks


@pytest.fixture(scope="module")
def maps():
    return load_split("val")[:40]


def idle(world):
    shape = world.alive.shape
    return (np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool), world.aim.copy(),
            np.zeros(shape, dtype=bool))


def open_room(maps, cfg=CFG, positions=((10.0, 30.0), (20.0, 30.0))):
    world = World.single(maps[0], cfg)
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[40:, :] = True
    world.grids[0] = grid
    for i, (x, y) in enumerate(positions):
        world.x[0, i], world.y[0, i] = x, y
    return world


# --- settings ---

@pytest.mark.parametrize("kwargs", [dict(duration=0), dict(decision_ticks=0), dict(fall_credit_window=-1)])
def test_invalid_match_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MatchConfig(**kwargs)


def test_duration_must_be_whole_decisions():
    with pytest.raises(ValueError):
        EngineConfig(match=MatchConfig(duration=1.0, decision_ticks=4))   # 30 ticks
    with pytest.raises(ValueError):
        EngineConfig(match=MatchConfig(duration=60.01))
    assert STEPS == 900


# --- the clock ---

def test_a_step_holds_the_action_for_two_ticks(maps):
    world = World.single(maps[0], CFG)
    x0 = world.x[0, 0]
    move, jet, aim, trigger = idle(world)
    move[0, 0] = 1
    match_step(world, move, jet, aim, trigger)
    assert world.tick[0] == 2
    assert world.x[0, 0] == pytest.approx(x0 + 2 * CFG.physics.run_speed * CFG.dt)


def test_match_ends_after_900_steps_and_must_be_restarted(maps):
    world = World.from_maps(maps[:2], CFG)
    for step in range(STEPS):
        ev = match_step(world, *idle(world))
        assert ev.done.all() == (step == STEPS - 1)
    assert ev.outcome.tolist() == [[0, 0], [0, 0]]          # nobody scored: draws
    with pytest.raises(ValueError):
        match_step(world, *idle(world))
    world.reset([0])
    with pytest.raises(ValueError):
        match_step(world, *idle(world))                     # match 1 is still over
    world.reset([1])
    assert not match_step(world, *idle(world)).done.any()


def test_holding_the_trigger_fires_every_three_steps(maps):
    world = World.single(maps[0], CFG)
    move, jet, aim, trigger = idle(world)
    trigger[:] = True
    shots = [match_step(world, move, jet, aim, trigger).shots[0, 0] for _ in range(12)]
    assert shots == [1, 0, 0] * 4                           # 6 ticks between shots, 2 ticks per step


# --- scoring ---

def test_kills_score_and_decide_the_match(maps):
    world = open_room(maps)
    move, jet, aim, trigger = idle(world)
    aim[0, 0], trigger[0, 0] = 0.0, True
    kills = deaths = 0
    for step in range(STEPS):
        ev = match_step(world, move, jet, aim, trigger)
        kills += ev.kills[0, 0]
        deaths += ev.deaths[0, 1]
        world.x[0, 0], world.y[0, 0] = 10.0, 30.0           # hold both in mid-air, target in front
        if world.alive[0, 1]:
            world.x[0, 1], world.y[0, 1] = 20.0, 30.0
    assert kills == deaths == world.score[0, 0] > 0 and world.score[0, 1] == 0
    assert ev.outcome[0].tolist() == [1, -1]
    assert ev.damage_dealt[0, 0] >= 0 and ev.hits[0, 0] <= ev.shots[0, 0]


def test_a_trade_scores_for_both_teams(maps):
    world = open_room(maps)
    world.health[0] = CFG.rifle.damage
    move, jet, aim, trigger = idle(world)
    aim[0] = [0.0, math.pi]
    trigger[0] = True
    ev = match_step(world, move, jet, aim, trigger)
    assert ev.kills[0].tolist() == [1, 1] and world.score[0].tolist() == [1, 1]


def fall_after(maps, delay_steps):
    """Agent 0 hits agent 1 once, then agent 1 drops through a pit `delay_steps` later."""
    world = open_room(maps)
    world.grids[0, 40:, 30:34] = False                      # a pit in the floor
    move, jet, aim, trigger = idle(world)
    aim[0, 0], trigger[0, 0] = 0.0, True
    match_step(world, move, jet, aim, trigger)
    trigger[:] = False
    for _ in range(delay_steps):
        match_step(world, move, jet, aim, trigger)
    world.x[0, 1], world.y[0, 1] = 32.0, world.kill_y[0] - 0.01
    world.vy[0, 1] = CFG.physics.max_fall_speed
    return world, match_step(world, move, jet, aim, trigger)


def test_a_fall_soon_after_damage_scores_for_the_damager(maps):
    world, ev = fall_after(maps, 10)
    assert ev.fell[0, 1] and ev.deaths[0, 1] == 1
    assert ev.kills[0, 0] == 1 and world.score[0].tolist() == [1, 0]
    assert ev.death_log[0].scorer == 0 and ev.death_log[0].death.killer == -1


def test_a_fall_long_after_damage_scores_for_nobody(maps):
    window = CFG.ticks(CFG.match.fall_credit_window) // CFG.match.decision_ticks
    world, ev = fall_after(maps, window + 1)
    assert ev.fell[0, 1] and ev.kills.sum() == 0 and world.score[0].tolist() == [0, 0]
    assert ev.death_log[0].scorer == -1


def test_an_accidental_fall_scores_for_nobody(maps):
    world = open_room(maps)
    world.y[0, 1], world.vy[0, 1] = world.kill_y[0] - 0.01, CFG.physics.max_fall_speed
    world.grids[0] = False
    ev = match_step(world, *idle(world))
    assert ev.fell[0, 1] and ev.kills.sum() == 0


# --- new maps ---

def test_load_maps_restarts_only_the_given_matches(maps):
    world = World.from_maps(maps[:3], CFG)
    for _ in range(30):
        match_step(world, *idle(world))
    world.score[:] = 2
    world.load_maps([1], [maps[10]])
    assert world.seeds.tolist() == [maps[0].seed, maps[10].seed, maps[2].seed]
    np.testing.assert_array_equal(world.grids[1], maps[10].grid.astype(bool))
    assert world.kill_y[1] == maps[10].kill_y
    assert world.tick.tolist() == [60, 0, 60] and world.score[:, 0].tolist() == [2, 0, 2]
    fresh = World.single(maps[10], CFG)
    np.testing.assert_array_equal(world.x[1], fresh.x[0])
    np.testing.assert_array_equal(world.spawns[1], fresh.spawns[0])
    assert world.episode[1] == 0
    with pytest.raises(ValueError):
        world.load_maps([0, 1], [maps[5]])


# --- determinism and fairness ---

def random_actions(rng, shape):
    return (rng.integers(-1, 2, size=shape), rng.random(shape) < 0.6,
            rng.uniform(-math.pi, math.pi, size=shape), rng.random(shape) < 0.7)


BATCH = 5


def play(world, steps, seed, log_match):
    """Random actions drawn for a batch of BATCH matches, so match 0 gets the same actions
    whatever the world's real batch size."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        actions = random_actions(rng, (BATCH, world.alive.shape[1]))
        ev = match_step(world, *(a[:world.n_matches] for a in actions))
        m = log_match
        out.append((world.x[m].copy(), world.y[m].copy(), world.health[m].copy(), world.score[m].copy(),
                    [(t, s.shooter, s.angle, s.target) for t, s in ev.shot_log if s.match == m]))
    return out


def test_same_seed_same_match(maps):
    cfg = EngineConfig(team_size=2)
    a = play(World.from_maps(maps[:5], cfg, rng_seed=3), 300, 0, log_match=2)
    b = play(World.from_maps(maps[:5], cfg, rng_seed=3), 300, 0, log_match=2)
    for (x1, y1, h1, s1, l1), (x2, y2, h2, s2, l2) in zip(a, b):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(s1, s2)
        assert l1 == l2


def test_a_match_does_not_depend_on_its_batch(maps):
    """Match 0 plays out bit for bit the same alone and in a batch of 5, given the same actions."""
    cfg = EngineConfig(team_size=2)
    alone = play(World.from_maps(maps[:1], cfg), 450, 1, log_match=0)
    batched = play(World.from_maps(maps[:5], cfg), 450, 1, log_match=0)
    assert any(r[4] for r in alone)
    for (x1, y1, h1, s1, l1), (x2, y2, h2, s2, l2) in zip(alone, batched):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(h1, h2)
        assert l1 == l2


def test_mirrored_actions_give_mirrored_play(maps):
    """With spread off, team B mirroring team A's actions keeps the whole match mirrored:
    positions, health and shots (until a respawn's random spawn choice breaks the symmetry)."""
    cfg = EngineConfig(rifle=RifleConfig(spread_deg=0.0))
    width = 96
    checked = 0
    for m in maps[:20]:
        world = World.single(m, cfg)
        world.x[0, 1], world.y[0, 1] = width - world.x[0, 0], world.y[0, 0]
        rng = np.random.default_rng(m.seed)
        for _ in range(300):
            move, jet, aim, trigger = random_actions(rng, (1, 2))
            move[0, 1], jet[0, 1], aim[0, 1], trigger[0, 1] = -move[0, 0], jet[0, 0], math.pi - aim[0, 0], trigger[0, 0]
            ev = match_step(world, move, jet, aim, trigger)
            if ev.respawn_log:
                break
            assert world.x[0, 1] == pytest.approx(width - world.x[0, 0], abs=1e-9)
            assert world.y[0, 1] == pytest.approx(world.y[0, 0], abs=1e-9)
            assert world.health[0, 0] == world.health[0, 1]
            assert ev.shots[0, 0] == ev.shots[0, 1] and ev.hits[0, 0] == ev.hits[0, 1]
            checked += 1
    assert checked > 3000


def test_events_add_up_over_full_matches(maps):
    """Two full 2v2 matches on 20 maps with random fighters starting on each other's spawns."""
    cfg = EngineConfig(team_size=2)
    world = World.from_maps(maps[:20], cfg)
    world.x[:, 2:], world.y[:, 2:] = world.x[:, :2], world.y[:, :2]
    rng = np.random.default_rng(4)
    enemy = (np.arange(4) + 2) % 4
    totals = {k: 0 for k in ("kills", "deaths", "falls", "credited_falls", "dealt", "taken")}
    for step in range(STEPS):
        move, jet, _, trigger = random_actions(rng, world.alive.shape)
        aim = np.arctan2(world.y[:, enemy] - world.y, world.x[:, enemy] - world.x) + rng.uniform(-0.1, 0.1, (20, 4))
        ev = match_step(world, move, jet, aim, trigger)
        totals["kills"] += ev.kills.sum()
        totals["deaths"] += ev.deaths.sum()
        totals["falls"] += ev.fell.sum()
        totals["credited_falls"] += sum(r.death.fell and r.scorer >= 0 for r in ev.death_log)
        totals["dealt"] += ev.damage_dealt.sum()
        totals["taken"] += ev.damage_taken.sum()
        for r in ev.death_log:
            assert r.scorer < 0 or world.team[r.scorer] != world.team[r.death.agent]
    assert ev.done.all()
    assert world.score.sum() == totals["kills"]
    assert totals["kills"] == totals["deaths"] - totals["falls"] + totals["credited_falls"]
    assert totals["dealt"] == totals["taken"] and totals["kills"] > 50
    lead = np.sign(world.score[:, 0] - world.score[:, 1])
    np.testing.assert_array_equal(ev.outcome, np.stack([lead, -lead], axis=1))
