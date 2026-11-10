import numpy as np
import pytest

from rl_donut.env import EngineConfig, Navigator, RouteBot, World, on_ground, run_routes, sample_routes
from rl_donut.mapgen import load_split
from rl_donut.mapgen.reach import SIDE_COST, UP_COST

CFG = EngineConfig()
MAPS, PER_MAP = 30, 5
ARRIVE_RATE = 0.99


@pytest.fixture(scope="module")
def navs():
    return [Navigator(m, CFG) for m in load_split("val")[:MAPS]]


@pytest.fixture(scope="module")
def routes(navs):
    return sample_routes(navs, PER_MAP)


@pytest.fixture(scope="module")
def results(routes):
    return run_routes(routes, CFG, full_tank=True)


def test_sampled_routes_join_different_main_segments(navs, routes):
    assert len(routes) == MAPS * PER_MAP
    for nav, start, goal in routes:
        assert nav.reach.main[nav.segments[start]] and nav.reach.main[goal]
        assert nav.segments[start] != goal


def test_sampling_is_seeded(navs, routes):
    again = sample_routes(navs, PER_MAP)
    assert [(n.map.seed, s, g) for n, s, g in routes] == [(n.map.seed, s, g) for n, s, g in again]
    other = sample_routes(navs, PER_MAP, seed=1)
    assert [(s, g) for _, s, g in routes] != [(s, g) for _, s, g in other]


def test_next_segment_leads_to_goal(routes):
    for nav, start, goal in routes:
        a, hops = int(nav.segments[start]), 0
        while a != goal:
            b = nav.next_segment(a, goal)
            assert nav.reach.reach[a, b]
            a, hops = b, hops + 1
            assert hops <= len(nav.reach.sizes)


def test_hop_is_a_chain_the_generator_believed_in(routes):
    """One-box steps through clearance boxes, ending on the target floor within one tank,
    and the steps' model costs add up to the landing's flight cost."""
    for nav, start, goal in routes:
        a = int(nav.segments[start])
        b = nav.next_segment(a, goal)
        boxes = nav.hop(a, b, start)
        assert boxes[0] == start
        assert nav.segments[boxes[-1]] == b
        assert all(nav.fits[box] for box in boxes)
        spent = 0
        for (r0, c0), (r1, c1) in zip(boxes, boxes[1:]):
            assert abs(r1 - r0) + abs(c1 - c0) == 1
            if nav.segments[r0, c0] != a or nav.segments[r1, c1] != a:
                spent += UP_COST if r1 < r0 else SIDE_COST if r1 == r0 else 0
        assert spent == nav.hop_cost(a, boxes[-1]) <= nav.tank


def test_bot_waits_on_the_ground_until_refuelled(navs):
    nav = navs[0]
    spot = tuple(nav.main_spots()[0])
    goal = next(int(s) for s in np.unique(nav.segments[tuple(nav.main_spots().T)]) if s != nav.segments[spot])
    bot = RouteBot(nav, goal, full_tank=True)
    x, y = nav.standing_centre(spot)
    assert bot.decide(x, y, 0.0, CFG.physics.fuel_capacity - 1, True) == (0, False)
    assert bot.status == "flying" and bot.hops == 1


def test_standing_box_finds_the_spot_an_agent_stands_on(navs):
    for nav in navs[:5]:
        world = World.single(nav.map, CFG)
        for spot in map(tuple, nav.main_spots()[::50]):
            world.x[0, 0], world.y[0, 0] = nav.standing_centre(spot)
            assert on_ground(world)[0, 0]
            assert nav.standing_box(world.x[0, 0], world.y[0, 0]) is not None


def test_routes_arrive(results):
    """Step 2.4's contract: the engine can fly the routes the generator promised."""
    arrived = sum(r.status == "arrived" for r in results)
    failed = [(r.seed, r.start, r.goal, r.status) for r in results if r.status != "arrived"]
    assert arrived / len(results) >= ARRIVE_RATE, failed


def test_route_results_are_deterministic(routes, results):
    again = run_routes(routes[:20], CFG, full_tank=True)
    for a, b in zip(results[:20], again):
        assert (a.status, a.seconds, a.hops) == (b.status, b.seconds, b.hops)
        np.testing.assert_array_equal(a.path, b.path)
