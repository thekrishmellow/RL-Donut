"""Route planning on the generator's flight model, and a bot that flies the plans.

The planner rebuilds exactly the flights the map generator believed possible (mapgen.reach):
from a floor segment, the cheapest-fuel chain of clearance-box positions to a standing spot
on another segment. The bot follows that chain under real engine physics, so the route
check (step 2.4) tests the generator's claim itself.

A route is flown hop by hop: plan one flight from the floor the bot stands on, fly it, land,
and plan again from wherever it actually landed.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from rl_donut.env.collision import on_ground
from rl_donut.env.config import EngineConfig
from rl_donut.env.physics import physics_tick
from rl_donut.env.state import World
from rl_donut.mapgen import GeneratedMap
from rl_donut.mapgen.reach import SIDE_COST, UP_COST, analyse_reachability, flight_costs, standing_spots

# Extra fuel the bot waits for before take-off, as a fraction of the tank.
FUEL_RESERVE = 0.1


class Navigator:
    """The generator's flight model for one map."""

    def __init__(self, m: GeneratedMap, cfg: EngineConfig):
        self.cfg, self.map = cfg, m
        regions = cfg.map.regions
        self.box_w, self.box_h = regions.clearance_w, regions.clearance_h
        self.tank = 2 * cfg.map.climb_budget
        self.fits, self.supported = standing_spots(m.grid, self.box_w, self.box_h)
        self.reach = analyse_reachability(m.grid, self.box_w, self.box_h, cfg.map.climb_budget)
        self.segments = self.reach.segments
        self.costs = flight_costs(self.fits, self.supported, self.segments, len(self.reach.sizes), self.tank)

    # --- coordinates ---

    def standing_centre(self, box: tuple[int, int]) -> tuple[float, float]:
        r, c = box
        return c + self.box_w / 2, r + self.box_h - self.cfg.hitbox_h / 2

    def flying_centre(self, box: tuple[int, int]) -> tuple[float, float]:
        r, c = box
        return c + self.box_w / 2, r + self.box_h / 2

    def standing_box(self, x: float, y: float) -> tuple[int, int] | None:
        """The standing spot under an agent whose feet rest on a floor, if any."""
        row = int(round(y + self.cfg.hitbox_h / 2)) - self.box_h
        if not 0 <= row < self.segments.shape[0]:
            return None
        half = self.cfg.hitbox_w / 2
        inside = range(int(np.ceil(x + half - self.box_w - 1e-9)), int(np.floor(x - half + 1e-9)) + 1)
        overlapping = range(int(np.floor(x - half)) - self.box_w + 1, int(np.floor(x + half)) + 1)
        for col in [*inside, *overlapping]:
            if 0 <= col < self.segments.shape[1] and self.segments[row, col] >= 0:
                return row, col
        return None

    # --- planning ---

    def main_spots(self) -> np.ndarray:
        """(row, col) of every standing spot in the main reachable group."""
        main_ids = np.nonzero(self.reach.main)[0]
        return np.argwhere(np.isin(self.segments, main_ids))

    def next_segment(self, start: int, goal: int) -> int | None:
        """First segment after `start` on a fewest-hops route to `goal`."""
        parent = {start: None}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            if a == goal:
                while parent[a] != start:
                    a = parent[a]
                return a
            for b in np.nonzero(self.reach.reach[a])[0]:
                if b not in parent:
                    parent[int(b)] = a
                    queue.append(int(b))
        return None

    def hop(self, start: int, target: int, from_box: tuple[int, int]) -> list[tuple[int, int]]:
        """Box positions from `from_box` (on segment `start`) to the cheapest landing on `target`."""
        cost = self.costs[start]
        candidates = np.argwhere((self.segments == target) & (cost <= self.tank))
        if not len(candidates):
            raise ValueError(f"segment {target} is not one flight from segment {start}")
        best = cost[candidates[:, 0], candidates[:, 1]]
        ties = candidates[best == best.min()]
        r, c = min(map(tuple, ties), key=lambda p: abs(p[1] - from_box[1]) + abs(p[0] - from_box[0]))

        # Walking backwards from the landing, prefer falls: among equally cheap flights this
        # picks the one that drops onto the landing floor, which needs no braking. A fall that
        # ends in mid-air costs real fuel to stop, which the generator's model doesn't charge.
        own = self.segments == start
        flows = self.fits & (~self.supported | own)
        path = [(int(r), int(c))]
        while not (own[r, c] and cost[r, c] == 0):
            v = cost[r, c]
            for (pr, pc), step in (((r - 1, c), 0), ((r, c - 1), SIDE_COST), ((r, c + 1), SIDE_COST),
                                   ((r + 1, c), UP_COST)):
                if 0 <= pr < cost.shape[0] and 0 <= pc < cost.shape[1] and flows[pr, pc] \
                        and cost[pr, pc] + step == v:
                    r, c = pr, pc
                    break
            else:
                raise RuntimeError(f"no predecessor for {(r, c)} at cost {v}; flight model mismatch")
            path.append((int(r), int(c)))
        path.reverse()

        # Walk along the start floor to where the flight leaves it.
        walk_row, walk_to = path[0]
        step = 1 if walk_to >= from_box[1] else -1
        walk = [(walk_row, col) for col in range(from_box[1], walk_to, step)]
        return walk + path

    def hop_cost(self, start: int, box: tuple[int, int]) -> int:
        return int(self.costs[start][box])


@dataclass
class RouteBot:
    """Flies one route: hop by hop, following each planned chain of boxes."""
    nav: Navigator
    goal: int
    # Wait for a full tank before every take-off instead of just enough for the planned hop.
    full_tank: bool = False
    status: str = "flying"
    hops: int = 0
    segment: int = -1
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[int, int]] = field(default_factory=list)
    progress: int = 0
    fuel_needed: float = 0.0

    def plan(self, segment: int, box: tuple[int, int]) -> None:
        nxt = self.nav.next_segment(segment, self.goal)
        if nxt is None:
            self.status = "stranded"
            return
        boxes = self.nav.hop(segment, nxt, box)
        # Walk on the start floor; in the air, aim at the middle of each box (aiming at the
        # landing height itself would leave the feet a hair below a step's top edge).
        own = self.nav.segments == segment
        self.waypoints = [self.nav.standing_centre(b) if own[b] else self.nav.flying_centre(b) for b in boxes]
        self.boxes, self.progress, self.segment = boxes, 0, segment
        p = self.nav.cfg.physics
        self.fuel_needed = p.fuel_capacity if self.full_tank else min(
            p.fuel_capacity,
            self.nav.hop_cost(segment, boxes[-1]) / self.nav.tank * p.fuel_capacity + FUEL_RESERVE * p.fuel_capacity)
        self.hops += 1

    def run_end(self, k: int) -> tuple[int, bool]:
        """Last box of the straight run starting at box `k`, and whether that run goes down."""
        last = len(self.boxes) - 1
        if k >= last:
            return last, False
        step = np.subtract(self.boxes[k + 1], self.boxes[k])
        j = k + 1
        while j < last and (np.subtract(self.boxes[j + 1], self.boxes[j]) == step).all():
            j += 1
        return j, bool(step[0] > 0)

    def inside(self, k: int, x: float, y: float) -> bool:
        r, c = self.boxes[k]
        hw, hh, tol = self.nav.cfg.hitbox_w / 2, self.nav.cfg.hitbox_h / 2, 0.05
        return (x - hw >= c - tol and x + hw <= c + self.nav.box_w + tol
                and y - hh >= r - tol and y + hh <= r + self.nav.box_h + tol)

    def decide(self, x: float, y: float, vy: float, fuel: float, grounded: bool) -> tuple[int, bool]:
        if grounded:
            box = self.nav.standing_box(x, y)
            segment = -1 if box is None else int(self.nav.segments[box])
            if segment == self.goal:
                self.status = "arrived"
                return 0, False
            if segment >= 0 and segment != self.segment:
                self.plan(segment, box)
                if self.status != "flying":
                    return 0, False
        if not self.waypoints:
            self.status = "lost"
            return 0, False
        window = range(self.progress, min(self.progress + 10, len(self.boxes)))
        reached = [k for k in window if self.inside(k, x, y)]
        if reached:
            self.progress = max(reached)
        end, falling = self.run_end(self.progress)
        tx, ty = self.waypoints[end]
        move = 0 if abs(tx - x) < 0.1 else (1 if tx > x else -1)
        # Thrust when below the target, or when the current vertical speed would carry the
        # agent past it: rising is stopped by gravity, falling only by thrust minus gravity.
        # Aiming at the end of the straight run lets the agent free-fall and brake late.
        p = self.nav.cfg.physics
        decel = p.thrust - p.gravity if vy > 0 else p.gravity
        jet = (y - ty) + vy * abs(vy) / (2 * decel) > 0.05
        last = len(self.boxes) - 1
        if self.progress == last or (falling and end == last):
            jet = False  # dropping onto the landing floor needs no braking
        if grounded and fuel < self.fuel_needed:
            return 0, False  # walking on could step off a ledge before the tank refills
        return move, jet


@dataclass(frozen=True)
class RouteResult:
    seed: int
    start: tuple[int, int]
    goal: int
    status: str            # arrived, fell, stranded, lost, timeout
    seconds: float
    hops: int
    path: np.ndarray       # (decisions, 2) hitbox centres, for pictures


def sample_routes(navs: list[Navigator], per_map: int, seed: int = 0):
    """Random (navigator, start spot, goal segment) triples, start and goal on different main segments."""
    routes = []
    for nav in navs:
        rng = np.random.default_rng([seed, nav.map.seed])
        spots = nav.main_spots()
        if len(np.unique(nav.segments[spots[:, 0], spots[:, 1]])) < 2:
            continue
        for _ in range(per_map):
            while True:
                a, b = spots[rng.integers(len(spots), size=2)]
                if nav.segments[tuple(a)] != nav.segments[tuple(b)]:
                    break
            routes.append((nav, (int(a[0]), int(a[1])), int(nav.segments[tuple(b)])))
    return routes


def run_routes(routes, cfg: EngineConfig | None = None, time_limit: float = 60.0,
               decision_ticks: int = 2, full_tank: bool = False) -> list[RouteResult]:
    """Fly every route at once (one match per route, team B's agent removed)."""
    cfg = cfg or EngineConfig()
    world = World.from_maps([nav.map for nav, _, _ in routes], cfg)
    world.alive[:, 1:] = False
    bots = []
    for i, (nav, start, goal) in enumerate(routes):
        world.x[i, 0], world.y[i, 0] = nav.standing_centre(start)
        bots.append(RouteBot(nav, goal, full_tank=full_tank))
    n = len(routes)
    move, jet = np.zeros((n, cfg.n_agents), dtype=np.int64), np.zeros((n, cfg.n_agents), dtype=bool)
    paths = [[] for _ in range(n)]
    finished_at = np.full(n, time_limit)
    ticks = int(time_limit * cfg.tick_rate)
    for tick in range(ticks):
        if tick % decision_ticks == 0:
            grounded = on_ground(world)[:, 0]
            for i, bot in enumerate(bots):
                if bot.status != "flying":
                    continue
                x, y = world.x[i, 0], world.y[i, 0]
                paths[i].append((x, y))
                move[i, 0], jet[i, 0] = bot.decide(x, y, world.vy[i, 0], world.fuel[i, 0], bool(grounded[i]))
                if bot.status != "flying":
                    finished_at[i] = tick / cfg.tick_rate
                    world.alive[i, 0] = False
        physics_tick(world, move, jet)
        fell = world.alive[:, 0] & (world.y[:, 0] > world.kill_y)
        for i in np.nonzero(fell)[0]:
            bots[i].status = "fell"
            finished_at[i] = tick / cfg.tick_rate
            world.alive[i, 0] = False
        if not world.alive[:, 0].any():
            break
    return [RouteResult(nav.map.seed, start, goal, "timeout" if bot.status == "flying" else bot.status,
                        float(finished_at[i]), bot.hops, np.array(paths[i]))
            for i, ((nav, start, goal), bot) in enumerate(zip(routes, bots))]
