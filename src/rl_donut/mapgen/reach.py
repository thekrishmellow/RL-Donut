"""Jetpack reachability between standing spots.

Flight model (conservative, checked against real physics in Phase 2):
- moving up one tile costs 2 fuel units, sideways 1, down 0 (gravity);
- a full tank holds 2 * climb_budget units;
- touching any standing spot refuels, so a flight ends there.

Standing spots are clearance-box positions resting on rock. Spots on the same row that touch
are one *floor segment*: an agent can walk between them for free. The map is fully reachable
when the directed segment graph is strongly connected: every segment can reach every other
and get back. It is trap-free when every segment reachable from the main group can return.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.mapgen.regions import box_rock_counts, covered_by_boxes

UP_COST = 2
SIDE_COST = 1
UNREACHED = np.int16(np.iinfo(np.int16).max // 2)


@dataclass(frozen=True)
class Reachability:
    segments: np.ndarray       # [row, col] segment id per standing spot, -1 elsewhere
    sizes: np.ndarray          # standing spots per segment
    reach: np.ndarray          # reach[a, b]: one flight from segment a can land on segment b
    closure: np.ndarray        # closure[a, b]: a can reach b, possibly via other segments
    mutual: np.ndarray         # mutual[a, b]: a and b can reach each other, possibly via others
    main: np.ndarray           # segments in the largest mutually-reachable group (by spots)

    @property
    def fully_reachable(self) -> bool:
        return bool(self.main.all())

    @property
    def stranded_fraction(self) -> float:
        total = self.sizes.sum()
        return float(self.sizes[~self.main].sum() / total) if total else 0.0

    @property
    def trap_free(self) -> bool:
        """Every segment reachable from the main group can get back to it."""
        if not self.main.any():
            return True
        reachable = self.closure[self.main].any(axis=0)
        return not (reachable & ~self.main).any()

    def acceptable(self, max_stranded: float) -> bool:
        return self.trap_free and self.stranded_fraction <= max_stranded


def standing_spots(grid: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(fits, supported): usable box positions, and those whose box rests on rock."""
    height = grid.shape[0]
    fits = box_rock_counts(grid, w, h) == 0
    under = box_rock_counts(grid, w, 1)
    on_rock = np.zeros(grid.shape, dtype=bool)
    on_rock[: height - h] = under[h:] > 0
    return fits, fits & on_rock


def floor_segments(supported: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label horizontal runs of standing spots; returns (segment ids, sizes)."""
    segments = np.full(supported.shape, -1, dtype=np.int32)
    sizes = []
    for y in range(supported.shape[0]):
        row = supported[y]
        x = 0
        while x < row.size:
            if row[x]:
                start = x
                while x + 1 < row.size and row[x + 1]:
                    x += 1
                segments[y, start: x + 1] = len(sizes)
                sizes.append(x - start + 1)
            x += 1
    return segments, np.array(sizes, dtype=np.int64)


def flight_costs(fits: np.ndarray, supported: np.ndarray, segments: np.ndarray,
                 n_segments: int, tank: int) -> np.ndarray:
    """Minimum fuel to reach every position from each segment, shape (segments, H, W).

    Batched over all segments. Each sweep applies free falling to closure, then one costed
    move (up or sideways); a path has at most `tank` costed moves, so sweeps are bounded.
    """
    height, width = fits.shape
    own = segments[None, :, :] == np.arange(n_segments)[:, None, None]
    # A flight continues through a position unless it lands on another segment there.
    propagates = fits[None] & (~supported[None] | own)
    cost = np.where(own, 0, UNREACHED).astype(np.int16)
    for _ in range(tank + 2):
        before = cost.copy()
        for y in range(1, height):
            fallen = np.where(propagates[:, y - 1], cost[:, y - 1], UNREACHED)
            cost[:, y] = np.where(fits[y], np.minimum(cost[:, y], fallen), cost[:, y])
        src = np.where(propagates, cost, UNREACHED)
        moved = np.full_like(cost, UNREACHED)
        moved[:, :-1] = src[:, 1:] + UP_COST
        moved[:, :, 1:] = np.minimum(moved[:, :, 1:], src[:, :, :-1] + SIDE_COST)
        moved[:, :, :-1] = np.minimum(moved[:, :, :-1], src[:, :, 1:] + SIDE_COST)
        moved = np.where(fits[None] & (moved <= tank), moved, UNREACHED)
        cost = np.minimum(cost, moved)
        if np.array_equal(cost, before):
            break
    return cost


def transitive_closure(reach: np.ndarray) -> np.ndarray:
    closure = reach | np.eye(len(reach), dtype=bool)
    while True:
        step = closure | ((closure.astype(np.int32) @ closure.astype(np.int32)) > 0)
        if np.array_equal(step, closure):
            return closure
        closure = step


def occupiable_tiles(grid: np.ndarray, reach: Reachability, w: int, h: int, climb_budget: int) -> np.ndarray:
    """Tiles some agent's box can cover on a flight that starts from the main group."""
    fits, supported = standing_spots(grid, w, h)
    if not reach.main.any():
        return np.zeros(grid.shape, dtype=bool)
    tank = 2 * climb_budget
    main_ids = np.nonzero(reach.main)[0]
    # Relabel main segments 0..k-1 so flights are only computed from the main group; flights
    # still land on (and stop at) every other standing spot because `supported` is unchanged.
    relabel = np.full(len(reach.sizes), -1, dtype=np.int32)
    relabel[main_ids] = np.arange(len(main_ids))
    relabelled = np.where(reach.segments >= 0, relabel[np.maximum(reach.segments, 0)], -1)
    cost = flight_costs(fits, supported, relabelled, len(main_ids), tank)
    return covered_by_boxes((cost <= tank).any(axis=0), w, h)


def analyse_reachability(grid: np.ndarray, w: int, h: int, climb_budget: int) -> Reachability:
    fits, supported = standing_spots(grid, w, h)
    segments, sizes = floor_segments(supported)
    n = len(sizes)
    if n == 0:
        empty = np.zeros((0, 0), dtype=bool)
        return Reachability(segments, sizes, empty, empty, empty, np.zeros(0, dtype=bool))

    cost = flight_costs(fits, supported, segments, n, tank=2 * climb_budget)
    landed = (cost <= 2 * climb_budget) & supported[None]
    reach = np.zeros((n, n), dtype=bool)
    for a in range(n):
        reach[a, np.unique(segments[landed[a]])] = True

    closure = transitive_closure(reach)
    mutual = closure & closure.T
    weight = mutual.astype(np.int64) @ sizes
    main = mutual[int(np.argmax(weight))]
    return Reachability(segments, sizes, reach, closure, mutual, main)
