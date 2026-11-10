"""Step 2.4's route check: can the engine fly the routes the map generator promised?

Flies random routes between main floor segments on validation maps, with the route bot
waiting for a full tank before every hop, and draws every failure.

    .venv/bin/python scripts/route_check.py                 # 200 maps x 10 routes
    .venv/bin/python scripts/route_check.py --maps 30 --per-map 5 --just-enough
"""

import argparse
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_donut.env import EngineConfig, Navigator, RouteBot, run_routes, sample_routes
from rl_donut.mapgen import load_split

MAX_PICTURES = 8


def draw_failures(failed, cfg, path):
    plans = []
    original = RouteBot.plan

    def recording_plan(self, segment, box):
        original(self, segment, box)
        plans.append(list(self.boxes))

    RouteBot.plan = recording_plan
    try:
        fig, axes = plt.subplots(len(failed), 1, figsize=(12, 6.3 * len(failed)), squeeze=False)
        for ax, (nav, start, goal, full_tank) in zip(axes[:, 0], failed):
            plans.clear()
            r = run_routes([(nav, start, goal)], cfg, full_tank=full_tank)[0]
            ax.imshow(nav.map.grid, cmap="Greys", vmin=0, vmax=1.6, extent=(0, nav.map.grid.shape[1], nav.map.grid.shape[0], 0))
            goal_spots = np.argwhere(nav.segments == goal)
            ax.plot(*np.transpose([nav.standing_centre(tuple(s)) for s in goal_spots]), "g.", ms=4)
            for boxes in plans:
                ax.plot(*np.transpose([nav.flying_centre(b) for b in boxes]), "r-", lw=0.6, alpha=0.6)
            if len(r.path):
                ax.plot(r.path[:, 0], r.path[:, 1], "b-", lw=0.9)
                ax.plot(*r.path[-1], "bo")
            ax.set_ylim(nav.map.grid.shape[0] + 5, 0)
            ax.set_title(f"seed {r.seed}: {r.status} after {r.hops} hops, {r.seconds:.1f} s "
                         "(red = planned, blue = flown, green = goal floor)")
        fig.tight_layout()
        fig.savefig(path, dpi=65)
    finally:
        RouteBot.plan = original


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=int, default=200)
    parser.add_argument("--per-map", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--just-enough", action="store_true",
                        help="also fly with the bot taking off on just enough fuel for each hop")
    parser.add_argument("--out", default="outputs/route_failures.png")
    args = parser.parse_args()

    cfg = EngineConfig()
    t = time.time()
    navs = [Navigator(m, cfg) for m in load_split("val")[:args.maps]]
    routes = sample_routes(navs, args.per_map, seed=args.seed)
    print(f"{len(routes)} routes on {len(navs)} val maps (setup {time.time() - t:.0f} s)")

    failed = []
    for full_tank in [True] + ([False] if args.just_enough else []):
        t = time.time()
        results = run_routes(routes, cfg, full_tank=full_tank)
        counts = Counter(r.status for r in results)
        rule = "full tank" if full_tank else "just enough"
        print(f"{rule:>11}: {100 * counts['arrived'] / len(results):.2f}% arrived  {dict(counts)}  "
              f"({time.time() - t:.0f} s)")
        for (nav, start, goal), r in zip(routes, results):
            if r.status != "arrived":
                print(f"             seed {r.seed} start {r.start} goal segment {r.goal}: {r.status} "
                      f"after {r.hops} hops")
                failed.append((nav, start, goal, full_tank))

    if failed:
        draw_failures(failed[:MAX_PICTURES], cfg, args.out)
        print(f"failure pictures -> {args.out}")


if __name__ == "__main__":
    main()
