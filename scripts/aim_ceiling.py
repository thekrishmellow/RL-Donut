"""Best possible hit rate of each aim design, for a perfect aimer firing the real rifle.

Pairs of positions where an agent fits are drawn on val maps; pairs in line of sight within
range give the real distribution of engagement distances and directions. Each aim design
then fires once per pair (in an empty room, so only aim and spread decide), picking the
allowed angle closest to the true direction.

    .venv/bin/python scripts/aim_ceiling.py --maps 100
"""

import argparse
import math

import numpy as np

from rl_donut.env import EngineConfig, Navigator, World, fire
from rl_donut.mapgen import line_of_sight, load_split


def engagement_pairs(maps, cfg, per_map, rng):
    pairs = []
    for m in maps:
        nav = Navigator(m, cfg)
        spots = np.argwhere(nav.fits)
        for _ in range(per_map):
            a, b = (nav.flying_centre(tuple(spots[rng.integers(len(spots))])) for _ in range(2))
            d = math.dist(a, b)
            if 0 < d <= cfg.rifle.range and line_of_sight(m.grid, a, b):
                pairs.append((d, math.atan2(b[1] - a[1], b[0] - a[0])))
    return pairs


def hit_rate(world, pairs, aim_of):
    hits = 0
    for d, theta in pairs:
        world.x[0] = [48.0, 48.0 + d * math.cos(theta)]
        world.y[0] = [24.0, 24.0 + d * math.sin(theta)]
        world.cooldown[0] = world.reload[0] = 0
        world.ammo[0] = world.cfg.rifle.magazine
        hits += fire(world, np.array([[aim_of(theta), 0.0]]), np.array([[True, False]]))[0].target == 1
    return hits / len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--per-map", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = EngineConfig()
    maps = load_split("val")[:args.maps]
    pairs = engagement_pairs(maps, cfg, args.per_map, np.random.default_rng(args.seed))
    d = np.array([p[0] for p in pairs])
    print(f"{len(pairs)} visible pairs within {cfg.rifle.range:.0f} tiles on {len(maps)} val maps; distance "
          + ", ".join(f"p{p} {np.percentile(d, p):.1f}" for p in (10, 25, 50, 75, 90)))
    world = World.single(maps[0], cfg)
    world.grids[0] = False
    print("| Aim design | Best possible hit rate |\n|---|---|")
    for k in (16, 32, 64, 128):
        step = 2 * math.pi / k
        rate = hit_rate(world, pairs, lambda th: round(th / step) * step)
        print(f"| {k} fixed directions ({360 / k:.1f}° apart) | {100 * rate:.0f}% |")
    print(f"| Continuous angle | {100 * hit_rate(world, pairs, lambda th: th):.0f}% |")


if __name__ == "__main__":
    main()
