"""Rifle shots on real maps: a fan of shots from one agent, ending at rock, range or an enemy.

    .venv/bin/python scripts/preview_rifle.py --count 3
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from rl_donut.env import EngineConfig, Navigator, RifleConfig, World, fire
from rl_donut.mapgen import line_of_sight, load_split

RAYS = 180


def rock_distance(grid, x, y, angle, limit):
    """Distance to the first rock along a ray, to 0.01 tiles (by bisection; pictures only)."""
    end = lambda t: (x + t * math.cos(angle), y + t * math.sin(angle))
    if line_of_sight(grid, (x, y), end(limit)):
        return limit
    lo, hi = 0.0, limit
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if line_of_sight(grid, (x, y), end(mid)) else (lo, mid)
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/rifle_preview.png")
    args = parser.parse_args()

    cfg = EngineConfig(team_size=2, rifle=RifleConfig(spread_deg=0.0))
    rng = np.random.default_rng(args.seed)
    maps = load_split("val")[:args.count]
    fig, axes = plt.subplots(len(maps), 1, figsize=(12, 6.4 * len(maps)), squeeze=False)
    for ax, m in zip(axes[:, 0], maps):
        world = World.single(m, cfg)
        nav = Navigator(m, cfg)
        spots = nav.main_spots()
        shooter = spots[rng.integers(len(spots))]
        world.x[0, 0], world.y[0, 0] = nav.standing_centre(tuple(shooter))
        # Enemies on standing spots within range of the shooter; the teammate stays out of it.
        near = spots[np.hypot(*(np.array([nav.standing_centre(tuple(s)) for s in spots]) -
                                (world.x[0, 0], world.y[0, 0])).T) < cfg.rifle.range * 0.8]
        for j, s in zip((2, 3), near[rng.choice(len(near), 2, replace=False)]):
            world.x[0, j], world.y[0, j] = nav.standing_centre(tuple(s))
        world.alive[0, 1] = False

        ax.imshow(m.grid, cmap="Greys", vmin=0, vmax=1.6, extent=(0, m.grid.shape[1], m.grid.shape[0], 0))
        x, y = world.x[0, 0], world.y[0, 0]
        hit_count = 0
        for angle in np.linspace(-math.pi, math.pi, RAYS, endpoint=False):
            aim = world.aim.copy()
            aim[0, 0] = angle
            world.cooldown[0, 0] = world.reload[0, 0] = 0
            world.ammo[0, 0] = cfg.rifle.magazine
            shot = fire(world, aim, np.array([[True, False, False, False]]))[0]
            if shot.target >= 0:
                hit_count += 1
                ax.plot([x, shot.hit[0]], [y, shot.hit[1]], "r-", lw=1.0)
            else:
                t = rock_distance(m.grid, x, y, angle, cfg.rifle.range)
                ax.plot([x, x + t * math.cos(angle)], [y, y + t * math.sin(angle)], "-", color="tab:blue", lw=0.4, alpha=0.6)
        ax.add_patch(Circle((x, y), cfg.rifle.range, fill=False, ls="--", color="tab:blue", lw=0.8))
        for j, colour in ((0, "tab:blue"), (2, "tab:red"), (3, "tab:red")):
            ax.add_patch(Rectangle((world.x[0, j] - cfg.hitbox_w / 2, world.y[0, j] - cfg.hitbox_h / 2),
                                   cfg.hitbox_w, cfg.hitbox_h, color=colour))
        ax.set_xlim(0, m.grid.shape[1])
        ax.set_ylim(m.grid.shape[0], 0)
        ax.set_title(f"seed {m.seed}: {RAYS} shots from the blue agent; blue = stopped by rock or range, "
                     f"red = hit an enemy ({hit_count})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=65)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
