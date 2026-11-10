"""One observation, drawn: the map with each agent's tile window, the windows as the agents
receive them (team B's is mirrored), and their main numbers. Team A stands near a pit with
the enemy in sight.

    .venv/bin/python scripts/preview_observation.py
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from rl_donut.env import EngineConfig, Navigator, World
from rl_donut.env.observe import CROP_H, CROP_W, FEATURE_NAMES, Observer
from rl_donut.env.visibility import visibility
from rl_donut.mapgen import load_split

SHOWN = ("map_x", "map_y", "tile_x", "health", "aim_cos", "aim_sin", "enemy0_seen_by_me",
         "enemy0_remembered", "enemy0_dx", "enemy0_dy", "enemy0_health", "enemy0_aim_cos", "time_left")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/observation_preview.png")
    args = parser.parse_args()

    cfg = EngineConfig()
    rng = np.random.default_rng(args.seed)
    seen = FEATURE_NAMES.index("enemy0_seen_by_me")
    for m in load_split("val"):
        if not m.pits:
            continue
        world = World.single(m, cfg)
        nav = Navigator(m, cfg)
        spots = np.array([nav.standing_centre(tuple(s)) for s in nav.main_spots()])
        pit_x = np.array([p.x0 + p.width / 2 for p in m.pits])
        near_pit = spots[(np.abs(spots[:, 0, None] - pit_x).min(axis=1) < 6) & (spots[:, 1] > 38)]
        if not len(near_pit):
            continue
        world.x[0, 0], world.y[0, 0] = near_pit[rng.integers(len(near_pit))]
        for x, y in spots[rng.permutation(len(spots))]:
            world.x[0, 1], world.y[0, 1] = x, y
            world.aim[0] = [math.atan2(y - world.y[0, 0], x - world.x[0, 0]),
                            math.atan2(world.y[0, 0] - y, world.x[0, 0] - x)]
            observer = Observer(world)
            obs = observer.observe(visibility(world))
            if obs[0, 0, seen] and math.dist((x, y), (world.x[0, 0], world.y[0, 0])) > 8:
                break
        else:
            continue
        break
    world.health[0, 1] = 70.0
    obs = observer.observe(visibility(world))

    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(m.grid, cmap="Greys", vmin=0, vmax=1.6, extent=(0, m.grid.shape[1], m.grid.shape[0], 0))
    colours = ("tab:blue", "tab:red")
    for i, colour in enumerate(colours):
        x, y = world.x[0, i], world.y[0, i]
        ax.add_patch(Rectangle((x - cfg.hitbox_w / 2, y - cfg.hitbox_h / 2), cfg.hitbox_w, cfg.hitbox_h, color=colour))
        ax.add_patch(Rectangle((math.floor(x) - CROP_W // 2, math.floor(y) - CROP_H // 2), CROP_W, CROP_H,
                               fill=False, ec=colour, lw=1.5, ls="--"))
    ax.plot(world.x[0], world.y[0], ":", color="tab:green")
    ax.set_xlim(0, m.grid.shape[1])
    ax.set_ylim(m.grid.shape[0], 0)
    ax.set_title(f"seed {m.seed}: agents, their {CROP_W} x {CROP_H} tile windows, and the line of sight between them")

    crops = observer.crop()[0]
    tile_colours = ListedColormap(["tab:red", "white", "0.45"])          # -1 below the map, 0 air, 1 rock
    for i, (colour, team) in enumerate(zip(colours, ("A", "B (mirrored)"))):
        cax = fig.add_subplot(grid[1, i])
        cax.imshow(crops[i], cmap=tile_colours, vmin=-1.5, vmax=1.5)
        cax.add_patch(Rectangle((CROP_W // 2 - 0.5, CROP_H // 2 - 0.5), 1, 1, fill=False, ec=colour, lw=2))
        cax.set_title(f"team {team}: tile window as received\n(grey rock, white air, red below the map)", fontsize=9)
        cax.set_xticks([]), cax.set_yticks([])
    tax = fig.add_subplot(grid[1, 2])
    tax.axis("off")
    rows = [f"{name:<20}{obs[0, 0, FEATURE_NAMES.index(name)]:>7.2f}{obs[0, 1, FEATURE_NAMES.index(name)]:>7.2f}"
            for name in SHOWN]
    tax.text(0, 1, f"{'feature':<20}{'A':>7}{'B':>7}\n" + "\n".join(rows), family="monospace", fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(args.out, dpi=70)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
