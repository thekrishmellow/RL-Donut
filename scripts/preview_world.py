"""Draw engine worlds: tiles, spawns and each agent's hitbox at its current position.

    .venv/bin/python scripts/preview_world.py --split val --count 4 --team-size 2
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rl_donut.env import EngineConfig, World
from rl_donut.mapgen import load_split

TEAM_COLOURS = {0: "#17becf", 1: "#e377c2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--out", default="outputs/world_preview.png")
    args = parser.parse_args()

    cfg = EngineConfig(team_size=args.team_size)
    world = World.from_maps(load_split(args.split)[args.start: args.start + args.count], cfg)
    fig, axes = plt.subplots(world.n_matches, 1, figsize=(12, 6.2 * world.n_matches), squeeze=False)
    for m, ax in enumerate(axes[:, 0]):
        ax.imshow(world.grids[m], cmap="Greys", vmin=0, vmax=1.6, extent=(0, cfg.map.width, cfg.map.height, 0))
        for team, colour in TEAM_COLOURS.items():
            sx, sy = world.spawns[m, team, :, 0], world.spawns[m, team, :, 1]
            ax.scatter(sx, sy, s=30, facecolors="none", edgecolors=colour, linewidths=1)
        for a in range(cfg.n_agents):
            x, y = world.x[m, a], world.y[m, a]
            ax.add_patch(Rectangle((x - cfg.hitbox_w / 2, y - cfg.hitbox_h / 2), cfg.hitbox_w, cfg.hitbox_h,
                                   color=TEAM_COLOURS[int(world.team[a])]))
        ax.axvline(cfg.map.width / 2, color="#999", lw=0.5, ls="--")
        ax.set_title(f"seed {world.seeds[m]}, tick {world.tick[m]}: filled boxes = agents, "
                     "circles = spawn points (blue = team A, pink = team B)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print(args.out)


if __name__ == "__main__":
    main()
