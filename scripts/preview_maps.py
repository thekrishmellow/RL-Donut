"""Render a sheet of generated maps for a range of seeds to outputs/maps_preview.png."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from rl_donut.mapgen import MapConfig, generate_map

# Tile classes drawn in the preview.
AIR, ROCK, DUG, FILLED, PLATFORM, UNREACHABLE, PIT = 0, 1, 2, 3, 4, 5, 6
COLOURS = ["white", "black", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
TEAM_A, TEAM_B = "#17becf", "#e377c2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="first seed")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--plain", action="store_true", help="draw only rock and air")
    parser.add_argument("--out", type=Path, default=Path("outputs/maps_preview.png"))
    args = parser.parse_args()

    cfg = MapConfig()
    rows = -(-args.count // args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(4.6 * args.cols, 2.8 * rows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, seed in zip(axes.flat, range(args.start, args.start + args.count)):
        game_map = generate_map(seed, cfg)
        image = game_map.grid.astype(int)
        if not args.plain:
            image[game_map.layers["dug"]] = DUG
            image[game_map.layers["filled"]] = FILLED
            image[game_map.layers["platforms"]] = PLATFORM
            image[game_map.layers["unreachable"]] = UNREACHABLE
            image[game_map.layers["pits"]] = PIT
        ax.imshow(image, cmap=ListedColormap(COLOURS), vmin=0, vmax=len(COLOURS) - 1, interpolation="nearest")
        if not args.plain:
            w, h = cfg.regions.clearance_w, cfg.regions.clearance_h
            for team, colour in ((game_map.spawns.team_a, TEAM_A), (game_map.spawns.team_b, TEAM_B)):
                for s in team:
                    ax.add_patch(Rectangle((s.x - 0.5, s.y - 0.5), w, h, facecolor=colour, edgecolor="white",
                                           linewidth=0.5))
        if game_map.fully_reachable:
            status = "reachable"
        else:
            status = f"{game_map.stranded_fraction:.0%} stranded" + ("" if game_map.trap_free else ", TRAP")
        pits = f"{len(game_map.pits)}/{game_map.pits_requested} pits"
        spawns = f"{len(game_map.spawns.team_a)}/{cfg.spawns.per_team} spawns"
        ax.set_title(f"seed {seed}: {game_map.tunnels} tunnels, {pits}, {len(game_map.platforms)} platforms, "
                     f"{spawns}, {status}", fontsize=8)
    if not args.plain:
        fig.legend(handles=[Patch(color=COLOURS[DUG], label="tunnel dug"),
                            Patch(color=COLOURS[FILLED], label="dead air filled"),
                            Patch(color=COLOURS[PLATFORM], label="platform"),
                            Patch(color=COLOURS[UNREACHABLE], label="unreachable air filled"),
                            Patch(color=COLOURS[PIT], label="pit shaft"),
                            Patch(color=TEAM_A, label="team A spawn"),
                            Patch(color=TEAM_B, label="team B spawn")],
                   loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Generated maps (black = rock)", fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
