"""Render a sheet of generated maps for a range of seeds to outputs/maps_preview.png."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from rl_donut.mapgen import MapConfig, generate_map

# Tile classes drawn in the preview.
AIR, ROCK, DUG, FILLED, PLATFORM, UNREACHABLE = 0, 1, 2, 3, 4, 5
COLOURS = ["white", "black", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


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
        ax.imshow(image, cmap=ListedColormap(COLOURS), vmin=0, vmax=len(COLOURS) - 1, interpolation="nearest")
        if game_map.fully_reachable:
            status = "reachable"
        else:
            status = f"{game_map.stranded_fraction:.0%} stranded" + ("" if game_map.trap_free else ", TRAP")
        ax.set_title(f"seed {seed}: rock {game_map.rock_fraction:.0%}, {game_map.tunnels} tunnels, "
                     f"{len(game_map.platforms)} platforms, {status}", fontsize=9)
    if not args.plain:
        fig.legend(handles=[Patch(color=COLOURS[DUG], label="tunnel dug"),
                            Patch(color=COLOURS[FILLED], label="dead air filled"),
                            Patch(color=COLOURS[PLATFORM], label="platform"),
                            Patch(color=COLOURS[UNREACHABLE], label="unreachable air filled")],
                   loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Generated maps (black = rock)", fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
