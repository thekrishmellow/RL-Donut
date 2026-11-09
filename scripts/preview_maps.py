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
AIR, ROCK, DUG, FILLED = 0, 1, 2, 3
COLOURS = ["white", "black", "#1f77b4", "#ff7f0e"]


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
        ax.imshow(image, cmap=ListedColormap(COLOURS), vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"seed {seed}: rock {game_map.rock_fraction:.0%}, bias {game_map.params.height_bias:.2f}, "
                     f"{game_map.tunnels} tunnels, {game_map.crossings} crossings", fontsize=9)
    if not args.plain:
        fig.legend(handles=[Patch(color=COLOURS[DUG], label="tunnel dug"),
                            Patch(color=COLOURS[FILLED], label="dead air filled")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Generated maps (black = rock)", fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
