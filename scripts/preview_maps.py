"""Render a sheet of generated maps for a range of seeds to outputs/maps_preview.png."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_donut.mapgen import MapConfig, generate_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="first seed")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/maps_preview.png"))
    args = parser.parse_args()

    cfg = MapConfig()
    rows = -(-args.count // args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(4.6 * args.cols, 2.7 * rows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, seed in zip(axes.flat, range(args.start, args.start + args.count)):
        game_map = generate_map(seed, cfg)
        params = game_map.params
        ax.imshow(game_map.grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"seed {seed}: rock {params.rock_fraction:.0%} -> {game_map.terrain_rock_fraction:.0%} "
                     f"(+border {game_map.rock_fraction:.0%}), bias {params.height_bias:.2f}", fontsize=9)
    fig.suptitle("Generated maps (black = rock)", fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
