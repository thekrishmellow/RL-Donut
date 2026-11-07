"""Render a comparison sheet of noise settings to outputs/noise_preview.png."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_donut.mapgen import MapConfig, Stream, fbm, make_rng

ROCK_FRACTION = 0.40


def noise_for(cfg, seed, **changes):
    return fbm(cfg.width, cfg.height, replace(cfg.noise, **changes), make_rng(seed, Stream.TERRAIN))


def as_rock(noise):
    return noise >= np.quantile(noise, 1 - ROCK_FRACTION)


def gradient_magnitude(noise):
    gy, gx = np.gradient(noise)
    return np.hypot(gx, gy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs/noise_preview.png"))
    args = parser.parse_args()

    cfg = MapConfig()
    rows = [
        ("Octaves stacked", [
            (f"{k} octave{'s' if k > 1 else ''}", noise_for(cfg, args.seed, octaves=k), "noise")
            for k in range(1, 5)
        ]),
        ("Fade curve (1 octave)", [
            ("linear", noise_for(cfg, args.seed, octaves=1, fade="linear"), "noise"),
            ("quintic", noise_for(cfg, args.seed, octaves=1, fade="quintic"), "noise"),
            ("linear: slope", gradient_magnitude(noise_for(cfg, args.seed, octaves=1, fade="linear")), "slope"),
            ("quintic: slope", gradient_magnitude(noise_for(cfg, args.seed, octaves=1, fade="quintic")), "slope"),
        ]),
        ("Cell size (4 octaves)", [
            (f"{w:g} x {h:g}", noise_for(cfg, args.seed, cell_w=w, cell_h=h), "noise")
            for w, h in [(12, 8), (16, 10), (32, 20)]
        ]),
        (f"Round vs stretched, {ROCK_FRACTION:.0%} rock", [
            ("round 16 x 16", noise_for(cfg, args.seed, cell_w=16, cell_h=16), "noise"),
            ("stretched 16 x 10", noise_for(cfg, args.seed, cell_w=16, cell_h=10), "noise"),
            ("round 16 x 16: rock", as_rock(noise_for(cfg, args.seed, cell_w=16, cell_h=16)), "rock"),
            ("stretched 16 x 10: rock", as_rock(noise_for(cfg, args.seed, cell_w=16, cell_h=10)), "rock"),
        ]),
    ]

    cmaps = {"noise": "terrain", "slope": "magma", "rock": "Greys"}
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 11), constrained_layout=True)
    for (row_title, panels), row_axes in zip(rows, axes):
        for ax in row_axes:
            ax.axis("off")
        for ax, (title, image, kind) in zip(row_axes, panels):
            ax.imshow(image, cmap=cmaps[kind], interpolation="nearest")
            ax.set_title(title, fontsize=10)
        row_axes[0].text(-0.08, 0.5, row_title, transform=row_axes[0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle(f"Noise preview, seed {args.seed}, {cfg.width} x {cfg.height} tiles", fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
