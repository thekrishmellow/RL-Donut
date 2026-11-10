"""What one agent can see: every standing spot on the map, coloured by whether an enemy
standing there would be seen (centre visible, only a hitbox corner visible, or hidden).

    .venv/bin/python scripts/preview_visibility.py --count 3
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from rl_donut.env import EngineConfig, Navigator
from rl_donut.env.raycast import segments
from rl_donut.env.visibility import target_points
from rl_donut.mapgen import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/visibility_preview.png")
    args = parser.parse_args()

    cfg = EngineConfig()
    points = target_points(cfg)
    rng = np.random.default_rng(args.seed)
    maps = load_split("val")[:args.count]
    fig, axes = plt.subplots(len(maps), 1, figsize=(12, 6.4 * len(maps)), squeeze=False)
    for ax, m in zip(axes[:, 0], maps):
        nav = Navigator(m, cfg)
        spots = nav.main_spots()
        x, y = nav.standing_centre(tuple(spots[rng.integers(len(spots))]))
        centres = np.array([nav.standing_centre(tuple(s)) for s in spots])
        ends = (centres[:, None, :] + points[None]).reshape(-1, 2)
        n = len(ends)
        clear, _ = segments((m.grid == 1)[None], np.zeros(n, dtype=int), np.full(n, x), np.full(n, y), ends[:, 0], ends[:, 1])
        clear = clear.reshape(len(centres), len(points))
        centre, corner_only = clear[:, 0], clear.any(axis=1) & ~clear[:, 0]

        ax.imshow(m.grid, cmap="Greys", vmin=0, vmax=1.6, extent=(0, m.grid.shape[1], m.grid.shape[0], 0))
        hidden = ~clear.any(axis=1)
        ax.scatter(*centres[hidden].T, s=3, color="0.75", label=f"hidden ({hidden.sum()})")
        ax.scatter(*centres[centre].T, s=5, color="tab:green", label=f"centre in sight ({centre.sum()})")
        ax.scatter(*centres[corner_only].T, s=14, color="tab:orange",
                   label=f"only a corner in sight: missed by a centre-only check ({corner_only.sum()})")
        ax.add_patch(Rectangle((x - cfg.hitbox_w / 2, y - cfg.hitbox_h / 2), cfg.hitbox_w, cfg.hitbox_h, color="tab:blue"))
        ax.set_xlim(0, m.grid.shape[1])
        ax.set_ylim(m.grid.shape[0], 0)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"seed {m.seed}: what the blue agent sees of an enemy standing at each spot")
    fig.tight_layout()
    fig.savefig(args.out, dpi=65)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
