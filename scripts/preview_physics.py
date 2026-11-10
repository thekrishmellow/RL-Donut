"""What one full tank can do: flight paths in an empty room, against the generator's needs.

    .venv/bin/python scripts/preview_physics.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_donut.env import EngineConfig, World, physics_tick
from rl_donut.mapgen import load_split

FLOOR, START_X = 46, 4.0


def fly(world_template, start_y, policy, ticks=400):
    world = World.single(world_template)
    grid = np.zeros((48, 96), dtype=bool)
    grid[0, :] = grid[:, 0] = grid[:, -1] = True
    grid[FLOOR:, :] = True
    world.grids[0] = grid
    world.x[0, 0], world.y[0, 0] = START_X, start_y
    world.alive[0, 1] = False
    xs, ys, empty_at = [world.x[0, 0]], [world.y[0, 0]], None
    jet = False
    for tick in range(ticks):
        if tick % 2 == 0:
            move, jet = policy(world, tick, start_y)
        physics_tick(world, np.array([[move, 0]]), np.array([[jet, False]]))
        xs.append(world.x[0, 0])
        ys.append(world.y[0, 0])
        if empty_at is None and world.fuel[0, 0] == 0:
            empty_at = len(xs) - 1
    return np.array(xs), np.array(ys), empty_at


def main():
    cfg = EngineConfig()
    m = load_split("val")[0]
    stand = FLOOR - cfg.hitbox_h / 2
    level_y = stand - 10
    paths = {
        "full thrust, straight up": (stand, lambda w, t, y0: (0, True)),
        "full thrust, moving right": (stand, lambda w, t, y0: (1, True)),
        "level flight, moving right": (level_y, lambda w, t, y0: (1, bool(w.y[0, 0] >= y0 or w.vy[0, 0] > 0))),
        "thrust 3 of every 4 decisions, right": (stand, lambda w, t, y0: (1, (t // 2) % 4 != 3)),
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axhspan(FLOOR, 48, color="#888")
    for (label, (start_y, policy)), colour in zip(paths.items(), ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]):
        xs, ys, empty = fly(m, start_y, policy)
        ax.plot(xs, ys, color=colour, label=label)
        if empty is not None:
            ax.plot(xs[empty], ys[empty], "o", color=colour, ms=6)
    budget = cfg.map.climb_budget
    ax.axhline(stand - budget, color="k", ls="--", lw=0.8)
    ax.text(70, stand - budget - 0.6, f"generator needs: climb {budget}", fontsize=9)
    ax.plot([START_X, START_X + 2 * budget], [level_y + 2, level_y + 2], color="k", lw=2)
    ax.text(START_X + 0.5, level_y + 3.4, f"generator needs: {2 * budget} tiles of level flight", fontsize=9)
    ax.set_xlim(0, 96)
    ax.set_ylim(48, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (tiles)")
    ax.set_ylabel("y (tiles, down)")
    ax.set_title("One full tank (dot = tank empty); the agent decides every 2 ticks")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/physics_preview.png", dpi=80)
    print("outputs/physics_preview.png")


if __name__ == "__main__":
    main()
