"""Watch one match: plays it with simple scripted fighters and saves an animated GIF and a
filmstrip PNG. A debugging aid only; the real renderer is Phase 4's.

The fighters move at random (changing their mind about twice a second), aim at the nearest
enemy they can see and fire while it stays in sight.

    .venv/bin/python scripts/view_match.py                        # 1v1 on the first val map
    .venv/bin/python scripts/view_match.py --map 3 --team-size 2 --close --seconds 20
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from rl_donut.env import EngineConfig, World, match_step
from rl_donut.mapgen import line_of_sight, load_split

TEAM_COLOURS = ("tab:blue", "tab:red")
STRIP_FRAMES = 6


class Fighters:
    """Random movement, aim at the nearest visible enemy, fire while it's visible."""

    def __init__(self, world: World, seed: int):
        self.world, self.rng = world, np.random.default_rng(seed)
        shape = world.alive.shape
        self.move = np.zeros(shape, dtype=np.int64)
        self.jet = np.zeros(shape, dtype=bool)

    def act(self):
        w, rng = self.world, self.rng
        change = rng.random(w.alive.shape) < 0.15
        self.move = np.where(change, rng.integers(-1, 2, size=w.alive.shape), self.move)
        self.jet = np.where(change, rng.random(w.alive.shape) < 0.55, self.jet)
        aim, trigger = w.aim.copy(), np.zeros_like(w.alive)
        for m in range(w.n_matches):
            for i in np.nonzero(w.alive[m])[0]:
                best = None
                for j in np.nonzero(w.alive[m] & (w.team != w.team[i]))[0]:
                    d = math.hypot(w.x[m, j] - w.x[m, i], w.y[m, j] - w.y[m, i])
                    if d <= w.cfg.rifle.range and (best is None or d < best[0]) and \
                            line_of_sight(w.grids[m], (w.x[m, i], w.y[m, i]), (w.x[m, j], w.y[m, j])):
                        best = (d, j)
                if best is not None:
                    j = best[1]
                    aim[m, i] = math.atan2(w.y[m, j] - w.y[m, i], w.x[m, j] - w.x[m, i])
                    trigger[m, i] = True
        return self.move, self.jet, aim, trigger


def record(world: World, steps: int, seed: int):
    """Play the match and keep a snapshot of everything drawn, once per decision step."""
    fighters = Fighters(world, seed)
    frames = []
    for _ in range(steps):
        ev = match_step(world, *fighters.act())
        frames.append(dict(
            x=world.x[0].copy(), y=world.y[0].copy(), aim=world.aim[0].copy(), alive=world.alive[0].copy(),
            health=world.health[0].copy(), fuel=world.fuel[0].copy(), protection=world.protection[0].copy(),
            score=world.score[0].copy(), tick=int(world.tick[0]),
            shots=[(s.shooter, s.angle, s.hit) for _, s in ev.shot_log],
            deaths=[(r.death.agent, r.scorer, r.death.fell) for r in ev.death_log],
        ))
        if ev.done[0]:
            break
    return frames


def shot_end(grid, x, y, angle, limit):
    """Where a missed shot stops: rock or range (by bisection on line_of_sight, to 0.02 tiles)."""
    end = lambda t: (x + t * math.cos(angle), y + t * math.sin(angle))
    if line_of_sight(grid, (x, y), end(limit)):
        return end(limit)
    lo, hi = 0.0, limit
    while hi - lo > 0.02:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if line_of_sight(grid, (x, y), end(mid)) else (lo, mid)
    return end(lo)


class Painter:
    """Draws one frame on an axis; the map is drawn once, agents and shots are updated."""

    def __init__(self, ax, world: World, title: str):
        cfg = world.cfg
        self.ax, self.world, self.cfg, self.title = ax, world, cfg, title
        grid = world.grids[0]
        height, width = grid.shape
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1.6, extent=(0, width, height, 0))
        ax.axhline(world.kill_y[0], color="tab:purple", lw=0.8, ls=":")
        ax.set_xlim(0, width)
        ax.set_ylim(height + 3, -2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for team, colour in enumerate(TEAM_COLOURS):
            sx, sy = world.spawns[0, team].T
            ax.plot(sx, sy, "^", color=colour, ms=4, alpha=0.35)
        self.dynamic = []

    def draw(self, f):
        for artist in self.dynamic:
            artist.remove()
        self.dynamic = []
        add = self.dynamic.append
        cfg, ax = self.cfg, self.ax
        hw, hh = cfg.hitbox_w / 2, cfg.hitbox_h / 2
        for shooter, angle, hit in f["shots"]:
            x0, y0 = f["x"][shooter], f["y"][shooter]
            x1, y1 = hit if hit else shot_end(self.world.grids[0], x0, y0, angle, cfg.rifle.range)
            add(ax.plot([x0, x1], [y0, y1], "-", color="red" if hit else "gold", lw=1.0, alpha=0.9)[0])
        for i in range(len(f["x"])):
            colour = TEAM_COLOURS[self.world.team[i]]
            x, y = f["x"][i], f["y"][i]
            if not f["alive"][i]:
                add(ax.plot(x, min(y, self.world.grids.shape[1] + 2), "x", color=colour, ms=7, mew=2)[0])
                continue
            add(ax.add_patch(Rectangle((x - hw, y - hh), cfg.hitbox_w, cfg.hitbox_h, color=colour,
                                       ec="white" if f["protection"][i] else colour, lw=1.2)))
            add(ax.plot([x, x + 1.6 * math.cos(f["aim"][i])], [y, y + 1.6 * math.sin(f["aim"][i])],
                        "-", color=colour, lw=1.2)[0])
            for row, value, full, bar_colour in ((1.4, f["health"][i], cfg.life.max_health, "limegreen"),
                                                 (1.0, f["fuel"][i], cfg.physics.fuel_capacity, "orange")):
                add(ax.add_patch(Rectangle((x - 0.8, y - hh - row), 1.6, 0.3, color="black", alpha=0.4)))
                add(ax.add_patch(Rectangle((x - 0.8, y - hh - row), 1.6 * max(value, 0) / full, 0.3, color=bar_colour)))
        for agent, scorer, fell in f["deaths"]:
            add(ax.text(f["x"][agent], f["y"][agent] - 2.6, "fell" if fell else "killed", ha="center",
                        fontsize=7, color="black", weight="bold"))
        seconds = f["tick"] / cfg.tick_rate
        add(ax.text(1, -1, f"{self.title}   t = {seconds:4.1f} s   blue {f['score'][0]} : {f['score'][1]} red",
                    fontsize=9, va="center"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=int, default=0, help="index into the val split")
    parser.add_argument("--team-size", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--close", action="store_true", help="start team B on team A's spawns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/match")
    args = parser.parse_args()

    cfg = EngineConfig(team_size=args.team_size)
    m = load_split("val")[args.map]
    world = World.single(m, cfg, rng_seed=args.seed)
    if args.close:
        k = cfg.team_size
        world.x[0, k:], world.y[0, k:] = world.x[0, :k] + 1.5, world.y[0, :k]
    steps = int(args.seconds * cfg.tick_rate) // cfg.match.decision_ticks
    frames = record(world, steps, args.seed)
    title = f"seed {m.seed}, {args.team_size}v{args.team_size}"

    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    painter = Painter(ax, world, title)
    fps = cfg.tick_rate // cfg.match.decision_ticks
    FuncAnimation(fig, lambda k: painter.draw(frames[k]), frames=len(frames)).save(
        f"{args.out}.gif", writer=PillowWriter(fps=fps), dpi=60)
    plt.close(fig)

    picks = np.linspace(0, len(frames) - 1, STRIP_FRAMES).astype(int)
    busiest = sorted(range(len(frames)), key=lambda k: -len(frames[k]["shots"]) - 5 * len(frames[k]["deaths"]))
    picks = sorted(set(busiest[:STRIP_FRAMES // 2]) | set(picks[:STRIP_FRAMES - STRIP_FRAMES // 2]))[:STRIP_FRAMES]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.6, 5.0 * len(picks)))
    for ax, k in zip(np.atleast_1d(axes), picks):
        Painter(ax, world, title).draw(frames[k])
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=90)
    last = frames[-1]
    print(f"{len(frames)} steps, final score {last['score'][0]} : {last['score'][1]} -> {args.out}.gif, {args.out}.png")


if __name__ == "__main__":
    main()
