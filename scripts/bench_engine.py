"""Engine speed: match decision steps per second, by batch size and format, split by component.

Actions are random, with every agent aiming roughly at an enemy and pulling the trigger half
the time, so the rifle does realistic work. Timings depend on the machine's load, which is
printed with the results.

    .venv/bin/python scripts/bench_engine.py
    .venv/bin/python scripts/bench_engine.py --batches 64 --seconds 3
    .venv/bin/python scripts/bench_engine.py --visibility    # also compute who sees whom every step
"""

import argparse
import os
import platform
import time

import numpy as np

import rl_donut.env.match as match_module
from rl_donut.env import EngineConfig, World, match_step
from rl_donut.env.visibility import visibility
from rl_donut.mapgen import load_split

COMPONENTS = ("physics_tick", "fire", "life_tick")


def timed(name, fn, spent):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        spent[name] += time.perf_counter() - t
        return out
    return wrapper


def bench(maps, team_size, n, seconds, with_visibility=False):
    cfg = EngineConfig(team_size=team_size)
    world = World.from_maps([maps[i % len(maps)] for i in range(n)], cfg)
    rng = np.random.default_rng(0)
    enemy = (np.arange(cfg.n_agents) + team_size) % cfg.n_agents
    shape = world.alive.shape
    spent = dict.fromkeys(COMPONENTS + ("visibility",), 0.0)
    originals = {name: getattr(match_module, name) for name in COMPONENTS}
    for name, fn in originals.items():
        setattr(match_module, name, timed(name, fn, spent))
    steps, start = 0, time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            move, jet, trigger = rng.integers(-1, 2, size=shape), rng.random(shape) < 0.6, rng.random(shape) < 0.5
            aim = np.arctan2(world.y[:, enemy] - world.y, world.x[:, enemy] - world.x) + rng.uniform(-0.1, 0.1, shape)
            ev = match_step(world, move, jet, aim, trigger)
            if with_visibility:
                timed("visibility", visibility, spent)(world)
            steps += 1
            if ev.done.any():
                world.reset(np.nonzero(ev.done)[0])
    finally:
        for name, fn in originals.items():
            setattr(match_module, name, fn)
    total = time.perf_counter() - start
    return n * steps / total, total / steps, {k: v / total for k, v in spent.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--seconds", type=float, default=4.0, help="time per measurement")
    parser.add_argument("--visibility", action="store_true", help="also compute visibility every step")
    args = parser.parse_args()

    maps = load_split("val")
    load = os.getloadavg()
    print(f"{platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
          f"load average {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f}; one process\n")
    parts = COMPONENTS + ("visibility",) * args.visibility
    names = ["physics", "rifle", "life"] + ["visibility"] * args.visibility
    print("| Format | Matches | Match steps/s | ms per batch step | " + " | ".join(names) + " | other |")
    print("|---" * (len(names) + 5) + "|")
    for team_size in (1, 2):
        for n in args.batches:
            rate, per_step, share = bench(maps, team_size, n, args.seconds, args.visibility)
            other = 1 - sum(share.values())
            print(f"| {team_size}v{team_size} | {n} | {rate:,.0f} | {1e3 * per_step:.2f} | "
                  + " | ".join(f"{100 * share[k]:.0f}%" for k in parts) + f" | {100 * other:.0f}% |",
                  flush=True)


if __name__ == "__main__":
    main()
