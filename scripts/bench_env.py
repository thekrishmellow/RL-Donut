"""Environment speed: full `MatchEnv.step` match steps per second, by batch size and format.

Each step decodes actions, runs the engine, computes visibility and observations, and
counts episode stats; auto-reset is included. Actions are uniform random. Timings depend
on the machine's load, which is printed with the results.

    .venv/bin/python scripts/bench_env.py
    .venv/bin/python scripts/bench_env.py --batches 256 --seconds 6
    .venv/bin/python scripts/bench_env.py --state                      # also call get_state() every step
    .venv/bin/python scripts/bench_env.py --processes 1 2 4 8 --batches 256   # independent processes at once
"""

import argparse
import math
import multiprocessing
import os
import platform
import time

import numpy as np

import rl_donut.env.match_env as env_module
from rl_donut.env import DISCRETE_NVEC
from rl_donut.env.observe import Observer
from rl_donut.mapgen import load_split

# (owner, attribute, column) of every timed part of a step.
PARTS = ((env_module, "decode", "actions"), (env_module, "match_step", "engine"),
         (env_module, "visibility", "visibility"), (Observer, "record", "memory"),
         (Observer, "observe", "observations"))
ACTION_SETS = 50


def timed(fn, spent, column):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        spent[column] += time.perf_counter() - t
        return out
    return wrapper


def random_actions(env, rng):
    shape = (env.n_matches, env.n_agents)
    discrete = np.stack([rng.integers(0, n, size=shape) for n in DISCRETE_NVEC], axis=-1)
    return discrete, rng.uniform(-math.pi, math.pi, size=shape + (1,))


def bench(team_size, n, seconds, state=False, breakdown=True):
    """Returns match steps per second, ms per batch step and each part's share of the time."""
    env = env_module.MatchEnv(n, team_size=team_size, split="val")
    rng = np.random.default_rng(0)
    actions = [random_actions(env, rng) for _ in range(ACTION_SETS)]
    columns = [column for _, _, column in PARTS] + ["state"] * state
    spent = dict.fromkeys(columns, 0.0)
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in PARTS] if breakdown else []
    for (owner, name, fn), (_, _, column) in zip(originals, PARTS):
        setattr(owner, name, timed(fn, spent, column))
    get_state = timed(env.get_state, spent, "state") if state else None
    try:
        env.reset()
        steps, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            env.step(*actions[steps % ACTION_SETS])
            if state:
                get_state()
            steps += 1
    finally:
        for owner, name, fn in originals:
            setattr(owner, name, fn)
    total = time.perf_counter() - start
    return n * steps / total, 1e3 * total / steps, {k: v / total for k, v in spent.items()}


def worker(job):
    team_size, n, seconds = job
    return bench(team_size, n, seconds, breakdown=False)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--seconds", type=float, default=4.0, help="time per measurement")
    parser.add_argument("--state", action="store_true", help="also call get_state() every step")
    parser.add_argument("--processes", type=int, nargs="+",
                        help="instead run this many independent processes at once and add up their rates")
    args = parser.parse_args()

    load = os.getloadavg()
    print(f"{platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
          f"load average {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f}\n")
    if args.processes:
        print("| Format | Matches per process | Processes | Total match steps/s | Per process |")
        print("|---|---|---|---|---|")
        context = multiprocessing.get_context("spawn")
        for team_size in (1, 2):
            for n in args.batches:
                for p in args.processes:
                    with context.Pool(p) as pool:
                        rates = pool.map(worker, [(team_size, n, args.seconds)] * p)
                    print(f"| {team_size}v{team_size} | {n} | {p} | {sum(rates):,.0f} | {np.mean(rates):,.0f} |", flush=True)
        return
    columns = [column for _, _, column in PARTS] + ["state"] * args.state
    print("| Format | Matches | Match steps/s | Agent steps/s | ms per batch step | "
          + " | ".join(columns) + " | other |")
    print("|---" * (len(columns) + 6) + "|")
    for team_size in (1, 2):
        for n in args.batches:
            rate, ms, share = bench(team_size, n, args.seconds, args.state)
            other = 1 - sum(share.values())
            print(f"| {team_size}v{team_size} | {n} | {rate:,.0f} | {2 * team_size * rate:,.0f} | {ms:.2f} | "
                  + " | ".join(f"{100 * share[k]:.0f}%" for k in columns) + f" | {100 * other:.0f}% |", flush=True)


if __name__ == "__main__":
    main()
