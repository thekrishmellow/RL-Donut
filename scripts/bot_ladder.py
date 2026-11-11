"""Bot versus bot, sides swapped, so the curriculum gates have a measured yardstick.

    .venv/bin/python scripts/bot_ladder.py
    .venv/bin/python scripts/bot_ladder.py --matches 16 --seconds 20
"""

import argparse

import numpy as np

from rl_donut.env import EngineConfig, MatchConfig, MatchEnv
from rl_donut.env.bots import RandomBot, RusherBot, StaticTargetBot, TurretBot
from rl_donut.env.policy import play_matches

BOTS = {
    "random": RandomBot,
    "static": StaticTargetBot,
    "turret": TurretBot,
    "rusher": RusherBot,
}
PAIRS = (("turret", "static"), ("rusher", "static"), ("rusher", "turret"), ("random", "turret"))


def rates(maps, left, right, matches: int, seconds: float, seed: int) -> dict:
    """Win / draw / loss for `left`, over `matches` with the sides swapped halfway."""
    half = matches // 2
    cfg = EngineConfig(match=MatchConfig(duration=seconds))
    outcomes = []
    kills = []
    for side, (a, b) in enumerate(((left, right), (right, left))):
        env = MatchEnv(half, seed=seed + side, cfg=cfg, maps=maps)
        infos = play_matches(env, a(), b(), seed=seed)
        sign = 1 if side == 0 else -1
        outcomes += [sign * int(np.sign(info["outcome"][0])) for info in infos]
        kills += [float(info["agents"]["kills"][0 if side == 0 else 1]) for info in infos]
    outcomes = np.array(outcomes)
    return {"wins": float(np.mean(outcomes > 0)), "draws": float(np.mean(outcomes == 0)),
            "losses": float(np.mean(outcomes < 0)), "kills": float(np.mean(kills))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", type=int, default=32, help="even number; half per side")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.matches % 2:
        raise SystemExit("--matches must be even so each side is played equally")
    from rl_donut.mapgen import load_split
    maps = load_split("val")
    print(f"{args.matches} matches, {args.seconds:.0f}s, sides swapped halfway\n")
    print("| Left | Right | Wins | Draws | Losses | Kills per match |")
    print("|---|---|---|---|---|---|")
    for left, right in PAIRS:
        result = rates(maps, BOTS[left], BOTS[right], args.matches, args.seconds, args.seed)
        print(f"| {left} | {right} | {result['wins']:.2f} | {result['draws']:.2f} | "
              f"{result['losses']:.2f} | {result['kills']:.2f} |", flush=True)


if __name__ == "__main__":
    main()
