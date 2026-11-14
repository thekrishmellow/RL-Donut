"""Record one match between two bots and draw it, or draw an existing replay.

    .venv/bin/python scripts/render_replay.py                          # turret vs rusher, 15 s
    .venv/bin/python scripts/render_replay.py --replay outputs/match.json.gz
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_match  # noqa: E402

from rl_donut.env import EngineConfig, MatchConfig, MatchEnv  # noqa: E402
from rl_donut.env.bots import RandomBot, RusherBot, StaticTargetBot, TurretBot  # noqa: E402
from rl_donut.env.replay import load, record, save, snapshots  # noqa: E402
from rl_donut.mapgen import load_split  # noqa: E402

BOTS = {"random": RandomBot, "static": StaticTargetBot, "turret": TurretBot, "rusher": RusherBot}


def load_checkpoint_policy(path: str | Path, name: str = "checkpoint"):
    import torch
    from rl_donut.rl.model import ActorCritic
    from rl_donut.rl.ppo import LearnedPolicy

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", 76)
    action_nvec = ckpt.get("action_nvec", [5, 2, 2])
    cont_dim = ckpt.get("cont_dim", 1)
    model = ActorCritic(obs_dim, action_nvec, hidden_dim=256, cont_dim=cont_dim)
    model.load_state_dict(ckpt["model_state"])
    return LearnedPolicy(model, name=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", help="draw this file instead of playing a new match")
    parser.add_argument("--a", default="turret", choices=sorted(BOTS))
    parser.add_argument("--b", default="rusher", choices=sorted(BOTS))
    parser.add_argument("--a-checkpoint", type=str, default=None, help="load a LearnedPolicy checkpoint for team A")
    parser.add_argument("--b-checkpoint", type=str, default=None, help="load a LearnedPolicy checkpoint for team B")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--team-size", type=int, default=1)
    parser.add_argument("--map", type=int, default=0, help="index into the val split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/replay")
    args = parser.parse_args()

    maps = load_split("val")
    if args.replay:
        replay = load(args.replay)
    else:
        cfg = EngineConfig(team_size=args.team_size, match=MatchConfig(duration=args.seconds))
        env = MatchEnv(1, args.team_size, seed=args.seed, cfg=cfg, maps=maps[args.map:args.map + 1])
        policy_a = load_checkpoint_policy(args.a_checkpoint, "agent_a") if args.a_checkpoint else BOTS[args.a]()
        policy_b = load_checkpoint_policy(args.b_checkpoint, "agent_b") if args.b_checkpoint else BOTS[args.b]()
        replay = record(env, policy_a, policy_b, seed=args.seed)
        save(f"{args.out}.json.gz", replay)
    env, frames = snapshots(replay, maps)
    title = f"seed {replay['map_seed']}  {replay['score'][0]} : {replay['score'][1]}"
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    painter = view_match.Painter(ax, env.world, title)
    fps = max(env.world.cfg.tick_rate // env.world.cfg.match.decision_ticks, 1)
    FuncAnimation(fig, lambda k: painter.draw(frames[k]), frames=len(frames)).save(
        f"{args.out}.gif", writer=PillowWriter(fps=fps), dpi=60)
    plt.close(fig)
    picks = np.linspace(0, len(frames) - 1, view_match.STRIP_FRAMES).astype(int)
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.6, 5.0 * len(picks)))
    for axis, k in zip(np.atleast_1d(axes), picks):
        view_match.Painter(axis, env.world, title).draw(frames[k])
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=90)
    print(f"{len(frames)} steps, score {replay['score'][0]} : {replay['score'][1]} -> {args.out}.gif")


if __name__ == "__main__":
    main()
