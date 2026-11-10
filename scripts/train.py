"""PPO training CLI script for RL-Donut against baseline bots.

Usage:
    python scripts/train.py --n-matches 8 --total-updates 50 --opponent static
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner
from rl_donut.rl.stand_in import SPEC_VERSION, StandInEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO against scripted bots in RL-Donut.")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier (defaults to timestamp)")
    parser.add_argument("--split", type=str, default="train", help="Map split (train/val)")
    parser.add_argument("--n-matches", type=int, default=8, help="Number of concurrent matches")
    parser.add_argument("--team-size", type=int, default=1, help="Agents per team (1 for 1v1, 2 for 2v2)")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout steps per update")
    parser.add_argument("--total-updates", type=int, default=50, help="Total PPO updates")
    parser.add_argument("--opponent", type=str, choices=["static", "random"], default="static", help="Opponent bot")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory for runs")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting RL-Donut PPO Training Run: {run_id} ===")
    print(f"Matches: {args.n_matches} | Split: {args.split} | Opponent: {args.opponent}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment
    env = StandInEnv(
        n_matches=args.n_matches,
        team_size=args.team_size,
        split=args.split,
        seed=args.seed,
    )

    # Select opponent
    if args.opponent == "random":
        opponent = RandomBot()
    else:
        opponent = StaticTargetBot()

    # Model and trainer
    device = torch.device("cpu")
    model = ActorCritic(env.obs_dim, env.action_nvec, hidden_dim=256)
    cfg = PPOConfig(lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda)
    trainer = PPOTrainer(model, cfg, device=device)

    total_decision_steps = args.total_updates * args.n_steps * (args.n_matches * args.team_size)
    runner = Runner(
        env=env,
        learner=model,
        opponent=opponent,
        trainer=trainer,
        n_steps=args.n_steps,
        total_training_steps=total_decision_steps,
        device=device,
    )

    # Save run config
    config_record = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "seed": args.seed,
        "n_matches": args.n_matches,
        "team_size": args.team_size,
        "n_steps": args.n_steps,
        "total_updates": args.total_updates,
        "opponent": args.opponent,
        "lr": args.lr,
        "obs_dim": env.obs_dim,
        "action_nvec": list(env.action_nvec),
        "created_at": datetime.now().isoformat(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_file = run_dir / "metrics.jsonl"
    start_time = time.time()

    for update in range(1, args.total_updates + 1):
        m = runner.step_rollout()
        wall_time = time.time() - start_time
        env_steps = runner.global_step

        record = {
            "update": update,
            "env_steps": env_steps,
            "wall_time": round(wall_time, 2),
            "loss_policy": round(m.loss_policy, 4),
            "loss_value": round(m.loss_value, 4),
            "entropy": round(m.entropy, 4),
            "approx_kl": round(m.approx_kl, 5),
            "clip_fraction": round(m.clip_fraction, 4),
            "explained_variance": round(m.explained_variance, 4),
            "mean_return": round(m.mean_return, 3),
            "mean_length": round(m.mean_length, 1),
            "alpha": round(m.alpha, 3),
            "completed_episodes": m.completed_episodes,
        }

        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Update {update:3d}/{args.total_updates:3d} | "
            f"Steps: {env_steps:6d} | "
            f"Return: {m.mean_return:6.2f} | "
            f"PLoss: {m.loss_policy:6.3f} | "
            f"VLoss: {m.loss_value:6.3f} | "
            f"Entropy: {m.entropy:5.2f} | "
            f"Time: {wall_time:5.1f}s"
        )

        # Save checkpoint periodically
        if update % 25 == 0 or update == args.total_updates:
            ckpt_path = run_dir / f"ckpt_{update:04d}.pt"
            trainer.save_checkpoint(ckpt_path, extra_meta={"update": update, "env_steps": env_steps})
            print(f"  -> Saved checkpoint: {ckpt_path.name}")

    print(f"=== Training completed in {time.time() - start_time:.1f}s ===")


if __name__ == "__main__":
    main()
