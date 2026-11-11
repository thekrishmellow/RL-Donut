"""Self-play PPO training CLI script for RL-Donut (Phase 6 Step 6.4).

Features:
- Dynamic opponent sampling (50% latest self, 30% PFSP historical pool, 20% baseline bots)
- Checkpoint admission testing (must beat previous snapshot >= 55% to join pool)
- Periodic offline Elo round-robin evaluation
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch

from rl_donut.rl.adaptive import AdaptiveConfig, AdaptiveMatchmaker
from rl_donut.rl.checkpoints import CheckpointManager
from rl_donut.rl.elo import run_round_robin
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.pool import OpponentPool
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner
from rl_donut.rl.stand_in import SPEC_VERSION, StandInEnv
from rl_donut.rl.styles import CombatStyle, STYLE_CONFIGS
from rl_donut.rl.watchlist import WatchlistTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play training for RL-Donut.")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier")
    parser.add_argument("--split", type=str, default="train", help="Map split")
    parser.add_argument("--n-matches", type=int, default=8, help="Matches per batch")
    parser.add_argument("--team-size", type=int, default=1, help="Agents per team (1 for 1v1, 2 for 2v2)")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout steps per update")
    parser.add_argument("--total-updates", type=int, default=50, help="Total updates")
    parser.add_argument("--snapshot-interval", type=int, default=25, help="Snapshot evaluation interval")
    parser.add_argument("--style", type=str, default="balanced", choices=["balanced", "aggressive", "defensive", "sniper"], help="Combat behavioral style")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive matchmaking (maintains 50-60%% win rate)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or ("selfplay_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting RL-Donut Self-Play Training Run: {run_id} ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = StandInEnv(
        n_matches=args.n_matches,
        team_size=args.team_size,
        split=args.split,
        seed=args.seed,
    )

    device = torch.device("cpu")
    model = ActorCritic(env.obs_dim, env.action_nvec, hidden_dim=256)
    cfg = PPOConfig(lr=args.lr)
    trainer = PPOTrainer(model, cfg, device=device)
    ckpt_manager = CheckpointManager(run_dir=run_dir, max_to_keep=5, milestone_interval=50)

    pool = OpponentPool(max_snapshots=30, admission_win_rate=0.55)
    # Add initial policy as snapshot 0
    initial_policy = LearnedPolicy(
        ActorCritic(env.obs_dim, env.action_nvec, hidden_dim=256), name="snapshot_0000"
    )
    initial_policy.model.load_state_dict(model.state_dict())
    pool.consider_admission(initial_policy, update=0)

    total_decision_steps = args.total_updates * args.n_steps * (args.n_matches * args.team_size)
    
    style = CombatStyle(args.style)
    style_cfg = STYLE_CONFIGS[style]
    matchmaker = AdaptiveMatchmaker(pool) if args.adaptive else None
    current_opponent = matchmaker.select_opponent() if matchmaker else pool.sample_opponent()

    runner = Runner(
        env=env,
        learner=model,
        opponent=current_opponent,
        trainer=trainer,
        n_steps=args.n_steps,
        weights=style_cfg.weights,
        total_training_steps=total_decision_steps,
        device=device,
    )

    # Save run config
    config_record = {
        "run_id": run_id,
        "mode": "self_play",
        "spec_version": SPEC_VERSION,
        "seed": args.seed,
        "n_matches": args.n_matches,
        "team_size": args.team_size,
        "n_steps": args.n_steps,
        "total_updates": args.total_updates,
        "snapshot_interval": args.snapshot_interval,
        "style": args.style,
        "adaptive": args.adaptive,
        "created_at": datetime.now().isoformat(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_file = run_dir / "metrics.jsonl"
    start_time = time.time()

    for update in range(1, args.total_updates + 1):
        # Sample opponent (adaptive or PFSP standard)
        if matchmaker is not None:
            opponent = matchmaker.select_opponent()
        else:
            opponent = pool.sample_opponent()
        runner.opponent = opponent

        m = runner.step_rollout()
        wall_time = time.time() - start_time
        env_steps = runner.global_step

        if matchmaker is not None:
            won = m.mean_return > 0.0
            lost = m.mean_return < 0.0
            matchmaker.record_match_outcome(won=won, lost=lost)

        record = {
            "update": update,
            "env_steps": env_steps,
            "wall_time": round(wall_time, 2),
            "opponent": opponent.name,
            "loss_policy": round(m.loss_policy, 4),
            "loss_value": round(m.loss_value, 4),
            "entropy": round(m.entropy, 4),
            "approx_kl": round(m.approx_kl, 5),
            "clip_fraction": round(m.clip_fraction, 4),
            "explained_variance": round(m.explained_variance, 4),
            "mean_return": round(m.mean_return, 3),
            "mean_length": round(m.mean_length, 1),
            "pool_size": len(pool.snapshots),
        }

        with open(metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Update {update:3d}/{args.total_updates:3d} | "
            f"Opponent: {opponent.name:15s} | "
            f"Return: {m.mean_return:6.2f} | "
            f"Pool: {len(pool.snapshots):2d} | "
            f"Time: {wall_time:5.1f}s"
        )

        # Snapshot evaluation gate
        if update % args.snapshot_interval == 0:
            candidate_policy = LearnedPolicy(
                ActorCritic(env.obs_dim, env.action_nvec, hidden_dim=256),
                name=f"candidate_{update:04d}",
            )
            candidate_policy.model.load_state_dict(model.state_dict())

            admitted = pool.consider_admission(candidate_policy, update=update, n_eval_matches=8)
            status = "ADMITTED to pool" if admitted else "REJECTED (win rate < 55%)"
            print(f"  -> Snapshot Gate @ update {update}: {status}")

            ckpt_manager.save(model, update=update, extra_meta={"update": update, "admitted": admitted})

    print(f"=== Self-Play completed in {time.time() - start_time:.1f}s ===")


if __name__ == "__main__":
    main()

