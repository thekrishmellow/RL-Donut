"""CLI script for offline Elo round-robin evaluation between snapshots and bots.

Usage:
    python scripts/elo_round_robin.py --run-dir runs/20261110_164535 --n-matches 10
"""

import argparse
from pathlib import Path
import torch

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.elo import run_round_robin
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline Elo round-robin tournament.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing ckpt_*.pt files")
    parser.add_argument("--n-matches", type=int, default=10, help="Matches per pair")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split (val/test)")
    parser.add_argument("--seed", type=int, default=50_000, help="Tournament seed")
    parser.add_argument("--out", type=str, default="", help="Output elo.json path")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_files = sorted(run_dir.glob("ckpt_*.pt"))

    policies = [RandomBot(), StaticTargetBot()]

    for ckpt_path in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ActorCritic(ckpt["obs_dim"], ckpt["action_nvec"], hidden_dim=256)
        model.load_state_dict(ckpt["model_state"])
        policy = LearnedPolicy(model, name=ckpt_path.stem)
        policies.append(policy)

    print(f"=== Starting Elo Round-Robin with {len(policies)} participants ({args.n_matches} matches/pair) ===")
    for p in policies:
        print(f"  - {p.name}")

    report = run_round_robin(
        policies=policies,
        n_matches_per_pair=args.n_matches,
        split=args.split,
        seed=args.seed,
    )

    print("\n=== Final Elo Ratings ===")
    for name, rating in sorted(report.ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {rating:6.1f}")

    out_file = Path(args.out) if args.out else run_dir / "elo.json"
    report.to_json(out_file)
    print(f"\nSaved ratings to: {out_file}")


if __name__ == "__main__":
    main()
