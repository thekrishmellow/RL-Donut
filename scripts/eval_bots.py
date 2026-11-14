"""CLI script to evaluate a checkpoint against scripted bots on held-out maps.

Usage:
    python scripts/eval_bots.py --checkpoint runs/20261110_164535/ckpt_0003.pt --opponent static --n-matches 20
"""

import argparse
from pathlib import Path
import torch

from rl_donut.env.bots import RandomBot, RusherBot, StaticTargetBot, TurretBot
from rl_donut.rl.benchmark import guard_test_split
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against baseline bots.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--opponent", type=str, choices=["static", "random", "turret", "rusher"], default="static", help="Opponent")
    parser.add_argument("--n-matches", type=int, default=20, help="Number of evaluation matches")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split (val/test)")
    parser.add_argument("--seed", type=int, default=10_000, help="Evaluation seed")
    parser.add_argument("--final", action="store_true", help="Required flag when evaluating on test split")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation on test split")
    return parser.parse_args()


def main():
    args = parse_args()
    guard_test_split(args.split, final=args.final, force=args.force)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", 330)
    action_nvec = ckpt.get("action_nvec", (3, 2, 2))

    model = ActorCritic(obs_dim, action_nvec, hidden_dim=256, cont_dim=ckpt.get("cont_dim", 1))
    state_dict = ckpt["model_state"] if "model_state" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.load_state_dict(state_dict)
    policy = LearnedPolicy(model, name="checkpoint_eval")

    if args.opponent == "random":
        opponent = RandomBot()
    elif args.opponent == "turret":
        opponent = TurretBot()
    elif args.opponent == "rusher":
        opponent = RusherBot()
    else:
        opponent = StaticTargetBot()

    print(f"=== Evaluating {ckpt_path.name} vs {opponent.name} on {args.split} split ({args.n_matches} matches) ===")
    report = evaluate_policy(
        learner=policy,
        opponent=opponent,
        n_matches=args.n_matches,
        split=args.split,
        seed=args.seed,
    )

    print(f"Matches: {report.n_matches}")
    print(f"Wins: {report.wins} | Losses: {report.losses} | Draws: {report.draws}")
    print(f"Win Rate: {report.win_rate:.1%} (95% CI: [{report.wilson_ci_lower:.1%}, {report.wilson_ci_upper:.1%}])")
    print(f"Mean Kills: {report.mean_kills:.2f} | Deaths: {report.mean_deaths:.2f}")
    print(f"Mean Damage Dealt: {report.mean_damage_dealt:.1f} | Taken: {report.mean_damage_taken:.1f}")


if __name__ == "__main__":
    main()

