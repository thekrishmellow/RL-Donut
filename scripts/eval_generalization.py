"""Generalization evaluation CLI script for RL-Donut (Phase 8).

Evaluates a policy checkpoint across:
- In-distribution train split
- In-distribution held-out val split
- Out-of-Distribution physics shifts (gravity ±25%, thrust ±25%)
- Out-of-Distribution seeds (30,000+)

Computes and reports the headline Generalization Gap.
"""

import argparse
import json
from pathlib import Path
import torch

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.generalization import evaluate_generalization
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy
from rl_donut.rl.stand_in import StandInEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generalization gap for RL-Donut.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "static"], help="Baseline bot")
    parser.add_argument("--n-matches", type=int, default=20, help="Matches per condition")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--out", type=str, default="", help="Output json path (default: <ckpt_dir>/generalization.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dummy_env = StandInEnv(n_matches=1, split="val")
    model = ActorCritic(dummy_env.obs_dim, dummy_env.action_nvec, hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    learner = LearnedPolicy(model, name=ckpt_path.stem)
    opponent = RandomBot() if args.opponent == "random" else StaticTargetBot()

    print(f"=== Evaluating Generalization for {learner.name} vs {opponent.name} ===")
    print(f"Matches per condition: {args.n_matches}")

    report = evaluate_generalization(
        learner=learner,
        opponent=opponent,
        n_matches_per_condition=args.n_matches,
        seed=args.seed,
    )

    print("\n--- Generalization Evaluation Results ---")
    print(f"{'Condition':20s} | {'Win Rate':9s} | {'95% CI':16s} | {'Kills':6s} | {'Deaths':6s}")
    print("-" * 65)

    for cond_name, res in report.conditions.items():
        ci_str = f"[{res['wilson_ci_lower']:.1%}, {res['wilson_ci_upper']:.1%}]"
        print(
            f"{cond_name:20s} | "
            f"{res['win_rate']:8.1%} | "
            f"{ci_str:16s} | "
            f"{res['mean_kills']:6.2f} | "
            f"{res['mean_deaths']:6.2f}"
        )

    print("-" * 65)
    print(f"Headline In-Distribution Gap (Δ_val): {report.generalization_gap_val:+.4f}")
    print(f"Headline Out-of-Distribution Gap (Δ_ood): {report.generalization_gap_ood:+.4f}")

    out_path = Path(args.out) if args.out else ckpt_path.parent / "generalization.json"
    report.save(out_path)
    print(f"\nSaved generalization report to: {out_path}")


if __name__ == "__main__":
    main()
