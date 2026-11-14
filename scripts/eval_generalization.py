"""Generalization evaluation CLI script for RL-Donut (Phase 8).

Evaluates a policy checkpoint across five tiers:
- In-distribution train split (seeds 0-9,999)
- In-distribution held-out val split (seeds 10,000-10,999)
- Out-of-Distribution physics shift: tied scale 0.75x (gravity & thrust 0.75x)
- Out-of-Distribution physics shift: tied scale 1.25x (gravity & thrust 1.25x)
- Out-of-Distribution unseen seeds: 30,000+ generated on the fly

Computes and reports headline Generalization Gaps with confidence intervals.
"""

import argparse
import json
from pathlib import Path
import torch

from rl_donut.env.bots import RusherBot, TurretBot
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.generalization import evaluate_generalization
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generalization gap for RL-Donut.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "static", "turret", "rusher"],
        help="Opponent bot",
    )
    parser.add_argument("--n-matches", type=int, default=20, help="Matches per condition")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--out", type=str, default="", help="Output json path (default: <ckpt_dir>/generalization.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ActorCritic(
        ckpt["obs_dim"], ckpt["action_nvec"], hidden_dim=256, cont_dim=ckpt.get("cont_dim", 1)
    )
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    learner = LearnedPolicy(model, name=ckpt_path.stem)
    if args.opponent == "random":
        opponent = RandomBot()
    elif args.opponent == "static":
        opponent = StaticTargetBot()
    elif args.opponent == "turret":
        opponent = TurretBot()
    elif args.opponent == "rusher":
        opponent = RusherBot()
    else:
        raise ValueError(f"Unknown opponent: {args.opponent}")

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
    print(f"In-Distribution Gap (Δ_val): {report.generalization_gap_val:+.4f}")
    unseen_ci_str = f"[{report.gap_unseen_ci[0]:+.4f}, {report.gap_unseen_ci[1]:+.4f}]"
    print(f"Unseen Seeds Gap (Δ_unseen):  {report.gap_unseen:+.4f} (95% CI: {unseen_ci_str})")
    print(f"Physics Shift Gap (Δ_physics):{report.gap_physics:+.4f}")
    print(f"Mean OOD Gap (Δ_ood):         {report.generalization_gap_ood:+.4f}")

    out_path = Path(args.out) if args.out else ckpt_path.parent / "generalization.json"
    report.save(out_path)
    print(f"\nSaved generalization report to: {out_path}")


if __name__ == "__main__":
    main()
