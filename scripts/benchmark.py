"""Phase 10: Formal evaluation benchmark script for RL-Donut (Milestone 12).

Evaluates a policy checkpoint across standard opponents:
- Scripted bots: RandomBot, StaticTargetBot, TurretBot, RusherBot
- Historical snapshots: First admitted snapshot, Phase 5 best checkpoint (if provided)

Runs 200 matches per pairing on held-out maps (val for development, test once with --final).
Enforces the test split guard and records comprehensive combat/movement metrics.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC
from rl_donut.env.bots import RandomBot, RusherBot, StaticTargetBot, TurretBot
from rl_donut.env.observe import OBS_DIM
from rl_donut.env.policy import Policy
from rl_donut.mapgen import load_split
from rl_donut.rl.benchmark import difference_interval, guard_test_split, play_pairing, summarise
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run formal benchmark on RL-Donut checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Map split (val for dev, test for final)")
    parser.add_argument("--n-matches", type=int, default=200, help="Matches per pairing (must be even, default 200)")
    parser.add_argument("--seconds", type=float, default=60.0, help="Match duration in seconds (default 60.0)")
    parser.add_argument("--seed", type=int, default=90_000, help="Base seed (default 90,000)")
    parser.add_argument("--first-snapshot", type=str, default="", help="Optional path to first admitted snapshot (.pt)")
    parser.add_argument("--phase5-best", type=str, default="", help="Optional path to Phase 5 best checkpoint (.pt)")
    parser.add_argument("--final", action="store_true", help="Required flag when evaluating on the test split")
    parser.add_argument("--force", action="store_true", help="Force re-run even if test split was already used")
    parser.add_argument("--out", type=str, default="", help="Output json path (default: <run_dir>/benchmark.json)")
    return parser.parse_args()


def load_policy(path: str | Path, name: str | None = None) -> LearnedPolicy:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", OBS_DIM)
    action_nvec = ckpt.get("action_nvec", DISCRETE_NVEC)
    cont_dim = ckpt.get("cont_dim", CONTINUOUS_DIM)

    model = ActorCritic(obs_dim=obs_dim, action_nvec=action_nvec, cont_dim=cont_dim)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.load_state_dict(state_dict)
    return LearnedPolicy(model, name=name or p.stem)


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    # 1. Enforce test split guard
    guard_test_split(args.split, final=args.final, force=args.force)

    # 2. Load learner
    learner = load_policy(ckpt_path)

    # 3. Assemble opponent list
    opponents: list[Policy] = [
        RandomBot(),
        StaticTargetBot(),
        TurretBot(),
        RusherBot(),
    ]
    if args.first_snapshot:
        opponents.append(load_policy(args.first_snapshot, name="snapshot_first"))
    if args.phase5_best:
        opponents.append(load_policy(args.phase5_best, name="phase5_best"))

    # 4. Load maps
    maps = load_split(args.split)

    print(f"=== Starting RL-Donut Formal Benchmark ===")
    print(f"Learner: {learner.name} ({ckpt_path})")
    print(f"Split: {args.split} | Matches per pairing: {args.n_matches} | Duration: {args.seconds}s")
    print(f"Opponents: {[op.name for op in opponents]}")

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_hash = "unknown"

    pairings_data: dict[str, dict] = {}

    print("\n" + "=" * 95)
    print(f"{'Opponent':18s} | {'Win Rate':9s} | {'95% CI':16s} | {'Kills':6s} | {'Deaths':6s} | {'Acc':6s} | {'Eng Dist':8s}")
    print("-" * 95)

    start_time = time.time()
    for idx, opp in enumerate(opponents):
        pair_seed = args.seed + 1000 * idx
        records = play_pairing(
            learner=learner,
            opponent=opp,
            maps=maps,
            n_matches=args.n_matches,
            seed=pair_seed,
            seconds=args.seconds,
        )
        summary = summarise(records)
        pairings_data[opp.name] = {
            "summary": summary,
            "records": [asdict(r) for r in records],
        }

        ci_str = f"[{summary['ci_lower']:.1%}, {summary['ci_upper']:.1%}]"
        dist_str = f"{summary['mean_fire_distance']:.1f}" if summary["mean_fire_distance"] is not None else "N/A"
        print(
            f"{opp.name:18s} | "
            f"{summary['win_rate']:8.1%} | "
            f"{ci_str:16s} | "
            f"{summary['mean_kills']:6.2f} | "
            f"{summary['mean_deaths']:6.2f} | "
            f"{summary['accuracy']:6.1%} | "
            f"{dist_str:8s}"
        )

    total_time = time.time() - start_time
    print("=" * 95)
    print(f"Benchmark completed in {total_time:.1f}s ({total_time / len(opponents):.1f}s per pairing)")

    # 5. Output report
    if args.out:
        out_path = Path(args.out)
    elif args.final:
        out_path = Path("runs/final/benchmark.json")
    else:
        out_path = ckpt_path.parent / "benchmark.json"

    report_payload = {
        "checkpoint": str(ckpt_path),
        "git_commit": commit_hash,
        "split": args.split,
        "seed": args.seed,
        "seconds": args.seconds,
        "n_matches": args.n_matches,
        "pairings": pairings_data,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved benchmark report to: {out_path}")


if __name__ == "__main__":
    main()

