"""Copy RusherBot onto the fair observation, then score it against a static bot.

The clone never receives the rusher's private enemy position. PPO is a later step:
this script stops at the validation matches.

    .venv/bin/python scripts/clone_rusher.py --matches 500 --eval-matches 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_donut.env.match_env import SPEC_VERSION
from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.clone import (
    ClonedPolicy,
    OffsetClonedPolicy,
    collect_rusher_demos,
    fit_clone,
    opening_weights,
    split_demonstrations,
)
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.rl.model import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="Behaviour-clone RusherBot on the fair observation.")
    parser.add_argument("--matches", type=int, default=500, help="Train-split matches to record")
    parser.add_argument("--seconds", type=float, default=60.0, help="Match length while recording")
    parser.add_argument("--batch-size", type=int, default=8, help="Matches played at once")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without a better val log-prob")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of recorded matches held out of the fit")
    parser.add_argument("--opening-steps", type=int, default=30, help="First decisions of each match to upweight. 30 is 2 seconds")
    parser.add_argument("--opening-weight", type=float, default=10.0, help="Loss weight on the opening. 1 weights every step equally")
    parser.add_argument("--enemy-offset", action="store_true", help="Append the true enemy offset. This is a diagnostic, not a fair agent")
    parser.add_argument("--eval-matches", type=int, default=16, help="Val matches against the static bot. 0 skips the check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/bc-rusher")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Recording {args.matches} matches of the rusher against a static bot on train maps")
    demos = collect_rusher_demos(
        args.matches,
        split="train",
        seed=args.seed,
        seconds=args.seconds,
        batch_size=args.batch_size,
        verbose=True,
        with_enemy_offset=args.enemy_offset,
    )
    train, val = split_demonstrations(demos, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Samples: {len(demos)}  train {len(train)}  held-out {len(val)}  obs {demos.obs.shape[-1]}")
    weights = opening_weights(train.match_id, args.opening_steps, args.opening_weight)
    print(
        f"Opening weight {args.opening_weight:g} on the first {args.opening_steps} steps "
        f"({int((weights > 1).sum())} of {len(train)} training rows)"
    )

    model = ActorCritic(demos.obs.shape[-1], (3, 2, 2), hidden_dim=256, cont_dim=1)
    fit = fit_clone(
        model,
        train,
        val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.minibatch,
        patience=args.patience,
        seed=args.seed,
        sample_weight=weights,
    )
    for row in fit.history:
        val_text = "none" if row["val_log_prob"] is None else f"{row['val_log_prob']:.3f}"
        print(f"epoch {row['epoch']:02d}  train nll {row['train_nll']:.3f}  val log-prob {val_text}")
    print(f"Restored epoch {fit.best_epoch}")

    ckpt = out / "ckpt_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "update": fit.best_epoch,
        "obs_dim": model.obs_dim,
        "action_nvec": model.action_nvec,
        "cont_dim": model.cont_dim,
        "extra_meta": {
            "kind": "behaviour_clone",
            "teacher": "rusher",
            "observation": "enemy_offset" if args.enemy_offset else "fair",
            "opponent": "static",
            "opening_steps": args.opening_steps,
            "opening_weight": args.opening_weight,
        },
    }, ckpt)

    report = None
    if args.eval_matches:
        print(f"Scoring the mode of the clone on {args.eval_matches} val matches against a static bot")
        policy = OffsetClonedPolicy(model) if args.enemy_offset else ClonedPolicy(model)
        report = evaluate_policy(
            policy,
            StaticTargetBot(),
            n_matches=args.eval_matches,
            split="val",
            seed=10_000,
            match_duration_seconds=60.0,
        )
        print(
            f"wins {report.wins}/{report.n_matches}  "
            f"kills {report.mean_kills:.2f}  damage {report.mean_damage_dealt:.1f}  "
            f"wilson [{report.wilson_ci_lower:.2f}, {report.wilson_ci_upper:.2f}]"
        )

    record = {
        "spec_version": SPEC_VERSION,
        "seed": args.seed,
        "matches": args.matches,
        "samples": len(demos),
        "train_samples": len(train),
        "val_samples": len(val),
        "obs_dim": int(demos.obs.shape[-1]),
        "observation": "enemy_offset" if args.enemy_offset else "fair",
        "opening_steps": args.opening_steps,
        "opening_weight": args.opening_weight,
        "best_epoch": fit.best_epoch,
        "best_val_log_prob": fit.best_val_log_prob,
        "history": fit.history,
        "checkpoint": str(ckpt),
        "eval": None if report is None else {
            "n_matches": report.n_matches,
            "wins": report.wins,
            "losses": report.losses,
            "draws": report.draws,
            "mean_kills": report.mean_kills,
            "mean_damage_dealt": report.mean_damage_dealt,
            "wilson_ci_lower": report.wilson_ci_lower,
            "wilson_ci_upper": report.wilson_ci_upper,
        },
    }
    (out / "clone.json").write_text(json.dumps(record, indent=2))
    print(f"Wrote {out / 'clone.json'}")


if __name__ == "__main__":
    main()
