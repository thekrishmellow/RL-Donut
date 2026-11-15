"""Train PPO against a static bot, widening the spawn gap as the policy starts winning.

The observation is the fair 330 numbers. A rung advances when the policy wins at least half
of 16 validation matches of 60 seconds on that rung.

    .venv/bin/python scripts/train_spawn_curriculum.py --updates 400
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_donut.env.match_env import SPEC_VERSION, MatchEnv
from rl_donut.mapgen import load_split
from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.rl.spawn_curriculum import RUNG_LADDER, SpawnCurriculum, maps_for_rung


def parse_args():
    parser = argparse.ArgumentParser(description="Spawn-distance curriculum against a static bot.")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-matches", type=int, default=16)
    parser.add_argument("--gate", type=float, default=0.5, help="Win rate that opens the next rung")
    parser.add_argument("--n-matches", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--map-limit", type=int, default=64, help="Maps kept per rung for training and for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=str, default="runs/spawn-curriculum")
    return parser.parse_args()


def make_runner(model, trainer, maps, n_matches, n_steps, total_steps, seed, step):
    env = MatchEnv(n_matches, team_size=1, split="train", seed=seed, maps=maps)
    runner = Runner(
        env=env,
        learner=model,
        opponent=StaticTargetBot(),
        trainer=trainer,
        n_steps=n_steps,
        total_training_steps=total_steps,
    )
    runner.global_step = step
    return runner


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Placing spawn rungs on train and val maps")
    train_maps = load_split("train")
    val_maps = load_split("val")
    train_decks = {rung.name: maps_for_rung(train_maps, rung, args.map_limit) for rung in RUNG_LADDER}
    val_decks = {rung.name: maps_for_rung(val_maps, rung, args.map_limit) for rung in RUNG_LADDER}
    for rung in RUNG_LADDER:
        print(f"  {rung.name}: {len(train_decks[rung.name])} train, {len(val_decks[rung.name])} val")

    curriculum = SpawnCurriculum(gate=args.gate)
    probe = MatchEnv(1, team_size=1, seed=args.seed, maps=train_decks[curriculum.rung.name][:1])
    device = torch.device("cpu")
    model = ActorCritic(probe.obs_dim, probe.action_nvec, hidden_dim=256, cont_dim=probe.action_cont_dim)
    trainer = PPOTrainer(model, PPOConfig(lr=args.lr), device=device)
    total_steps = args.updates * args.n_steps * args.n_matches
    runner = make_runner(
        model, trainer, train_decks[curriculum.rung.name],
        args.n_matches, args.n_steps, total_steps, args.seed, 0,
    )

    metrics = out / "metrics.jsonl"
    metrics.write_text("")
    best_rate = -1.0
    rung_best = -1.0
    history = []
    print(f"Rung {curriculum.rung.name}. Gate {args.gate:.0%} of {args.eval_matches} val matches.")

    for update in range(1, args.updates + 1):
        metrics_row = runner.step_rollout()
        record = {
            "update": update,
            "rung": curriculum.rung.name,
            "mean_return": round(metrics_row.mean_return, 3),
            "entropy": round(metrics_row.entropy, 4),
            "alpha": round(metrics_row.alpha, 3),
        }
        if update % args.eval_every == 0 or update == args.updates:
            report = evaluate_policy(
                LearnedPolicy(model, device=device),
                StaticTargetBot(),
                n_matches=args.eval_matches,
                split="val",
                seed=10_000,
                match_duration_seconds=60.0,
                custom_maps=val_decks[curriculum.rung.name],
            )
            record.update({
                "eval_wins": report.wins,
                "eval_matches": report.n_matches,
                "eval_win_rate": report.win_rate,
                "eval_kills": report.mean_kills,
                "eval_damage": report.mean_damage_dealt,
            })
            print(
                f"update {update:3d}  {curriculum.rung.name:8}  "
                f"wins {report.wins}/{report.n_matches}  "
                f"kills {report.mean_kills:.2f}  damage {report.mean_damage_dealt:.1f}"
            )
            rung_name = curriculum.rung.name
            meta = {"update": update, "rung": rung_name, "win_rate": report.win_rate}
            if report.win_rate > best_rate:
                best_rate = report.win_rate
                trainer.save_checkpoint(out / "ckpt_best.pt", extra_meta=meta)
            if report.win_rate > rung_best:
                rung_best = report.win_rate
                trainer.save_checkpoint(out / f"ckpt_{rung_name}.pt", extra_meta=meta)
            if curriculum.consider(report.win_rate):
                print(f"  gate cleared. Next rung: {curriculum.rung.name}")
                runner = make_runner(
                    model, trainer, train_decks[curriculum.rung.name],
                    args.n_matches, args.n_steps, total_steps, args.seed + curriculum.index,
                    runner.global_step,
                )
                rung_best = -1.0
        else:
            print(
                f"update {update:3d}  {curriculum.rung.name:8}  "
                f"return {metrics_row.mean_return:6.2f}  entropy {metrics_row.entropy:5.2f}"
            )
        history.append(record)
        with metrics.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "spec_version": SPEC_VERSION,
        "seed": args.seed,
        "gate": args.gate,
        "final_rung": curriculum.rung.name,
        "finished_official": curriculum.finished,
        "updates": args.updates,
    }
    (out / "curriculum.json").write_text(json.dumps(summary, indent=2))
    print(f"Stopped on {curriculum.rung.name}. Wrote {out / 'curriculum.json'}")


if __name__ == "__main__":
    main()
