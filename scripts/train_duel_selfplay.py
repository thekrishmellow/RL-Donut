"""Train a fighter against a moving copy of itself, on caves where the two can already see.

The opponent is refreshed from the learner at every update. It moves and shoots.
Aim is trained onto an enemy the observation says is visible, and saved human
matches contribute only the controls that do not push into rock.

    .venv/bin/python scripts/train_duel_selfplay.py --updates 160
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_donut.mapgen import load_split
from rl_donut.play.arena import load_actor
from rl_donut.play.selfplay import SelfOpponent, pretrain_aim, self_play_updates
from rl_donut.play.store import DemoStore
from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer
from rl_donut.rl.spawn_curriculum import RUNG_LADDER, maps_for_rung


def _save(path: Path, model: ActorCritic, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_nvec": model.action_nvec,
        "cont_dim": model.cont_dim,
        "extra_meta": meta,
    }, path)


def _eval(model, maps, opponent, seconds: float) -> dict:
    policy = LearnedPolicy(model, name="learner")
    report = evaluate_policy(
        policy, opponent, n_matches=8, custom_maps=maps, seed=11000,
        match_duration_seconds=seconds,
    )
    return {
        "wins": report.wins,
        "n": report.n_matches,
        "kills": report.mean_kills,
        "damage": report.mean_damage_dealt,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-play on in-sight duel caves.")
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--n-matches", type=int, default=6)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/spawn-curriculum-14/ckpt_tiles_8.pt"))
    parser.add_argument("--demos", type=Path, default=Path("runs/human-play/demos.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/duel-selfplay"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("Loading caves.", flush=True)
    train_all = load_split("train")
    val_all = load_split("val")
    start = args.checkpoint if args.checkpoint.exists() else None
    model = load_actor(start)
    demos = None
    if args.demos.exists():
        saved = DemoStore(args.demos).demonstrations()
        demos = saved
        print(f"Human rows on file: {len(saved)}", flush=True)
    close = next(rung for rung in RUNG_LADDER if rung.name == "tiles_8")
    print("Learning the aim from enemies the policy can see.", flush=True)
    error = pretrain_aim(model, maps_for_rung(train_all, close, 12), rounds=80, seed=args.seed)
    print(f"Aim median error after that pass: {error:.1f} deg", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "metrics.jsonl"
    best_score = -1.0
    trainer = PPOTrainer(model, PPOConfig(lr=1e-4))
    stages = [("tiles_8", 80), ("tiles_14", 60), ("tiles_20", 60), ("official", 80)]
    completed = 0

    def run_stage(name, count):
        nonlocal completed, best_score
        rung = next(item for item in RUNG_LADDER if item.name == name)
        train_maps = maps_for_rung(train_all, rung, 16)
        val_maps = maps_for_rung(val_all, rung, 8)
        print(f"Stage {name}: {count} updates against a moving copy.", flush=True)

        def on_update(live, row):
            nonlocal best_score
            update = completed + row["update"]
            row["update"] = update
            row["stage"] = name
            if row["update"] % args.eval_every == 0 or row["update"] == completed + count:
                mirror = SelfOpponent(live)
                mirror.pull(live)
                versus_self = _eval(live, val_maps, mirror, args.seconds)
                versus_still = _eval(live, val_maps, StaticTargetBot(), args.seconds)
                row["versus_self"] = versus_self
                row["versus_still"] = versus_still
                print(
                    f"{name} update {update:3d}  vs self {versus_self['wins']}/8  "
                    f"kills {versus_self['kills']:.2f}  dmg {versus_self['damage']:.0f}  "
                    f"vs still {versus_still['wins']}/8  kills {versus_still['kills']:.2f}  "
                    f"dmg {versus_still['damage']:.0f}",
                    flush=True,
                )
                score = versus_self["kills"] + versus_still["kills"]
                _save(args.out / f"ckpt_{update:04d}.pt", live, row)
                if score > best_score:
                    best_score = score
                    _save(args.out / "ckpt_best.pt", live, row)
            else:
                print(
                    f"{name} update {update:3d}  return {row['mean_return']:.3f}  "
                    f"aim loss {row['aim_loss']:.3f}",
                    flush=True,
                )
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

        self_play_updates(
            model, train_maps, count,
            n_matches=args.n_matches,
            n_steps=args.n_steps,
            seconds=args.seconds,
            seed=args.seed + completed,
            aim_coef=0.2,
            demos=demos,
            trainer=trainer,
            on_update=on_update,
        )
        completed += count

    for stage_name, stage_count in stages:
        run_stage(stage_name, stage_count)
    _save(args.out / "ckpt_latest.pt", model, {"updates": completed, "best_score": best_score})
    print(f"Best combined kills: {best_score:.2f}. Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
