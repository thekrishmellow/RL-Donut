"""Play a cave in the browser, then teach the learned weights from that match.

You are the blue fighter. Red is the hunter: it flies the cave toward you and shoots
on line of sight. Your controls are saved with every other person's. End and teach
runs four practice updates. The bout afterwards is the new weights against the weights
from before the lesson, with nobody at the keyboard.

    .venv/bin/python scripts/play_human.py
    .venv/bin/python scripts/play_human.py --checkpoint runs/spawn-curriculum-14/ckpt_tiles_8.pt

The first launch saves that brain as runs/human-play/ckpt_base.pt. Later launches
continue from runs/human-play/ckpt_latest.pt. --restart loads the original brain again.
Saved matches stay in runs/human-play/demos.npz, which is git-ignored with the rest of runs/.
"""

import argparse
from pathlib import Path

from rl_donut.mapgen import load_split
from rl_donut.play.arena import HumanArena, load_actor
from rl_donut.play.server import serve


DEFAULT_CKPT = Path("runs/spawn-curriculum-14/ckpt_tiles_8.pt")


def resolve_checkpoint(run_dir: Path, explicit: Path | None, restart: bool) -> tuple[Path | None, str]:
    latest = run_dir / "ckpt_latest.pt"
    base = run_dir / "ckpt_base.pt"
    if restart and base.exists():
        return base, f"original brain at {base}"
    if latest.exists() and not restart:
        return latest, f"last lesson at {latest}"
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"checkpoint not found: {explicit}")
        return explicit, f"checkpoint {explicit}"
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT, f"checkpoint {DEFAULT_CKPT}"
    return None, "a fresh network, because no checkpoint was found"


def main():
    parser = argparse.ArgumentParser(description="Play the agent in the browser and teach it from your matches.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="weights to start from when no lesson is saved yet")
    parser.add_argument("--restart", action="store_true", help="load the original brain instead of the last lesson")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/human-play"))
    args = parser.parse_args()

    path, label = resolve_checkpoint(args.run_dir, args.checkpoint, args.restart)
    print(f"Loading caves and {label}.", flush=True)
    deck = list(load_split("val")[:32])
    placement = "Official spawns. The teams start where this cave placed them."
    print(f"{len(deck)} caves. {placement}", flush=True)
    arena = HumanArena(
        args.run_dir,
        load_actor(path),
        deck,
        seconds=args.seconds,
        placement=placement,
        brain=label,
        # A match is saved and filtered here. The self-play fit is scripts/train_duel_selfplay.py,
        # so a lesson in the browser cannot wipe a fighter that has already learned to shoot.
        practice_updates=4,
    )
    print(f"Saved matches: {arena.store.matches()}. Decisions on file: {arena.store.count()}.", flush=True)
    serve(arena, args.host, args.port)


if __name__ == "__main__":
    main()
