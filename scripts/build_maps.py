"""Generate, validate and cache every map of a split, then write a stats report.

    .venv/bin/python scripts/build_maps.py --split val --workers 8

Work is done in chunks saved under map_cache/build/<split>/, so an interrupted build resumes
where it stopped. The finished split is written to data/maps/<split>.npz and .json, with a
summary in data/maps/<split>.md and charts in outputs/map_stats_<split>.png.
"""

import argparse
import pickle
import time
from collections import Counter
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_donut.mapgen import AIR, MapConfig, MapGenerationError, generate_valid_map
from rl_donut.mapgen.cache import DATA_DIR, save_split
from rl_donut.mapgen.splits import SPLITS

BUILD_DIR = Path("map_cache/build")


def build_one(seed: int):
    try:
        # Layers are debug-only and large; drop them before sending maps between processes.
        return "ok", replace(generate_valid_map(seed), layers={})
    except MapGenerationError as e:
        return "skipped", {"seed": seed, "error": str(e)}


def build_chunk(path: Path, seeds: list[int], workers: int) -> None:
    with Pool(workers) as pool:
        results = pool.map(build_one, seeds, chunksize=4)
    path.write_bytes(pickle.dumps(results))


def report(name: str, maps, skipped, cfg: MapConfig) -> str:
    attempts = np.array([m.attempt + 1 for m in maps])
    reasons = Counter(r for m in maps for rej in m.rejections for r in rej)
    pits = Counter(m.params.pit_count for m in maps)
    rock = np.array([m.params.rock_fraction for m in maps])
    air = np.array([(m.grid == AIR).mean() for m in maps])
    platforms = np.array([len(m.platforms) for m in maps])
    stranded = np.array([m.stranded_fraction for m in maps])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    panels = [
        (attempts, "attempts per map", np.arange(1, attempts.max() + 2) - 0.5),
        (rock, "design rock fraction", 15),
        (air, "final air fraction", 20),
        (platforms, "platforms per map", np.arange(platforms.max() + 2) - 0.5),
        (stranded * 100, "unreachable floor (%)", 20),
    ]
    for ax, (values, title, bins) in zip(axes.flat, panels):
        ax.hist(values, bins=bins, color="#1f77b4")
        ax.set_title(title)
    ax = axes.flat[5]
    ax.bar([str(k) for k in sorted(pits)], [pits[k] for k in sorted(pits)], color="#d62728")
    ax.set_title("pit count")
    fig.suptitle(f"split '{name}': {len(maps)} maps")
    out = Path("outputs") / f"map_stats_{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=90)

    split = SPLITS[name]
    lines = [
        f"# Split `{name}`",
        "",
        f"Seeds {split.start:,}–{split.start + split.count - 1:,}: {len(maps):,} maps, "
        f"{len(skipped)} skipped (attempt cap {cfg.max_attempts}).",
        f"Charts: `outputs/map_stats_{name}.png` (not committed; rebuild to regenerate).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Attempts per map | mean {attempts.mean():.2f}, median {np.median(attempts):.0f}, "
        f"p99 {np.percentile(attempts, 99):.0f}, max {attempts.max()} |",
        f"| Rejection reasons (all failed attempts) | "
        + ", ".join(f"{k} {v}" for k, v in reasons.most_common()) + " |",
        f"| Pit count mix | " + ", ".join(f"{k}: {pits[k] / len(maps):.1%}" for k in sorted(pits)) + " |",
        f"| Design rock fraction | mean {rock.mean():.3f}, range {rock.min():.3f}–{rock.max():.3f} |",
        f"| Final air fraction | p10 {np.percentile(air, 10):.0%}, median {np.median(air):.0%}, "
        f"p90 {np.percentile(air, 90):.0%} |",
        f"| Platforms per map | mean {platforms.mean():.2f} |",
        f"| Fully reachable | {np.mean([m.fully_reachable for m in maps]):.1%} |",
        f"| Unreachable floor | mean {stranded.mean():.2%}, max {stranded.max():.2%} |",
    ]
    if skipped:
        lines += ["", "Skipped seeds: " + ", ".join(str(s["seed"]) for s in skipped)]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=200)
    args = parser.parse_args()

    cfg = MapConfig()
    seeds = list(SPLITS[args.split].seeds)
    build_dir = BUILD_DIR / args.split
    build_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    chunks = [seeds[i: i + args.chunk] for i in range(0, len(seeds), args.chunk)]
    for i, chunk in enumerate(chunks):
        path = build_dir / f"chunk_{chunk[0]:06d}.pkl"
        if path.exists():
            continue
        t = time.time()
        build_chunk(path, chunk, args.workers)
        print(f"chunk {i + 1}/{len(chunks)} (seeds {chunk[0]}–{chunk[-1]}) in {time.time() - t:.0f} s", flush=True)

    maps, skipped = [], []
    for chunk in chunks:
        for status, value in pickle.loads((build_dir / f"chunk_{chunk[0]:06d}.pkl").read_bytes()):
            (maps if status == "ok" else skipped).append(value)
    save_split(args.split, maps, skipped, cfg)
    summary = report(args.split, maps, skipped, cfg)
    (DATA_DIR / f"{args.split}.md").write_text(summary)
    print(summary)
    print(f"finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
