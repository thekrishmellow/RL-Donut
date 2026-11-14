"""Generate high-quality, lightweight figures (< 200 KB) for docs/REPORT.md."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from rl_donut.runs import load_metrics, load_json

FIGS_DIR = Path("docs/figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_comparison():
    runs_dir = Path("runs")
    pfsp = load_metrics(runs_dir / "sp-pfsp")
    adaptive = load_metrics(runs_dir / "sp-adaptive")
    sniper = load_metrics(runs_dir / "sp-sniper")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)

    # 1. Mean Return
    ax = axes[0]
    if pfsp:
        ax.plot([r["update"] for r in pfsp], [r["mean_return"] for r in pfsp], label="sp-pfsp (baseline)", color="#1f77b4")
    if adaptive:
        ax.plot([r["update"] for r in adaptive], [r["mean_return"] for r in adaptive], label="sp-adaptive", color="#2ca02c")
    if sniper:
        ax.plot([r["update"] for r in sniper], [r["mean_return"] for r in sniper], label="sp-sniper", color="#d62728")
    ax.set_title("Mean Episodic Return per Update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Policy Entropy
    ax = axes[1]
    if pfsp:
        ax.plot([r["update"] for r in pfsp], [r["entropy"] for r in pfsp], label="sp-pfsp", color="#1f77b4")
    if adaptive:
        ax.plot([r["update"] for r in adaptive], [r["entropy"] for r in adaptive], label="sp-adaptive", color="#2ca02c")
    if sniper:
        ax.plot([r["update"] for r in sniper], [r["entropy"] for r in sniper], label="sp-sniper", color="#d62728")
    ax.set_title("Action Distribution Entropy")
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Eval Damage vs RusherBot
    ax = axes[2]
    for r_data, label, col in [(pfsp, "sp-pfsp", "#1f77b4"), (adaptive, "sp-adaptive", "#2ca02c"), (sniper, "sp-sniper", "#d62728")]:
        updates = [r["update"] for r in r_data if "eval_damage" in r]
        damages = [r["eval_damage"] for r in r_data if "eval_damage" in r]
        if updates:
            ax.plot(updates, damages, marker="o", label=label, color=col)
    ax.set_title("Validation Damage vs RusherBot (60s)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Damage Dealt")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    out = FIGS_DIR / "training_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def plot_generalization():
    gen_file = Path("runs/sp-adaptive/generalization.json")
    if not gen_file.exists():
        print(f"Skipping {gen_file} (not found)")
        return
    with open(gen_file) as f:
        data = json.load(f)

    tiers = list(data["conditions"].keys())
    win_rates = [data["conditions"][t]["win_rate"] * 100 for t in tiers]
    ci_low = [data["conditions"][t]["wilson_ci_lower"] * 100 for t in tiers]
    ci_high = [data["conditions"][t]["wilson_ci_upper"] * 100 for t in tiers]
    yerr = [
        [max(0.0, w - l) for w, l in zip(win_rates, ci_low)],
        [max(0.0, h - w) for w, h in zip(win_rates, ci_high)],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bars = ax.bar(tiers, win_rates, yerr=yerr, capsize=5, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_title("Generalization Across 5 Tiers (sp-adaptive vs RusherBot)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(-2, 35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, h in zip(bars, ci_high):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"CI upper: {h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGS_DIR / "generalization_tiers.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def plot_benchmark_test():
    bench_file = Path("runs/final/benchmark.json")
    if not bench_file.exists():
        print(f"Skipping {bench_file} (not found)")
        return
    with open(bench_file) as f:
        data = json.load(f)

    pairings = data["pairings"]
    opponents = list(pairings.keys())
    ci_high = [pairings[o]["summary"]["ci_upper"] * 100 for o in opponents]
    kills = [pairings[o]["summary"]["mean_kills"] for o in opponents]
    deaths = [pairings[o]["summary"]["mean_deaths"] for o in opponents]
    airborne = [pairings[o]["summary"]["mean_airborne_s"] for o in opponents]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)

    # 1. Kills vs Deaths on Test Split
    ax = axes[0]
    x = np.arange(len(opponents))
    w = 0.35
    ax.bar(x - w/2, kills, width=w, label="Learner Kills", color="#2ca02c", edgecolor="black")
    ax.bar(x + w/2, deaths, width=w, label="Learner Deaths", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents)
    ax.set_title("Test Split: Kills vs Deaths per Match (60s)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Mobility Profile (Airborne Time)
    ax = axes[1]
    ax.bar(opponents, airborne, color="#9467bd", edgecolor="black", alpha=0.85)
    ax.set_title("Learner Airborne Time per 60s Match")
    ax.set_ylabel("Airborne Seconds")
    ax.set_ylim(0, 60)
    ax.axhline(60, color="gray", linestyle=":", label="Total Duration (60s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    out = FIGS_DIR / "benchmark_summary.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    plot_training_comparison()
    plot_generalization()
    plot_benchmark_test()
