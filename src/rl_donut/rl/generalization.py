"""Generalization evaluation across splits and out-of-distribution (OOD) shifts (Phase 8).

Defines evaluation tiers:
1. In-distribution train seeds (seeds 0-9,999)
2. In-distribution held-out val seeds (seeds 10,000-10,999)
3. Out-of-Distribution (OOD):
   - Physics scale shifts: gravity and thrust together, at 0.75x and 1.25x
   - Unseen OOD map seeds (seeds 30,000+)
Computes headline Generalization Gaps:
  Δ_val = win_rate_train - win_rate_val
  Δ_ood = win_rate_train - win_rate_ood
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from rl_donut.env.config import EngineConfig, PhysicsConfig
from rl_donut.mapgen.generator import GeneratedMap, generate_map
from rl_donut.env.policy import Policy
from rl_donut.rl.evaluate import EvalReport, evaluate_policy


@dataclass
class GeneralizationReport:
    learner_name: str
    opponent_name: str
    train_win_rate: float
    val_win_rate: float
    generalization_gap_val: float
    unseen_win_rate: float
    gap_unseen: float
    gap_unseen_ci: tuple[float, float]
    physics_win_rates: dict[str, float]
    gap_physics: float
    mean_ood_win_rate: float
    generalization_gap_ood: float
    conditions: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def difference_interval(
    p1: float, ci1: tuple[float, float], p2: float, ci2: tuple[float, float]
) -> tuple[float, float]:
    """Newcombe's hybrid score interval for the difference of two independent proportions."""
    l1, u1 = ci1
    l2, u2 = ci2
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(np.round(lower, 4)), float(np.round(upper, 4))


def get_physics_shifts() -> dict[str, EngineConfig]:
    """Tied gravity and thrust scales: 0.75x and 1.25x. Gravity and thrust move together."""
    return {
        "physics_0.75": EngineConfig(physics=PhysicsConfig(scale_range=(0.75, 0.75))),
        "physics_1.25": EngineConfig(physics=PhysicsConfig(scale_range=(1.25, 1.25))),
    }


def get_ood_engine_configs() -> dict[str, EngineConfig]:
    """Backward-compatible view with old four-name keys."""
    return {
        "gravity_low": EngineConfig(physics=PhysicsConfig(scale_range=(0.75, 0.75))),
        "gravity_high": EngineConfig(physics=PhysicsConfig(scale_range=(1.25, 1.25))),
        "thrust_low": EngineConfig(physics=PhysicsConfig(scale_range=(0.75, 0.75))),
        "thrust_high": EngineConfig(physics=PhysicsConfig(scale_range=(1.25, 1.25))),
    }


def build_ood_maps(start_seed: int = 30_000, count: int = 16) -> list[GeneratedMap]:
    """Generate on-the-fly unseen OOD cave maps from seeds 30,000+."""
    return [generate_map(seed) for seed in range(start_seed, start_seed + count)]


def compute_generalization_gap(train_win_rate: float, eval_win_rate: float) -> float:
    """Headline metric: Δ = win_rate_train - win_rate_eval."""
    return float(np.round(train_win_rate - eval_win_rate, 4))


def evaluate_generalization(
    learner: Policy,
    opponent: Policy,
    n_matches_per_condition: int = 20,
    seed: int = 42,
    include_physics_shifts: bool = True,
    include_ood_maps: bool = True,
    match_duration_seconds: float = 60.0,
) -> GeneralizationReport:
    """Comprehensive generalization benchmark evaluating learner against an opponent across 5 tiers."""
    conditions_results: dict[str, dict[str, Any]] = {}

    # 1. In-distribution train split
    rep_train = evaluate_policy(
        learner=learner,
        opponent=opponent,
        n_matches=n_matches_per_condition,
        split="train",
        seed=seed,
        match_duration_seconds=match_duration_seconds,
    )
    conditions_results["train"] = asdict(rep_train)

    # 2. In-distribution held-out val split
    rep_val = evaluate_policy(
        learner=learner,
        opponent=opponent,
        n_matches=n_matches_per_condition,
        split="val",
        seed=seed + 100,
        match_duration_seconds=match_duration_seconds,
    )
    conditions_results["val"] = asdict(rep_val)

    # 3. Tied physics scale, low and high. Seed offsets +200, +300.
    physics_win_rates: dict[str, float] = {}
    shift_configs = get_physics_shifts()
    if include_physics_shifts:
        offsets = {"physics_0.75": 200, "physics_1.25": 300}
        for name, cfg in shift_configs.items():
            rep_shift = evaluate_policy(
                learner=learner,
                opponent=opponent,
                n_matches=n_matches_per_condition,
                split="val",
                seed=seed + offsets[name],
                cfg=cfg,
                match_duration_seconds=match_duration_seconds,
            )
            conditions_results[name] = asdict(rep_shift)
            physics_win_rates[name] = rep_shift.win_rate

    # 4. Out-of-Distribution seed maps (30,000+). Seed offset +400.
    unseen_win_rate = rep_val.win_rate
    ci_unseen = (rep_val.wilson_ci_lower, rep_val.wilson_ci_upper)
    if include_ood_maps:
        ood_maps = build_ood_maps(start_seed=30_000, count=max(n_matches_per_condition, 8))
        rep_unseen = evaluate_policy(
            learner=learner,
            opponent=opponent,
            n_matches=n_matches_per_condition,
            seed=seed + 400,
            custom_maps=ood_maps,
            match_duration_seconds=match_duration_seconds,
        )
        conditions_results["unseen"] = asdict(rep_unseen)
        unseen_win_rate = rep_unseen.win_rate
        ci_unseen = (rep_unseen.wilson_ci_lower, rep_unseen.wilson_ci_upper)

    train_wr = rep_train.win_rate
    val_wr = rep_val.win_rate

    gap_val = compute_generalization_gap(train_wr, val_wr)
    gap_unseen = compute_generalization_gap(train_wr, unseen_win_rate)
    gap_unseen_ci = difference_interval(
        train_wr,
        (rep_train.wilson_ci_lower, rep_train.wilson_ci_upper),
        unseen_win_rate,
        ci_unseen,
    )

    phys_mean = float(np.mean(list(physics_win_rates.values()))) if physics_win_rates else val_wr
    gap_physics = compute_generalization_gap(train_wr, phys_mean)

    ood_pool = list(physics_win_rates.values()) + ([unseen_win_rate] if include_ood_maps else [])
    mean_ood_wr = float(np.mean(ood_pool)) if ood_pool else val_wr
    gap_ood = compute_generalization_gap(train_wr, mean_ood_wr)

    return GeneralizationReport(
        learner_name=learner.name,
        opponent_name=opponent.name,
        train_win_rate=train_wr,
        val_win_rate=val_wr,
        generalization_gap_val=gap_val,
        unseen_win_rate=unseen_win_rate,
        gap_unseen=gap_unseen,
        gap_unseen_ci=gap_unseen_ci,
        physics_win_rates=physics_win_rates,
        gap_physics=gap_physics,
        mean_ood_win_rate=mean_ood_wr,
        generalization_gap_ood=gap_ood,
        conditions=conditions_results,
    )
