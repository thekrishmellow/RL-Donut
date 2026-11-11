"""Generalization evaluation across splits and out-of-distribution (OOD) shifts (Phase 8).

Defines evaluation tiers:
1. In-distribution train seeds (seeds 0-9,999)
2. In-distribution held-out val seeds (seeds 10,000-10,999)
3. Out-of-Distribution (OOD):
   - Physics parameter shifts: gravity ±25%, jetpack thrust ±25%
   - Unseen OOD map seeds (seeds 30,000+)
Computes headline Generalization Gaps:
  Δ_val = win_rate_train - win_rate_val
  Δ_ood = win_rate_train - win_rate_ood
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np

from rl_donut.env.config import EngineConfig, PhysicsConfig
from rl_donut.mapgen.generator import GeneratedMap, generate_map
from rl_donut.rl.evaluate import EvalReport, evaluate_policy
from rl_donut.rl.stand_in import Policy


@dataclass
class GeneralizationReport:
    learner_name: str
    opponent_name: str
    train_win_rate: float
    val_win_rate: float
    generalization_gap_val: float
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


def get_ood_engine_configs() -> dict[str, EngineConfig]:
    """Returns physics shift configurations (±25% gravity and thrust)."""
    base_physics = PhysicsConfig()
    return {
        "gravity_low": EngineConfig(
            physics=PhysicsConfig(
                gravity=base_physics.gravity * 0.75,
                thrust=base_physics.thrust,
            )
        ),
        "gravity_high": EngineConfig(
            physics=PhysicsConfig(
                gravity=base_physics.gravity * 1.25,
                thrust=base_physics.thrust,
            )
        ),
        "thrust_low": EngineConfig(
            physics=PhysicsConfig(
                gravity=base_physics.gravity,
                thrust=base_physics.thrust * 0.75,
            )
        ),
        "thrust_high": EngineConfig(
            physics=PhysicsConfig(
                gravity=base_physics.gravity,
                thrust=base_physics.thrust * 1.25,
            )
        ),
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
    """Comprehensive generalization benchmark evaluating learner against an opponent."""
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

    ood_win_rates: list[float] = []

    # 3. Physics parameter shifts (±25% gravity and thrust)
    if include_physics_shifts:
        for name, cfg in get_ood_engine_configs().items():
            rep_shift = evaluate_policy(
                learner=learner,
                opponent=opponent,
                n_matches=n_matches_per_condition,
                split="val",
                seed=seed + 200,
                cfg=cfg,
                match_duration_seconds=match_duration_seconds,
            )
            conditions_results[name] = asdict(rep_shift)
            ood_win_rates.append(rep_shift.win_rate)

    # 4. Out-of-Distribution seed maps (30,000+)
    if include_ood_maps:
        ood_maps = build_ood_maps(start_seed=30_000, count=max(8, n_matches_per_condition // 2))
        rep_ood_maps = evaluate_policy(
            learner=learner,
            opponent=opponent,
            n_matches=n_matches_per_condition,
            seed=seed + 300,
            custom_maps=ood_maps,
            match_duration_seconds=match_duration_seconds,
        )
        conditions_results["ood_seeds_30k"] = asdict(rep_ood_maps)
        ood_win_rates.append(rep_ood_maps.win_rate)

    train_wr = rep_train.win_rate
    val_wr = rep_val.win_rate
    mean_ood_wr = float(np.mean(ood_win_rates)) if ood_win_rates else val_wr

    gap_val = compute_generalization_gap(train_wr, val_wr)
    gap_ood = compute_generalization_gap(train_wr, mean_ood_wr)

    return GeneralizationReport(
        learner_name=learner.name,
        opponent_name=opponent.name,
        train_win_rate=train_wr,
        val_win_rate=val_wr,
        generalization_gap_val=gap_val,
        mean_ood_win_rate=mean_ood_wr,
        generalization_gap_ood=gap_ood,
        conditions=conditions_results,
    )
