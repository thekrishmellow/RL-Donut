"""Phase 10: Benchmark harness, statistics, and test split guard.

Provides:
- MatchRecord: Comprehensive per-match statistics for combat, navigation, and survival.
- play_pairing: 50/50 side-swapped evaluation harness using MatchEnv.
- summarise: Aggregate stats, win rates with Wilson 95% intervals, accuracy, and distances.
- difference_interval: Newcombe's hybrid score interval for differences of two proportions.
- guard_test_split: Prevents accidental contamination of the held-out test split.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Sequence
import numpy as np

from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.env.policy import Policy, play_matches
from rl_donut.mapgen import GeneratedMap
from rl_donut.rl.evaluate import wilson_score_interval


@dataclass
class MatchRecord:
    map_seed: int
    learner_team: int
    outcome: int                # +1 / 0 / -1 for the learner
    score: tuple[int, int]      # learner, opponent
    kills: int
    deaths: int
    falls: int
    damage_dealt: float
    damage_taken: float
    shots: int
    hits: int
    airborne_s: float
    hover_s: float
    fire_distance: float        # mean distance to the nearest seen enemy when firing; nan if never
    spawn_kills: int


def _record_from_info(info: dict, learner_team: int, env: MatchEnv) -> MatchRecord:
    map_seed = int(info["map_seed"])
    score_arr = info["score"]
    score = (int(score_arr[learner_team]), int(score_arr[1 - learner_team]))
    if score[0] > score[1]:
        outcome = 1
    elif score[0] < score[1]:
        outcome = -1
    else:
        outcome = 0

    agents_stat = info["agents"]
    learner_slots = [a for a in range(env.n_agents) if env.team[a] == learner_team]

    kills = int(sum(agents_stat["kills"][a] for a in learner_slots))
    deaths = int(sum(agents_stat["deaths"][a] for a in learner_slots))
    falls = int(sum(agents_stat["falls"][a] for a in learner_slots))
    damage_dealt = float(sum(agents_stat["damage_dealt"][a] for a in learner_slots))
    damage_taken = float(sum(agents_stat["damage_taken"][a] for a in learner_slots))
    shots = int(sum(agents_stat["shots"][a] for a in learner_slots))
    hits = int(sum(agents_stat["hits"][a] for a in learner_slots))
    airborne_s = float(np.mean([agents_stat["airborne_s"][a] for a in learner_slots]))
    hover_s = float(np.mean([agents_stat["hover_s"][a] for a in learner_slots]))
    spawn_kills = int(sum(agents_stat["spawn_kills"][a] for a in learner_slots))

    dist_vals = [agents_stat["fire_distance"][a] for a in learner_slots if np.isfinite(agents_stat["fire_distance"][a])]
    fire_distance = float(np.mean(dist_vals)) if dist_vals else float(np.nan)

    return MatchRecord(
        map_seed=map_seed,
        learner_team=learner_team,
        outcome=outcome,
        score=score,
        kills=kills,
        deaths=deaths,
        falls=falls,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        shots=shots,
        hits=hits,
        airborne_s=airborne_s,
        hover_s=hover_s,
        fire_distance=fire_distance,
        spawn_kills=spawn_kills,
    )


def play_pairing(
    learner: Policy,
    opponent: Policy,
    *,
    maps: Sequence[GeneratedMap],
    n_matches: int,
    seed: int,
    seconds: float = 60.0,
    team_size: int = 1,
    cfg: EngineConfig | None = None,
) -> list[MatchRecord]:
    """Play n_matches between learner and opponent, exactly half with learner on Team A and half on Team B."""
    if n_matches % 2 != 0:
        raise ValueError(f"n_matches must be even for half-split side swapping, got {n_matches}")

    half = n_matches // 2
    if cfg is None:
        cfg = EngineConfig(team_size=team_size, match=MatchConfig(duration=seconds))
    elif cfg.match.duration != seconds:
        cfg = EngineConfig(
            team_size=team_size,
            hitbox_w=cfg.hitbox_w,
            hitbox_h=cfg.hitbox_h,
            physics=cfg.physics,
            rifle=cfg.rifle,
            life=cfg.life,
            match=MatchConfig(
                duration=seconds,
                decision_ticks=cfg.match.decision_ticks,
                fall_credit_window=cfg.match.fall_credit_window,
            ),
            map=cfg.map,
        )

    records: list[MatchRecord] = []

    # 1. First half: learner is Team A (learner_team=0), opponent is Team B
    env_a = MatchEnv(n_matches=half, maps=list(maps), cfg=cfg, seed=seed)
    infos_a = play_matches(env_a, team_a=learner, team_b=opponent, seed=seed)
    for info in infos_a:
        records.append(_record_from_info(info, learner_team=0, env=env_a))

    # 2. Second half: opponent is Team A, learner is Team B (learner_team=1)
    env_b = MatchEnv(n_matches=half, maps=list(maps), cfg=cfg, seed=seed + 1000)
    infos_b = play_matches(env_b, team_a=opponent, team_b=learner, seed=seed + 1000)
    for info in infos_b:
        records.append(_record_from_info(info, learner_team=1, env=env_b))

    return records


def summarise(records: list[MatchRecord]) -> dict[str, Any]:
    """Summarise a collection of MatchRecord objects into summary statistics and Wilson CI."""
    n = len(records)
    if n == 0:
        return {
            "n_matches": 0, "wins": 0, "draws": 0, "losses": 0,
            "win_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
        }

    wins = sum(1 for r in records if r.outcome > 0)
    draws = sum(1 for r in records if r.outcome == 0)
    losses = sum(1 for r in records if r.outcome < 0)
    win_rate = wins / n
    ci_low, ci_high = wilson_score_interval(wins, n)

    shots = sum(r.shots for r in records)
    hits = sum(r.hits for r in records)
    accuracy = (hits / shots) if shots > 0 else 0.0

    valid_dists = [r.fire_distance for r in records if np.isfinite(r.fire_distance)]
    mean_fire_dist = float(np.mean(valid_dists)) if valid_dists else float(np.nan)

    return {
        "n_matches": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": float(np.round(win_rate, 4)),
        "ci_lower": float(np.round(ci_low, 4)),
        "ci_upper": float(np.round(ci_high, 4)),
        "mean_kills": float(np.round(np.mean([r.kills for r in records]), 2)),
        "mean_deaths": float(np.round(np.mean([r.deaths for r in records]), 2)),
        "mean_falls": float(np.round(np.mean([r.falls for r in records]), 2)),
        "mean_damage_dealt": float(np.round(np.mean([r.damage_dealt for r in records]), 2)),
        "mean_damage_taken": float(np.round(np.mean([r.damage_taken for r in records]), 2)),
        "accuracy": float(np.round(accuracy, 4)),
        "mean_shots": float(np.round(shots / n, 2)),
        "mean_hits": float(np.round(hits / n, 2)),
        "mean_airborne_s": float(np.round(np.mean([r.airborne_s for r in records]), 2)),
        "mean_hover_s": float(np.round(np.mean([r.hover_s for r in records]), 2)),
        "mean_fire_distance": float(np.round(mean_fire_dist, 2)) if np.isfinite(mean_fire_dist) else None,
        "mean_spawn_kills": float(np.round(np.mean([r.spawn_kills for r in records]), 2)),
    }


def difference_interval(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score confidence interval for difference of two proportions (p1 - p2)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 0.0
    p1 = w1 / n1
    p2 = w2 / n2
    d = p1 - p2
    l1, u1 = wilson_score_interval(w1, n1)
    l2, u2 = wilson_score_interval(w2, n2)

    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(np.round(lower, 4)), float(np.round(upper, 4))


def guard_test_split(split: str, final: bool = False, force: bool = False, marker_dir: Path | str = "runs/final") -> None:
    """Enforce strict protection of the test split (decision D10.2)."""
    if split != "test":
        return

    if not final:
        raise RuntimeError(
            "Evaluation on 'test' split requires the --final flag. "
            "The test split must only be evaluated once for the final report."
        )

    marker_path = Path(marker_dir) / "TEST_USED"
    if marker_path.exists() and not force:
        raise RuntimeError(
            f"Test split already used! Found marker file: {marker_path}. "
            "Pass --force to override if this is intentional."
        )

    # Write marker record
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_hash = "unknown"

    content = (
        f"git_commit: {commit_hash}\n"
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"machine: {platform.node()} ({platform.system()} {platform.machine()})\n"
        f"forced: {force}\n"
    )
    marker_path.write_text(content)

