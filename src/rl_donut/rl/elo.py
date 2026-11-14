"""Elo rating system and Bradley-Terry MLE tournament solver (Phases 6 & 10)."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np

from rl_donut.env.policy import Policy
from rl_donut.rl.evaluate import evaluate_policy

ANCHOR_BOT_NAME = "random_bot"
ANCHOR_RATING = 1200.0


@dataclass
class MatchResult:
    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int


@dataclass
class EloReport:
    ratings: dict[str, float]
    matches: list[MatchResult]
    anchor_name: str
    anchor_rating: float
    ratings_ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ratings": {k: round(v, 1) for k, v in sorted(self.ratings.items(), key=lambda x: -x[1])},
            "ratings_ci": {k: [round(ci[0], 1), round(ci[1], 1)] for k, ci in self.ratings_ci.items()} if self.ratings_ci else {},
            "anchor": {"name": self.anchor_name, "rating": self.anchor_rating},
            "matches": [
                {
                    "player_a": m.player_a,
                    "player_b": m.player_b,
                    "wins_a": m.wins_a,
                    "wins_b": m.wins_b,
                    "draws": m.draws,
                }
                for m in self.matches
            ],
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of player A against player B under the logistic Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _fit_bradley_terry(
    players: list[str],
    W: np.ndarray,
    iterations: int = 200,
    eps: float = 1e-4,
) -> np.ndarray:
    """Fit Bradley-Terry latent strengths theta using Hunter's MM algorithm."""
    K = len(players)
    if K <= 1:
        return np.zeros(K, dtype=np.float64)

    # Tiny regularizer for numerical stability when a player has 0 wins
    w_wins = np.sum(W, axis=1) + eps
    N = W + W.T + 2 * eps

    gamma = np.ones(K, dtype=np.float64)

    for _ in range(iterations):
        denom = np.zeros(K, dtype=np.float64)
        for i in range(K):
            mask = (np.arange(K) != i) & (N[i] > 0)
            if mask.any():
                denom[i] = np.sum(N[i, mask] / (gamma[i] + gamma[mask]))
            else:
                denom[i] = 1.0

        new_gamma = w_wins / np.maximum(denom, 1e-12)
        geo_mean = np.exp(np.mean(np.log(np.maximum(new_gamma, 1e-12))))
        new_gamma /= max(geo_mean, 1e-12)

        if np.max(np.abs(new_gamma - gamma)) < 1e-7:
            gamma = new_gamma
            break
        gamma = new_gamma

    return np.log(np.maximum(gamma, 1e-12))


def compute_elo_ratings(
    matches: Sequence[MatchResult],
    anchor_name: str = ANCHOR_BOT_NAME,
    anchor_rating: float = ANCHOR_RATING,
    k_factor: float = 32.0,
    iterations: int = 500,
) -> dict[str, float]:
    """Compute Bradley-Terry maximum-likelihood Elo ratings from match outcomes, normalized to anchor."""
    players = sorted(list({m.player_a for m in matches} | {m.player_b for m in matches}))
    if not players:
        return {}

    name_to_idx = {name: i for i, name in enumerate(players)}
    K = len(players)
    W = np.zeros((K, K), dtype=np.float64)

    for m in matches:
        i, j = name_to_idx[m.player_a], name_to_idx[m.player_b]
        W[i, j] += m.wins_a + 0.5 * m.draws
        W[j, i] += m.wins_b + 0.5 * m.draws

    theta = _fit_bradley_terry(players, W, iterations=iterations)

    scale = 400.0 / math.log(10.0)
    theta_anchor = theta[name_to_idx[anchor_name]] if anchor_name in name_to_idx else 0.0

    ratings = {}
    for i, name in enumerate(players):
        ratings[name] = float(np.round(anchor_rating + scale * (theta[i] - theta_anchor), 1))

    return ratings


def bootstrap_elo_ci(
    matches: Sequence[MatchResult],
    anchor_name: str = ANCHOR_BOT_NAME,
    anchor_rating: float = ANCHOR_RATING,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Bradley-Terry Elo ratings."""
    players = sorted(list({m.player_a for m in matches} | {m.player_b for m in matches}))
    if not players:
        return {}

    rng = np.random.default_rng(seed)
    boot_ratings = {p: [] for p in players}

    for _ in range(n_bootstraps):
        resampled_matches = []
        for m in matches:
            tot = m.wins_a + m.wins_b + m.draws
            if tot == 0:
                continue
            probs = np.array([m.wins_a, m.wins_b, m.draws], dtype=np.float64) / tot
            counts = rng.multinomial(tot, probs)
            resampled_matches.append(
                MatchResult(
                    player_a=m.player_a,
                    player_b=m.player_b,
                    wins_a=int(counts[0]),
                    wins_b=int(counts[1]),
                    draws=int(counts[2]),
                )
            )

        r = compute_elo_ratings(resampled_matches, anchor_name=anchor_name, anchor_rating=anchor_rating)
        for p in players:
            boot_ratings[p].append(r.get(p, anchor_rating))

    ci_dict = {}
    for p in players:
        vals = boot_ratings[p]
        if vals:
            ci_low = float(np.round(np.percentile(vals, 2.5), 1))
            ci_high = float(np.round(np.percentile(vals, 97.5), 1))
            ci_dict[p] = (ci_low, ci_high)
        else:
            ci_dict[p] = (anchor_rating, anchor_rating)

    return ci_dict


def run_round_robin(
    policies: Sequence[Policy],
    n_matches_per_pair: int = 20,
    split: str = "val",
    seed: int = 50_000,
    anchor_name: str = ANCHOR_BOT_NAME,
    anchor_rating: float = ANCHOR_RATING,
    match_seconds: float = 60.0,
    bootstrap_ci: bool = True,
) -> EloReport:
    """Execute pairwise round-robin tournament with half-split side swapping and Bradley-Terry fit."""
    n = len(policies)
    matches: list[MatchResult] = []

    for i in range(n):
        for j in range(i + 1, n):
            pol_a = policies[i]
            pol_b = policies[j]
            report = evaluate_policy(
                learner=pol_a,
                opponent=pol_b,
                n_matches=n_matches_per_pair,
                split=split,
                seed=seed + i * 100 + j,
                match_duration_seconds=match_seconds,
            )
            matches.append(
                MatchResult(
                    player_a=pol_a.name,
                    player_b=pol_b.name,
                    wins_a=report.wins,
                    wins_b=report.losses,
                    draws=report.draws,
                )
            )

    ratings = compute_elo_ratings(matches, anchor_name=anchor_name, anchor_rating=anchor_rating)
    ratings_ci = bootstrap_elo_ci(matches, anchor_name=anchor_name, anchor_rating=anchor_rating) if bootstrap_ci else {}

    return EloReport(
        ratings=ratings,
        matches=matches,
        anchor_name=anchor_name,
        anchor_rating=anchor_rating,
        ratings_ci=ratings_ci,
    )
