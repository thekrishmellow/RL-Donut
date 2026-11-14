"""Elo rating system and offline round-robin tournament runner (Phase 6 Step 6.3)."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence
import numpy as np

from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.env.policy import Policy

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

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ratings": {k: round(v, 1) for k, v in sorted(self.ratings.items(), key=lambda x: -x[1])},
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


def compute_elo_ratings(
    matches: Sequence[MatchResult],
    anchor_name: str = ANCHOR_BOT_NAME,
    anchor_rating: float = ANCHOR_RATING,
    k_factor: float = 32.0,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute iterative Elo ratings from a list of match outcomes, normalized to anchor."""
    players = set()
    for m in matches:
        players.add(m.player_a)
        players.add(m.player_b)

    ratings = {p: 1500.0 for p in players}
    if anchor_name in ratings:
        ratings[anchor_name] = anchor_rating

    # Iterative refinement over all pairwise matches
    for _ in range(iterations):
        for m in matches:
            ra = ratings[m.player_a]
            rb = ratings[m.player_b]
            ea = compute_expected_score(ra, rb)
            eb = 1.0 - ea

            # Actual score: 1.0 per win, 0.5 per draw
            total = m.wins_a + m.wins_b + m.draws
            if total == 0:
                continue
            sa = (m.wins_a + 0.5 * m.draws) / total
            sb = (m.wins_b + 0.5 * m.draws) / total

            ratings[m.player_a] += k_factor * (sa - ea)
            ratings[m.player_b] += k_factor * (sb - eb)

    # Re-anchor to fixed anchor rating
    if anchor_name in ratings:
        shift = anchor_rating - ratings[anchor_name]
        ratings = {p: r + shift for p, r in ratings.items()}

    return ratings


def run_round_robin(
    policies: Sequence[Policy],
    n_matches_per_pair: int = 20,
    split: str = "val",
    seed: int = 50_000,
    anchor_name: str = ANCHOR_BOT_NAME,
    anchor_rating: float = ANCHOR_RATING,
    match_seconds: float = 60.0,
) -> EloReport:
    """Execute pairwise round-robin tournament with half-split side swapping."""
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
    return EloReport(
        ratings=ratings,
        matches=matches,
        anchor_name=anchor_name,
        anchor_rating=anchor_rating,
    )

