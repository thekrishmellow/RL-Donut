"""Adaptive matchmaking and dynamic difficulty adjustment (Phase 7 Step 7.2).

Maintains learner win rate inside the optimal learning zone (50% - 60%)
by dynamically tuning opponent selection probabilities from the pool.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np

from rl_donut.rl.pool import OpponentPool
from rl_donut.rl.stand_in import Policy


@dataclass(frozen=True)
class AdaptiveConfig:
    target_win_rate_min: float = 0.50
    target_win_rate_max: float = 0.60
    window_size: int = 20
    difficulty_shift_rate: float = 0.05


class AdaptiveMatchmaker:
    """Selects opponents from the OpponentPool to maintain target learner win rate."""

    def __init__(
        self,
        pool: OpponentPool,
        cfg: AdaptiveConfig | None = None,
    ):
        self.pool = pool
        self.cfg = cfg or AdaptiveConfig()
        self.recent_outcomes = deque(maxlen=self.cfg.window_size)
        # Difficulty bias: 0.0 = balanced, +1.0 = maximum difficulty, -1.0 = easiest
        self.difficulty_bias: float = 0.0

    @property
    def current_win_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.55
        return float(np.mean(self.recent_outcomes))

    def record_match_outcome(self, won: bool, lost: bool) -> None:
        """Record outcome (1.0 for win, 0.0 for loss, 0.5 for draw) and update difficulty."""
        score = 1.0 if won else (0.0 if lost else 0.5)
        self.recent_outcomes.append(score)

        if len(self.recent_outcomes) >= 5:
            wr = self.current_win_rate
            if wr > self.cfg.target_win_rate_max:
                # Learner is dominating: increase difficulty
                self.difficulty_bias = min(1.0, self.difficulty_bias + self.cfg.difficulty_shift_rate)
            elif wr < self.cfg.target_win_rate_min:
                # Learner is struggling: decrease difficulty
                self.difficulty_bias = max(-1.0, self.difficulty_bias - self.cfg.difficulty_shift_rate)

    def select_opponent(self, rng: np.random.Generator | None = None) -> Policy:
        """Select an opponent biased by current difficulty level."""
        rng = rng or np.random.default_rng()
        snaps = self.pool.snapshots
        bots = self.pool.bots

        if not snaps:
            return rng.choice(bots).policy if bots else self.pool.sample_opponent(rng)

        # Baseline probabilities: 50% latest self, 30% PFSP historical, 20% bots
        p_latest = 0.50 + 0.30 * max(0.0, self.difficulty_bias)
        p_bots = 0.20 + 0.30 * max(0.0, -self.difficulty_bias)
        p_historical = max(0.10, 1.0 - p_latest - p_bots)

        total_p = p_latest + p_historical + p_bots
        probs = [p_latest / total_p, p_historical / total_p, p_bots / total_p]

        category = rng.choice(["latest", "historical", "bot"], p=probs)
        latest = self.pool.latest_snapshot

        if category == "latest" and latest is not None:
            return latest.policy
        elif category == "bot" and bots:
            idx = rng.choice(len(bots))
            return bots[idx].policy
        elif snaps:
            # PFSP historical selection
            weights = np.array([(1.0 - m.learner_win_rate) ** 2 for m in snaps], dtype=np.float64)
            if weights.sum() == 0:
                weights = np.ones(len(snaps), dtype=np.float64)
            p = weights / weights.sum()
            idx = rng.choice(len(snaps), p=p)
            return snaps[idx].policy

        return self.pool.sample_opponent(rng)

