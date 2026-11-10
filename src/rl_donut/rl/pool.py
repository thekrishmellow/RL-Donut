"""Opponent pool manager and Prioritized Fictitious Self-Play (PFSP) sampling (Phase 6 Steps 6.1 & 6.2)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.rl.stand_in import Policy


@dataclass
class PoolMember:
    name: str
    policy: Policy
    update: int
    is_permanent: bool = False
    learner_wins: int = 0
    learner_losses: int = 0
    learner_draws: int = 0

    @property
    def total_matches(self) -> int:
        return self.learner_wins + self.learner_losses + self.learner_draws

    @property
    def learner_win_rate(self) -> float:
        """Win rate of the learner against this opponent."""
        if self.total_matches == 0:
            return 0.5
        return self.learner_wins / self.total_matches

    @property
    def opponent_win_rate(self) -> float:
        """Win rate of this opponent against the learner."""
        if self.total_matches == 0:
            return 0.5
        return self.learner_losses / self.total_matches


class OpponentPool:
    """Manages pool snapshots and PFSP opponent sampling."""

    def __init__(
        self,
        max_snapshots: int = 30,
        admission_win_rate: float = 0.55,
        permanent_bots: Sequence[Policy] | None = None,
    ):
        self.max_snapshots = max_snapshots
        self.admission_win_rate = admission_win_rate
        self.members: list[PoolMember] = []

        # Add permanent benchmark bots
        bots = permanent_bots if permanent_bots is not None else [StaticTargetBot(), RandomBot()]
        for bot in bots:
            self.members.append(
                PoolMember(name=bot.name, policy=bot, update=0, is_permanent=True)
            )

    @property
    def snapshots(self) -> list[PoolMember]:
        return [m for m in self.members if not m.is_permanent]

    @property
    def bots(self) -> list[PoolMember]:
        return [m for m in self.members if m.is_permanent]

    @property
    def latest_snapshot(self) -> PoolMember | None:
        snaps = self.snapshots
        return snaps[-1] if snaps else None

    def consider_admission(
        self,
        candidate_policy: Policy,
        update: int,
        n_eval_matches: int = 16,
        split: str = "val",
        seed: int = 30_000,
    ) -> bool:
        """Evaluate candidate against the latest snapshot. Adds to pool if win rate >= threshold."""
        latest = self.latest_snapshot
        if latest is None:
            # First snapshot is always admitted
            self._add_snapshot(candidate_policy, update)
            return True

        report = evaluate_policy(
            learner=candidate_policy,
            opponent=latest.policy,
            n_matches=n_eval_matches,
            split=split,
            seed=seed,
            match_duration_seconds=10.0,
        )

        if report.win_rate >= self.admission_win_rate:
            self._add_snapshot(candidate_policy, update)
            return True
        return False

    def _add_snapshot(self, policy: Policy, update: int) -> None:
        member = PoolMember(name=f"snapshot_{update:04d}", policy=policy, update=update)
        self.members.append(member)

        # Prune oldest snapshot if exceeding max_snapshots
        snaps = self.snapshots
        if len(snaps) > self.max_snapshots:
            oldest = snaps[0]
            self.members.remove(oldest)

    def sample_opponent(self, rng: np.random.Generator | None = None) -> Policy:
        """Sample opponent using: 50% latest self, 30% PFSP historical pool, 20% bots."""
        rng = rng or np.random.default_rng()
        r = rng.uniform(0.0, 1.0)

        latest = self.latest_snapshot
        snaps = self.snapshots

        # 1. 50% latest self (if available)
        if r < 0.50 and latest is not None:
            return latest.policy

        # 2. 30% PFSP from historical pool
        if r < 0.80 and snaps:
            # PFSP: weight toward opponents the learner struggles against
            # Weight = (1 - learner_win_rate)^2
            weights = np.array([(1.0 - m.learner_win_rate) ** 2 for m in snaps], dtype=np.float64)
            if weights.sum() == 0 or np.isnan(weights.sum()):
                weights = np.ones(len(snaps), dtype=np.float64)
            probs = weights / weights.sum()
            idx = rng.choice(len(snaps), p=probs)
            return snaps[idx].policy

        # 3. 20% permanent bots (or fallback if no snapshots exist)
        bots = self.bots
        if bots:
            idx = rng.choice(len(bots))
            return bots[idx].policy

        return latest.policy if latest is not None else StaticTargetBot()

    def record_match_result(self, opponent_name: str, learner_won: bool, learner_lost: bool) -> None:
        """Update match history against a specific opponent."""
        for m in self.members:
            if m.name == opponent_name:
                if learner_won:
                    m.learner_wins += 1
                elif learner_lost:
                    m.learner_losses += 1
                else:
                    m.learner_draws += 1
                break

