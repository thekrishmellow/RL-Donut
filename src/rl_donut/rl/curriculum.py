"""Curriculum manager gating policy advancement from static targets to hazard maps (Phase 5 Step 5.7)."""

from dataclasses import dataclass
from typing import Callable
from rl_donut.mapgen import GeneratedMap
from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.evaluate import evaluate_policy
from rl_donut.env.policy import Policy


@dataclass
class CurriculumStage:
    name: str
    opponent: Policy
    gate_win_rate: float
    map_filter: Callable[[GeneratedMap], bool] | None
    description: str


class CurriculumManager:
    """Manages training curriculum gates before self-play."""

    def __init__(self):
        self.stages = [
            CurriculumStage(
                name="stage_1_static_target_no_pits",
                opponent=StaticTargetBot(),
                gate_win_rate=0.90,
                map_filter=lambda m: len(m.pits) == 0,
                description="Hit and eliminate stationary targets on flat cave maps",
            ),
            CurriculumStage(
                name="stage_2_random_bot_no_pits",
                opponent=RandomBot(),
                gate_win_rate=0.80,
                map_filter=lambda m: len(m.pits) == 0,
                description="Chase and shoot erratic random-motion bots without pit hazards",
            ),
            CurriculumStage(
                name="stage_3_random_bot_all_maps",
                opponent=RandomBot(),
                gate_win_rate=0.75,
                map_filter=None,  # All maps, including bottom pits
                description="Navigate and fight across maps with bottom pits (fuel preservation)",
            ),
        ]
        self.current_stage_idx = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def is_curriculum_complete(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def evaluate_gate(self, policy: Policy, n_matches: int = 20, seed: int = 20_000) -> tuple[bool, float]:
        """Test whether the policy passes the current curriculum gate."""
        stage = self.current_stage
        report = evaluate_policy(
            learner=policy,
            opponent=stage.opponent,
            n_matches=n_matches,
            split="val",
            seed=seed,
            map_filter=stage.map_filter,
            match_duration_seconds=10.0,  # Fast eval matches
        )
        passed = report.win_rate >= stage.gate_win_rate
        if passed and not self.is_curriculum_complete:
            self.current_stage_idx += 1
        return passed, report.win_rate

