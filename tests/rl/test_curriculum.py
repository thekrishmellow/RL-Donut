from rl_donut.rl.bots import StaticTargetBot
from rl_donut.rl.curriculum import CurriculumManager


def test_curriculum_manager_stages():
    cm = CurriculumManager()
    assert len(cm.stages) == 3
    assert cm.current_stage.name == "stage_1_static_target_no_pits"
    assert cm.current_stage.gate_win_rate == 0.90
    assert not cm.is_curriculum_complete

    # Test gate evaluation
    bot = StaticTargetBot()
    passed, win_rate = cm.evaluate_gate(bot, n_matches=2)
    assert isinstance(passed, bool)
    assert 0.0 <= win_rate <= 1.0
