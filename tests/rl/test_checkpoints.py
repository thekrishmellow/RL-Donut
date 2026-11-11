from pathlib import Path
import pytest
import torch

from rl_donut.rl.checkpoints import CheckpointManager
from rl_donut.rl.model import ActorCritic


def test_checkpoint_pruning(tmp_path: Path):
    model = ActorCritic(obs_dim=8, action_nvec=(3, 2), hidden_dim=16)
    manager = CheckpointManager(run_dir=tmp_path, max_to_keep=2, milestone_interval=5)

    # Save 4 non-milestone checkpoints (1, 2, 3, 4)
    for u in [1, 2, 3, 4]:
        manager.save(model, update=u)

    files = [f.name for f in tmp_path.glob("ckpt_*.pt")]
    # Max to keep is 2, so only 3 and 4 should remain
    assert sorted(files) == ["ckpt_0003.pt", "ckpt_0004.pt"]

    # Save milestone 5 (should be preserved permanently)
    manager.save(model, update=5)
    # Save 6, 7
    manager.save(model, update=6)
    manager.save(model, update=7)

    files_after = [f.name for f in tmp_path.glob("ckpt_*.pt")]
    assert "ckpt_0005.pt" in files_after  # milestone preserved
    assert "ckpt_0006.pt" in files_after
    assert "ckpt_0007.pt" in files_after
    assert "ckpt_0003.pt" not in files_after  # pruned

