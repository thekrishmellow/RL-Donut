"""Checkpoint management and disk pruning (Phase 5 Step 5.6)."""

from pathlib import Path
from typing import Any
import torch

from rl_donut.rl.model import ActorCritic


class CheckpointManager:
    """Saves checkpoints and automatically prunes old ones to conserve disk space."""

    def __init__(
        self,
        run_dir: Path | str,
        max_to_keep: int = 5,
        milestone_interval: int = 100,
    ):
        self.run_dir = Path(run_dir)
        self.max_to_keep = max_to_keep
        self.milestone_interval = milestone_interval
        self.recent_checkpoints: list[Path] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: ActorCritic,
        update: int,
        optimizer: torch.optim.Optimizer | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint and prune old non-milestone checkpoints."""
        ckpt_path = self.run_dir / f"ckpt_{update:04d}.pt"
        payload = {
            "model_state": model.state_dict(),
            "update": update,
            "obs_dim": model.obs_dim,
            "action_nvec": model.action_nvec,
            "cont_dim": model.cont_dim,
            "extra_meta": extra_meta or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()

        torch.save(payload, ckpt_path)

        is_milestone = (update % self.milestone_interval == 0)
        if not is_milestone:
            self.recent_checkpoints.append(ckpt_path)
            # Prune if exceeding max_to_keep
            while len(self.recent_checkpoints) > self.max_to_keep:
                oldest = self.recent_checkpoints.pop(0)
                if oldest.exists():
                    oldest.unlink()

        return ckpt_path

    @staticmethod
    def load(path: Path | str, model: ActorCritic) -> dict[str, Any]:
        """Load weights into model from checkpoint file."""
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        return ckpt

