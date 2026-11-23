"""Human matches, saved so a later lesson and a later launch can use them."""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from rl_donut.rl.clone import Demonstrations


@dataclass
class HumanDemos:
    obs: np.ndarray
    discrete: np.ndarray
    continuous: np.ndarray
    match_id: np.ndarray
    seed: np.ndarray
    player: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])


class DemoStore:
    """One npz file. Each teach appends a match. The file is replaced atomically."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> HumanDemos | None:
        if not self.path.exists():
            return None
        with np.load(self.path) as data:
            return HumanDemos(
                obs=np.array(data["obs"]),
                discrete=np.array(data["discrete"]),
                continuous=np.array(data["continuous"]),
                match_id=np.array(data["match_id"]),
                seed=np.array(data["seed"]),
                player=np.array(data["player"]),
            )

    def demonstrations(self) -> Demonstrations:
        saved = self.load()
        if saved is None or len(saved) == 0:
            raise ValueError("no saved matches")
        return Demonstrations(saved.obs, saved.discrete, saved.continuous, saved.match_id)

    def count(self) -> int:
        saved = self.load()
        return 0 if saved is None else len(saved)

    def matches(self) -> int:
        saved = self.load()
        if saved is None or len(saved) == 0:
            return 0
        return int(len(np.unique(saved.match_id)))

    def players(self) -> list[str]:
        saved = self.load()
        if saved is None or len(saved) == 0:
            return []
        seen: list[str] = []
        for name in saved.player.tolist():
            text = str(name)
            if text not in seen:
                seen.append(text)
        return seen

    def append(
        self,
        obs: np.ndarray,
        discrete: np.ndarray,
        continuous: np.ndarray,
        player: str,
        seed: int,
    ) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        discrete = np.asarray(discrete, dtype=np.int64)
        continuous = np.asarray(continuous, dtype=np.float32)
        if obs.ndim != 2 or len(obs) == 0:
            raise ValueError("a lesson needs at least one decision")
        if discrete.shape != (len(obs), 3) or continuous.shape != (len(obs), 1):
            raise ValueError("action shapes do not match the observations")
        previous = self.load()
        match_id = 0 if previous is None or len(previous) == 0 else int(previous.match_id.max()) + 1
        rows = len(obs)
        block = HumanDemos(
            obs=obs,
            discrete=discrete,
            continuous=continuous,
            match_id=np.full(rows, match_id, dtype=np.int64),
            seed=np.full(rows, int(seed), dtype=np.int64),
            player=np.full(rows, player, dtype="U40"),
        )
        if previous is not None and len(previous) > 0:
            block = HumanDemos(
                obs=np.concatenate([previous.obs, block.obs]),
                discrete=np.concatenate([previous.discrete, block.discrete]),
                continuous=np.concatenate([previous.continuous, block.continuous]),
                match_id=np.concatenate([previous.match_id, block.match_id]),
                seed=np.concatenate([previous.seed, block.seed]),
                player=np.concatenate([previous.player.astype("U40"), block.player]),
            )
        self._write(block)
        return match_id

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def _write(self, demos: HumanDemos) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".writing.npz")
        np.savez(
            temporary,
            obs=demos.obs,
            discrete=demos.discrete,
            continuous=demos.continuous,
            match_id=demos.match_id,
            seed=demos.seed,
            player=demos.player,
        )
        os.replace(temporary, self.path)
