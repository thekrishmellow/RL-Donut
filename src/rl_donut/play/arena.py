"""One person against the current network, then that network against the weights it just left.

The person is team A. The opponent flies the cave toward them and shoots when it
can see them. After the match the learned policy practices in the match engine, and
human controls are kept only when they do not push the body into rock.
"""

import json
from pathlib import Path

import numpy as np
import torch

from rl_donut.env.bots import RusherBot
from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.env.observe import OBS_DIM
from rl_donut.env.perception import Reaction
from rl_donut.env.policy import PolicyContext, team_slots
from rl_donut.mapgen.generator import GeneratedMap
from rl_donut.play.frames import grid_rows, scoreboard, snapshot
from rl_donut.play.physics_data import hold_in_air
from rl_donut.play.store import DemoStore
from rl_donut.rl.evaluate import legal_duration
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy
from rl_donut.rl.spawn_curriculum import with_gap_spawns


DUEL_NOTE = "Fighters start 6–16 tiles apart, already in sight."
OFFICIAL_NOTE = "Official spawns. The teams start hidden from each other."


def clean_player(name: str) -> str:
    text = " ".join(str(name).split())[:40]
    return text or "you"


def duel_deck(maps: list[GeneratedMap], limit: int = 32) -> tuple[list[GeneratedMap], str]:
    """Caves from `maps` with the two fighters placed where they can already see each other."""
    if limit < 1:
        raise ValueError("limit must be positive")
    placed: list[GeneratedMap] = []
    for game_map in maps:
        moved = with_gap_spawns(game_map, 6.0, 16.0, True)
        if moved is not None:
            placed.append(moved)
        if len(placed) >= limit:
            break
    if placed:
        return placed, DUEL_NOTE
    if not maps:
        raise ValueError("no maps")
    return [maps[0]], OFFICIAL_NOTE


def model_from_state(template: ActorCritic, state: dict) -> ActorCritic:
    hidden = template.actor_backbone[0].out_features
    copy = ActorCritic(template.obs_dim, template.action_nvec, hidden_dim=hidden, cont_dim=template.cont_dim)
    copy.load_state_dict(state)
    copy.eval()
    return copy


class HuntingPolicy:
    """Flies the cave's routes toward the player and shoots when a shot can land.

    The learned policy waits on its spawn. This is the search the match was missing:
    the same route planner the rusher uses, which stays in the air the map allows.
    """

    name = "hunter"

    def __init__(self):
        self.rusher = RusherBot(aim_sigma_deg=2.0, reaction=1)

    def reset(self) -> None:
        self.rusher._routes.clear()
        self.rusher.reaction = Reaction()

    def act(self, obs, ctx=None):
        if ctx is None:
            raise ValueError("the hunter reads the match")
        return self.rusher.act(obs, ctx)


class AirPolicy(LearnedPolicy):
    """The learned policy, with a step into rock turned off before the engine sees it."""

    def act(self, obs, ctx=None):
        discrete, continuous = super().act(obs, ctx)
        return hold_in_air(obs, discrete, continuous)


class HumanArena:
    """Holds the live network, the saved matches, and the match the browser is stepping."""

    def __init__(
        self,
        run_dir: Path | str,
        model: ActorCritic,
        maps: list[GeneratedMap],
        *,
        seconds: float = 20.0,
        placement: str | None = None,
        brain: str = "the agent",
        practice_updates: int = 8,
    ):
        if model.obs_dim != OBS_DIM or not model.cont_dim:
            raise ValueError("the play agent needs the 330-number observation and an aim angle")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model.eval()
        self.hunter = HuntingPolicy()
        self.store = DemoStore(self.run_dir / "demos.npz")
        self.seconds = legal_duration(seconds)
        if placement is None:
            self.maps, self.placement = duel_deck(list(maps))
        else:
            if not maps:
                raise ValueError("no maps")
            self.maps = list(maps)
            self.placement = placement
        self.brain = brain
        self.practice_updates = practice_updates
        self.rng = np.random.default_rng(0)
        self.cursor = 0
        self._lesson = 0
        self.env: MatchEnv | None = None
        self.obs: np.ndarray | None = None
        self.rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.playing = False
        self.saved = False
        self.taught = False
        base = self.run_dir / "ckpt_base.pt"
        if not base.exists():
            self._save_state(base, self.model.state_dict())

    def status(self) -> dict:
        return {
            "rows": self.store.count(),
            "matches": self.store.matches(),
            "players": self.store.players(),
            "playing": self.playing,
            "brain": self.brain,
            "seconds": self.seconds,
            "placement": self.placement,
        }

    def start(self, seed: int | None = None, seconds: float | None = None) -> dict:
        if self.rows and not self.taught:
            raise ValueError("teach this match before starting another")
        if seconds is not None:
            seconds = float(seconds)
            if not 2 <= seconds <= 60:
                raise ValueError("match length must be between 2 and 60 seconds")
            self.seconds = legal_duration(seconds)
        index = self.cursor if seed is None else int(seed)
        self.cursor = index + 1
        game_map = self.maps[index % len(self.maps)]
        self.env = self._make_env(game_map, index)
        self.obs = self.env.reset()
        self.rows = []
        self.playing = True
        self.saved = False
        self.taught = False
        self.hunter.reset()
        return self._opening()

    def step(self, move: int, jet: bool, shoot: bool, aim: float) -> dict:
        if not self.playing or self.env is None or self.obs is None:
            raise ValueError("start a match first")
        move = int(move)
        if move not in (-1, 0, 1):
            raise ValueError("move must be -1, 0, or 1")
        aim = float(aim)
        if not np.isfinite(aim):
            raise ValueError("aim must be a finite angle")
        human_obs = np.asarray(self.obs[0, 0], dtype=np.float32).copy()
        human_discrete = np.array([move + 1, int(bool(jet)), int(bool(shoot))], dtype=np.int64)
        human_continuous = np.array([aim], dtype=np.float32)
        self.rows.append((human_obs, human_discrete, human_continuous))

        discrete = np.zeros((1, self.env.n_agents, 3), dtype=np.int64)
        continuous = np.zeros((1, self.env.n_agents, 1), dtype=np.float32)
        discrete[0, 0] = human_discrete
        continuous[0, 0] = human_continuous
        slots = team_slots(self.env, 1)
        agent_discrete, agent_continuous = self.hunter.act(
            self.obs[0, 1:2], PolicyContext(self.env, slots, self.rng)
        )
        discrete[0, 1] = agent_discrete[0]
        continuous[0, 1] = agent_continuous[0]
        result = self.env.step(discrete, continuous, restart_done=False)
        self.obs = result.obs
        done = bool(result.done[0])
        if done:
            self.playing = False
        return {
            "frame": snapshot(self.env, result.events, done),
            "decisions": len(self.rows),
            "board": scoreboard(self.env),
        }

    def teach(self, player: str = "you") -> dict:
        if self.taught:
            raise ValueError("this match is already in the agent")
        if not self.rows and not self.saved:
            raise ValueError("play at least one decision before teaching")
        if self.env is None:
            raise ValueError("start a match first")
        self.playing = False
        player = clean_player(player)
        if not self.saved:
            obs = np.stack([row[0] for row in self.rows])
            discrete = np.stack([row[1] for row in self.rows])
            continuous = np.stack([row[2] for row in self.rows])
            self.store.append(obs, discrete, continuous, player, int(self.env.world.seeds[0]))
            self.saved = True
        human_board = scoreboard(self.env)
        before = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        self._lesson += 1
        from rl_donut.play.selfplay import practice_round
        report = practice_round(
            self.model,
            self.store.demonstrations(),
            self.maps,
            updates=self.practice_updates,
            seconds=self.seconds,
            seed=self._lesson,
        )
        self._save_state(self.run_dir / "ckpt_previous.pt", before)
        self._save_state(self.run_dir / "ckpt_latest.pt", self.model.state_dict())
        ai = self._ai_match(before, report.improved)
        if self.practice_updates:
            from rl_donut.play.physics_data import air_legal
            demos = self.store.demonstrations()
            kept = int(np.count_nonzero(air_legal(demos)))
            dropped = len(demos) - kept
            ai["message"] = (
                f"Practiced {self.practice_updates} updates inside the match engine, "
                f"against a moving copy. {dropped} saved controls pointed into rock and were left out. "
                + ai["message"]
            )
        self.taught = True
        self.rows = []
        payload = {
            "learned": report.as_dict(),
            "human": human_board,
            "ai": ai,
            "status": self.status(),
        }
        with (self.run_dir / "lessons.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "player": player,
                "decisions": report.latest_rows,
                "rows": report.rows,
                "logprob_before": report.logprob_before,
                "logprob_after": report.logprob_after,
                "improved": report.improved,
                "ai_score": ai["board"]["score"],
                "ai_damage": ai["board"]["damage"],
            }) + "\n")
        return payload

    def restore_original(self) -> dict:
        if self.playing or (self.rows and not self.taught):
            raise ValueError("teach or finish the current match first")
        path = self.run_dir / "ckpt_base.pt"
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._save_state(self.run_dir / "ckpt_latest.pt", self.model.state_dict())
        self.brain = "original brain"
        return self.status()

    def forget(self) -> dict:
        if self.playing or (self.rows and not self.taught):
            raise ValueError("teach or finish the current match first")
        self.store.clear()
        return self.status()

    def _opening(self) -> dict:
        assert self.env is not None
        grid = self.env.world.grids[0]
        return {
            "seed": int(self.env.world.seeds[0]),
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "grid": grid_rows(grid),
            "kill_y": float(self.env.world.kill_y[0]),
            "seconds": self.seconds,
            "placement": self.placement,
            "frame": snapshot(self.env, None, False),
            "decisions": 0,
            "board": scoreboard(self.env),
            "status": self.status(),
            "you_are": "blue",
            "opponent": "the current agent",
        }

    def _make_env(self, game_map: GeneratedMap, rng_seed: int) -> MatchEnv:
        cfg = EngineConfig(team_size=1, match=MatchConfig(duration=self.seconds))
        return MatchEnv(1, team_size=1, seed=int(rng_seed), cfg=cfg, maps=[game_map])

    def _ai_match(self, before_state: dict, improved: bool) -> dict:
        """The updated actor plays its previous weights. No person is in this match."""
        current = AirPolicy(self.model, name="after")
        previous = AirPolicy(model_from_state(self.model, before_state), name="before")
        index = self.cursor
        self.cursor += 1
        game_map = self.maps[index % len(self.maps)]
        env = self._make_env(game_map, 10_000 + index)
        obs = env.reset()
        rng = np.random.default_rng(10_000 + index)
        frames = []
        while True:
            discrete = np.zeros((1, env.n_agents, 3), dtype=np.int64)
            continuous = np.zeros((1, env.n_agents, 1), dtype=np.float32)
            for team, policy in ((0, current), (1, previous)):
                slots = team_slots(env, team)
                chosen, aim = policy.act(obs[0, team:team + 1], PolicyContext(env, slots, rng))
                discrete[0, team] = chosen[0]
                continuous[0, team] = aim[0]
            result = env.step(discrete, continuous, restart_done=False)
            frames.append(snapshot(env, result.events, bool(result.done[0])))
            obs = result.obs
            if result.done[0]:
                break
        if improved:
            message = (
                "Blue is the policy after this practice. Red is the policy from before it. "
                "Both move and shoot from what they learned. Nobody is playing."
            )
        else:
            message = (
                "Your match is saved. Both fighters are the learned policy. Nobody is playing."
            )
        return {
            "seed": int(env.world.seeds[0]),
            "placement": self.placement,
            "message": message,
            "improved": improved,
            "board": scoreboard(env),
            "frames": frames,
            "roles": ["after", "before"],
        }

    def _save_state(self, path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": state,
            "obs_dim": self.model.obs_dim,
            "action_nvec": self.model.action_nvec,
            "cont_dim": self.model.cont_dim,
            "extra_meta": {"source": "human-play"},
        }, path)


def load_actor(path: Path | None) -> ActorCritic:
    """The curriculum checkpoints use a width-256 actor. A missing file starts a new one."""
    if path is None:
        model = ActorCritic(OBS_DIM, (3, 2, 2), hidden_dim=256, cont_dim=1)
        model.eval()
        return model
    checkpoint = torch.load(path, map_location="cpu")
    hidden = 256
    model = ActorCritic(
        int(checkpoint["obs_dim"]),
        tuple(checkpoint["action_nvec"]),
        hidden_dim=hidden,
        cont_dim=int(checkpoint.get("cont_dim", 1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
