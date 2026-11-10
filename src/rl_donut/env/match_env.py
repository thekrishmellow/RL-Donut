"""The batched environment the trainer talks to (handoff contract, section 4).

`MatchEnv` runs `n_matches` matches side by side. Every step takes one action per agent
(discrete parts and the aim, in each agent's own frame), plays one decision step, and
returns observations, events, `done`, `final_obs` and per-match `info`. A finished match
restarts at once on the next map from a shuffled deck, so arrays always have the full batch
shape. Matches all last the same time and start together, so they also end together.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC, decode
from rl_donut.env.collision import on_ground
from rl_donut.env.config import EngineConfig
from rl_donut.env.match import MatchEvents, match_step
from rl_donut.env.observe import OBS_DIM, Observer
from rl_donut.env.state import TEAM_A, TEAM_B, World
from rl_donut.env.visibility import visibility
from rl_donut.mapgen import GeneratedMap, load_split

SPEC_VERSION = "0.3"
STATE_SLOTS = 4              # own team, then enemies, two each; empty slots in 1v1
STATE_AGENT = 12
STATE_DIM = STATE_SLOTS * STATE_AGENT + 2
HOVER_SPEED = 1.0            # tiles/s: airborne slower than this counts as hovering
SPAWN_CAMP_RADIUS = 8.0      # tiles from an enemy spawn
STATS = ("kills", "deaths", "falls", "damage_dealt", "damage_taken", "shots", "hits",
         "airborne_s", "hover_s", "fire_distance", "spawn_kills")


@dataclass
class StepResult:
    obs: np.ndarray          # (M, A, OBS_DIM) float32; for a match that just ended, the first obs of the next one
    events: MatchEvents      # what happened on this decision step
    done: np.ndarray         # (M,) bool: the match ended on this step (time up)
    final_obs: np.ndarray    # (M, A, OBS_DIM): the last obs of each match before any restart
    info: list[dict]         # per match; episode stats where done, else empty


class MapDeck:
    """Deals maps in a shuffled order, every map once per pass, reshuffling between passes."""

    def __init__(self, maps: list[GeneratedMap], rng: np.random.Generator):
        if not maps:
            raise ValueError("no maps to deal")
        self.maps, self.rng = maps, rng
        self.order, self.next = rng.permutation(len(maps)), 0

    def draw(self, n: int) -> list[GeneratedMap]:
        out = []
        for _ in range(n):
            if self.next == len(self.order):
                self.order, self.next = self.rng.permutation(len(self.maps)), 0
            out.append(self.maps[self.order[self.next]])
            self.next += 1
        return out


class MatchEnv:
    obs_dim = OBS_DIM
    action_nvec = DISCRETE_NVEC
    action_cont_dim = CONTINUOUS_DIM
    state_dim = STATE_DIM
    spec_version = SPEC_VERSION

    def __init__(self, n_matches: int, team_size: int = 1, split: str = "train", seed: int = 0,
                 map_filter=None, cfg: EngineConfig | None = None, maps: list[GeneratedMap] | None = None):
        cfg = cfg or EngineConfig(team_size=team_size)
        if cfg.team_size != team_size:
            raise ValueError("cfg.team_size and team_size disagree")
        maps = list(maps) if maps is not None else load_split(split)
        if map_filter is not None:
            maps = [m for m in maps if map_filter(m)]
        self.deck = MapDeck(maps, np.random.default_rng([seed, 1]))
        self.world = World.from_maps(self.deck.draw(n_matches), cfg, rng_seed=seed)
        self.observer = Observer(self.world)
        self.n_matches, self.n_agents = n_matches, cfg.n_agents
        self.team = self.world.team.copy()
        self.mirror = self.team == TEAM_B
        self.stats = {name: np.zeros((n_matches, self.n_agents)) for name in STATS}
        self.fired_near = np.zeros((n_matches, self.n_agents))    # shots fired with an enemy in sight
        self.vis = visibility(self.world)
        self.started = False

    def reset(self) -> np.ndarray:
        """Start every match (the first time on the maps dealt at construction, later on the
        next maps of the deck); returns obs (M, A, OBS_DIM)."""
        everything = np.arange(self.n_matches)
        if self.started:
            self.world.load_maps(everything, self.deck.draw(self.n_matches))
        else:
            self.world.reset(everything)
        self.observer.reset(everything)
        self._clear_stats(everything)
        self.vis = visibility(self.world)
        self.started = True
        return self.observer.observe(self.vis)

    def step(self, discrete: np.ndarray, continuous: np.ndarray) -> StepResult:
        if not self.started:
            raise RuntimeError("call reset() before step()")
        w = self.world
        c = decode(discrete, continuous, mirror=self.mirror)
        nearest = self._nearest_visible_enemy()
        ev = match_step(w, c.move, c.jet, c.aim, c.trigger)
        self.observer.record(ev)
        self._count(ev, nearest)
        self.vis = visibility(w)
        obs = self.observer.observe(self.vis)
        final_obs = obs.copy()
        info = [{} for _ in range(self.n_matches)]
        done = ev.done.copy()
        if done.any():
            ended = np.nonzero(done)[0]
            for m in ended:
                info[m] = self._episode_info(m, ev)
            w.load_maps(ended, self.deck.draw(len(ended)))
            self.observer.reset(ended)
            self._clear_stats(ended)
            self.vis = visibility(w)
            obs[ended] = self.observer.observe(self.vis)[ended]
        return StepResult(obs=obs, events=ev, done=done, final_obs=final_obs, info=info)

    def get_state(self) -> np.ndarray:
        """(M, 2, STATE_DIM) float32: the true state, not limited by sight, once per team in that
        team's frame (team B mirrored): own agents first, then enemies, then time left and score."""
        w, cfg = self.world, self.world.cfg
        M, W, H = self.n_matches, w.grids.shape[2], w.grids.shape[1]
        out = np.zeros((M, 2, STATE_DIM), dtype=np.float32)
        time_left = 1.0 - w.tick / cfg.ticks(cfg.match.duration)
        for t in (TEAM_A, TEAM_B):
            s = -1.0 if t == TEAM_B else 1.0
            x = W - w.x if t == TEAM_B else w.x
            own, foes = np.nonzero(self.team == t)[0], np.nonzero(self.team != t)[0]
            for slot, j in [(k, a) for k, a in enumerate(own)] + [(2 + k, a) for k, a in enumerate(foes)]:
                out[:, t, slot * STATE_AGENT:(slot + 1) * STATE_AGENT] = np.stack([
                    np.ones(M), w.alive[:, j], x[:, j] / W, w.y[:, j] / H,
                    s * w.vx[:, j] / cfg.physics.run_speed, w.vy[:, j] / cfg.physics.max_fall_speed,
                    w.health[:, j] / cfg.life.max_health, w.ammo[:, j] / cfg.rifle.magazine,
                    w.fuel[:, j] / cfg.physics.fuel_capacity, s * np.cos(w.aim[:, j]), np.sin(w.aim[:, j]),
                    w.respawn[:, j] / cfg.ticks(cfg.life.respawn_time)], axis=-1)
            out[:, t, -2] = time_left
            out[:, t, -1] = (w.score[:, t] - w.score[:, 1 - t]) / 10.0
        return out

    # --- episode stats ---

    def _clear_stats(self, matches: np.ndarray) -> None:
        for v in self.stats.values():
            v[matches] = 0.0
        self.fired_near[matches] = 0.0

    def _nearest_visible_enemy(self) -> np.ndarray:
        """(M, A) distance to the nearest enemy this agent sees now; inf when none."""
        w = self.world
        d = np.hypot(w.x[:, None, :] - w.x[:, :, None], w.y[:, None, :] - w.y[:, :, None])
        return np.where(self.vis.own, d, np.inf).min(axis=2)

    def _count(self, ev: MatchEvents, nearest: np.ndarray) -> None:
        w, cfg, st = self.world, self.world.cfg, self.stats
        seconds = cfg.match.decision_ticks / cfg.tick_rate
        for name, value in (("kills", ev.kills), ("deaths", ev.deaths), ("falls", ev.fell),
                            ("damage_dealt", ev.damage_dealt), ("damage_taken", ev.damage_taken),
                            ("shots", ev.shots), ("hits", ev.hits)):
            st[name] += value
        airborne = w.alive & ~on_ground(w)
        st["airborne_s"] += airborne * seconds
        st["hover_s"] += (airborne & (np.abs(w.vy) < HOVER_SPEED)) * seconds
        near = np.isfinite(nearest) & (ev.shots > 0)
        st["fire_distance"] += np.where(near, nearest, 0.0) * ev.shots
        self.fired_near += np.where(near, ev.shots, 0)
        for record in ev.death_log:
            m, scorer = record.death.match, record.scorer
            if scorer >= 0:
                enemy_spawns = w.spawns[m, 1 - self.team[scorer]]
                gap = np.hypot(enemy_spawns[:, 0] - w.x[m, scorer], enemy_spawns[:, 1] - w.y[m, scorer]).min()
                st["spawn_kills"][m, scorer] += gap <= SPAWN_CAMP_RADIUS

    def _episode_info(self, m: int, ev: MatchEvents) -> dict:
        per_agent = {name: v[m].copy() for name, v in self.stats.items()}
        fired = self.fired_near[m]
        per_agent["fire_distance"] = np.where(fired > 0, per_agent["fire_distance"] / np.maximum(fired, 1), np.nan)
        return {"map_seed": int(self.world.seeds[m]), "score": self.world.score[m].copy(),
                "outcome": ev.outcome[m].copy(), "agents": per_agent}
