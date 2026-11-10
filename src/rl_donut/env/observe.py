"""What each agent observes: a fixed-length float32 vector per agent, in the agent's own frame.

Layout (`FEATURE_NAMES`, then the tile grid):

- self (20): position on the map and inside its tile, velocity, on-ground, fuel, health,
  ammo, reload and cooldown left, aim, alive, respawn and protection left, the direction the
  last hit came from and how long ago;
- 2 enemy slots (13 each), in agent order: present (the slot exists in this format), alive,
  seen by me, seen by my team, remembered (seen within `MEMORY_SECONDS`), relative position
  (live when the team sees it, last known when remembered, else 0), and only while seen:
  relative velocity, health, aim; then time since seen;
- 1 ally slot (9): present, alive, relative position, velocity, health, aim;
- match (2): time left, score difference;
- tile grid: `CROP_H` x `CROP_W` tiles centred on the agent's tile, row by row, 1 = rock,
  0 = air, -1 = below the map (falling there kills).

Team B sees a mirror image (x -> width - x, aim a -> pi - a, tile grid flipped left-right),
so both sides look the same to one policy; `actions.decode(..., mirror=...)` flips its
actions back. Numbers are divided by fixed physical scales so they stay roughly in [-1, 1].
"""

import numpy as np

from rl_donut.env.collision import on_ground
from rl_donut.env.match import MatchEvents
from rl_donut.env.state import TEAM_B, World
from rl_donut.env.visibility import Visibility

ENEMY_SLOTS = 2
MEMORY_SECONDS = 10.0       # sightings and hits older than this are forgotten
CROP_W, CROP_H = 21, 13
POSITION_SCALE = 30.0       # rifle range, in tiles
SCORE_SCALE = 10.0
BELOW_MAP = -1.0

SELF_FEATURES = ("map_x", "map_y", "tile_x", "tile_y", "vx", "vy", "on_ground", "fuel", "health", "ammo",
                 "reload", "cooldown", "aim_cos", "aim_sin", "alive", "respawn", "protection",
                 "hit_from_cos", "hit_from_sin", "since_hit")
ENEMY_FEATURES = ("present", "alive", "seen_by_me", "seen_by_team", "remembered", "dx", "dy", "vx", "vy",
                  "health", "aim_cos", "aim_sin", "since_seen")
ALLY_FEATURES = ("present", "alive", "dx", "dy", "vx", "vy", "health", "aim_cos", "aim_sin")
MATCH_FEATURES = ("time_left", "score_diff")
FEATURE_NAMES = (SELF_FEATURES
                 + tuple(f"enemy{k}_{f}" for k in range(ENEMY_SLOTS) for f in ENEMY_FEATURES)
                 + tuple(f"ally_{f}" for f in ALLY_FEATURES) + MATCH_FEATURES)
OBS_DIM = len(FEATURE_NAMES) + CROP_W * CROP_H


class Observer:
    """Builds observations for one `World` and keeps the per-match memory they need."""

    def __init__(self, world: World):
        self.world = world
        cfg = world.cfg
        M, A = world.x.shape
        self.memory = cfg.ticks(MEMORY_SECONDS)
        self.seen_x, self.seen_y = np.zeros((M, A, A)), np.zeros((M, A, A))
        self.since_seen = np.full((M, A, A), self.memory)       # [m, i, j]: i's team last saw j
        self.hit_from = np.zeros((M, A))                         # direction the last hit came from
        self.since_hit = np.full((M, A), self.memory)
        self.pad = max(CROP_W, CROP_H) // 2 + 1
        self.tiles = np.zeros((M, world.grids.shape[1] + 2 * self.pad, world.grids.shape[2] + 2 * self.pad),
                              dtype=np.float32)
        self.mirror = world.team == TEAM_B
        team_size = cfg.team_size
        self.enemies = np.full((A, ENEMY_SLOTS), -1)
        self.ally = np.full(A, -1)
        for i in range(A):
            foes = np.nonzero(world.team != world.team[i])[0]
            self.enemies[i, :len(foes)] = foes
            mates = [k for k in np.nonzero(world.team == world.team[i])[0] if k != i]
            if mates:
                self.ally[i] = mates[0]
        assert team_size <= ENEMY_SLOTS
        self.reset()

    def reset(self, matches: np.ndarray | None = None) -> None:
        """Forget everything for the given matches (default: all) and reload their tiles; call
        after `World.reset` or `World.load_maps`."""
        matches = np.arange(self.world.n_matches) if matches is None else np.asarray(matches)
        self.since_seen[matches] = self.memory
        self.since_hit[matches] = self.memory
        self.hit_from[matches] = 0.0
        p, (H, W) = self.pad, self.world.grids.shape[1:]
        self.tiles[matches] = 1.0
        self.tiles[matches, p + H:, :] = BELOW_MAP
        self.tiles[matches, p:p + H, p:p + W] = self.world.grids[matches]

    def record(self, events: MatchEvents) -> None:
        """Age the memory by one decision step and note the hits of that step."""
        step = self.world.cfg.match.decision_ticks
        self.since_seen = np.minimum(self.since_seen + step, self.memory)
        self.since_hit = np.minimum(self.since_hit + step, self.memory)
        for _, shot in events.shot_log:
            if shot.target >= 0:
                self.hit_from[shot.match, shot.target] = shot.angle + np.pi
                self.since_hit[shot.match, shot.target] = 0

    def observe(self, vis: Visibility) -> np.ndarray:
        """(M, A, OBS_DIM) float32 for the world's current state; also stores current sightings."""
        w, cfg = self.world, self.world.cfg
        M, A = w.x.shape
        H, W = w.grids.shape[1:]
        seen = vis.team
        self.seen_x = np.where(seen, w.x[:, None, :], self.seen_x)
        self.seen_y = np.where(seen, w.y[:, None, :], self.seen_y)
        self.since_seen = np.where(seen, 0, self.since_seen)

        s = np.where(self.mirror, -1.0, 1.0)                     # sign of x components, (A,)
        mx = np.where(self.mirror, W - w.x, w.x)
        speed_x, speed_y = cfg.physics.run_speed, cfg.physics.max_fall_speed
        hit_recent = self.since_hit < self.memory
        self_part = [
            mx / W, w.y / H, mx - np.floor(mx), w.y - np.floor(w.y),
            s * w.vx / speed_x, w.vy / speed_y, on_ground(w),
            w.fuel / cfg.physics.fuel_capacity, w.health / cfg.life.max_health, w.ammo / cfg.rifle.magazine,
            w.reload / cfg.reload_ticks, w.cooldown / cfg.shot_interval_ticks,
            s * np.cos(w.aim), np.sin(w.aim),
            w.alive, w.respawn / cfg.ticks(cfg.life.respawn_time), w.protection / cfg.ticks(cfg.life.protection_time),
            np.where(hit_recent, s * np.cos(self.hit_from), 0.0), np.where(hit_recent, np.sin(self.hit_from), 0.0),
            self.since_hit / self.memory,
        ]

        m = np.arange(M)[:, None]
        i = np.arange(A)[None, :]
        enemy_parts = []
        for k in range(ENEMY_SLOTS):
            j = self.enemies[:, k]
            present = np.broadcast_to(j >= 0, (M, A))
            jj = np.maximum(j, 0)[None, :]
            live = seen[m, i, jj] & present
            known = (self.since_seen[m, i, jj] < self.memory) & present
            px = np.where(live, w.x[m, jj], self.seen_x[m, i, jj])
            py = np.where(live, w.y[m, jj], self.seen_y[m, i, jj])
            enemy_parts += [
                present, w.alive[m, jj] & present, vis.own[m, i, jj] & present, live, known,
                np.where(known, s * (px - w.x) / POSITION_SCALE, 0.0), np.where(known, (py - w.y) / POSITION_SCALE, 0.0),
                np.where(live, s * (w.vx[m, jj] - w.vx) / speed_x, 0.0), np.where(live, (w.vy[m, jj] - w.vy) / speed_y, 0.0),
                np.where(live, w.health[m, jj] / cfg.life.max_health, 0.0),
                np.where(live, s * np.cos(w.aim[m, jj]), 0.0), np.where(live, np.sin(w.aim[m, jj]), 0.0),
                np.where(present, self.since_seen[m, i, jj] / self.memory, 1.0),
            ]

        j = self.ally
        present = np.broadcast_to(j >= 0, (M, A))
        jj = np.maximum(j, 0)[None, :]
        on = lambda v: np.where(present, v, 0.0)
        ally_part = [
            present, w.alive[m, jj] & present,
            on(s * (w.x[m, jj] - w.x) / POSITION_SCALE), on((w.y[m, jj] - w.y) / POSITION_SCALE),
            on(s * w.vx[m, jj] / speed_x), on(w.vy[m, jj] / speed_y), on(w.health[m, jj] / cfg.life.max_health),
            on(s * np.cos(w.aim[m, jj])), on(np.sin(w.aim[m, jj])),
        ]

        own_team = w.team[None, :]
        match_part = [
            np.broadcast_to(1.0 - w.tick[:, None] / cfg.ticks(cfg.match.duration), (M, A)),
            (w.score[m, own_team] - w.score[m, 1 - own_team]) / SCORE_SCALE,
        ]

        features = np.stack([np.broadcast_to(np.asarray(v, dtype=np.float32), (M, A))
                             for v in self_part + enemy_parts + ally_part + match_part], axis=-1)
        return np.concatenate([features, self.crop().reshape(M, A, -1)], axis=-1)

    def crop(self) -> np.ndarray:
        """(M, A, CROP_H, CROP_W) tiles around each agent's tile, mirrored for team B."""
        w, p = self.world, self.pad
        W = w.grids.shape[2]
        # Columns are chosen in the agent's own frame and mapped back (column c' of the mirror
        # image is column W - 1 - c'); flipping a window centred on floor(x) instead would be
        # one column off whenever x is a whole number, as it is on every spawn.
        mx = np.where(self.mirror, W - w.x, w.x)
        cols = np.floor(mx).astype(np.int64)[..., None, None] + np.arange(CROP_W)[None, :] - CROP_W // 2
        cols = np.where(self.mirror[:, None, None], W - 1 - cols, cols) + p
        rows = np.floor(w.y).astype(np.int64)[..., None, None] + p + np.arange(CROP_H)[:, None] - CROP_H // 2
        rows = np.clip(rows, 0, self.tiles.shape[1] - 1)
        cols = np.clip(cols, 0, self.tiles.shape[2] - 1)
        return self.tiles[np.arange(w.n_matches)[:, None, None, None], rows, cols]
