"""One drawable moment of a match, as plain numbers for the browser."""

import math

import numpy as np

from rl_donut.env.match import MatchEvents
from rl_donut.env.match_env import MatchEnv
from rl_donut.mapgen import line_of_sight


def _shot_end(grid, x: float, y: float, angle: float, limit: float, hit) -> tuple[float, float]:
    if hit is not None:
        return float(hit[0]), float(hit[1])
    end = lambda t: (x + t * math.cos(angle), y + t * math.sin(angle))
    if line_of_sight(grid, (x, y), end(limit)):
        return end(limit)
    lo, hi = 0.0, limit
    while hi - lo > 0.05:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if line_of_sight(grid, (x, y), end(mid)) else (lo, mid)
    return end(lo)


def grid_rows(grid: np.ndarray) -> list[str]:
    rock = np.asarray(grid)
    return ["".join("1" if cell else "0" for cell in row) for row in rock]


def scoreboard(env: MatchEnv) -> dict:
    stats = env.stats
    return {
        "score": [int(env.world.score[0, 0]), int(env.world.score[0, 1])],
        "kills": [int(stats["kills"][0, 0]), int(stats["kills"][0, 1])],
        "damage": [round(float(stats["damage_dealt"][0, 0]), 1), round(float(stats["damage_dealt"][0, 1]), 1)],
    }


def snapshot(env: MatchEnv, events: MatchEvents | None, done: bool) -> dict:
    world = env.world
    agents = []
    for i in range(env.n_agents):
        agents.append({
            "x": round(float(world.x[0, i]), 3),
            "y": round(float(world.y[0, i]), 3),
            "aim": round(float(world.aim[0, i]), 4),
            "alive": bool(world.alive[0, i]),
            "health": round(float(world.health[0, i]), 1),
            "fuel": round(float(world.fuel[0, i]), 1),
            "team": int(world.team[i]),
            "protect": bool(world.protection[0, i] > 0),
        })
    shots = []
    if events is not None:
        grid = world.grids[0]
        limit = float(world.cfg.rifle.range)
        for _, shot in events.shot_log:
            if shot.match != 0:
                continue
            x0 = float(world.x[0, shot.shooter])
            y0 = float(world.y[0, shot.shooter])
            x1, y1 = _shot_end(grid, x0, y0, float(shot.angle), limit, shot.hit)
            shots.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), int(shot.target >= 0)])
    tick = int(world.tick[0])
    duration = float(world.cfg.match.duration)
    return {
        "t": round(tick / world.cfg.tick_rate, 3),
        "time_left": round(max(0.0, duration - tick / world.cfg.tick_rate), 3),
        "done": bool(done),
        "score": [int(world.score[0, 0]), int(world.score[0, 1])],
        "agents": agents,
        "shots": shots,
    }
