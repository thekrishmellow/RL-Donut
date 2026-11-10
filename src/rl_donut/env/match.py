"""The match loop: one decision step for every match in a batch.

Each agent's action is held for `decision_ticks` ticks; every tick runs physics, then the
rifles, then health, death and respawn. Kills are scored per team; when the clock runs
out the team with more kills wins and equal kills is a draw.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.life import Death, life_tick
from rl_donut.env.physics import physics_tick
from rl_donut.env.rifle import Shot, fire
from rl_donut.env.state import World


@dataclass(frozen=True)
class DeathRecord:
    tick: int
    death: Death
    scorer: int             # agent credited with the kill, -1 = nobody


@dataclass
class MatchEvents:
    """What happened on one decision step. Arrays are (M, A) unless noted."""
    damage_dealt: np.ndarray
    damage_taken: np.ndarray
    kills: np.ndarray       # kills credited, pit falls included
    deaths: np.ndarray
    fell: np.ndarray        # died by falling through a pit
    shots: np.ndarray
    hits: np.ndarray        # shots that hit an enemy (protected or not)
    done: np.ndarray        # (M,) the match ended on this step
    outcome: np.ndarray     # (M, 2) +1 win, -1 loss, 0 draw per team where done; 0 elsewhere
    shot_log: list[tuple[int, Shot]]
    death_log: list[DeathRecord]
    respawn_log: list[tuple[int, int, int]]   # (tick, match, agent)


def scorer_of(world: World, d: Death) -> int:
    if d.killer >= 0:
        return d.killer
    window = world.cfg.ticks(world.cfg.match.fall_credit_window)
    if d.fell and d.last_damager >= 0 and d.ticks_since_damage <= window:
        return d.last_damager
    return -1


def match_step(world: World, move: np.ndarray, jet: np.ndarray, aim: np.ndarray,
               trigger: np.ndarray) -> MatchEvents:
    """Play one decision step in every match. Every match must still be running; a match
    that ends here must be reset (or given a new map) before the next step."""
    cfg = world.cfg
    end = cfg.ticks(cfg.match.duration)
    if (world.tick >= end).any():
        raise ValueError("a match is over; reset it or load a new map first")
    shape = world.alive.shape
    ev = MatchEvents(*(np.zeros(shape) for _ in range(2)), *(np.zeros(shape, dtype=np.int64) for _ in range(2)),
                     np.zeros(shape, dtype=bool), *(np.zeros(shape, dtype=np.int64) for _ in range(2)),
                     np.zeros(world.n_matches, dtype=bool), np.zeros((world.n_matches, 2), dtype=np.int64),
                     [], [], [])

    for _ in range(cfg.match.decision_ticks):
        physics_tick(world, move, jet)
        shots = fire(world, aim, trigger)
        life = life_tick(world, shots)
        for s in shots:
            ev.shots[s.match, s.shooter] += 1
            ev.hits[s.match, s.shooter] += s.target >= 0
            ev.shot_log.append((int(world.tick[s.match]), s))
        ev.damage_dealt += life.dealt
        ev.damage_taken += life.taken
        for d in life.deaths:
            scorer = scorer_of(world, d)
            ev.deaths[d.match, d.agent] += 1
            ev.fell[d.match, d.agent] |= d.fell
            if scorer >= 0:
                ev.kills[d.match, scorer] += 1
                world.score[d.match, world.team[scorer]] += 1
            ev.death_log.append(DeathRecord(int(world.tick[d.match]), d, scorer))
        ev.respawn_log += [(int(world.tick[m]), m, j) for m, j in life.respawns]
        world.tick += 1

    ev.done[...] = world.tick >= end
    lead = np.sign(world.score[:, 0] - world.score[:, 1])
    ev.outcome[ev.done] = np.stack([lead, -lead], axis=1)[ev.done]
    return ev
