"""Health, death and respawn.

`life_tick` runs once per tick, after the rifles have fired, in this order:

1. Firing ends the shooter's spawn protection.
2. Every hit fired on this tick is applied at once, so two agents can kill each other.
3. Agents at 0 health, or below the kill line after falling through a pit, die.
4. Living agents regenerate once enough time has passed without damage.
5. Spawn protection counts down.
6. Dead agents count down to respawn; when the timer ends they reappear on one of their
   team's spawns that no living enemy can see.
"""

import math
from dataclasses import dataclass

import numpy as np

from rl_donut.env.rifle import Shot
from rl_donut.env.state import World
from rl_donut.mapgen import line_of_sight


@dataclass(frozen=True)
class Death:
    match: int
    agent: int
    killer: int             # agent whose hit emptied the health bar; -1 for a fall
    fell: bool              # died below the kill line
    last_damager: int       # last agent to damage the victim, -1 = none since respawn
    ticks_since_damage: int  # how long before death that damage was (who gets a fall kill is G4)


@dataclass(frozen=True)
class LifeEvents:
    dealt: np.ndarray       # (M, A) damage dealt to enemies on this tick
    taken: np.ndarray       # (M, A) damage taken on this tick
    deaths: list[Death]
    respawns: list[tuple[int, int]]   # (match, agent) of agents that reappeared


def pick_spawn(world: World, m: int, agent: int) -> tuple[float, float]:
    """A random spawn of the agent's team that no living enemy can see (centre to centre,
    at any distance); if every spawn is seen, the one farthest from the nearest enemy."""
    spawns = world.spawns[m, world.team[agent]]
    enemies = np.nonzero(world.alive[m] & (world.team != world.team[agent]))[0]
    rng = world.rngs[m]
    if not len(enemies):
        return tuple(spawns[rng.integers(len(spawns))])
    eyes = [(float(world.x[m, j]), float(world.y[m, j])) for j in enemies]
    safe = [k for k, (sx, sy) in enumerate(spawns)
            if not any(line_of_sight(world.grids[m], eye, (float(sx), float(sy))) for eye in eyes)]
    if safe:
        return tuple(spawns[safe[rng.integers(len(safe))]])
    nearest = [min(math.hypot(sx - ex, sy - ey) for ex, ey in eyes) for sx, sy in spawns]
    return tuple(spawns[int(np.argmax(nearest))])


def life_tick(world: World, shots: list[Shot]) -> LifeEvents:
    cfg, life = world.cfg, world.cfg.life
    dealt, taken = np.zeros_like(world.health), np.zeros_like(world.health)

    for s in shots:
        world.protection[s.match, s.shooter] = 0

    # Damage this tick per (match, victim) per shooter; the killer is whoever dealt the most,
    # ties broken at random so agent order never matters.
    by_shooter: dict[tuple[int, int], dict[int, float]] = {}
    for s in shots:
        if s.target >= 0 and world.protection[s.match, s.target] == 0:
            by_shooter.setdefault((s.match, s.target), {}).setdefault(s.shooter, 0.0)
            by_shooter[s.match, s.target][s.shooter] += cfg.rifle.damage
    damaged = np.zeros_like(world.alive)
    for (m, j), shooters in by_shooter.items():
        top = max(shooters.values())
        tied = sorted(i for i, d in shooters.items() if d == top)
        world.last_damager[m, j] = tied[world.rngs[m].integers(len(tied))] if len(tied) > 1 else tied[0]
        for i, d in shooters.items():
            dealt[m, i] += d
        taken[m, j] = sum(shooters.values())
        damaged[m, j] = True
    world.health -= taken
    world.since_damage[...] = np.where(damaged, 0, world.since_damage + 1)

    shot_dead = world.alive & (world.health <= 0)
    fell = world.alive & ~shot_dead & (world.y > world.kill_y[:, None])
    dying = shot_dead | fell
    deaths = [Death(int(m), int(j), int(world.last_damager[m, j]) if shot_dead[m, j] else -1, bool(fell[m, j]),
                    int(world.last_damager[m, j]), int(world.since_damage[m, j]))
              for m, j in zip(*np.nonzero(dying))]

    regen = world.alive & ~dying & (world.since_damage >= cfg.ticks(life.regen_delay))
    world.health[...] = np.where(regen, np.minimum(world.health + life.regen_rate * cfg.dt, life.max_health), world.health)

    world.protection[...] = np.maximum(world.protection - 1, 0)

    waiting = ~world.alive & (world.respawn > 0)
    world.respawn[waiting] -= 1
    returning = waiting & (world.respawn == 0)
    world.alive[dying] = False
    world.respawn[dying] = cfg.ticks(life.respawn_time)
    world.vx[dying] = world.vy[dying] = 0.0

    respawns = []
    for m, j in zip(*np.nonzero(returning)):
        m, j = int(m), int(j)
        world.x[m, j], world.y[m, j] = pick_spawn(world, m, j)
        world.vx[m, j] = world.vy[m, j] = 0.0
        world.aim[m, j] = 0.0 if world.x[m, j] < world.grids.shape[2] / 2 else math.pi
        world.fuel[m, j] = cfg.physics.fuel_capacity
        world.ammo[m, j] = cfg.rifle.magazine
        world.cooldown[m, j] = world.reload[m, j] = 0
        world.health[m, j] = life.max_health
        world.since_damage[m, j] = 0
        world.last_damager[m, j] = -1
        world.protection[m, j] = cfg.ticks(life.protection_time)
        world.alive[m, j] = True
        respawns.append((m, j))
    return LifeEvents(dealt, taken, deaths, respawns)
