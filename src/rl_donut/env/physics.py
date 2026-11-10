"""One physics tick: movement controls, gravity, jetpack thrust, speed limits and fuel.

Speeds are updated first and positions second (semi-implicit Euler): stable, exact for a
fixed timestep, and identical every time for the same inputs.
"""

import numpy as np

from rl_donut.env.collision import move, on_ground
from rl_donut.env.state import World


def physics_tick(world: World, move_dir: np.ndarray, jet: np.ndarray) -> None:
    """Advance every match by one tick.

    `move_dir` is -1 (left), 0 or 1 (right) and `jet` is True to fire the jetpack, both of
    shape (matches, agents). Dead agents ignore their controls.
    """
    move_dir, jet = np.asarray(move_dir), np.asarray(jet, dtype=bool)
    if not np.isin(move_dir, (-1, 0, 1)).all():
        raise ValueError("move_dir must be -1, 0 or 1")
    p, dt, alive = world.cfg.physics, world.cfg.dt, world.alive
    thrusting = jet & alive & (world.fuel > 0)

    world.vx[...] = np.where(alive, move_dir * p.run_speed, 0.0)
    accel = (p.gravity - p.thrust * thrusting) * world.physics_scale[:, None]
    world.vy[...] = np.where(alive, np.clip(world.vy + accel * dt, -p.max_rise_speed, p.max_fall_speed), 0.0)
    move(world, dt)

    refilling = on_ground(world) & ~thrusting
    world.fuel[...] = np.clip(world.fuel - p.fuel_drain * dt * thrusting + p.fuel_refill * dt * refilling,
                              0.0, p.fuel_capacity)
