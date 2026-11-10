"""Who can see whom: an agent sees a living enemy when a line from its hitbox centre
reaches the enemy's centre or any of its four hitbox corners without touching rock, at any
distance. The rifle fires from the same centre and judges blocking with the same grid walk,
so an enemy that can be shot is almost always seen (on real maps, 0.1% of shootable enemies
are reported unseen, against 10.6% when checking centres only).

In 2v2, team sight shares sightings: an enemy is known when this agent or a living teammate
sees it. Observations keep the agent's own sighting as a separate flag.
"""

from dataclasses import dataclass

import numpy as np

from rl_donut.env.raycast import segments
from rl_donut.env.state import World

# Corners sit this far inside the hitbox: an agent standing on a floor has its bottom
# corners exactly on the rock's surface, where the grid walk counts them as inside rock.
CORNER_INSET = 0.05


@dataclass(frozen=True)
class Visibility:
    """Each (M, A, A): [m, i, j] is about agent i looking at agent j; False unless j is a
    living enemy of a living i."""
    own: np.ndarray     # i itself sees j
    team: np.ndarray    # i or a living teammate sees j


def target_points(cfg) -> np.ndarray:
    """(5, 2) offsets from a hitbox centre: the centre and the four inset corners."""
    hx, hy = cfg.hitbox_w / 2 - CORNER_INSET, cfg.hitbox_h / 2 - CORNER_INSET
    return np.array([(0, 0), (-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)], dtype=np.float64)


def visibility(world: World) -> Visibility:
    M, A = world.x.shape
    enemy = world.team[:, None] != world.team[None, :]
    pairs = world.alive[:, :, None] & world.alive[:, None, :] & enemy[None]
    m, i, j = np.nonzero(pairs)
    points = target_points(world.cfg)
    p = len(points)
    clear, _ = segments(world.grids, np.repeat(m, p),
                        np.repeat(world.x[m, i], p), np.repeat(world.y[m, i], p),
                        (world.x[m, j][:, None] + points[:, 0]).ravel(),
                        (world.y[m, j][:, None] + points[:, 1]).ravel())
    own = np.zeros((M, A, A), dtype=bool)
    own[m, i, j] = clear.reshape(-1, p).any(axis=1)
    ally = (world.team[:, None] == world.team[None, :])[None] & world.alive[:, None, :]
    team = np.einsum("mik,mkj->mij", ally.astype(np.int64), own.astype(np.int64)) > 0
    return Visibility(own=own, team=team & pairs)
