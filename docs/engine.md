# Game engine (Phase 2)

The engine plays matches on the maps from Phase 1: it moves agents, resolves collisions,
fires the rifle, applies damage and respawns, and keeps score. It has no rendering code
and runs many matches at once, because training speed is the project's biggest risk.

This guide explains each step: what it does, why, and the alternatives considered.
Decisions are logged in [`DESIGN.md`](../DESIGN.md) (IDs starting with `P`); the
whole-project plan is in [`PLAN.md`](PLAN.md).

## Promises the engine must keep

The map generator assumed these when it decided which maps are valid. Each one becomes a test.

| Promise | Why the generator needs it | Step |
|---|---|---|
| Climb at least `climb_budget` (12) tiles on one tank | "Reachable" in the generator must mean reachable in the game (M26) | 3 |
| Nothing moves 1 tile or more per tick | The 1-tile border and 1-tile platforms can't be passed through (M16) | 3 |
| A scripted bot reaches ≥ 99% of route targets | The generator's flight model holds under real physics (M4b) | 4 |
| Hitscan is blocked exactly when `geometry.line_of_sight` is | "Can't see" means "can't be shot"; spawns are hidden by this rule (M40) | 5 |

## Steps

| Step | Status | Code |
|---|---|---|
| 1. World and state model | done | `src/rl_donut/env/config.py`, `src/rl_donut/env/state.py` |
| 2. Tile collision | done | `src/rl_donut/env/collision.py` |
| 3. Movement, jetpack and fuel | next | — |
| 4. Route check | planned | — |
| 5. Aiming and hitscan rifle | planned | — |
| 6. Health, death and respawn | planned | — |
| 7. Match loop | planned | — |
| 8. Debug viewer and speed check | planned | — |

---

## Step 1: World and state model (`config.py`, `state.py`)

### Many matches in one set of arrays

`World` holds a whole batch of matches. Each per-agent quantity is one numpy array of shape
(matches, agents), for example `world.x[match, agent]`. Each per-match quantity has shape
(matches,). One numpy operation then updates every agent of every match.

**Why.** We measured one physics tick (gravity, thrust, speed cap, box collision) with
4 agents per match, on a busy laptop:

| Layout | Ticks per second |
|---|---|
| One Python object per agent, 1 match | ~48,000 |
| numpy arrays, 1 match | ~12,000 |
| numpy arrays, 64 matches together | ~470,000 (all matches) |

numpy has a fixed cost of about 1 µs per operation. With only 4 numbers per array that cost
dominates, so arrays for a single match are the slowest option. Across 64 matches it is
paid once for 256 agents, about 10× faster than plain Python on one core. Python objects
could catch up by running several processes, but this laptop is usually busy, so extra
processes scale poorly. Switching layouts later would mean rewriting the engine and its tests.

**Cost.** Rules are written for all matches at once, using masks ("respawn every agent
whose timer ran out") instead of `if` statements. For tests and debugging,
`World.single(map)` makes a batch of one, and `world.agent(match, agent)` returns one
agent's state as a small readable record (`AgentView`).

**Alternative kept in reserve.** numba (a compiler for Python loops) would give
object-style code at array speed, but it is a new dependency tied to specific numpy
versions, and ours is pinned.

### Agent order

Agents are ordered team A first, then team B: with 2 per team, slots are `[A, A, B, B]`
(`world.team`). Agent `i` and agent `i + team_size` are mirror partners.

### Coordinates

- Positions are continuous, in tiles; x grows to the right and y grows **downward**, like
  the grid's rows, so a position's tile is simply `grid[floor(y), floor(x)]`.
- An agent's position is the **centre of its hitbox** (0.8 × 1.6 tiles).
  - Mirroring is then just `x -> width - x`, which makes fairness tests simple.
  - Aiming and line of sight start from the centre anyway.
  - The alternative, the top-left corner (as the generator's boxes use), makes collision
    code slightly shorter but needs `width - x - hitbox_w` to mirror.
- Values are 64-bit floats (numpy's default), which keeps replays exact. Observations are
  converted to 32-bit when handed to the neural network.

### Starting positions

`spawn_centre` converts a spawn's 2 × 3 clearance box at (row y, column x) to the hitbox
centre `(x + 1, y + 3 − 0.8)`: horizontally centred in the box, feet exactly on the box's
floor. The generator calls a spot "standing" when rock is under either column of the box.
The hitbox spans 0.6–1.4 across the box, so it overlaps both columns and is always supported.

`reset` puts agent `i` of each team on spawn `i` of its team. Team B's spawns mirror
team A's, so every match starts as an exact mirror image. Choosing spawns for respawns is
step 6.

### Settings (`EngineConfig`)

| Setting | Default | Meaning |
|---|---|---|
| `team_size` | 1 | Agents per team (1v1 or 2v2); at most the map's 6 spawns per team |
| `tick_rate` | 30 | Physics ticks per second of game time |
| `hitbox_w`, `hitbox_h` | 0.8, 1.6 | Agent size in tiles; must fit inside the generator's 2 × 3 clearance box |
| `map` | `MapConfig()` | The generator settings the maps were built with |

Physics constants are added in step 3, when they are decided.

**Picture:** `scripts/preview_world.py` draws the tiles, spawn points and agents.

**Tests** (`tests/test_env_state.py`):

- invalid settings are rejected, including a hitbox larger than the clearance box or more
  agents than spawns;
- `spawn_centre` puts feet exactly on the box floor;
- on 50 val maps with 1 and 2 agents per team, every agent starts with its hitbox in air
  and rock directly under its feet;
- start positions are mirror images;
- a batch equals the same maps built one at a time;
- array shapes and types match the documentation;
- `reset` restarts only the chosen matches;
- `AgentView` reads one agent;
- maps of different sizes can't share a batch.

---

## Step 2: Tile collision (`collision.py`)

Each tick, every living agent tries to move by its velocity × the tick length. Collision
decides how far it really gets without its hitbox entering rock.

### One axis at a time

`move(world, dt)` moves every agent horizontally first, then vertically. On each axis:

1. Move the box to its target position.
2. Find the column (or row) that the box's **leading edge** is now in, for example the
   right edge when moving right.
3. If any tile in that column, across the rows the box spans, is rock, snap the box back
   so its edge sits exactly on that tile's edge, and set its speed on this axis to 0.

**Why checking one line is enough.** Nothing moves a full tile in one tick (the wall
promise), so the leading edge can enter at most one new column or row per tick, and the box
was clear before it moved. `move` refuses any move of a full tile or more with a
`ValueError` instead of silently letting an agent pass through a wall. Step 3 must choose
speed limits that keep this true.

**Why horizontal first.** It's the usual platformer order, and resolving vertical motion
last means the final position decides whether the agent is standing. The order only
matters when a box clips a corner diagonally, and then by less than one tick of motion.

**Alternatives considered.** An exact sweep (finding the moment the box first touches any
tile) is slightly more accurate at corners but much harder to write for many matches at
once. Small sub-steps are simple but cost several times as much and still aren't exact.

### Touching is not overlapping

A box whose right edge is exactly at x = 21 does not touch column 21, so an agent pressed
against a wall can still slide along it, and an agent on a floor can run along it.
Snapped positions such as `21 − 0.4` are not exact in binary floating point, so tile
ranges are computed with a tolerance (`EPS` = 1e-9 tiles).

### Standing (`on_ground`)

An agent is standing when its feet are on a tile edge (within 1e-6) and any tile directly
under its hitbox is rock. It's computed from the position alone, so it is correct right
after a reset or respawn, and for an agent standing still. The alternative, a flag set
when a fall was blocked, would be wrong on the tick after a spawn and whenever gravity is 0.

### Outside the map

- Left, right and top: solid, like the border (M16).
- Below the bottom edge: air. The bottom border row is solid except at pits, so pits are
  the only way down there; the agent falls until it passes the kill line (step 6 kills it).
  An agent that falls through a pit and flies back up before the kill line can re-enter
  through the same pit.

### Agents pass through each other

Agents don't collide with each other, as in Mini Militia. Solid bodies would let agents
block doorways or push each other into pits, which RL would quickly learn to exploit.
Bullets still hit agents (step 5).

### Checking our own work

`overlaps_rock(world)` checks every tile under every hitbox by brute force. Tests use it
to confirm the fast leading-edge check never misses anything.

**Picture:** `scripts/preview_world.py --trace 900` moves agents with random velocities
through the collision code and draws their paths: they stop flat on floors and ceilings and
slide along walls.

**Tests** (`tests/test_collision.py`):

- outside-the-map rules;
- falling agents land exactly on the floor; rising ones stop exactly at the ceiling;
- a wall stops horizontal speed only;
- an agent touching a wall can slide along it and leave it; running along a floor isn't blocked;
- 300 ticks of repeated landing don't drift off the edge;
- everyone stands after a reset; a hovering agent doesn't;
- dead agents neither move nor stand;
- falling through a pit goes below the map;
- moving a full tile in one tick is refused;
- on 40 val maps, 600 ticks of random motion with 1 and 2 agents per team never enter rock
  (and agents really do hit floors, walls and ceilings: about 12,000 contacts per run);
- a move is blocked exactly when the unblocked target would overlap rock, compared with
  the brute-force check over more than 10,000 moves.
