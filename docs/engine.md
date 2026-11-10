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
| 3. Movement, jetpack and fuel | done | `src/rl_donut/env/physics.py`, `src/rl_donut/env/config.py` |
| 4. Route check | next | — |
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
| `physics` | `PhysicsConfig()` | Movement constants (step 3) |
| `map` | `MapConfig()` | The generator settings the maps were built with |

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

---

## Step 3: Movement, jetpack and fuel (`physics.py`)

`physics_tick(world, move_dir, jet)` advances every match by one tick. `move_dir` is
−1, 0 or 1 (left, none, right) and `jet` fires the jetpack, both per agent.

### What the engine owes the generator

The generator judged maps with a simple fuel model (M27): climbing costs 2 units per tile,
sideways flight 1, falling nothing, from a tank of 2 × `climb_budget` = 24 units. So a full
tank must manage **12 tiles straight up** and **24 tiles of level flight**, and nothing may
move a tile or more per tick (M16).

### Choosing the constants

We simulated the draft numbers tick by tick (agent deciding every 2 ticks):

| Option | Climb on one tank | Fastest | Tiles per tick | Level flight |
|---|---|---|---|---|
| Draft (no rising limit) | 77 tiles | 38 tiles/s | 1.27: **passes through walls** | 24.4 tiles |
| Draft + rising limit 12 tiles/s | 28.5 | 12 | 0.40 | 24.4: **only 2% spare** |
| Rising limit 12 + drain 32/s (chosen) | 35.7 | 12 | 0.40 | 30.0: 25% spare |

- **Rising needs a speed limit.** Without one, a full tank reaches 38 tiles/s and climbs
  77 tiles, more than the whole 48-tile map.
- **Level flight was the real weak spot.** Holding height takes thrust 62.5% of the time
  (gravity 25 ÷ thrust 40), so at the draft's 40 units/s a tank lasts exactly 4 s, or 24
  tiles at running speed: the generator's assumption with nothing to spare. A learning
  agent switching the jetpack 15 times a second would fail gaps the generator accepted.
  Draining 32 units/s gives 30 tiles.

| Setting (`PhysicsConfig`) | Value |
|---|---|
| `gravity` | 25 tiles/s² |
| `thrust` | 40 tiles/s² (net 15 upward) |
| `max_rise_speed` | 12 tiles/s |
| `max_fall_speed` | 20 tiles/s (0.67 tiles per tick) |
| `run_speed` | 6 tiles/s, on the ground and in the air |
| `fuel_capacity` | 100 units |
| `fuel_drain` | 32 units/s while thrusting |
| `fuel_refill` | 25 units/s while standing and not thrusting (4 s from empty) |

`EngineConfig` refuses any speed limit of a tile per tick or more, and `PhysicsConfig`
refuses thrust that can't beat gravity.

### How a tick works

1. Horizontal speed is **set directly** from `move_dir` × `run_speed`, with no momentum. This
   is Mini Militia's feel, easier for RL to learn, and exactly predictable for the route
   check. Acceleration with friction was the alternative.
2. Vertical speed gains gravity, minus thrust if the jetpack fires and the tank isn't
   empty, then is clipped to the rising and falling limits.
3. Positions move by the new speeds through collision (step 2). Updating speed before
   position is **semi-implicit Euler**, the standard game rule: stable, and identical for
   the same inputs.
4. Fuel drains while thrusting and refills while standing without thrusting.

There is no jump and no fall damage, as in Mini Militia: the jetpack is the only way up,
and falling only kills through pits.

### A quirk: pulsing climbs higher

At the rising limit, extra thrust can't add speed but still burns fuel. So pulsing the
jetpack (3 decisions on, 1 off) climbs about 39 tiles, against 36 at full thrust. Both are
far above the 12 the generator needs; a skilled agent simply learns to pulse.

### For Phase 3: randomizing gravity and thrust

The plan varies gravity and thrust by ±15% between matches. At the worst corner (gravity
+15%, thrust −15%) level flight drops to 22 tiles, below the generator's 24. Those ranges
must be narrowed, or gravity and thrust varied together, and the contract tests must run
at the extremes.

**Picture:** `scripts/preview_physics.py` plots one tank's flight paths in an empty room
against the generator's needs.

**Tests** (`tests/test_physics.py`):

- **contracts:** climb ≥ 12 tiles (35.7); level flight ≥ 1.2 × 24 tiles (30.0, holding
  height within 0.14 tiles); the draft's 40/s drain would leave less than 20% spare;
  speed limits of a tile per tick or more are refused;
- invalid physics settings are rejected;
- rising and falling speeds stop exactly at their limits;
- horizontal speed is set directly on the ground and in the air;
- fuel drains at 32/s, an empty tank gives no thrust, and fuel refills only while standing
  (full after 4 s);
- standing still stays exactly on the floor; dead agents ignore controls;
- 900 ticks of random play with 2 agents per team on 40 val maps never enter rock, never
  exceed a tile per tick and keep fuel in range;
- the same controls give the same trajectory, bit for bit.
