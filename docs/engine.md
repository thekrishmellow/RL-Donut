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
| 2. Tile collision | next | — |
| 3. Movement, jetpack and fuel | planned | — |
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
