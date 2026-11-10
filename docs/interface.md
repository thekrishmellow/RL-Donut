# Phase 3 — the RL interface

Phase 2 built a game engine that plays many matches at once. Phase 3 wraps it into the
interface a learning agent uses: what an agent **does** (actions), what it **sees**
(observations), and the batched `MatchEnv` that the trainer steps. The contract it must
meet is section 4 of [`handoff_phase5_6.md`](handoff_phase5_6.md); decisions are logged
in [`DESIGN.md`](../DESIGN.md) with `I` IDs.

| Step | What | Status |
|---|---|---|
| 1 | Action space | done |
| 2 | Visibility | done |
| 3 | Observations | — |
| 4 | Batched `MatchEnv` | — |
| 5 | Shared contract test | — |
| 6 | Gravity and thrust randomization | — |
| 7 | Full environment speed benchmark | — |
| 8 | PettingZoo wrapper | — |

## Step 1 — Action space

Each decision step (2 engine ticks), every agent picks four things:

| Part | Kind | Values |
|---|---|---|
| move | discrete, 3 choices | 0 = left, 1 = none, 2 = right |
| jetpack | discrete, 2 | 0 = off, 1 = on |
| shoot | discrete, 2 | 0 = no, 1 = yes |
| aim | continuous, 1 number | angle in radians: 0 = right, π/2 = down |

`rl_donut.env.actions.decode(discrete, continuous)` turns a batch of actions, shaped
`(M, A, 3)` and `(M, A, 1)`, into the engine's `move`, `jet`, `aim` and `trigger` arrays.
`encode` goes the other way, for scripted bots that think in engine controls.

### Why the aim is continuous (I1)

The plan's draft used 16 fixed aim directions. These are 22.5° apart, while an enemy's hitbox 10
tiles away covers only about 4.6° of view. To measure what each design allows,
`scripts/aim_ceiling.py` draws 6,785 pairs of positions in line of sight within rifle range
on 100 val maps. The median pair is 10.8 tiles apart; 10% are under 3.6 tiles and 10% over 23.
For each pair, a perfect aimer fires one real rifle shot (with its ±2° spread) using the best angle
its design allows:

| Aim design | Best possible hit rate |
|---|---|
| 16 fixed directions (22.5° apart) | 46% |
| 32 fixed directions (11.2° apart) | 71% |
| 64 fixed directions (5.6° apart) | 88% |
| 128 fixed directions (2.8° apart) | 95% |
| Continuous angle | 98% |

The remaining 2% is the rifle's spread at long range (P16). With 16 directions, more than
half of perfect shots miss, so an agent could never learn to shoot well at mid range. The
alternatives considered were:
- **128 fixed directions:** a 95% ceiling and pure categorical PPO. But 128 unrelated
  choices give the network no sense that neighbouring angles are similar, and it learns
  more slowly.
- **Aim relative to the nearest visible enemy:** easy to learn, but it builds the targeting
  into the action. It stays the fallback if continuous aim learns too slowly.

A continuous angle has the best ceiling and keeps neighbouring angles close. In PPO, it is
one Gaussian head (a mean from the network plus a learned standard deviation) next to the
categorical heads. This is the standard setup for continuous control.

### Details that matter for learning

- **No clipping or wrapping.** The engine only uses the cosine and sine of the aim, so 0.3
  and 0.3 + 2π are the same shot (`tests/test_actions.py` plays 150 steps both ways and
  gets identical matches). A Gaussian sample can therefore go straight into the env.
- **Mirroring.** Team B's view of the map is a mirror image, and the mirror of angle a is
  π − a. Step 3.3 decides whether observations are mirrored per team. If they are, the
  same mirroring applies to team B's move and aim before `decode`.

Run the measurement again with `.venv/bin/python scripts/aim_ceiling.py --maps 100` (about 7 s).

## Step 2 — Visibility

`visibility(world)` in `rl_donut.env.visibility` answers "who can see whom" for the whole
batch. It returns two (M, A, A) arrays, where entry [m, i, j] is about agent i looking at agent j:
- `own`: i sees j itself;
- `team`: i or a living teammate sees j.

Both are False unless i and j are alive and on different teams. Step 3.3 uses them to decide
which enemy information goes into each observation: nothing about unseen enemies, so no wallhack.

### What counts as seeing (I3)

An agent sees an enemy when a line from its own hitbox centre reaches the enemy's centre
**or any of the four corners of its hitbox** without touching rock, **at any distance**.
The rifle fires from the same centre, so this is close to "could I hit some part of them".
Measured on 5,738 pairs of positions on val maps where some part of the enemy can be shot:

| What is checked | Shootable enemies reported as unseen |
|---|---|
| Centre only | 10.6% |
| Centre, head and feet | 2.5% |
| Centre and the 4 hitbox corners | 0.1% |

Checking the centre alone misses peeking enemies. These are mostly enemies on a ledge whose
head shows over the edge, and they can shoot you while the observation calls them hidden.
`scripts/preview_visibility.py` draws, for one agent, every standing spot on a map coloured
by what that agent would see of an enemy there. The orange spots are the ones a centre-only
check would miss.

The corners sit 0.05 tiles inside the hitbox. A standing agent's feet touch the floor exactly,
and the grid walk counts a point on a tile edge as inside that tile, so un-inset corners would
always be "in rock".

Any distance means only rock blocks sight. This matches the respawn rule (P21), and an agent
can see an enemy beyond rifle range and plan an approach. A screen-sized window would be an
invented limit with no tactical purpose.

### Team sight (I4)

In 2v2, an enemy is known when the agent **or a living teammate** sees it, as if players call
out positions. The observation still flags whether the agent sees the enemy itself, since
"I can shoot them now" and "my teammate says they're there" are different facts. Dead
teammates report nothing.

### A fast, exact ray caster (I5)

Every check above is a line of sight. The map generator's `line_of_sight` walks the grid tile
by tile along the line (the Amanatides and Woo method): it repeatedly asks whether the line
crosses a vertical or a horizontal tile edge next, steps into that tile, and stops at the
first rock. It is exact, but in pure Python it costs about 3.7 µs per line, and visibility
needs up to 5 lines per pair of enemies.

`rl_donut.env.raycast.segments` runs the same walk for **all lines at once** in numpy. Every
unfinished line advances one tile per loop iteration, using the same arithmetic in the same
order, so its answers are identical. A test compares the two on 100,000 random segments,
including segments through exact tile corners, along grid lines, of zero length, and leaving
the map. Two details make it fast:
- **A rock frame.** Each grid is padded with one tile of rock. A walk that starts inside the
  map stops on the frame before it can leave, so no step needs a bounds check.
- **Flat tile indices.** A tile is a single number that moves by 1 for a column step and by
  a row's length for a row step.

It also returns where along each line the first rock is entered. Step 3.3's distance rays will
use that.

Time per decision step for 64 matches (`scripts/bench_engine.py --visibility`):

| Format | Engine | Visibility, pure Python | Visibility, batched |
|---|---|---|---|
| 1v1 | 0.45 ms | 2.2 ms | 0.8 ms |
| 2v2 | 0.60 ms | 9.3 ms | 1.4 ms |

The batched cost is set by the **longest** walk, because the loop runs until the last line
finishes, and the fixed cost of each numpy call dominates. It therefore grows slowly with
batch size: at 256 matches in 1v1, engine plus visibility runs 110,000 match steps per second.
If this becomes the bottleneck, step 3.7 will look at it again.
