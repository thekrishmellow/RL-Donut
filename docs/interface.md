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
| 3 | Observations | done |
| 4 | Batched `MatchEnv` | done |
| 5 | Shared contract test | done |
| 6 | Gravity and thrust randomization | done |
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

## Step 3 — Observations

An observation is everything the network knows: one float32 vector of `OBS_DIM` = 330
numbers per agent per decision step. `Observer(world)` in `rl_donut.env.observe` builds
them and keeps the memory they need:

```python
observer = Observer(world)
obs = observer.observe(visibility(world))           # after reset: (M, A, 330)
ev = match_step(world, ...)
observer.record(ev)                                 # age memory, note this step's hits
obs = observer.observe(visibility(world))
observer.reset(matches)                             # after World.reset / load_maps of those matches
```

### Layout

The first 57 numbers have names in `FEATURE_NAMES`; then come the 273 tiles.

| Part | Count | Contents |
|---|---|---|
| Self | 20 | position on the map (x / width, y / height) and inside its tile; velocity; on-ground; fuel, health, ammo; reload and cooldown left; aim (cos, sin); alive; respawn and protection time left; direction the last hit came from (cos, sin) and time since |
| Enemy slot × 2 | 13 each | present, alive, seen by me, seen by my team, remembered; relative position; while seen: relative velocity, health, aim (cos, sin); time since seen |
| Ally slot | 9 | present, alive, relative position, velocity, health, aim (cos, sin) |
| Match | 2 | time left, score difference |
| Tile grid | 21 × 13 = 273 | tiles centred on the agent's tile, row by row: 1 rock, 0 air, −1 below the map |

`scripts/preview_observation.py` draws one observation: both agents' tile windows on the map,
the windows as each agent receives them, and the main numbers side by side.

### The choices behind it (I6–I11)

- **Terrain as a tile grid, not rays (I6).** For 64 matches, 24 distance rays cost
  1.0–1.4 ms per step. They are also coarse: at 10 tiles, neighbouring rays are 2.6 tiles
  apart, enough to miss a gap an agent fits through. A 21 × 13 window of tiles is a plain array
  lookup (0.1–0.2 ms) and exact. It can't see beyond ±10 × ±6 tiles, but enemies come
  through the enemy slots at any distance anyway. Because it is image-shaped, a small CNN
  can replace the first layer later.
- **Pits are marked (I6).** 67% of maps have pits. Below the map is a deadly drop, not
  rock, so those tiles are −1; the sides and top beyond the map are rock, as in the engine.
- **Team B sees a mirror image (I7).** The maps are mirror images, so team B's observation
  flips x: positions become width − x, velocities and aim cosines change sign, and the tile
  window is mirrored. Its actions are flipped back by `actions.decode(..., mirror=team ==
  TEAM_B)`. One policy then plays both sides identically, and "map x" always means distance
  from its own side. A test builds a mirror-image world with the teams swapped and checks
  every agent's observation equals its twin's. Building the windows needed care. Spawn
  positions are whole numbers of tiles, and simply flipping a window centred on the agent's
  tile is one column off exactly then, so the columns are chosen in the agent's own frame.
- **One layout for 1v1 and 2v2 (I8).** There are always 2 enemy slots and 1 ally slot, with
  "present" = 0 for slots that don't exist in 1v1. The same network can play both formats,
  which keeps the format decision (G3) open. Slots stay in agent order, so "enemy 0" is
  always the same player and its memory stays attached.
- **Memory instead of wallhacks (I9).** While an enemy is seen by the agent or a teammate,
  its slot holds live position, velocity, health and aim. When it is out of sight, only its
  last-known position and the time since are kept. After 10 s it is forgotten (position 0,
  "remembered" 0). Hits work the same way: the agent learns the direction the shot came from,
  not where the shooter is now.
- **No frame stacking (I10).** Velocity is in the observation, and memory covers what an
  old frame would say. Stacking 4 frames would make it about 1,300 numbers. If the trainer
  wants stacking or an LSTM, it adds them on its side.
- **Fixed scales (I11).** Positions are divided by 30 tiles (rifle range), velocities by
  run speed (6) and maximum fall speed (20), health by 100, the score difference by 10, and
  times by their full length. Everything stays within about ±3.5 (the largest is the
  distance across the map), and a test checks this on real matches. PPO's running
  normalization can still sit on top.

### Cost

Per decision step for 64 matches:

| Format | Engine | Visibility | Observations | Match steps per second, all three |
|---|---|---|---|---|
| 1v1 | 0.44 ms | 0.71 ms | 0.39 ms | 41,500 |
| 2v2 | 0.59 ms | 1.07 ms | 0.56 ms | 28,800 |

## Step 4 — The batched environment

`rl_donut.env.MatchEnv` is what the trainer uses. It follows section 4 of the handoff
contract (`spec_version` "0.3"):

```python
env = MatchEnv(n_matches=64, team_size=1, split="train", seed=0)
obs = env.reset()                                    # (64, 2, 330)
result = env.step(discrete, continuous)              # (64, 2, 3) int, (64, 2, 1) float, each agent's own frame
result.obs, result.events, result.done, result.final_obs, result.info
state = env.get_state()                              # (64, 2, 50), for a centralized critic
```

Each step:
1. Turns actions into engine controls, mirroring team B's back (`decode`).
2. Plays one decision step (`match_step`).
3. Updates the observers' memory, visibility and the episode stats.
4. Builds observations.

When a match ends, `final_obs` keeps its last observation and `info[m]` its episode stats. The
match then restarts on the next map and `obs[m]` becomes the new match's first observation.

### Decisions (I12–I15)

- **A shuffled map deck (I12).** The split's maps (after `map_filter`, e.g. "no pits" for
  an early curriculum) are shuffled with the env's seed and dealt in order. When the deck
  runs out it is reshuffled. Every map is played equally often, and a seed always gives the
  same order. A uniform random draw would repeat some maps and skip others for long stretches.
- **`get_state()` per team (I13).** A centralized critic sees the true state, not limited by
  sight. With mirrored observations, a critic for team B needs the state in team B's frame
  too, so there is one state per team: (M, 2, 50). Each has 4 agent slots (own team first)
  of 12 numbers, plus time left and score difference.
- **Episode stats (I14).** For each agent: kills, deaths, falls, damage dealt and taken,
  shots, hits, seconds airborne and hovering (airborne, vertical speed under 1 tile/s), mean
  distance to the nearest visible enemy when firing, and kills within 8 tiles of an enemy
  spawn. They are the raw material for the exploit watchlist (spawn camping, hovering out
  of reach, sniping from cover).
- **Matches end together (I15).** Every match lasts 900 steps and they start together, so
  a batch ends and restarts in one step. Staggering would need a shortened first match or a
  fake clock, which would distort "time left". The only effect is that stats arrive in bursts.

### Tests

`tests/test_match_env.py` plays a full 60 s 2v2 match on 8 maps and checks:
- auto-reset onto new maps;
- `final_obs` against the new first observation;
- episode stats against the per-step events summed over the match.

It also checks:
- the same seed gives the same run, and a different seed doesn't;
- the deck deals every map once per pass, and `map_filter` works;
- giving both sides the same actions on a symmetric map keeps the whole match a mirror image,
  for every observation and for `get_state`.

That last test caught a rounding issue. Running moves 0.2 tiles per tick, so positions often
land within 10⁻¹⁴ of a tile edge, and team B's mirrored x (width − x) rounds the other way.
The two sides then saw windows one column apart. Tiles are now found after a 10⁻⁹ nudge.

### Speed

With random actions, measured with the laptop's load average at about 4:

| Format | 16 matches | 64 matches | 256 matches |
|---|---|---|---|
| 1v1 | 9,000 | 22,000 | 36,300 |
| 2v2 | 6,400 | 19,400 | 34,500 |

These are match steps per second. Visibility is about 40% of the time, the engine 30% and
observations 25%. Step 3.7 measures this properly.

## Step 5 — The shared contract test

`tests/test_env_contract.py` is the interface's promise to the trainer, owned by both teams.
It runs every environment factory in `FACTORIES` (today only `MatchEnv`, in 1v1 and 2v2)
through the checks in section 4.6 of the handoff guide: shapes and kinds, invalid actions,
seeding, auto-reset with `final_obs` and `info`, and events adding up over a match.

Two choices:
- **Versioned layout (I16).** `SPECS` records what each `spec_version` promises: sizes, the
  episode stat names, and a fingerprint of the observation feature names. Changing the
  layout without bumping the version and adding an entry fails the test.
- **Kinds, not widths (I16).** Events are checked as float, integer or bool. The engine
  keeps damage in float64; a trainer that wants float32 casts it.

When this step started, Team B's branch turned out to already have a contract test and a
stand-in. Both were built on the first draft contract (16 aim directions, no mirroring, its
own observations), because the branch started before Phase 3. The spec 0.3 test above keeps
every check from Team B's version and adds the new ones. Section 0 of
[`handoff_phase5_6.md`](handoff_phase5_6.md) lists what changed since that branch started,
and the edits Team B's code needs.

## Step 6 — Gravity and thrust randomization

Varying physics between matches stops a policy from leaning on exact timings, and gives
Phase 8 physics shifts to evaluate on. The catch is the map generator: it accepted every
map assuming one tank climbs 12 tiles and flies 24 tiles level (P11, with 20% spare: 28.8).

Holding height means firing the jetpack a fixed share of the time, gravity ÷ thrust, so level
flight depends only on their ratio. Measured on one tank (tiles):

| Gravity \ thrust | ×0.85 | ×0.95 | ×1.0 | ×1.05 | ×1.15 |
|---|---|---|---|---|---|
| ×0.85 | 30.0 | 33.6 | 35.2 | 36.8 | 40.4 |
| ×1.0 | 25.6 | 28.4 | 30.0 | 31.6 | 34.4 |
| ×1.15 | 22.0 | 24.8 | 26.4 | 27.2 | 30.0 |

Independent ±15% (the draft) falls to 22 tiles and makes some accepted gaps uncrossable;
even ±5% each loses the 20% spare. Climbing stays at 26 tiles or more everywhere.

**Choice (I17): one factor per match multiplies both gravity and thrust.**
`PhysicsConfig.scale_range = (low, high)` draws it uniformly at each reset; the default
`(1, 1)` turns it off, and the trainer turns it on when it wants, e.g. `(0.85, 1.15)`.
Allowed values lie in `[0.5, 1.5]`, the range the tests cover:

| Factor | 0.5 | 0.7 | 0.85 | 1.0 | 1.15 | 1.3 | 1.5 |
|---|---|---|---|---|---|---|---|
| Climb (tiles, ≥ 12) | 33.8 | 34.9 | 35.3 | 35.7 | 35.9 | 36.1 | 36.3 |
| Level flight (tiles, ≥ 28.8) | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 |

Speed limits don't scale, so the tile-per-tick rule holds too. What the factor changes is how
heavy the game feels: a 10-tile fall takes 0.97 s at 0.85 and 0.83 s at 1.15 (0.89 s
normally), and reaching full rising speed takes 0.94 s against 0.70 s (0.8 s normally).

- **Hidden from agents.** The factor isn't in the observations, so the spec stays 0.3. A
  policy must play well across the range; fuel endurance and the hover rhythm don't change,
  so its plans still work. Physics shifts in Phase 8 then don't feed the network inputs it
  never saw.
- **Off by default.** Randomization usually slows early learning, so it is a training
  choice (for example a late curriculum stage), not part of the environment.
- **Its own random stream.** Each match draws the factor from `(rng_seed, map seed,
  episode, 1)`, so turning it on leaves rifle spread unchanged. `World.physics_scale` holds
  each match's factor.
- Scripted route-following (`navigation.py`) plans with the nominal physics.

**Tests** (`tests/test_physics.py`, `tests/test_match_env.py`): climb and level-flight
contracts at 0.5, 0.85, 1.15 and 1.5; the factor multiplies both accelerations; draws differ
per match and per episode, repeat for the same seed, and are 1 by default; rifle spread is
unchanged; bad ranges are refused; a batched run at 0.85–1.15 never enters rock and redraws
the factor at auto-reset.
