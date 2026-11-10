# Phase 3 — the RL interface

Phase 2 built a game engine that plays many matches at once. Phase 3 wraps it into the
interface a learning agent uses: what an agent **does** (actions), what it **sees**
(observations), and the batched `MatchEnv` that the trainer steps. The contract it must
meet is section 4 of [`handoff_phase5_6.md`](handoff_phase5_6.md); decisions are logged
in [`DESIGN.md`](../DESIGN.md) with `I` IDs.

| Step | What | Status |
|---|---|---|
| 1 | Action space | done |
| 2 | Visibility | — |
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
