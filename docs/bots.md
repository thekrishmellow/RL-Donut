# Phase 4 — scripted bots, replays and the renderer

Phase 3 built the environment a learner plays in. Phase 4 adds the opponents it is measured
against (scripted bots), files that record matches (replays) and a way to watch them (the
renderer). Decisions are logged in [`DESIGN.md`](../DESIGN.md) with `B` IDs.

| Step | What | Status |
|---|---|---|
| 1 | Policy interface, match harness, random and static-target bots | done |
| 2 | Bot perception helpers | done |
| 3 | Turret bot | done |
| 4 | Rusher bot | done |
| 5 | Bot ladder and difficulty calibration | done |
| 6 | Replay format, writer and reader | done |
| 7 | Renderer | done |
| 8 | Merge preparation with Team B | done |

The bots come first because Team B's curriculum needs them: hit a static target, then beat
the turret bot (70%), then the rusher bot (60%).

## Step 1 — Policy interface, harness, simple bots

Everything that controls agents (the learner, pool snapshots, scripted bots) has one
interface, in `rl_donut.env.policy` (B1):

```python
class Policy(Protocol):
    name: str
    def act(self, obs: np.ndarray, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]: ...
        # obs (n, obs_dim) -> discrete (n, 3) int64, continuous (n, 1) float32, in each agent's own frame

@dataclass
class PolicyContext:
    env: MatchEnv               # scripted bots read the true game state through it
    slots: np.ndarray           # (n, 2) (match, agent) of each row
    rng: np.random.Generator
```

This is the shape Team B's branch already uses (in its stand-in), with the Phase 3 return
type. At merge, their code imports it from here.

- **Privileged bots.** Scripted bots may read positions and line of sight through
  `ctx.env`; they are a yardstick, not a fair opponent.
- **New matches.** `MatchEnv` restarts finished matches by itself, so a bot that remembers
  things per slot (a cover spot, a route) compares `ctx.env.world.episode[m]` with the value
  it last saw. The protocol has no reset method, so Team B's callers don't change.
- **Harness.** `play_matches(env, team_a, team_b, seed)` resets the env, plays one match in
  every slot of the batch, and returns each match's episode info (score, outcome, per-agent
  stats). It checks the shape of every policy's actions. Swapping sides is the caller's job:
  call again with the policies exchanged. Team B's `evaluate_policy` (confidence intervals,
  reports) stays the official evaluation; this harness is for the bot ladder and tests.
  `team_slots(env, team)` lists a team's (match, agent) slots, match-major.

**Bots** (`rl_donut.env.bots`):

| Bot | Moves | Jetpack | Aim | Shoots |
|---|---|---|---|---|
| `RandomBot` | uniform | uniform | uniform in [−π, π] | uniform |
| `StaticTargetBot` | never | never | 0 (toward the enemy side, in its own frame) | never |

128 full matches on `val` maps:

| Format | Pairing | Draws | Kills per match | Pit falls per agent | Match steps/s |
|---|---|---|---|---|---|
| 1v1 | random vs static | 100% | 0.00 | 0.02 | 28,100 |
| 1v1 | random vs random | 100% | 0.00 | 0.04 | 27,800 |
| 2v2 | random vs static | 100% | 0.00 | 0.01 | 16,400 |
| 2v2 | random vs random | 99% | 0.01 | 0.05 | 17,400 |

A random bot practically never kills: seven hits are needed, health regenerates, and random
aim rarely points at the enemy. Pit falls with no recent damager score for nobody (P25). So
beating a random bot means actually getting kills, and draws against it are failures. The
harness adds little cost: 28,100 match steps/s against 35,000 for the bare env.

**Tests** (`tests/test_policy.py`): slot order; one info per match in 1v1 and 2v2 with valid
outcomes; the static target never scores, shoots or leaves the ground, and never moves while
alive; the same seed repeats exactly and another seed differs; actions of the wrong shape
are refused with the policy's name.

## Steps 2–4 — Perception, turret, rusher

`rl_donut.env.perception` is what every scripted bot shares (B2):

- **Shooting uses the bot's own sight**, the same five-point check as an agent. The aim
  point is the enemy's centre, or the first visible corner when the centre is behind rock.
  The trigger is pulled only when that point is inside the rifle's 30-tile range.
- **Chasing may use the true position** of the nearest living enemy. A bot is a yardstick,
  not a fair opponent.
- **Aim error and reaction.** The aim is that point plus Gaussian noise. `Reaction` counts
  how many decisions in a row the same enemy has been seen, and a new match zeroes it.
  `controls_to_actions` mirrors the result into each agent's own frame.
- **Routes are cached per map.** Building the flight planner costs about 112 ms. It is built
  once per map seed and reused, including across matches.

| Bot | Moves | Shoots | Defaults |
|---|---|---|---|
| `TurretBot` | never; it holds the ground it spawned on | a visible enemy, after 2 decisions (~0.13 s) | aim error 2° |
| `RusherBot` | flies toward the nearest living enemy with the generator's own route planner; walks in once it shares that enemy's floor | a visible enemy, after 1 decision | aim error 2° |

The 2° error is the same size as the rifle's own spread. It is the difficulty knob: a
perfect aim was measured earlier at a 54% hit rate on partly hidden targets, and 2° of
extra error roughly halved that.

**Tests** (`tests/test_bots.py`): an enemy in the open is seen and in range, and a rock
column hides it; team B's aim of π comes back as 0 in its own frame; the reaction counter
restarts on a new match; the turret fires only once it has watched for its reaction and
never moves; the rusher walks toward an enemy standing on its floor.

## Step 5 — Ladder

`scripts/bot_ladder.py` plays each pair with the sides swapped halfway. 16 full matches on
val maps:

| Left | Right | Wins | Draws | Losses | Kills per match |
|---|---|---|---|---|---|
| turret | static | 0.00 | 1.00 | 0.00 | 0.00 |
| rusher | static | 1.00 | 0.00 | 0.00 | 4.88 |
| rusher | turret | 0.12 | 0.50 | 0.38 | 1.62 |
| random | turret | 0.00 | 1.00 | 0.00 | 0.00 |

The turret and the static target never meet, because spawns cannot see each other and the
turret does not hunt. That is what the curriculum wants: a learner beats the turret by
going to it and winning the duel. The rusher does hunt, and it kills a static target in
every match. In a duel the turret wins more often than the rusher (0.38 against 0.12, half
the matches drawn). The planned order, turret gate then rusher gate, is worth checking once
a learner exists: the stationary aimer is the harder fight for anything that has to
approach.

## Steps 6–7 — Replays and the viewer

A match is deterministic, so a replay stores the map seed, the environment seed and the
action at every decision step, not every position (B3). `rl_donut.env.replay` records,
saves a `.json.gz`, loads, and plays the match back. `snapshots` keeps one frame per
decision for drawing. Playing a replay back reproduces the score.

`scripts/render_replay.py` records a match (or reads a file) and writes a GIF and a
filmstrip with the same drawing as `scripts/view_match.py`. Matplotlib only, which was
already the project's choice; no new library. An 8 second turret-versus-rusher replay shows
the rusher leaving its spawn and flying across the map while the turret holds still.

## Step 8 — Ready to merge

Team B's curriculum can import `TurretBot` and `RusherBot` from `rl_donut.env`. The replay
writer is `rl_donut.env.replay`. The handoff checklist (section 8) is unchanged: their
branch still has to move onto this environment, and results measured on the stand-in are
re-run afterwards. `requirements.txt` already has the PettingZoo pins; their branch adds
`torch==2.14.1`, which is kept at the merge.
