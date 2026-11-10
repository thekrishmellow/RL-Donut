# Phase 4 — scripted bots, replays and the renderer

Phase 3 built the environment a learner plays in. Phase 4 adds the opponents it is measured
against (scripted bots), files that record matches (replays) and a way to watch them (the
renderer). Decisions are logged in [`DESIGN.md`](../DESIGN.md) with `B` IDs.

| Step | What | Status |
|---|---|---|
| 1 | Policy interface, match harness, random and static-target bots | done |
| 2 | Bot perception helpers | — |
| 3 | Turret bot | — |
| 4 | Rusher bot | — |
| 5 | Bot ladder and difficulty calibration | — |
| 6 | Replay format, writer and reader | — |
| 7 | Renderer | — |
| 8 | Merge preparation with Team B | — |

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
