# Submission notes

15 November 2026. Updated 24 November 2026. For evaluators of RL-Donut.

This page is the project brief: what was built, which number to trust, how to run it,
and what is still open. The technical report is
[`docs/RL-Donut-technical-report.pdf`](RL-Donut-technical-report.pdf). The measurements
behind the scores are in [`docs/PROGRESS.md`](PROGRESS.md). Every approved design choice,
with the reason and the measurement, is in [`DESIGN.md`](../DESIGN.md).

Repository: [github.com/thekrishmellow/RL-Donut](https://github.com/thekrishmellow/RL-Donut),
branch `main`.

## What the project is

RL-Donut is a Mini Militia-style 2D deathmatch built as a reinforcement-learning
environment. Two agents with jetpacks and rifles fight for 60 seconds on a cave. The cave
is generated from an integer seed. The same seed always rebuilds the same cave. A different
seed builds a different one.

The agent does not see pixels. It receives 330 numbers: its own position, fuel, and health,
a 21 by 13 window of the tiles around it, and whether an enemy is seen or still remembered
from the last 10 seconds. Its actions are move (left, none, right), jetpack (off, on),
shoot (no, yes), and one aim angle. Zero is to the right. Team B sees a mirrored cave and
its actions are mirrored back, so one network can play either side.

A shot is a straight line from the shooter's centre, the same line-of-sight test the map
uses. Rock stops the shot. A hit does 15 damage. A kill is 7 hits, about 1.2 seconds of
accurate fire. Health is 100 and regenerates at 10 per second after 3 seconds without
damage. Falling through a bottom pit is a death. Physics runs at 30 ticks a second. The
agent decides every second tick, so a minute-long match is 900 decisions.

## What is complete

Phases 1 through 4 are finished and covered by tests.

**Maps.** Perlin noise and fractional Brownian motion are written in NumPy, not taken from
a noise library. Four octaves, each twice as fine and half as strong as the one before.
A quantile threshold turns the noisiest 40 to 55 percent of the tiles into rock. A fixed
cutoff, aimed at 40 percent, had produced anywhere from 31 to 50 percent. Three
cellular-automaton passes remove single-tile spikes. The left half is mirrored to the
right before that threshold, so neither team is given the ground. Disconnected caves are
joined by a Dijkstra tunnel that prefers existing air and turns a vertical shaft into
stairs, because a jetpack cannot climb a smooth pit. A map is kept only when every floor
an agent can reach, the agent can also leave. One seed is split into named random streams,
so a new use of randomness cannot change maps that already exist. NumPy is pinned at
2.5.4 for that reason.

**The match.** Positions, velocities, health, and fuel are NumPy arrays updated for many
matches at once. The full environment runs at about 35,000 one-versus-one decision steps
a second. A scripted flier completed 99.80 percent of 2,000 routes the generator claimed
were possible.

**Bots, replays, viewer.** A random bot, a bot that stands still, a turret, and a rusher.
The rusher flies the generator's own routes and, over 16 matches, killed a stationary
opponent every time (4.88 kills each). A replay stores the seed and the actions. Playing
it back reproduces the match.

**Seed splits.** They do not overlap.

| Split | Seeds | Use |
|---|---|---|
| train | 0–9,999 | Learning |
| val | 10,000–10,999 | Checks during training |
| test | 20,000–20,999 | Held for one final evaluation. Not used by the run below |
| OOD | 30,000+ | Shifted maps or physics. Code exists. Not the result below |

Built maps are committed in `data/maps/`.

## The training result to use

The learning code for the later phases is on `main`. Training is not finished. The run to
read is `runs/spawn-curriculum-14`.

PPO plays a static bot: the opponent stands on its spawn and does not shoot. Eight matches
run at once. Every 128 decisions the network is updated. Every 25 updates, training stops
and the current weights play 16 validation matches of 60 seconds, on maps from the val
split, with sides swapped. If the policy wins at least 8 of those 16, the enemy is placed
farther away. That rule is the gate.

The reward is a match outcome plus a fading shaping term. A win is +1, a loss is -1, a
draw is 0. While shaping is at full strength, a kill is +0.3, a death is -0.3, each point
of damage dealt is +0.003, each point taken is -0.003, every decision costs 0.0005, and a
pit fall costs 0.3. The weight on shaping, alpha, starts at 1 and falls toward 0.1 over
the first 60 percent of the scheduled steps.

| Rung | Best check | What happened next |
|---|---|---|
| 6–12 tiles, enemy already in sight | Update 175: 8 of 16 wins, 1.81 kills, 273 damage | The gate opened. Weights: `runs/spawn-curriculum-14/ckpt_tiles_8.pt` |
| 12–16 tiles, still in sight, still inside the 30-tile rifle | 4 of 16, more than once. First time is update 225: 0.31 kills, 78 damage. Update 325, also 4 of 16, has 0.62 kills and 120 damage | The gate stayed shut. Later checks are 1, 2, 3, or 4 of 16. `ckpt_tiles_14.pt` is update 225, the first 4 of 16, not the higher-damage later check |
| 16–28 tiles, then the official spawns | Not run | The ladder stopped here |

Both of those rungs start with a line of sight. The official spawns are about 64 tiles
apart and are placed so the teams cannot see each other. That setting was not trained.

A second scoring of the two checkpoints, same seed and match length, found that the 12–16
tile weights do not beat the 6–12 tile weights on the longer maps (3 of 16 wins and about
90 damage, against 3 of 16 and about 80 damage). Sixteen matches is a wide interval: 8 of
16 has a Wilson 95 percent interval of about 0.28 to 0.72. The damage drop is the clearer
fact. Entropy at the end of the log is 3.25. That number includes the aim cloud. A fresh
policy under the same definition starts near 3.9. The three button heads alone top out near
2.48, which is what the current code logs. The aim spread on these checkpoints is about 59 degrees.
A standing body at 14 tiles is about 7 degrees tall, so most samples miss even when the
mean points at the enemy.

## Changes that came after that run

Two changes are in the code and are not inside those checkpoints.

The aim spread is now initialized at 4 degrees and clamped between 1 and 6 degrees. The
entropy bonus no longer pays the policy to widen it. The rifle's own scatter, plus or
minus 2 degrees, is unchanged. A new network with that aim, before any training, scored
about one kill every 30 seconds when the enemy started in view, because an untrained aim
points to the right and those rungs place the enemy on the right. On four official maps
the same network dealt no damage.

When no enemy is seen or remembered, the learner flies a loop of the map's reachable
floors and walks each floor before leaving it. It does not shoot during that search. Sight
is a line from the body, so turning the gun does not reveal anyone. On four official maps
the roam covered 22 to 39 tiles in 60 seconds and never gained a line of sight to the
static enemy.

Loading an old checkpoint does not turn these changes off. The spread stored in the file
is still about one radian. The code clamps it to 6 degrees at the moment of aiming. The
roam runs whenever the observation has no enemy, including for those older weights.

## Code that is present and is not this result

Self-play, combat styles, generalization across unseen seeds and physics scales, a
one-shot test-split benchmark, and Bradley-Terry ratings are implemented. They were not
the evaluation of `spawn-curriculum-14`. The dashboard tabs for Elo, generalization, and
the benchmark are empty for this run because those files were not written.

[`archived/REPORT.md`](../archived/REPORT.md) is an earlier sign-off. It says the twelve
milestones are fulfilled, quotes an 81 percent close-range win rate, and writes a reward
with see-enemy and approach bonuses. That reward is not in `src/rl_donut/rl/rewards.py`.
The file also includes a test-split table of 0 wins. Its run folders are not in this
clone. [`docs/REPORT.md`](REPORT.md) only points at the archive. Do not grade the project
from either file. Grade it from this page, from [`docs/PROGRESS.md`](PROGRESS.md), and
from the PDF.

## Setup

Python 3.11 or newer. Developed on 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Do not upgrade NumPy off the pin in `requirements.txt`. Map generation depends on its
random streams, and the golden tests fail if they change.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run scripts/dashboard.py
```

The dashboard listens on http://localhost:8501 and rereads `runs/` every 3 seconds. Select
`spawn-curriculum-14`. The five numbers at the top are the end of the run. The best score
is on the Validation checks tab: 50 percent, which is 8 of 16, at update 175 on the 6–12
tile rung. Pictures and replays are in `outputs/`. A picture there is a rendering of the
game or of an earlier close-range fight. It is not a recording of update 600.

To continue training:

```bash
.venv/bin/python scripts/train_spawn_curriculum.py --updates 400
```

That starts a new network. It does not resume the saved checkpoint. The new run will use
the 6 degree aim cap and the floor roam.

A person can play at http://127.0.0.1:8765:

```bash
.venv/bin/python scripts/play_human.py
```

You are blue. Red, during that match, is the rusher. It flies the cave toward you and
shoots on line of sight. It is not the learned policy, and it is allowed your true
position, so the fight is not a validation match. End and teach runs four short updates
on the learned weights. The training dashboard does not show those lessons, because they
are not written to `metrics.jsonl`. A separate self-play log,
`runs/genuine-selfplay`, has 186 updates and no validation win rate. Do not read it as
the 8 of 16.

The PDF can be rebuilt after `pip install reportlab`, which is not required to train or
to view the dashboard:

```bash
.venv/bin/python scripts/build_technical_report.py
```

## What we would like an evaluator to look at

1. `runs/spawn-curriculum-14/metrics.jsonl` and the Validation checks tab. That is the training result.
2. [`docs/RL-Donut-technical-report.pdf`](RL-Donut-technical-report.pdf). Decisions and the same numbers, in paper form.
3. A map preview and a replay in `outputs/`, to see the cave and a match.
4. `DESIGN.md`, if the question is why a generator or engine choice was made. Each row names the alternative and the measurement.

## What is still open

| Open item | Why it matters |
|---|---|
| Retrain with the 6 degree aim cap | The saved weights were trained at a 59 degree spread |
| Stop at the peak | Every rung so far rose and then fell. Update 600 is not the policy to keep |
| A smaller step from 8 tiles to 14 tiles | The longer in-sight rung peaked at 4 of 16 |
| The official hidden spawns | This is the actual search problem. It has not been trained. A measured 0 would be a result. It is not a missing file |

525 tests are collected under `tests/`. Phases 1–4 are logged as done in `DESIGN.md`.
Phases 5–10 have code on `main` and are not logged there as finished.
