# RL-Donut — progress report

15 November 2026. Updated 24 November 2026.

RL-Donut is a two-person project: a Mini Militia-style 2D game built as a reinforcement
learning environment. Agents with jetpacks and rifles fight on cave maps that are generated
from a seed, learn by playing, and are later tested on maps they have never seen.

The plan has twelve phases. The environment side is complete, including the scripted
opponents, replays and a viewer. The learning code is on `main` and steps that environment.
Training has started and is not finished. The measured curriculum is below. The earlier
sign-off is in `archived/REPORT.md`. It does not replace these numbers. Its reward formula,
with see-enemy and approach bonuses, is not the reward in `src/rl_donut/rl/rewards.py`.

## How the work is done

Each piece is discussed before it is built: why it is needed, what the alternatives are, and
a measurement that checks the choice. It is then implemented with tests, reviewed, and
written up. Every design choice is recorded, with its reason, in `DESIGN.md`. There are 104
recorded decisions so far.

The work is split so the two halves can proceed at the same time. One person builds the maps,
the game, and the interface the learner uses. The other builds the learning algorithms. They
meet through a written interface contract and a shared test that both sides must pass.

## What we learned and implemented

These are the techniques behind the finished work. Each one was studied, compared with an
alternative, and then written by us. Nothing in the map generator or the game engine is a
black-box library call.

### Making maps from noise

- **Perlin noise, written ourselves in NumPy.** Perlin noise fills a grid with smooth random
  hills. At each grid corner we place a random gradient, and between corners we blend them
  with a quintic fade curve. The quintic curve matters: a simpler blend leaves a visible
  crease on every cell edge. Cells are stretched sideways (16 tiles wide, 10 tall) so the
  hills become ledges, which suits a side-view game. We did not use a noise library. Owning
  the code means a seed produces the same map on any machine, and we are not tied to a
  library that is still changing.
- **Fractional Brownian motion (FBM).** One Perlin grid is one scale of hills. FBM adds four
  layers, each twice as fine and half as strong as the one before (lacunarity 2, persistence
  0.5). Large layers shape the caves; small layers add roughness. Extra layers keep the big
  shape and only add detail.
- **Quantile threshold.** The noise is turned into rock and air by keeping the rockiest
  fraction of tiles, between 40% and 55% per map. A fixed cutoff was measured and rejected:
  aimed at 40% rock, it produced anywhere from 31% to 50% across 1,000 seeds. A quantile
  hits the fraction exactly. A height bias, also drawn per map, pushes rock toward the
  bottom so there is ground below and open space above.
- **Cellular automata smoothing.** Three passes of a neighbour rule (a tile becomes rock when
  5 of its 8 neighbours are rock, air when 3 or fewer are). Rough, single-tile jags fell from
  about 72 per map to about 1. Erosion and dilation were the alternative; they cut thin
  bridges that agents need.
- **Mirror, then threshold.** Noise is generated for the left half and copied to the right,
  and only then turned into rock. Smoothing a half map first treats the centre as a wall and
  grows a spine down the middle. Left-right mirroring was chosen over a 180° rotation,
  because rotation plus the height bias would give one team the ground and the other the sky.
- **Named random streams.** One map seed is split with NumPy's `SeedSequence` into separate
  PCG64 generators: terrain, tunnels, platforms, pits, spawns, and the per-map settings.
  Adding a new use of randomness cannot change maps that already exist. A failed map is
  retried with a new attempt number, not by adding to the seed, which NumPy warns against.
  NumPy itself is pinned to one exact version, and a few maps are stored as known-good
  "golden" values, because NumPy only promises identical random numbers on the same build.

### Making every map playable

- **Flood fill on a clearance box.** Connectivity is not "is this tile air". An agent is
  0.8 by 1.6 tiles, so we flood-fill (4 directions, not diagonals) only where a 2 by 3 box
  fits. On raw terrain that stricter box left 64% of the air usable, against 85% for a box
  the agent's own size, which is why smoothing and tunnels have to recover space.
- **Dijkstra tunnels.** Disconnected caves are joined by the cheapest path: each step costs
  1, plus 12 for every rock tile dug, so the path reuses existing caves. Digging straight
  up is charged triple, which turns shafts into staircases. A plain vertical shaft taller
  than the jetpack can climb is a one-way trap. That one change took fully reachable maps
  from 78% to about 94%, without digging any more rock (about 39 tiles per map).
- **A fuel model of flight.** Climbing one tile costs 2 fuel, moving sideways costs 1, and
  falling is free. The tank is 24. A map is accepted when every floor you can reach, you can
  also leave, and at most 2% of floor is stranded. Demanding 100% would reject maps over a
  two-tile notch.
- **Exact line of sight (Amanatides–Woo).** A grid walk that steps from tile to tile along a
  line and checks both tiles when the line clips a corner. Sampling the line every quarter
  tile, the obvious shortcut, missed 46 corner clips out of 10,000. The same function is
  what the rifle uses, so "cannot see" and "cannot be shot" are the same fact.

### The game, simulated in batches

- **One array for every match.** Positions, velocities, health and fuel are NumPy arrays of
  shape (matches, agents), updated together. Measured: about 470,000 ticks per second for 64
  matches this way, against 48,000 with ordinary Python objects and 12,000 with one array
  per match. A compiled accelerator (Numba) was considered and deferred; it is a new
  dependency, and the array version is already fast enough.
- **Semi-implicit Euler.** Each tick updates speed first, then position. At a fixed
  timestep this is stable and gives the same result every time. There is no jump and no
  fall damage: the jetpack is the only way up, as in Mini Militia.
- **Collision one axis at a time.** Horizontal, then vertical. If the leading edge enters
  rock, the agent is placed on the tile edge and that speed is zeroed. This is exact as long
  as nothing moves a full tile in one tick, which the speed limits enforce (the fastest
  motion is 0.67 tiles per tick).
- **Hitscan shooting.** A shot is a ray from the shooter's centre. It hits the nearest enemy
  body it enters, unless rock is in the way. Damage is 15, so a kill is 7 hits, about 1.2
  seconds at full fire. Hits on the same tick are applied together, so a trade kills both;
  resolving them in agent order would hand every tie to one side, and a learner would
  exploit that.

### What the learner sees and does

- **Structured observations instead of pixels.** The agent receives numbers (position, fuel,
  health, a small tile window), not a screenshot. A 2 by 256 network can learn this with
  PPO. Pixels would need a convolutional network and far more training.
- **A continuous aim angle, sampled from a Gaussian.** Sixteen fixed aim directions, the
  first design, hit only 46% of perfectly aimed shots at real fight distances. A free angle
  hits 98%. The network predicts the mean angle; a learned spread is added when sampling.
- **The same ray walk, vectorized.** Who-sees-whom runs the Amanatides–Woo walk for every
  pair at once. It matches the exact Python version on 100,000 random lines, and it is
  several times faster.
- **A tile window instead of distance rays.** Casting 24 rays around the agent is slower and,
  at 10 tiles out, the rays are 2.6 tiles apart, so they skip walls. A 21 by 13 window of
  the tiles themselves is exact and about ten times cheaper.
- **PettingZoo's parallel interface.** PettingZoo is the standard library for multi-agent
  reinforcement learning. Its turn-based mode (one agent acts at a time) fits chess; its
  parallel mode fits a game where everyone acts together. We implemented the parallel one,
  with Gymnasium spaces describing legal observations and actions, and it passes the
  library's own tests. Training does not use it: one match at a time is about 700 steps per
  second, against 35,000 for the batched environment.
- **A centralized critic.** The learner's value network may see the true state of the whole
  match, including enemies behind walls. The policy that acts cannot. That split is the
  standard way to train cooperative teams.

### What the learning half uses

Built by the teammate, on the draft interface, and not yet run on this environment:

- **PPO** (proximal policy optimization) with **GAE** (generalized advantage estimation) for
  the learning update, and a network with a separate head for each action.
- **MAPPO** for 2v2: one policy shared by teammates, with the centralized critic above.
- **PFSP** (prioritized fictitious self-play): the opponent is sometimes the latest self,
  sometimes a harder past version, sometimes a fixed bot, so the agent cannot forget how to
  beat old styles.
- **Elo** ratings anchored to the random bot, so scores from different runs compare.
- **Wilson score intervals** on win rates, so a result is reported with a confidence range
  rather than a single percentage.

## What is finished

### Map generator

A map is produced entirely from a seed, so the same seed always gives the same map. The
pipeline is: Perlin noise with FBM on the left half, mirror it, quantile threshold, cellular
automata smoothing, a solid border, tunnels, pits, a reachability check with platforms, spawn
points, then validation.

- Every map is a mirror image left to right, so neither side starts with an advantage.
- Disconnected caves are joined by tunnels, dead ends are filled, and every area is
  reachable with one tank of jetpack fuel. Platforms and staircase tunnels fill the gaps
  the jetpack cannot cross.
- About a third of maps have no pits, a third have one, and a third have two. Pits are
  carved only where an agent can actually reach them.
- The two teams spawn where they cannot see or shoot each other.
- A map that fails these checks is regenerated. Of 10,000 training seeds, 3 failed and were
  skipped. The validation and test sets, 1,000 maps each, had no failures.

11,997 maps are saved and frozen: 9,997 for training, 1,000 for validation, and 1,000 held
out for the final test.

### Game engine

The engine runs many matches at once, with no graphics, so training can be fast. It is
deterministic: the same inputs always give the same match.

- Movement, jetpack thrust, fuel, and collision with the rock.
- A rifle with aim, a small random spread, a magazine, reload, and shots that stop at rock.
  "Cannot see" and "cannot be shot" use the same line-of-sight check.
- Health, regeneration after a few seconds out of combat, death, and respawn at a hidden
  spawn with a short period of protection.
- A 60 second deathmatch. The team with more kills wins. Falling into a pit scores for the
  last opponent who damaged the falling agent.

The engine was checked against the promises the map generator makes. One full tank climbs
35.7 tiles, where 12 is the minimum the maps assume, and flies 30.0 tiles level, where 28.8
is required. A scripted flier completed 99.80% of 2,000 routes the generator claimed were
possible. On this laptop the engine alone runs about 74,000 match steps per second in 1v1.

### Learning interface

This is the part a learning agent actually uses.

- **Actions.** Each decision, an agent chooses left, stay, or right; jetpack on or off;
  shoot or not; and an aim angle. Aim is a continuous angle rather than a fixed set of
  directions. Measured on real maps, a perfect shot hits 98% of the time with a continuous
  angle, and 46% of the time with 16 fixed directions.
- **Vision.** An agent sees an enemy when a clear line reaches the enemy's centre or any
  corner of its body, at any distance. Checking corners as well as the centre misses 0.1%
  of enemies that could be shot; checking the centre only misses 10.6%. In 2v2, teammates
  share what they see.
- **Observations.** Each agent receives 330 numbers: its own state, two enemy slots, one
  teammate slot, the match score and time, and a 21 by 13 window of the tiles around it.
  The second team sees a mirror image of the map, so one network can play either side.
  Enemies that go out of sight are remembered for 10 seconds, then forgotten.
- **The training environment.** It runs a batch of matches, deals maps from a shuffled deck
  so every map is seen equally often, restarts a finished match immediately, and reports
  per-agent statistics at the end (kills, damage, time in the air, spawn camping, and
  others). A shared test locks this layout down, so a change cannot pass silently.
- **Physics variation.** Each match can scale gravity and thrust by the same factor. Scaling
  them separately by 15% was measured and rejected: level flight fell to 22 tiles, below the
  24 the maps were accepted with, so some gaps would become impossible. Scaling them
  together keeps level flight at 30 tiles for every factor from half to one and a half. It
  is off unless training turns it on, and agents are not told the factor.
- **Speed.** The full environment, including vision and observations, runs about 35,000
  match steps per second in 1v1 and 20,000 in 2v2, at 256 matches in one process. That is
  about as fast as the learning update is expected to consume them, so speeding the
  environment up further waits until training shows it is the slow part.
- **A standard wrapper.** A PettingZoo environment sits on top of one match, for outside
  tools and demos. It passes PettingZoo's own interface and seeding tests. Training uses
  the batched environment, which is about fifty times faster.

### Opponents, recordings, and a viewer

The learner now has fixed opponents, saved matches, and a way to watch them.

- **Random and static.** The random opponent drew every 1v1 over 128 matches and almost
  never scored. Beating it means the learner is actually landing kills.
- **Turret.** Stands still and shoots whatever comes into sight, with 2° of aim error and a
  short reaction delay. It uses the same line-of-sight check as an agent.
- **Rusher.** Flies toward the nearest living enemy using the map generator's own route
  planner, cached so the 112 ms planning cost is paid once per map, and shoots on the way.
  It killed a stationary opponent in all 16 test matches (4.88 kills each). In a duel the
  turret won more often than the rusher.
- **Replays.** Because a match is deterministic, a replay stores the seed and the actions,
  and playing it back reproduces the score. A matplotlib viewer writes a GIF and a filmstrip.

## What the teammate has built

In parallel, the learning half has been written through to self-play:

- a reward (win or loss, plus shaping that fades as training goes on);
- PPO, the policy-gradient algorithm, with a network that outputs the four action parts;
- a training loop, logging, checkpoints, and checks for degenerate behaviour such as camping
  or shooting blindly;
- a curriculum: hit a static target, beat a random opponent, then stronger scripted
  opponents, then self-play;
- evaluation with sides swapped and confidence intervals;
- an opponent pool, Elo ratings, and a 2v2 team-play architecture;
- styled opponents (aggressive, defensive, sniper) and tests on maps and physics the agent
  was not trained on.

That code steps the real environment: discrete move, jetpack and shoot, plus a continuous
aim angle, and the 330-number observation. The curriculum numbers below were measured on
that environment, against a static bot.

## Training so far

`runs/spawn-curriculum-14` trains PPO against a static bot and widens the gap between
spawns after the policy wins at least half of 16 validation matches of 60 seconds. Both
rungs that were actually trained keep the enemy in sight and inside the 30-tile rifle range.

| Rung | Best validation result | What happened next |
|---|---|---|
| 6–12 tiles, enemy in sight | 8 of 16 wins at update 175, 1.81 kills, 273 damage | The 50% gate opened. Checkpoint: `ckpt_tiles_8.pt` |
| 12–16 tiles, still in sight | 4 of 16 wins, more than once | The first 4 of 16 is update 225: 0.31 kills, 78 damage. That file is `ckpt_tiles_14.pt`. A later 4 of 16, update 325, has 0.62 kills and 120 damage. Checks after that are 1, 2, 3, or 4 of 16. The gate stayed shut |
| 16–28 tiles, beyond rifle range, and the official hidden spawns | Not run | The ladder stopped on the 12–16 tile rung |

A separate check of those two checkpoints, same seed and match length, found that the
12–16 tile weights do not beat the 6–12 tile weights on the longer maps (3 of 16 wins and
about 90 damage, against 3 of 16 and about 80 damage). Both checkpoints aimed with a spread
of about 59 degrees. A standing body at 14 tiles is about 7 degrees tall. Sixteen matches
is a wide interval: 8 of 16 has a Wilson 95 percent range of about 0.28 to 0.72.

The entropy in this log includes the aim cloud. A fresh policy under that definition sits
near 3.9. Update 175 logs 3.47, and the run ends at 3.25. The three button heads alone
cannot exceed about 2.48. The current code logs only those heads, so a later file near 2.0
is not “half as random” as 3.47.

After that run, the aim spread was capped at 6 degrees and the entropy bonus stopped
paying the policy to widen it. When no enemy is seen or remembered, the learner roams the
map's floors instead of standing still, and the learned aim returns as soon as an enemy is
in the observation. Those two changes are not inside the checkpoints above. Loading an old
checkpoint still clamps the stored spread to 6 degrees at aim time, which is a different
gun from the one that was trained. A new network with the 4-degree aim, before any
training, scored about one kill every 30 seconds when the enemy started in view. On four
official maps the roam covered 22–39 tiles in 60 seconds and did not see the static enemy.

## Self-play log

`runs/genuine-selfplay/metrics.jsonl` has 186 updates and walks the 6–12, 12–16, and 16–28
tile stages. It records mean return, entropy, and an aim loss. It does not record a
validation win rate. The last line is update 186, stage tiles_20, mean return about −0.05,
entropy about 2.0 on the discrete-only scale. That file is a health log of the update, not
a win-rate result. The opponent pool and Elo are in the code. They are not this run, and
they are not the 8 of 16.

## The play page

`scripts/play_human.py` serves http://127.0.0.1:8765. You are blue. The caves are the first
32 official validation maps, so the teams start where the cave placed them. The red fighter
during that match is the rusher: it flies the generator's routes toward your true position
and shoots on line of sight, with about 2 degrees of aim error. It is not the learned
network. The learned weights are what End and teach updates, four practice steps at
learning rate 0.0001. Rock presses are left out. The aim fit is squared angle error, not
the narrow Gaussian, because that Gaussian wiped weights to non-finite values. After the
lesson, the new weights play the previous weights. The weight card is the average absolute
parameter. It is a fingerprint of whether a lesson touched the file, not a skill score.
Restore reloads the original brain and keeps the saved matches. Forget deletes the matches
and keeps the brain.

The training dashboard on port 8501 reads `runs/*/metrics.jsonl` only. A play lesson writes
`runs/human-play/lessons.jsonl` and does not appear on that dashboard.

## Tests and write-up

525 tests were collected when this note was first written. Guides describe the map
generator, the engine, the learning interface, and the handoff between the two halves.
`archived/REPORT.md` is the earlier sign-off, including a test-split table of 0 wins. Use
this file for the current result.

## What is left

| Remaining work | Why it matters |
|---|---|
| Retrain with the 6-degree aim cap, and stop when wins stop rising | The saved checkpoints were trained while the spread stayed near 59 degrees. Update 600 is not the policy to keep |
| Make a finished kill worth more than healable damage | 100 damage and one kill are both 0.30 in the shaping term, and health returns at 10 per second after 3 quiet seconds |
| A wider validation check | 8 of 16 opened the gate. The Wilson interval is about 0.28 to 0.72 |
| A rung the policy can clear between 8 tiles and 16 tiles | The 12–16 tile rung never held 8 of 16 |
| Search from the official spawns | The teams start hidden. Roaming the floors did not produce a sighting in the four maps that were checked |
| Self-play with a validation win rate, after a policy that can shoot | The code is on main. `genuine-selfplay` does not record wins |

The route planner is cached per map. The next measurement is a training run that starts
from the capped aim, on the 6–12 tile rung, and stops when the validation wins stop rising.
