# RL-Donut — Design & Decision Log

Mini Militia-inspired 2D multi-agent RL arena: jetpack movement, hitscan combat,
seeded procedural cave maps, PPO + self-play, generalization to unseen maps.

This file logs approved decisions. The whole-project plan (architecture, draft designs for
later phases, infrastructure, open decisions) is in [`docs/PLAN.md`](docs/PLAN.md).

## Phases

| Phase | Scope | Milestones |
|---|---|---|
| 0 | Project setup | — |
| 1 | Seeded procedural map generator (incl. bottom pits) | 2 |
| 2 | Game engine (physics, jetpack, rifle, regen, respawns) | 1 |
| 3 | RL interface (observations, actions, PettingZoo wrapper) | 3 |
| 4 | Scripted bots, renderer, replays | — |
| 5 | Rewards + single-agent PPO vs bots | 4, 5 |
| 6 | Self-play, opponent pool, Elo, 1v1/2v2 | 6, 7, 8 |
| 7 | Adaptive opponents, style policies | 9 |
| 8 | Generalization evaluation | 10 |
| 9 | Dashboard | 11 |
| 10 | Benchmarks & report | 12 |

### Phase 1 steps

1. Seed system — done
2. Perlin noise + FBM — done
3. Mirrored noise + quantile threshold to tile grid — done
4. Cellular-automata smoothing — done
5. Border + kill line — done
6. Regions and corridors: usable positions, region joining, dead-air filling, centre crossings — done
7. Placement, split in three:
   - 7a. Jetpack reachability + floating platforms — done
   - 7b. Bottom pits — done (runs before 7a in the pipeline)
   - 7c. Spawn points — done
8. Validation + retry — done
9. Batch generation, seed splits, cache, stats report — done

### Phase 2 steps

Each step ends with a test or picture; its decisions are discussed when it starts.

1. World and state model — done
2. Tile collision — done
3. Movement, jetpack and fuel (climb and < 1 tile per tick contracts) — done
4. Route check (a scripted bot flies generator routes on many seeds, M4b): 99.80% arrive — done
5. Aiming and hitscan rifle (uses `geometry.line_of_sight`, M40) — done
6. Health, death and respawn — done
7. Match loop: tick order, timer, scoring, action repeat, event log; determinism test — done
8. Debug viewer (script only) and engine speed check — done

## Decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| G1 | Game style | 2D Mini Militia-like, structured observations | Fast to simulate; PPO converges without CNNs |
| G2 | Health regeneration | On | Matches Mini Militia; rewards retreat/cover tactics |
| G3 | Final format | Deferred (1v1 and/or 2v2) | Decide once the engine exists |
| E1 | Python env | venv + exactly pinned `requirements.txt` | Zero extra tooling |
| M1 | Noise source | Own vectorized numpy Perlin + FBM | We own the seed -> map contract; no stale/pre-1.0 deps; maps are tiny so speed is irrelevant |
| M2 | Seeding | `SeedSequence(master, spawn_key=(stream, attempt))` + explicit `PCG64` | Independent named streams; adding components never changes existing ones; retries use `attempt`, not `seed + k` |
| M3 | Reproducibility | Pin numpy, `GENERATOR_VERSION`, golden-value tests, cache val/test maps to disk | numpy only guarantees identical streams on the same build |
| M4 | Flood-fill connectivity | 4-connectivity over positions where a clearance box fits entirely in air; default box 2 wide × 3 tall (agent hitbox 0.8 × 1.6 plus margin), configurable; tunnels carved to the same clearance | Diagonal corners and tight cracks are not passable; the margin stops agents wedging in gaps exactly their size. On raw terrain the 1×2 box left 85% of air usable vs 64% for 2×3, so smoothing and carving must recover space |
| M4b | Physical route check (Phase 2) | Scripted bot follows routes between random points on many seeds; ≥ 99% must arrive | Proves the route search holds under real physics and momentum |
| M5 | Jetpack reachability | Standing-spot graph with a fuel/climb budget must be strongly connected (every spot reachable and returnable) | Falling is free but climbing costs fuel, so deep smooth pits are traps; approximation gets validated against real physics in Phase 2. Implemented by M26–M27; acceptance refined by M31 |
| M7 | Noise parameters | Random-angle unit gradients, quintic fade, 16×10 tile cells (stretched horizontally), 4 octaves, persistence 0.5, lacunarity 2 | Quintic removes visible cell-edge creases; stretched cells give ledge/shelf shapes suited to a side-scroller |
| M8 | Rock cutoff | Quantile threshold (exact rock fraction), fraction sampled per seed: train 40–55%, OOD 30–35% / 60–65% | A fixed threshold gave 31–50% rock across 1,000 seeds at a 40% average; sampling keeps variety but makes it controlled and measurable |
| M9 | Height bias | Linear bias toward rock at the bottom, sampled per seed from 0.0–0.3 | Ground below, open sky above, like Mini Militia; sampling stops the agent assuming the ground is always at the bottom |
| M10 | Per-map parameters | Drawn from a dedicated `PARAMS` stream, always at attempt 0 (M41); includes the pit count | Keeps the `TERRAIN` noise stream unchanged |
| M12 | Smoothing | Cellular automata, 3 passes: rock if ≥ 5 of 8 neighbours are rock, air if ≤ 3, unchanged at 4; outside the map counts as rock | Rough tiles per map fell 72 → 1 by pass 3, with little change after; gentler than erode/dilate, which can cut thin bridges |
| M13 | Mirror order | Generate left-half noise, mirror it, then threshold and smooth the full map | Smoothing a half map treats the seam as rock and grows a wall down the centre; the smoothing rule is symmetric, so symmetry survives every later step |
| M14 | Symmetry type | Left/right mirror (not 180° rotation) | With height bias, rotation would give one side the ground and the other the sky |
| M15 | Rock fraction after smoothing | Accept drift (mean +1%, max +3.4%); record target and final fraction (`Terrain.final_rock_fraction`) | Forcing exactness would undo part of the smoothing. Mirrored rock also comes in pairs, so the pre-smoothing count can be one tile over an odd target |
| M16 | Border | 1 tile of rock on all four sides (solid ceiling); engine also treats beyond top/sides as solid | Fastest motion is 20 tiles/s ÷ 30 ticks = 0.67 tiles per tick, so a 1-tile wall can't be skipped (Phase 2 test keeps this < 1); 1 tile costs ~6% of the map vs ~12% for 2; a solid ceiling stops off-screen camping |
| M17 | Pit layout | Count drawn uniformly from 0–2: 1 = centred on the seam, 2 = mirrored pair; width 3–6 tiles; shaft carved up from the bottom edge, at most 10 tiles deep; fewer pits if none fit, actual count recorded (superseded by M36: retry instead) | Keeps symmetry; ≤ 6 tiles is crossable with ~1 s of jetpack; a centre pit creates a contested hazard |
| M18 | Pit timing | Pits are carved in step 7, after pocket filling, and must open into the main play region | The prototype carved a pit into a sealed air bubble that no one could reach |
| M19 | Kill line | Dead once the whole hitbox is more than `kill_margin` (default 3) tiles below the bottom edge; stored as `GeneratedMap.kill_y` | The fall stays visible briefly; one value shared by generator and engine |
| M20 | Tunnel digging | Cheapest path (Dijkstra over box positions): each step costs 1 plus 12 per rock tile in the box; every tunnel is mirrored and 2 × 3 wide; the border is never dug; refined by M30 (staircases) | Reuses existing caves and digs only gaps (mean 38 tiles dug per map); straight lines dig too much and look artificial; random walks are unpredictable |
| M21 | Small regions | Regions with < 60 usable positions are filled; larger ones are joined to the main region | "Fill small, join large" (M6) with a concrete threshold |
| M22 | Dead air | Fill every air tile no agent box in the main region can cover | Sealed bubbles and thin cracks are unusable; cracks would let bullets through gaps no agent can enter; what you see is exactly the playable space |
| M23 | Centre crossings | At least 2 separate routes across the centre line, or one opening ≥ 12 rows tall; extra crossings dug ≥ 6 rows from existing ones | A single narrow choke point makes camping it the dominant strategy. The first rule counted a 35–44-row open centre as "1 route" on 39 of 300 maps; those are wide open, not choke points |
| M24 | Step order | Step 6 = regions and corridors; step 7 = platforms, pits, spawns | Joining regions is the connectivity work; platforms must be placed in the final open spaces without blocking tunnels |
| M11 | Falling off the map | 0–2 bottom pits per seed (3–6 tiles wide; count is a per-seed parameter, positions from the `PITS` stream); engine kill line below the map is always on; curriculum starts with 0 pits | Fall deaths teach fuel management; sampling stops the agent assuming either layout; the kill line is also a safety net for physics bugs |
| G4 | Kill credit for falls | Deferred to Phase 5 (rewards) | It is a reward-design question |
| E2 | Preview images | matplotlib (scripts only, never imported by the generator) | Labeled grids and colorbars; reused for training charts later |
| M25 | Step 7 split | 7a platforms + reachability, 7b pits, 7c spawns | Each piece is testable on its own; pits and spawns depend on knowing which floors are reachable |
| M26 | Climb contract | `MapConfig.climb_budget = 12` tiles; Phase 2 must prove the engine climbs at least that far on one full tank | Map generator and physics agree on one number, so "reachable" in the generator means reachable in the game |
| M27 | Flight model | Up costs 2 fuel units, sideways 1, down 0; tank = 2 × climb_budget; landing on any standing spot refuels | Simple, conservative (real flight can arc and use momentum); computed for all floors at once with numpy (~30 ms per map) |
| M28 | Platforms | Solid rock strips, 1 tile thick, 4–8 wide, 3 rows clear above and below, 2-tile side margin; mirrored pairs, or one centred platform near the seam | Solid (not one-way) keeps physics simple and gives cover; the clear air keeps standing room and the 2 × 3 clearance |
| M29 | Platform placement | Ledges only where reachability finds stranded floors (midpoint between the stranded floor and the nearest reachable one); cover at 1 per 800 air tiles, max 4; every placement keeps one region and the crossing rule, ledges must grow the reachable area, cover must never shrink it | A first metric (vertical drop only) passed 1 of 300 maps because it ignored sideways flight. "Stepping-stone" targeting was tried and removed: no gain, because no 8-row-tall pocket fits on the failing maps |
| M30 | Staircase tunnels | Digging vertically right after a vertical step costs 3× (`vertical_run_cost`), so tunnels climb as staircases | Straight dug shafts taller than the climb budget were one-way traps. Fully reachable maps went 78% → ~94%; digging per map is unchanged (39 vs 38 tiles) |
| M31 | Reachability acceptance | Accept a map when it is trap-free (every floor reachable from the main area can get back) and at most 2% of floor is unreachable; spawns only on the main area; step 8 rejects and retries the rest | Unreachable floor is never visited, so it is harmless in small amounts; traps are never acceptable. Strict 100% would reject maps for 2-tile notches |
| M32 | Unreachable air | Air no reachable flight can occupy is filled with rock when it is ≤ 3% of the map's air; larger unreachable caverns are left for step 8 to reject | Same idea as dead air (M22). Filling everything would delete up to 41% of a map's air (whole upper caverns behind tall necks) |
| M33 | Pit order | Pits are cut before platforms | Ledges can repair any reachability a pit breaks, and unreachable fill and cover see the final floors |
| M34 | Pit opening | Every pit column must open through floor in the main reachable group; the pit must keep one region and the crossing rule, strand no extra floor, and stay trap-free | Pits belong on walked paths; a pit under an unvisited cavern teaches nothing |
| M35 | No lids | Platforms may not use columns within 2 tiles of a pit | A ledge over a pit would make it harmless |
| M36 | Pit shortfall | If the requested count doesn't fit, step 8 rejects the map and retries with the next attempt (replaces "fewer pits" in M17) | Otherwise "1 pit" would quietly become "0 pits" on ~60% of maps and the curriculum would get fewer pits than it asks for. Costs ~1.4 attempts per finished map |
| M37 | Spawn layout | 6 spawns per team on main-group floor; team B mirrors team A; maps that can't fit 6 are retried | Mirroring makes spawns fair by construction; 6 covers 1v1 and 2v2 with spares for safe respawns |
| M38 | Spawn rules | No line of sight to any enemy spawn (either direction); ≥ 6 tiles from pits; ≥ 16 tiles from the centre line; ≥ 6 tiles between same-team spawns | Prevents spawn kills and spawning into hazards. Spacing 8 fit only 82.5% of maps; 6 (three agent widths) fits 92% |
| M39 | Spawn selection | Random order from the `SPAWNS` stream, keeping spots that pass every rule | Farthest-point picking repeats the same pattern on every map, which agents could memorise |
| M40 | Line of sight | Exact grid walk (Amanatides–Woo), corners check both side tiles; shared in `geometry.py` and to be reused by Phase 2 hitscan | Quarter-tile sampling missed 46 of 10,000 corner clips; one function means "can't see" and "can't be shot" always agree |
| M41 | Fixed design settings | A seed's rock fraction, height bias and pit count (`MapParams`) come from attempt 0; retries re-draw only geometry | Pass rates depend on these settings (34% with a centred pit vs 93% with none); re-drawing them would skew kept maps to a 49/18/34% pit mix and favour low rock. Costs 1.97 vs ~1.6 attempts per map |
| M42 | Validation checks | Reject on trap, > 2% stranded floor, pit shortfall, spawn shortfall; plus safety re-checks of symmetry, one region + crossings, and the border | The first four are the acceptance rules from M31, M36, M37; the re-checks are cheap and catch future bugs |
| M43 | Retry cap | 30 attempts per seed, then `MapGenerationError` | Worst seen in 3,000 seeds was 24; expected ~0.1 seeds per 12,000 hit the cap |
| M44 | Minimum open space | No check | Only 0.2% of maps have under 30% air |
| M45 | Map cache format | Per split: one compressed `.npz` of grids plus a `.json` manifest (versions, full `MapConfig`, one line per map with metadata and grid hash); loading refuses version, config, count or hash mismatches | Loading 10,000 maps takes seconds instead of hours; the checks stop stale maps being mixed with new settings. One line per map keeps git diffs readable. Pickle was rejected: unreadable, tied to class layouts, unsafe to load |
| M46 | Commit the cache | All splits, including train (~15 MB), are committed in `data/maps/` | Every clone trains and evaluates on identical maps, even if numpy's random streams change later |
| M47 | Freeze policy | Any change to generator output bumps `GENERATOR_VERSION` and rebuilds every split; a drift test regenerates a few val and test seeds and compares hashes | Results across generator versions are not comparable, so changes must be loud |
| M48 | Exhausted seeds | A seed that fails all attempts is skipped and listed in the manifest | Keeps seed ranges fixed; no hand-picked replacements. In practice 3 of 10,000 train seeds (all centred pit + rock ≥ 0.53 + height bias ≥ 0.25), 0 in val and test |
| M49 | OOD splits | Deferred to Phase 8 | Which shifts are worth testing depends on what the agents learn |
| P1 | Engine state layout | numpy arrays of shape (matches, agents) covering many matches at once; `World.single` and `World.agent` for tests and debugging | Measured: ~470,000 ticks/s for 64 matches together vs ~48,000 for Python objects and ~12,000 for single-match arrays (numpy's ~1 µs per call dominates small arrays); this laptop is usually loaded, so multiple processes scale poorly; switching later means a rewrite. numba kept as a fallback (new dependency, version-sensitive) |
| P2 | Coordinates | Continuous tile units, y downward, position = hitbox centre, float64 | Tile lookup is `grid[floor(y), floor(x)]`; mirroring is `width - x`; aim starts at the centre; float64 keeps replays exact |
| P3 | Start positions | Agent `i` of each team starts on spawn `i` of its team; team A slots first | Spawns are mirrored, so every match starts as an exact mirror image; respawn selection is decided in step 2.6 |
| P4 | Collision method | One axis at a time, horizontal then vertical; check only the tiles the leading edge enters; snap to the tile edge and zero that axis's speed; `move` refuses moves of a full tile or more | Exact given < 1 tile per tick (M16); cheap for many matches at once. An exact sweep is harder to batch for at most one tick of difference at corners |
| P5 | Standing | Feet on a tile edge (within 1e-6) with rock under the hitbox, computed from position | Correct right after reset or respawn and when standing still; a "fall was blocked" flag would not be |
| P6 | Outside the map | Left, right and top solid; below the bottom edge is air | Matches the border (M16); the bottom border is solid except at pits, so pits are the only way down |
| P7 | Agent-agent collision | None; agents pass through each other | As in Mini Militia; solid bodies would allow doorway blocking and pushing into pits |
| P8 | Physics constants | Gravity 25, thrust 40 (tiles/s²); rising limit 12 and falling limit 20 tiles/s; run 6 tiles/s; tank 100, drain 32/s, refill 25/s while standing and not thrusting | Simulated: without a rising limit a full tank reaches 38 tiles/s (1.27 tiles per tick, breaks M16); the draft's drain of 40/s gave 24.4 tiles of level flight against the generator's 24. Chosen numbers climb 35.7 tiles and fly 30 tiles level |
| P9 | Horizontal control | Speed set directly to ±run speed, on the ground and in the air; no momentum | Mini Militia feel; easier to learn; exactly predictable for the route check |
| P10 | Update rule | Semi-implicit Euler: speed, then position | Stable and deterministic at a fixed timestep |
| P11 | Engine contract tests | Climb ≥ climb_budget; level flight ≥ 1.2 × 2 × climb_budget; every speed limit < 1 tile per tick (enforced by config); no jump, no fall damage | Requires real margin over the generator's model, not a tie; gravity/thrust randomization (Phase 3) must keep these at its extremes, which ±15% on both does not (22 tiles of level flight) |
| P12 | Route planner | Rebuild the generator's own cheapest flight (`mapgen.reach` costs) hop by hop, preferring falls when tracing back from the landing; replan after every landing | Tests the generator's claim itself rather than whatever a cleverer search finds; a search over real physics would pass maps the model got wrong. Preferring falls avoids flights that end a fall in mid-air, whose braking the model doesn't charge |
| P13 | Route controller | Aim at the end of each straight run of boxes; thrust by stopping distance; on the ground, stand still (no walking, no thrust) until the tank holds the hop's fuel | Walking while refuelling stepped bots off ledges with 2–15 fuel: it caused most falls and the looping timeouts (98.55% → 99.80%) |
| P14 | Route check rule | 2,000 routes on 200 val maps (`scripts/route_check.py`), 150 on 30 maps in the tests; 60 s per route, pits included; full tank before each hop; ≥ 99% must arrive | Full tank tests whether every flight the generator promised fits in one tank. Result 99.80%; the 3 falls are 21–24-unit hops (tank 24) with 20+ tiles of sideways flight, 1 timeout loops. Just-enough fuel: 99.45% |
| P15 | Shot resolution | Hitscan from the hitbox centre: nearest living enemy hitbox the ray enters within range, then the generator's `line_of_sight` to the entry point; one shot at a time in Python | Exact and shares M40's rule. Measured on a loaded laptop: 4–19 µs per call, so even every agent firing at the maximum rate adds 95 µs (1v1) to 190 µs (2v2) to a 380 µs 64-match tick; a batched numpy ray walk would cost about 600 µs whatever the shot count. numba if step 8 says so. Checked: 0 of 36,000 spawn-to-enemy-spawn shots hit within range (0.9% would at unlimited range) |
| P16 | Rifle numbers | 15 damage, 0.2 s between shots, 30-round magazine, automatic 1.5 s reload when empty, unlimited spares, 30-tile range, uniform ±2° spread | 7 hits (about 1.2 s) to kill. Aimed at the centre, 10-tile shots always hit; at 30 tiles about 76% hit a target to the side (spread moves the shot across the 1.6-tile height) and 38% one straight above or below (0.8-tile width) |
| P17 | Aim | Any angle per agent, applied instantly; 0 = right, π/2 = down; team A starts aiming right, team B left | Keeps the engine independent of the Phase 3 action design (16 fixed directions are too coarse to hit at range) |
| P18 | Friendly fire and randomness | Teammates are neither hit nor in the way; spread comes from one random stream per match, seeded from (world seed, map seed, episode) | As in Mini Militia team modes, and RL can't learn to hurt teammates; per-match streams keep a match identical whatever else is in its batch |
| P19 | Same-tick hits | All hits fired on a tick are applied together, so trades kill both; the killer is the top damage dealer on the killing tick, ties broken by the match's random stream | Agent-order resolution would hand every tie to the lower-numbered agent, a bias RL would learn. In random 2v2 fights 45 of 286 kills were trades |
| P20 | Health and death | 100 health; regenerates 10/s after 3 s without damage; death at 0 health or below the kill line; a fall has no killer in the engine but records the last damager and how long before | Regeneration per G2; 13 s from almost dead to full makes retreat a real choice against a 1.2 s time to kill. Fall credit is G4's decision (Phase 5), so the engine only records the facts |
| P21 | Respawn | After 2 s, at a random own spawn no living enemy can see (`line_of_sight` centre to centre, any distance), else the one farthest from the nearest enemy; full health, fuel and ammo, aim towards the centre | Prevents spawn camping; random choice avoids a predictable spawn. Fallback used 5 times in 303 respawns of random fights |
| P22 | Spawn protection | 1 s without damage after respawning; firing ends it | As in Mini Militia; stops protection being used to attack for free |
| P23 | Decision rate | Agents decide every 2 ticks (15/s); each action is held for both; tick order physics → rifle → life | About human reaction time; 900 steps per 60 s match. Every tick doubles learning steps for little gain; every 3 ticks makes aim and jetpack coarse |
| P24 | Match format | 60 s timed deathmatch; kills score for the team; more kills wins, equal is a draw, no overtime; a finished match must be reset before it is stepped again | A fixed length keeps batched matches in step; first-to-N would vary episode length |
| P25 | Pit-fall scoring | The last enemy to damage the faller within 3 s (`fall_credit_window`) scores; otherwise nobody | Knocking someone into a pit counts; accidents give nothing. Death records keep the facts, so the reward (G4, Phase 5) can still use its own rule |
| P26 | New maps mid-batch | `World.load_maps(matches, maps)` swaps maps into chosen matches and restarts only those | Phase 3 can restart a finished match on a new map while the batch plays on |
| P27 | Debug viewer | `scripts/view_match.py`: matplotlib, saves a GIF and a filmstrip PNG, simple scripted fighters | No new dependency (Pillow comes with matplotlib); pictures drop into docs. A live window or real renderer is Phase 4's decision |
| P28 | Engine speed check | `scripts/bench_engine.py`, results in `docs/engine.md`, no timing assertion in tests; bar ≥ 50,000 match decision steps/s at 64 matches 1v1 | The laptop's load swings, so timing tests would fail at random. Measured 73,700 (1v1) and 55,200 (2v2) at 64 matches; the per-shot Python rifle is 25–59% of the time and the first thing to speed up |
| M6 | Disconnected regions | Fill small pockets, carve tunnels to large regions, reject if main region < 60% of open space; always carve passages across the mirror seam | Keeps maps playable without discarding too many seeds |

## Seed splits

Defined in `src/rl_donut/mapgen/splits.py`; built maps live in `data/maps/`.

| Split | Seeds |
|---|---|
| train | 0 – 9,999 |
| val | 10,000 – 10,999 |
| test | 20,000 – 20,999 (untouched until final eval) |
| OOD | 30,000+ with shifted generator/physics params (Phase 8) |
