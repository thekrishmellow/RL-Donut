# RL-Donut — Design & Decision Log

Mini Militia-inspired 2D multi-agent RL arena: jetpack movement, hitscan combat,
seeded procedural cave maps, PPO + self-play, generalization to unseen maps.

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
7. Floating platforms, bottom pits, spawn points
8. Validation (jetpack reachability, fairness, line of sight) + retry
9. Batch generation, seed splits, cache, stats report

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
| M5 | Jetpack reachability | Standing-spot graph with a fuel/climb budget must be strongly connected (every spot reachable and returnable) | Falling is free but climbing costs fuel, so deep smooth pits are traps; approximation gets validated against real physics in Phase 2 |
| M7 | Noise parameters | Random-angle unit gradients, quintic fade, 16×10 tile cells (stretched horizontally), 4 octaves, persistence 0.5, lacunarity 2 | Quintic removes visible cell-edge creases; stretched cells give ledge/shelf shapes suited to a side-scroller |
| M8 | Rock cutoff | Quantile threshold (exact rock fraction), fraction sampled per seed: train 40–55%, OOD 30–35% / 60–65% | A fixed threshold gave 31–50% rock across 1,000 seeds at a 40% average; sampling keeps variety but makes it controlled and measurable |
| M9 | Height bias | Linear bias toward rock at the bottom, sampled per seed from 0.0–0.3 | Ground below, open sky above, like Mini Militia; sampling stops the agent assuming the ground is always at the bottom |
| M10 | Per-map parameters | Drawn from a dedicated `PARAMS` stream | Keeps the `TERRAIN` noise stream unchanged |
| M12 | Smoothing | Cellular automata, 3 passes: rock if ≥ 5 of 8 neighbours are rock, air if ≤ 3, unchanged at 4; outside the map counts as rock | Rough tiles per map fell 72 → 1 by pass 3, with little change after; gentler than erode/dilate, which can cut thin bridges |
| M13 | Mirror order | Generate left-half noise, mirror it, then threshold and smooth the full map | Smoothing a half map treats the seam as rock and grows a wall down the centre; the smoothing rule is symmetric, so symmetry survives every later step |
| M14 | Symmetry type | Left/right mirror (not 180° rotation) | With height bias, rotation would give one side the ground and the other the sky |
| M15 | Rock fraction after smoothing | Accept drift (mean +1%, max +3.4%); record target and final fraction (`Terrain.final_rock_fraction`) | Forcing exactness would undo part of the smoothing. Mirrored rock also comes in pairs, so the pre-smoothing count can be one tile over an odd target |
| M16 | Border | 1 tile of rock on all four sides (solid ceiling); engine also treats beyond top/sides as solid | Fastest motion is 20 tiles/s ÷ 30 ticks = 0.67 tiles per tick, so a 1-tile wall can't be skipped (Phase 2 test keeps this < 1); 1 tile costs ~6% of the map vs ~12% for 2; a solid ceiling stops off-screen camping |
| M17 | Pit layout | Count drawn uniformly from 0–2: 1 = centred on the seam, 2 = mirrored pair; width 3–6 tiles; shaft carved up from the bottom edge, at most 10 tiles deep; fewer pits if none fit, actual count recorded | Keeps symmetry; ≤ 6 tiles is crossable with ~1 s of jetpack; a centre pit creates a contested hazard |
| M18 | Pit timing | Pits are carved in step 7, after pocket filling, and must open into the main play region | The prototype carved a pit into a sealed air bubble that no one could reach |
| M19 | Kill line | Dead once the whole hitbox is more than `kill_margin` (default 3) tiles below the bottom edge; stored as `GeneratedMap.kill_y` | The fall stays visible briefly; one value shared by generator and engine |
| M20 | Tunnel digging | Cheapest path (Dijkstra over box positions): each step costs 1 plus 12 per rock tile in the box; every tunnel is mirrored and 2 × 3 wide; the border is never dug | Reuses existing caves and digs only gaps (mean 38 tiles dug per map); straight lines dig too much and look artificial; random walks are unpredictable |
| M21 | Small regions | Regions with < 60 usable positions are filled; larger ones are joined to the main region | "Fill small, join large" (M6) with a concrete threshold |
| M22 | Dead air | Fill every air tile no agent box in the main region can cover | Sealed bubbles and thin cracks are unusable; cracks would let bullets through gaps no agent can enter; what you see is exactly the playable space |
| M23 | Centre crossings | At least 2 separate routes across the centre line, or one opening ≥ 12 rows tall; extra crossings dug ≥ 6 rows from existing ones | A single narrow choke point makes camping it the dominant strategy. The first rule counted a 35–44-row open centre as "1 route" on 39 of 300 maps; those are wide open, not choke points |
| M24 | Step order | Step 6 = regions and corridors; step 7 = platforms, pits, spawns | Joining regions is the connectivity work; platforms must be placed in the final open spaces without blocking tunnels |
| M11 | Falling off the map | 0–2 bottom pits per seed (3–6 tiles wide, own `PITS` stream); engine kill line below the map is always on; curriculum starts with 0 pits | Fall deaths teach fuel management; sampling stops the agent assuming either layout; the kill line is also a safety net for physics bugs |
| G4 | Kill credit for falls | Deferred to Phase 5 (rewards) | It is a reward-design question |
| E2 | Preview images | matplotlib (scripts only, never imported by the generator) | Labeled grids and colorbars; reused for training charts later |
| M6 | Disconnected regions | Fill small pockets, carve tunnels to large regions, reject if main region < 60% of open space; always carve passages across the mirror seam | Keeps maps playable without discarding too many seeds |

## Seed splits (planned)

| Split | Seeds |
|---|---|
| train | 0 – 9,999 |
| val | 10,000 – 10,999 |
| test | 20,000 – 20,999 (untouched until final eval) |
| OOD | 30,000+ with shifted generator/physics params |
