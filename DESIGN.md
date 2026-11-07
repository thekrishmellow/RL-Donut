# RL-Donut — Design & Decision Log

Mini Militia-inspired 2D multi-agent RL arena: jetpack movement, hitscan combat,
seeded procedural cave maps, PPO + self-play, generalization to unseen maps.

## Phases

| Phase | Scope | Milestones |
|---|---|---|
| 0 | Project setup | — |
| 1 | Seeded procedural map generator | 2 |
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
3. Quantile threshold to tile grid
4. Cellular-automata smoothing
5. Borders + left/right mirroring
6. Platforms and corridors
7. Connectivity + spawn detection
8. Validation + retry
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
| M4 | Flood-fill connectivity | 4-connectivity on "agent fits" tiles (open tile with open tile above; agent slot is 1×2) | Diagonal corners and 1-tile-high cracks are not passable |
| M5 | Jetpack reachability | Standing-spot graph with a fuel/climb budget must be strongly connected (every spot reachable and returnable) | Falling is free but climbing costs fuel, so deep smooth pits are traps; approximation gets validated against real physics in Phase 2 |
| M7 | Noise parameters | Random-angle unit gradients, quintic fade, 16×10 tile cells (stretched horizontally), 4 octaves, persistence 0.5, lacunarity 2 | Quintic removes visible cell-edge creases; stretched cells give ledge/shelf shapes suited to a side-scroller |
| E2 | Preview images | matplotlib (scripts only, never imported by the generator) | Labeled grids and colorbars; reused for training charts later |
| M6 | Disconnected regions | Fill small pockets, carve tunnels to large regions, reject if main region < 60% of open space; always carve passages across the mirror seam | Keeps maps playable without discarding too many seeds |

## Seed splits (planned)

| Split | Seeds |
|---|---|
| train | 0 – 9,999 |
| val | 10,000 – 10,999 |
| test | 20,000 – 20,999 (untouched until final eval) |
| OOD | 30,000+ with shifted generator/physics params |
