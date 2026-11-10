# RL-Donut

A Mini Militia-inspired 2D arena for multi-agent reinforcement learning. Agents with
jetpacks and rifles fight on procedurally generated cave maps, learn through PPO and
self-play, and are evaluated on maps they have never seen.

**Status:** Phase 1 (procedural map generator) complete. Phase 2 (game engine): steps 1–4 of 8 done.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | The whole project: roadmap, architecture, technical design per phase, infrastructure, risks, open decisions |
| [`DESIGN.md`](DESIGN.md) | Project phases and the decision log (every choice, with its reason) |
| [`docs/mapgen.md`](docs/mapgen.md) | How the map generator works, step by step, with code and the reasoning behind it |
| [`docs/engine.md`](docs/engine.md) | How the game engine works, step by step, and the promises it keeps to the map generator |

## Setup

Requires Python 3.11+ (developed on 3.13).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Dependency versions are pinned exactly. `numpy` in particular must not be upgraded casually:
map generation depends on its random number streams, and the golden tests will fail if they change.

## Everyday commands

```bash
.venv/bin/python -m pytest -q                                  # run all tests
.venv/bin/python scripts/preview_maps.py --start 0 --count 8   # render validated maps -> outputs/maps_preview.png (--raw: first attempts)
.venv/bin/python scripts/preview_noise.py --seed 42            # noise settings sheet -> outputs/noise_preview.png
.venv/bin/python scripts/build_maps.py --split val --workers 8 # (re)build a split -> data/maps/ (resumable)
.venv/bin/python scripts/preview_world.py --count 4            # engine worlds: agents and spawns -> outputs/world_preview.png (--trace N: random-motion paths)
.venv/bin/python scripts/preview_physics.py                    # one tank's flight paths vs the generator's needs -> outputs/physics_preview.png
.venv/bin/python scripts/route_check.py                        # route bot flies 2,000 generator routes; failures -> outputs/route_failures.png
```

The train, val and test maps are already built and committed in `data/maps/`; load them with
`rl_donut.mapgen.load_split("train")`. Rebuild only after changing the generator (see the
freeze policy in `docs/mapgen.md`, step 9).

## Repository layout

```
RL-Donut/
├── DESIGN.md                  Phases + decision log
├── docs/PLAN.md               Whole-project plan and technical design
├── docs/mapgen.md             Map generator guide
├── docs/engine.md             Game engine guide
├── requirements.txt           Pinned dependencies
├── pyproject.toml             Package + pytest config
├── src/rl_donut/
│   ├── env/                   Phase 2: game engine (many matches per batch)
│   │   ├── config.py          EngineConfig and PhysicsConfig: team size, tick rate, hitbox, physics constants
│   │   ├── state.py           World: per-agent arrays (matches, agents), spawns, reset
│   │   ├── collision.py       Hitbox vs tile grid, one axis at a time; standing detection
│   │   ├── physics.py         One tick: controls, gravity, jetpack, speed limits, fuel
│   │   └── navigation.py      Route planner on the generator's flight model, and a bot that flies it
│   └── mapgen/                Phase 1: procedural maps
│       ├── seeding.py         Master seed -> independent named random streams
│       ├── generator.py       Pipeline entry point: generate_map(seed) -> GeneratedMap
│       ├── config.py          All generator settings (MapConfig and one config per step)
│       ├── noise.py           Perlin noise + FBM
│       ├── terrain.py         Mirrored noise -> rock/air grid -> smoothing
│       ├── regions.py         One connected play space: staircase tunnels, dead-air filling, centre crossings
│       ├── pits.py            Bottom pits: shafts through the bottom edge, mirrored or centred
│       ├── reach.py           Jetpack reachability between floors: traps, stranded floors, occupiable air
│       ├── platforms.py       Floating ledges and cover, unreachable-air filling
│       ├── spawns.py          Mirrored spawn points with no line of sight between teams
│       ├── geometry.py        Exact line of sight on the grid (shared with future hitscan)
│       ├── validation.py      Accept/reject finished maps; generate_valid_map retries rejected seeds
│       ├── splits.py          Seed ranges for train / val / test
│       └── cache.py           Save and load built splits, with version, config and hash checks
├── data/maps/                 Built splits: <split>.npz grids, <split>.json manifest, <split>.md stats
├── scripts/                   Tools (matplotlib; never imported by the generator)
│   ├── build_maps.py          Generate a whole split in parallel and write its stats report
│   ├── preview_maps.py
│   ├── preview_world.py
│   ├── preview_physics.py
│   ├── route_check.py
│   └── preview_noise.py
├── tests/                     One test file per module, including golden-value tests
├── map_cache/                 Resumable build chunks (git-ignored)
└── outputs/                   Generated images and logs (git-ignored)
```

Later phases will add the PettingZoo wrapper to `src/rl_donut/env/`, plus
`src/rl_donut/rl/` (PPO and self-play), `src/rl_donut/eval/` and `dashboard/`.
