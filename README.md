# RL-Donut

A Mini Militia-inspired 2D arena for multi-agent reinforcement learning. Agents with
jetpacks and rifles fight on procedurally generated cave maps, learn through PPO and
self-play, and are evaluated on maps they have never seen.

**Status:** Phase 1 (procedural map generator) complete. Phase 2 (game engine) complete. Phase 3 (RL interface) in progress: action space, visibility, observations, the batched `MatchEnv`, the shared contract test, gravity/thrust randomization and the speed benchmark done.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | The whole project: roadmap, architecture, technical design per phase, infrastructure, risks, open decisions |
| [`DESIGN.md`](DESIGN.md) | Project phases and the decision log (every choice, with its reason) |
| [`docs/mapgen.md`](docs/mapgen.md) | How the map generator works, step by step, with code and the reasoning behind it |
| [`docs/engine.md`](docs/engine.md) | How the game engine works, step by step, and the promises it keeps to the map generator |
| [`docs/interface.md`](docs/interface.md) | The RL interface: actions, observations, the batched environment, with the measurements behind each choice |
| [`docs/handoff_phase5_6.md`](docs/handoff_phase5_6.md) | For the teammate building Phases 5–6: what exists, the interface contract, every step and decision, the merge checklist |

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
.venv/bin/python scripts/preview_rifle.py --count 3            # fans of rifle shots on real maps -> outputs/rifle_preview.png
.venv/bin/python scripts/view_match.py --team-size 2 --close   # watch a match: outputs/match.gif and a filmstrip outputs/match.png
.venv/bin/python scripts/bench_engine.py                       # engine speed by batch size, split by component
.venv/bin/python scripts/aim_ceiling.py                        # best possible hit rate of each aim design on real maps
.venv/bin/python scripts/preview_visibility.py                 # what one agent sees of enemies at every spot -> outputs/visibility_preview.png
.venv/bin/python scripts/bench_engine.py --visibility          # engine speed including visibility
.venv/bin/python scripts/preview_observation.py                # one observation drawn -> outputs/observation_preview.png
.venv/bin/python scripts/bench_env.py --state                  # full MatchEnv.step speed by batch size, split by part
.venv/bin/python scripts/bench_env.py --processes 1 2 4 8 --batches 256   # several env processes at once
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
├── docs/interface.md          RL interface guide
├── docs/handoff_phase5_6.md   Phases 5–6 handoff: contract, steps, merge plan
├── requirements.txt           Pinned dependencies
├── pyproject.toml             Package + pytest config
├── src/rl_donut/
│   ├── env/                   Phase 2: game engine (many matches per batch); Phase 3: RL interface
│   │   ├── actions.py         Action layout (3 discrete parts + continuous aim) and decode to engine controls
│   │   ├── raycast.py         Thousands of exact lines of sight at once (numpy grid walk)
│   │   ├── visibility.py      Who sees whom: centre or hitbox corners in sight, team sight
│   │   ├── observe.py         Observations: 57 named numbers + tile window, memory, team B mirrored
│   │   ├── match_env.py       MatchEnv: the batched RL environment (auto-reset, episode stats, critic state)
│   │   ├── config.py          EngineConfig and its parts: physics, rifle, life, match settings
│   │   ├── state.py           World: per-agent arrays (matches, agents), spawns, reset
│   │   ├── collision.py       Hitbox vs tile grid, one axis at a time; standing detection
│   │   ├── physics.py         One tick: controls, gravity, jetpack, speed limits, fuel
│   │   ├── rifle.py           Aiming, hitscan shots, cooldown, magazine and reload
│   │   ├── life.py            Damage, regeneration, death, respawn, spawn protection
│   │   ├── match.py           One decision step for every match: clock, scoring, events
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
│   ├── preview_rifle.py
│   ├── view_match.py          Debug viewer: one match as a GIF and a filmstrip
│   ├── bench_engine.py        Engine speed check
│   ├── bench_env.py           Full environment speed benchmark
│   ├── aim_ceiling.py         Hit-rate ceiling of each aim design
│   ├── preview_visibility.py  Visibility picture
│   ├── preview_observation.py Observation picture
│   └── preview_noise.py
├── tests/                     One test file per module, including golden-value tests
├── map_cache/                 Resumable build chunks (git-ignored)
└── outputs/                   Generated images and logs (git-ignored)
```

Later phases will add the PettingZoo wrapper to `src/rl_donut/env/`, plus
`src/rl_donut/rl/` (PPO and self-play), `src/rl_donut/eval/` and `dashboard/`.
