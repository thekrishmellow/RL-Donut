# RL-Donut

A Mini Militia-inspired 2D arena for multi-agent reinforcement learning. Agents with
jetpacks and rifles fight on procedurally generated cave maps, learn through PPO and
self-play, and are evaluated on maps they have never seen.

**Status:** Phase 1 (procedural map generator), steps 1–5 of 9 done.

## Documentation

| Document | What it covers |
|---|---|
| [`DESIGN.md`](DESIGN.md) | Project phases and the decision log (every choice, with its reason) |
| [`docs/mapgen.md`](docs/mapgen.md) | How the map generator works, step by step, with code and the reasoning behind it |

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
.venv/bin/python scripts/preview_maps.py --start 0 --count 8   # render maps -> outputs/maps_preview.png
.venv/bin/python scripts/preview_noise.py --seed 42            # noise settings sheet -> outputs/noise_preview.png
```

## Repository layout

```
RL-Donut/
├── DESIGN.md                  Phases + decision log
├── docs/mapgen.md             Map generator guide
├── requirements.txt           Pinned dependencies
├── pyproject.toml             Package + pytest config
├── src/rl_donut/
│   └── mapgen/                Phase 1: procedural maps
│       ├── seeding.py         Master seed -> independent named random streams
│       ├── generator.py       Pipeline entry point: generate_map(seed) -> GeneratedMap
│       ├── config.py          All generator settings (MapConfig, NoiseConfig, TerrainConfig, PitConfig)
│       ├── noise.py           Perlin noise + FBM
│       └── terrain.py         Mirrored noise -> rock/air grid -> smoothing
├── scripts/                   Preview tools (matplotlib; never imported by the generator)
│   ├── preview_maps.py
│   └── preview_noise.py
├── tests/                     One test file per module, including golden-value tests
└── outputs/                   Generated images (git-ignored)
```

Later phases will add `src/rl_donut/env/` (game engine and PettingZoo environment),
`src/rl_donut/rl/` (PPO and self-play), `src/rl_donut/eval/` and `dashboard/`.
