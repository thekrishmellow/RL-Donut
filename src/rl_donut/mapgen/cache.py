"""Save and load finished maps for a split.

Each split is two files in the cache directory: `<split>.npz` holds every grid (compressed),
and `<split>.json` holds a header (generator version, numpy version, full MapConfig) plus
one record per map with its metadata and a hash of its grid. Preview layers are not stored.
Loading refuses caches built with a different generator version or config, and checks
every grid against its hash.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl_donut.mapgen.config import MapConfig
from rl_donut.mapgen.generator import GeneratedMap
from rl_donut.mapgen.pits import Pit
from rl_donut.mapgen.platforms import Platform
from rl_donut.mapgen.seeding import GENERATOR_VERSION
from rl_donut.mapgen.spawns import Spawn, Spawns
from rl_donut.mapgen.terrain import MapParams

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "maps"


class CacheError(RuntimeError):
    pass


def grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()[:16]


def config_json(cfg: MapConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True)


def map_record(m: GeneratedMap) -> dict:
    return {
        "seed": m.seed,
        "attempt": m.attempt,
        "hash": grid_hash(m.grid),
        "params": asdict(m.params),
        "terrain_rock_fraction": m.terrain_rock_fraction,
        "kill_y": m.kill_y,
        "tunnels": m.tunnels,
        "crossings": m.crossings,
        "widest_crossing": m.widest_crossing,
        "pits": [asdict(p) for p in m.pits],
        "pits_requested": m.pits_requested,
        "platforms": [asdict(p) for p in m.platforms],
        "spawns": {"team_a": [asdict(s) for s in m.spawns.team_a],
                   "team_b": [asdict(s) for s in m.spawns.team_b]},
        "fully_reachable": m.fully_reachable,
        "trap_free": m.trap_free,
        "stranded_fraction": m.stranded_fraction,
        "rejections": [list(r) for r in m.rejections],
    }


def map_from_record(record: dict, grid: np.ndarray, generator_version: int) -> GeneratedMap:
    return GeneratedMap(
        seed=record["seed"],
        attempt=record["attempt"],
        generator_version=generator_version,
        grid=grid,
        params=MapParams(**record["params"]),
        terrain_rock_fraction=record["terrain_rock_fraction"],
        kill_y=record["kill_y"],
        tunnels=record["tunnels"],
        crossings=record["crossings"],
        widest_crossing=record["widest_crossing"],
        pits=tuple(Pit(**p) for p in record["pits"]),
        pits_requested=record["pits_requested"],
        platforms=tuple(Platform(**p) for p in record["platforms"]),
        spawns=Spawns(tuple(Spawn(**s) for s in record["spawns"]["team_a"]),
                      tuple(Spawn(**s) for s in record["spawns"]["team_b"])),
        fully_reachable=record["fully_reachable"],
        trap_free=record["trap_free"],
        stranded_fraction=record["stranded_fraction"],
        rejections=tuple(tuple(r) for r in record["rejections"]),
    )


def save_split(name: str, maps: list[GeneratedMap], skipped: list[dict], cfg: MapConfig,
               root: Path = DATA_DIR) -> None:
    """`skipped` lists seeds that exhausted their attempts, with their rejection reasons."""
    root.mkdir(parents=True, exist_ok=True)
    maps = sorted(maps, key=lambda m: m.seed)
    np.savez_compressed(root / f"{name}.npz", grids=np.stack([m.grid for m in maps]).astype(np.uint8))
    header = {
        "split": name,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "config": json.loads(config_json(cfg)),
        "skipped": skipped,
    }
    # One compact line per map keeps the file small and makes git diffs show which maps changed.
    records = ",\n".join("  " + json.dumps(map_record(m)) for m in maps)
    text = json.dumps(header, indent=1)[:-2] + ',\n "maps": [\n' + records + "\n ]\n}\n"
    (root / f"{name}.json").write_text(text)


def load_split(name: str, cfg: MapConfig | None = None, root: Path = DATA_DIR) -> list[GeneratedMap]:
    cfg = cfg or MapConfig()
    manifest_path, grids_path = root / f"{name}.json", root / f"{name}.npz"
    if not manifest_path.exists() or not grids_path.exists():
        raise CacheError(f"split '{name}' is not built; run scripts/build_maps.py --split {name}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise CacheError(f"split '{name}' was built by generator version {manifest['generator_version']}, "
                         f"current is {GENERATOR_VERSION}")
    if json.dumps(manifest["config"], sort_keys=True) != config_json(cfg):
        raise CacheError(f"split '{name}' was built with a different MapConfig")
    with np.load(grids_path) as data:
        grids = data["grids"]
    records = manifest["maps"]
    if len(grids) != len(records):
        raise CacheError(f"split '{name}': {len(grids)} grids but {len(records)} records")
    maps = []
    for record, grid in zip(records, grids):
        if grid_hash(grid) != record["hash"]:
            raise CacheError(f"split '{name}': grid for seed {record['seed']} does not match its hash")
        maps.append(map_from_record(record, grid, manifest["generator_version"]))
    return maps
