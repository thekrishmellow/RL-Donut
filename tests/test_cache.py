import json
from dataclasses import replace

import numpy as np
import pytest

from rl_donut.mapgen import MapConfig, generate_valid_map
from rl_donut.mapgen.cache import DATA_DIR, CacheError, grid_hash, load_split, map_record, save_split
from rl_donut.mapgen.splits import SPLITS

CFG = MapConfig()


@pytest.fixture(scope="module")
def maps():
    return [replace(generate_valid_map(s, CFG), layers={}) for s in (0, 1, 4)]


@pytest.fixture
def saved(tmp_path, maps):
    save_split("val", maps, [{"seed": 99, "error": "test"}], CFG, root=tmp_path)
    return tmp_path


# --- splits ---

def test_splits_never_overlap():
    ranges = sorted((s.start, s.start + s.count) for s in SPLITS.values())
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(ranges, ranges[1:]))


def test_split_sizes():
    assert {n: s.count for n, s in SPLITS.items()} == {"train": 10_000, "val": 1_000, "test": 1_000}


# --- round trip ---

def test_round_trip_keeps_every_field(saved, maps):
    loaded = load_split("val", CFG, root=saved)
    assert len(loaded) == len(maps)
    for a, b in zip(loaded, maps):
        assert np.array_equal(a.grid, b.grid)
        assert map_record(a) == map_record(b)
        assert (a.generator_version, a.params, a.spawns, a.pits, a.platforms) == \
               (b.generator_version, b.params, b.spawns, b.pits, b.platforms)


def test_manifest_records_skipped_seeds_and_versions(saved):
    manifest = json.loads((saved / "val.json").read_text())
    assert manifest["skipped"] == [{"seed": 99, "error": "test"}]
    assert manifest["numpy_version"] == np.__version__


def test_missing_split_raises(tmp_path):
    with pytest.raises(CacheError):
        load_split("val", CFG, root=tmp_path)


def test_different_config_is_refused(saved):
    with pytest.raises(CacheError):
        load_split("val", replace(CFG, max_stranded=0.05), root=saved)


def test_different_generator_version_is_refused(saved):
    path = saved / "val.json"
    manifest = json.loads(path.read_text())
    manifest["generator_version"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(CacheError):
        load_split("val", CFG, root=saved)


def test_corrupted_grid_is_refused(saved):
    with np.load(saved / "val.npz") as data:
        grids = data["grids"].copy()
    grids[0, 5, 5] ^= 1
    np.savez_compressed(saved / "val.npz", grids=grids)
    with pytest.raises(CacheError):
        load_split("val", CFG, root=saved)


# --- frozen splits in the repository ---

@pytest.mark.parametrize("name", ["val", "test"])
def test_frozen_split_matches_the_generator(name):
    """Fails when the generator's output drifts from the committed evaluation maps."""
    if not (DATA_DIR / f"{name}.json").exists():
        pytest.skip(f"split '{name}' not built")
    manifest = json.loads((DATA_DIR / f"{name}.json").read_text())
    for record in manifest["maps"][:3]:
        regenerated = generate_valid_map(record["seed"], CFG)
        assert regenerated.attempt == record["attempt"]
        assert grid_hash(regenerated.grid) == record["hash"]


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_committed_splits_load(name):
    if not (DATA_DIR / f"{name}.json").exists():
        pytest.skip(f"split '{name}' not built")
    maps = load_split(name, CFG)
    skipped = json.loads((DATA_DIR / f"{name}.json").read_text())["skipped"]
    assert len(maps) + len(skipped) == SPLITS[name].count
    assert [m.seed for m in maps] == sorted(m.seed for m in maps)
