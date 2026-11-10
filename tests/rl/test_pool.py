import numpy as np
import pytest

from rl_donut.rl.bots import RandomBot, StaticTargetBot
from rl_donut.rl.pool import OpponentPool


def test_pool_initialization():
    pool = OpponentPool()
    assert len(pool.bots) == 2
    assert len(pool.snapshots) == 0
    assert pool.latest_snapshot is None


def test_pool_sampling_distribution():
    pool = OpponentPool()
    bot1 = StaticTargetBot("bot_1")
    bot2 = StaticTargetBot("bot_2")
    pool._add_snapshot(bot1, update=50)
    pool._add_snapshot(bot2, update=100)

    assert len(pool.snapshots) == 2
    assert pool.latest_snapshot.name == "snapshot_0100"

    # Sample 100 opponents and check they are valid policies
    rng = np.random.default_rng(42)
    sampled = [pool.sample_opponent(rng).name for _ in range(100)]
    assert len(sampled) == 100


def test_pool_pruning_capacity():
    pool = OpponentPool(max_snapshots=3)
    for i in range(1, 6):
        pool._add_snapshot(StaticTargetBot(f"bot_{i}"), update=i * 10)

    # Max snapshots is 3, so oldest 2 should be pruned
    assert len(pool.snapshots) == 3
    names = [s.name for s in pool.snapshots]
    assert names == ["snapshot_0030", "snapshot_0040", "snapshot_0050"]

