import numpy as np

from rl_donut.env import EngineConfig, MatchConfig, MatchEnv
from rl_donut.env.bots import RandomBot, StaticTargetBot
from rl_donut.env.replay import load, play, record, save, snapshots
from rl_donut.mapgen import load_split


def test_replay_reproduces_the_match_and_roundtrips(tmp_path):
    maps = load_split("val")[:1]
    cfg = EngineConfig(match=MatchConfig(duration=4.0))

    def fresh():
        return MatchEnv(1, seed=3, cfg=cfg, maps=maps)

    replay = record(fresh(), RandomBot(), StaticTargetBot(), seed=1)
    info = play(replay, maps)
    assert info["score"].tolist() == replay["score"]
    assert info["outcome"].tolist() == replay["outcome"]
    assert record(fresh(), RandomBot(), StaticTargetBot(), seed=1)["steps"] == replay["steps"]

    path = tmp_path / "match.json.gz"
    save(path, replay)
    assert load(path)["map_seed"] == replay["map_seed"] == maps[0].seed

    _, frames = snapshots(replay, maps)
    assert len(frames) == len(replay["steps"])
    assert frames[-1]["tick"] == int(4.0 * cfg.tick_rate)
    assert np.array(frames[-1]["score"]).tolist() == replay["score"]