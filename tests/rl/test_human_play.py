"""A person can play the agent, and the next bout uses what they did."""

import json
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from rl_donut.env.observe import OBS_DIM
from rl_donut.mapgen import load_split
from rl_donut.play.arena import HumanArena, duel_deck
from rl_donut.play.server import serve
from rl_donut.play.store import DemoStore
from rl_donut.rl.model import ActorCritic


@pytest.fixture(scope="module")
def duel_maps():
    deck, note = duel_deck(load_split("val"), limit=4)
    assert deck and "in sight" in note
    return deck, note


def _small_model():
    torch.manual_seed(0)
    return ActorCritic(OBS_DIM, (3, 2, 2), hidden_dim=32, cont_dim=1)


def test_saved_matches_round_trip_and_keep_their_order(tmp_path):
    store = DemoStore(tmp_path / "demos.npz")
    first = np.zeros((2, OBS_DIM), dtype=np.float32)
    first[:, 0] = 1
    second = np.zeros((1, OBS_DIM), dtype=np.float32)
    second[:, 1] = 2
    assert store.append(first, np.zeros((2, 3), np.int64), np.zeros((2, 1), np.float32), "ada", 11) == 0
    assert store.append(second, np.ones((1, 3), np.int64), np.full((1, 1), 0.2, np.float32), "bo", 12) == 1
    saved = store.load()
    assert store.count() == 3 and store.matches() == 2 and store.players() == ["ada", "bo"]
    np.testing.assert_array_equal(saved.obs[:, 0], [1, 1, 0])
    assert int(saved.seed[-1]) == 12 and str(saved.player[-1]) == "bo"
    store.clear()
    assert store.count() == 0


def test_a_short_match_is_saved_and_played_back_by_two_agents(tmp_path, duel_maps):
    deck, note = duel_maps
    arena = HumanArena(
        tmp_path, _small_model(), deck,         seconds=2.0, placement=note, practice_updates=0,
    )
    opening = arena.start()
    assert opening["width"] == 96 and "in sight" in opening["placement"]
    for _ in range(4):
        step = arena.step(1, True, False, 0.25)
    assert step["decisions"] == 4
    lesson = arena.teach("Ada Lovelace")
    saved = arena.store.load()
    assert arena.store.count() == 4
    np.testing.assert_array_equal(saved.discrete, np.tile([2, 1, 0], (4, 1)))
    np.testing.assert_allclose(saved.continuous, 0.25)
    assert str(saved.player[0]) == "Ada Lovelace"
    weights = lesson["status"]["weights"]
    assert "original" in weights and "now" in weights
    assert len(lesson["ai"]["frames"]) == 30
    assert lesson["ai"]["roles"] == ["after", "before"]
    assert "Nobody is playing" in lesson["ai"]["message"]
    with pytest.raises(ValueError):
        arena.teach("Ada Lovelace")
    original = torch.load(tmp_path / "ckpt_base.pt", map_location="cpu")["model_state"]["actor_backbone.0.bias"]
    with torch.no_grad():
        arena.model.actor_backbone[0].bias.add_(1)
    arena.restore_original()
    torch.testing.assert_close(arena.model.actor_backbone[0].bias, original)
    arena.forget()
    assert arena.store.count() == 0
    assert (tmp_path / "lessons.jsonl").exists()


def test_restore_puts_the_weight_number_back(tmp_path, duel_maps):
    deck, note = duel_maps
    arena = HumanArena(tmp_path, _small_model(), deck, placement=note, practice_updates=0)
    original = arena.status()["weights"]
    assert original["matches_original"]
    with torch.no_grad():
        arena.model.actor_backbone[0].bias.add_(0.4)
    moved = arena.status()["weights"]
    assert not moved["matches_original"]
    assert moved["now"] != original["now"]
    restored = arena.restore_original()["weights"]
    assert restored["matches_original"]
    assert abs(restored["now"] - restored["original"]) < 1e-8


def test_the_page_can_start_step_and_teach(tmp_path, duel_maps):
    deck, note = duel_maps
    arena = HumanArena(
        tmp_path, _small_model(), deck,         seconds=2.0, placement=note, practice_updates=0,
    )
    httpd = serve(arena, "127.0.0.1", 0, background=True)
    port = httpd.server_address[1]

    def call(method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())

    try:
        page = urlopen(f"http://127.0.0.1:{port}/", timeout=5)
        assert b"Fight your agent" in page.read()
        started = call("POST", "/api/match/start", {"seconds": 2})
        assert started["frame"]["agents"][0]["team"] == 0
        stepped = call("POST", "/api/match/step", {"move": 0, "jet": False, "shoot": True, "aim": 1.2})
        assert stepped["decisions"] == 1
        taught = call("POST", "/api/match/teach", {"player": "bo"})
        assert taught["learned"]["latest_rows"] == 1
        assert len(taught["ai"]["frames"]) == 30
        status = call("GET", "/api/status")
        assert status["matches"] == 1 and status["players"] == ["bo"]
    finally:
        httpd.shutdown()
        httpd.server_close()
