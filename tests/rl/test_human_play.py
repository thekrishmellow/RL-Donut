"""A person can play the agent, and the next bout uses what they did."""

import json
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from rl_donut.env.observe import OBS_DIM
from rl_donut.mapgen import load_split
from rl_donut.play.arena import HumanArena, duel_deck
from rl_donut.play.learn import learn_from_demos, row_weights
from rl_donut.play.server import serve
from rl_donut.play.store import DemoStore
from rl_donut.rl.clone import Demonstrations
from rl_donut.rl.model import ActorCritic


@pytest.fixture(scope="module")
def duel_maps():
    deck, note = duel_deck(load_split("val"), limit=4)
    assert deck and "in sight" in note
    return deck, note


def _small_model():
    torch.manual_seed(0)
    return ActorCritic(OBS_DIM, (3, 2, 2), hidden_dim=32, cont_dim=1)


def test_row_weights_favour_the_newest_match():
    weights = row_weights(np.array([0, 0, 1, 1]), 4)
    np.testing.assert_array_equal(weights, [1, 1, 4, 4])


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


def test_learning_raises_the_chance_of_the_played_action():
    torch.manual_seed(0)
    model = ActorCritic(8, (3, 2, 2), hidden_dim=32, cont_dim=1)
    obs = np.zeros((64, 8), dtype=np.float32)
    obs[:, 0] = 1
    discrete = np.tile(np.array([[2, 1, 1]], dtype=np.int64), (64, 1))
    continuous = np.full((64, 1), 0.3, dtype=np.float32)
    demos = Demonstrations(obs, discrete, continuous, np.zeros(64, dtype=np.int64))
    report = learn_from_demos(
        model, demos, epochs=12, lr=1e-3, batch_size=32, anchor_coef=0.0, seed=0
    )
    assert report.improved
    assert report.logprob_after > report.logprob_before + 0.2


def test_a_lesson_that_does_not_fit_keeps_the_starting_weights():
    torch.manual_seed(0)
    model = ActorCritic(8, (3, 2, 2), hidden_dim=16, cont_dim=1)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    obs = np.zeros((16, 8), dtype=np.float32)
    demos = Demonstrations(
        obs, np.zeros((16, 3), dtype=np.int64), np.zeros((16, 1), dtype=np.float32),
        np.zeros(16, dtype=np.int64),
    )
    report = learn_from_demos(model, demos, epochs=2, lr=0.0, anchor_coef=0.0, seed=0)
    assert report.improved is False
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_a_short_match_is_saved_and_played_back_by_two_agents(tmp_path, duel_maps):
    deck, note = duel_maps
    arena = HumanArena(
        tmp_path, _small_model(), deck, seconds=2.0, placement=note,
        learn_kwargs={"epochs": 1, "lr": 1e-4, "anchor_coef": 0.25},
        practice_updates=0,
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


def test_the_page_can_start_step_and_teach(tmp_path, duel_maps):
    deck, note = duel_maps
    arena = HumanArena(
        tmp_path, _small_model(), deck, seconds=2.0, placement=note,
        learn_kwargs={"epochs": 1, "lr": 1e-4},
        practice_updates=0,
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
