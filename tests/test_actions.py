import math

import numpy as np
import pytest

from rl_donut.env import EngineConfig, World, match_step
from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC, Controls, decode, encode
from rl_donut.mapgen import load_split


def test_layout():
    assert DISCRETE_NVEC == (3, 2, 2) and CONTINUOUS_DIM == 1


def test_decode_maps_each_part():
    discrete = np.array([[[0, 0, 0], [1, 1, 0], [2, 1, 1]]])
    continuous = np.array([[[0.0], [math.pi / 2], [-7.5]]])
    c = decode(discrete, continuous)
    assert c.move.tolist() == [[-1, 0, 1]]
    assert c.jet.tolist() == [[False, True, True]]
    assert c.trigger.tolist() == [[False, False, True]]
    assert c.aim.tolist() == [[0.0, math.pi / 2, -7.5]]       # any real angle is accepted as is


def test_encode_inverts_decode():
    rng = np.random.default_rng(0)
    discrete = np.stack([rng.integers(0, n, size=(4, 2)) for n in DISCRETE_NVEC], axis=-1)
    continuous = rng.normal(size=(4, 2, 1)) * 5
    d2, c2 = encode(decode(discrete, continuous))
    np.testing.assert_array_equal(d2, discrete)
    np.testing.assert_array_equal(c2, continuous)


@pytest.mark.parametrize("discrete, continuous", [
    (np.zeros((1, 2, 2), dtype=int), np.zeros((1, 2, 1))),          # missing a discrete part
    (np.zeros((1, 2, 3), dtype=int), np.zeros((1, 2, 2))),          # two aim numbers
    (np.zeros((1, 2, 3), dtype=int), np.zeros((1, 3, 1))),          # different agents
    (np.array([[[3, 0, 0]]]), np.zeros((1, 1, 1))),                 # move out of range
    (np.array([[[0, 2, 0]]]), np.zeros((1, 1, 1))),                 # jetpack out of range
    (np.array([[[0, 0, -1]]]), np.zeros((1, 1, 1))),                # negative
    (np.zeros((1, 1, 3), dtype=int), np.array([[[np.nan]]])),       # aim not finite
])
def test_bad_actions_are_rejected(discrete, continuous):
    with pytest.raises(ValueError):
        decode(discrete, continuous)


def test_angles_a_full_turn_apart_play_identically():
    """The engine only uses cos and sin of the aim, so a Gaussian head needs no wrapping."""
    m = load_split("val")[0]
    worlds = [World.single(m, EngineConfig(team_size=2)) for _ in range(2)]
    rng = np.random.default_rng(1)
    for _ in range(150):
        discrete = np.stack([rng.integers(0, n, size=(1, 4)) for n in DISCRETE_NVEC], axis=-1)
        angle = rng.uniform(-math.pi, math.pi, size=(1, 4, 1))
        for world, turns in zip(worlds, (0, 3)):
            c = decode(discrete, angle + turns * 2 * math.pi)
            match_step(world, c.move, c.jet, c.aim, c.trigger)
    np.testing.assert_allclose(worlds[0].x, worlds[1].x, atol=1e-9)
    np.testing.assert_array_equal(worlds[0].health, worlds[1].health)
