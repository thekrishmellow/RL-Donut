import numpy as np
import pytest

from rl_donut.mapgen import Stream, make_rng


def draw(seed, stream, attempt=0, n=3):
    return make_rng(seed, stream, attempt).integers(0, 2**31, size=n).tolist()


def test_same_seed_same_stream_is_reproducible():
    assert draw(123, Stream.TERRAIN, n=100) == draw(123, Stream.TERRAIN, n=100)


def test_streams_are_distinct_for_same_seed():
    outputs = {tuple(draw(7, s, n=10)) for s in Stream}
    assert len(outputs) == len(Stream)


def test_neighbouring_seeds_are_uncorrelated():
    a = make_rng(1000, Stream.TERRAIN).random(10_000)
    b = make_rng(1001, Stream.TERRAIN).random(10_000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05


def test_attempts_give_fresh_streams():
    assert draw(42, Stream.TERRAIN, attempt=0) != draw(42, Stream.TERRAIN, attempt=1)


@pytest.mark.parametrize("bad", [-1, 1.5, "42", None])
def test_rejects_invalid_seeds(bad):
    with pytest.raises(ValueError):
        make_rng(bad, Stream.TERRAIN)


# Golden values recorded with numpy 2.5.4. If this fails, every seed now maps
# to a different world: pin numpy back or bump GENERATOR_VERSION deliberately.
GOLDEN_SEED_42 = {
    Stream.TERRAIN: [437549260, 1817907092, 483988055],
    Stream.PLATFORMS: [1669829392, 1224127273, 1569834193],
    Stream.CORRIDORS: [1121350195, 1730749946, 201347010],
    Stream.SPAWNS: [179228908, 524989694, 1848525639],
    Stream.DOMAIN: [1193351002, 168438378, 812722046],
}


@pytest.mark.parametrize("stream", list(GOLDEN_SEED_42))
def test_golden_values(stream):
    assert draw(42, stream) == GOLDEN_SEED_42[stream]


def test_golden_retry_attempt():
    assert draw(42, Stream.TERRAIN, attempt=1) == [872328959, 1622985186, 694690853]
