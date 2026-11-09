import pytest

from rl_donut.mapgen import MapConfig, generate_map


@pytest.fixture(scope="session")
def generated_maps():
    """First attempts for seeds 0-59 with the default config, shared by all test modules."""
    cfg = MapConfig()
    return [generate_map(s, cfg) for s in range(60)]
