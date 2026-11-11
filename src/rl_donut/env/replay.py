"""Match replays: the actions, not the positions.

A match is deterministic, so the map seed, the environment seed and the actions at every
decision step reproduce it exactly. That is much smaller than storing every agent every
tick, which matters on a nearly full disk. A reader rebuilds the match by stepping `MatchEnv`.
"""

import gzip
import json
from pathlib import Path

import numpy as np

from rl_donut.env.actions import CONTINUOUS_DIM, DISCRETE_NVEC, decode
from rl_donut.env.match import match_step
from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import SPEC_VERSION, MatchEnv
from rl_donut.env.policy import PolicyContext, team_slots


def record(env: MatchEnv, team_a, team_b, seed: int = 0) -> dict:
    """Play one match in a one-match env and return a JSON-ready replay of its actions."""
    if env.n_matches != 1:
        raise ValueError("a replay is one match; build the env with n_matches=1")
    rng = np.random.default_rng(seed)
    contexts = [(policy, PolicyContext(env, team_slots(env, team), rng)) for team, policy in enumerate((team_a, team_b))]
    discrete = np.zeros((1, env.n_agents, len(DISCRETE_NVEC)), dtype=np.int64)
    continuous = np.zeros((1, env.n_agents, CONTINUOUS_DIM))
    obs = env.reset()
    map_seed = int(env.world.seeds[0])
    steps = []
    while True:
        for policy, ctx in contexts:
            m, a = ctx.slots.T
            d, c = policy.act(obs[m, a], ctx)
            discrete[m, a], continuous[m, a] = d, c
        steps.append({"discrete": discrete[0].astype(int).tolist(),
                      "continuous": continuous[0].astype(float).tolist()})
        result = env.step(discrete, continuous)
        if result.done[0]:
            info = result.info[0]
            break
        obs = result.obs
    cfg = env.world.cfg
    return {
        "spec_version": SPEC_VERSION,
        "map_seed": map_seed,
        "team_size": cfg.team_size,
        "duration": cfg.match.duration,
        "rng_seed": env.world.rng_seed,
        "steps": steps,
        "score": [int(v) for v in info["score"]],
        "outcome": [int(v) for v in info["outcome"]],
    }


def save(path, replay: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(replay, handle)


def load(path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def replay_env(replay: dict, maps) -> MatchEnv:
    """A one-match env on the replay's map, reset and ready to step."""
    found = [m for m in maps if m.seed == replay["map_seed"]]
    if not found:
        raise ValueError(f"map seed {replay['map_seed']} is not in the maps given")
    cfg = EngineConfig(team_size=replay["team_size"], match=MatchConfig(duration=replay["duration"]))
    env = MatchEnv(1, replay["team_size"], seed=replay["rng_seed"], cfg=cfg, maps=found[:1])
    env.reset()
    return env


def play(replay: dict, maps) -> dict:
    """Step a replay to its end and return that match's episode info."""
    env = replay_env(replay, maps)
    result = None
    for step in replay["steps"]:
        result = env.step(np.asarray(step["discrete"])[None], np.asarray(step["continuous"])[None])
    if result is None or not result.done[0]:
        raise RuntimeError("replay ended before the match did")
    return result.info[0]


def snapshots(replay: dict, maps) -> tuple[MatchEnv, list[dict]]:
    """Replay against the engine directly, keeping one drawable frame per decision.

    Unlike `play`, this does not restart the match at the end, so the last frame is the
    final moment.
    """
    env = replay_env(replay, maps)
    world = env.world
    frames = []
    for step in replay["steps"]:
        controls = decode(np.asarray(step["discrete"])[None], np.asarray(step["continuous"])[None], mirror=env.mirror)
        ev = match_step(world, controls.move, controls.jet, controls.aim, controls.trigger)
        frames.append(dict(
            x=world.x[0].copy(), y=world.y[0].copy(), aim=world.aim[0].copy(), alive=world.alive[0].copy(),
            health=world.health[0].copy(), fuel=world.fuel[0].copy(), protection=world.protection[0].copy(),
            score=world.score[0].copy(), tick=int(world.tick[0]),
            shots=[(s.shooter, s.angle, s.hit) for _, s in ev.shot_log],
            deaths=[(r.death.agent, r.scorer, r.death.fell) for r in ev.death_log],
        ))
    return env, frames


def snapshots(replay: dict, maps) -> tuple[MatchEnv, list[dict]]:
    """Replay against the engine directly, keeping one drawable frame per decision.

    Unlike `play`, this does not restart the match at the end, so the last frame is the
    final moment.
    """
    env = replay_env(replay, maps)
    world = env.world
    frames = []
    for step in replay["steps"]:
        controls = decode(np.asarray(step["discrete"])[None], np.asarray(step["continuous"])[None], mirror=env.mirror)
        ev = match_step(world, controls.move, controls.jet, controls.aim, controls.trigger)
        frames.append(dict(
            x=world.x[0].copy(), y=world.y[0].copy(), aim=world.aim[0].copy(), alive=world.alive[0].copy(),
            health=world.health[0].copy(), fuel=world.fuel[0].copy(), protection=world.protection[0].copy(),
            score=world.score[0].copy(), tick=int(world.tick[0]),
            shots=[(s.shooter, s.angle, s.hit) for _, s in ev.shot_log],
            deaths=[(r.death.agent, r.scorer, r.death.fell) for r in ev.death_log],
        ))
    return env, frames
