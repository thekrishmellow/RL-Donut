"""Behaviour cloning from RusherBot onto the fair 330-number observation.

The rusher plans with the true enemy position. These demonstrations keep only the
observation a learner receives and the action the rusher then took, already in the
agent's own frame. The aim loss wraps the demonstrated angle onto the circle around
the network's current mean, because the Gaussian head treats angles as a line.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl_donut.env.bots import RusherBot, StaticTargetBot
from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.env.observe import POSITION_SCALE
from rl_donut.env.perception import nearest_living
from rl_donut.env.policy import PolicyContext, team_slots
from rl_donut.mapgen import load_split
from rl_donut.rl.evaluate import legal_duration
from rl_donut.rl.model import ActorCritic


@dataclass
class Demonstrations:
    """One row per decision of the rusher. `match_id` splits train and val by match."""

    obs: np.ndarray
    discrete: np.ndarray
    continuous: np.ndarray
    match_id: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def select(self, mask: np.ndarray) -> "Demonstrations":
        mask = np.asarray(mask, dtype=bool)
        return Demonstrations(self.obs[mask], self.discrete[mask], self.continuous[mask], self.match_id[mask])


@dataclass
class CloneFit:
    epochs_ran: int
    best_epoch: int
    best_val_log_prob: float | None
    history: list[dict]


def wrap_angle(target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Shift `target` by multiples of 2π so it sits within π of `mean`."""
    return mean + torch.atan2(torch.sin(target - mean), torch.cos(target - mean))


def _tensors(model: ActorCritic, obs, discrete, continuous):
    device = next(model.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    discrete_t = torch.as_tensor(discrete, dtype=torch.long, device=device)
    continuous_t = torch.as_tensor(continuous, dtype=torch.float32, device=device)
    return obs_t, discrete_t, continuous_t


def action_log_prob(model: ActorCritic, obs, discrete, continuous) -> torch.Tensor:
    """Log-probability of each demonstrated action. The aim is wrapped to the current mean."""
    if not model.cont_dim:
        raise ValueError("behaviour cloning expects the Gaussian aim head")
    obs_t, discrete_t, continuous_t = _tensors(model, obs, discrete, continuous)
    mean = model.aim_mean(model.actor_backbone(obs_t))
    wrapped = wrap_angle(continuous_t, mean.detach())
    _, _, log_prob, _, _ = model.get_action_and_value(obs_t, discrete_t, wrapped)
    return log_prob


def mode_actions(model: ActorCritic, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Most likely discrete action and the aim mean. This is the cloned policy's decision."""
    device = next(model.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    latent = model.actor_backbone(obs_t)
    discrete = torch.stack([head(latent).argmax(dim=-1) for head in model.action_heads], dim=-1)
    continuous = model.aim_mean(latent)
    return discrete.detach().cpu().numpy().astype(np.int64), continuous.detach().cpu().numpy().astype(np.float32)


class ClonedPolicy:
    """Plays the cloned actor. Evaluation uses the mode, so the score is the copy itself."""

    def __init__(self, model: ActorCritic, name: str = "cloned_rusher"):
        self.name = name
        self.model = model
        self.model.eval()

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None):
        with torch.no_grad():
            return mode_actions(self.model, obs)


def true_enemy_offset(ctx: PolicyContext) -> np.ndarray:
    """(n, 2) offset to the nearest living enemy, in the agent's own frame, divided by rifle range.

    This is the position the rusher plans from. It is filled in when the enemy is behind
    rock. A dead enemy leaves zeros.
    """
    index, ex, ey = nearest_living(ctx)
    w = ctx.env.world
    m, i = ctx.slots.T
    sign = np.where(ctx.env.mirror[i], -1.0, 1.0)
    offset = np.stack([
        sign * (ex - w.x[m, i]) / POSITION_SCALE,
        (ey - w.y[m, i]) / POSITION_SCALE,
    ], axis=-1).astype(np.float32)
    offset[index < 0] = 0.0
    return offset


class OffsetClonedPolicy:
    """The diagnostic clone. It sees the fair observation plus the true enemy offset."""

    def __init__(self, model: ActorCritic, name: str = "cloned_rusher_offset"):
        self.name = name
        self.model = model
        self.model.eval()

    def act(self, obs: np.ndarray, ctx: PolicyContext | None = None):
        if ctx is None:
            raise ValueError("the offset clone reads the enemy position from the match")
        row = np.concatenate([np.asarray(obs, dtype=np.float32), true_enemy_offset(ctx)], axis=-1)
        with torch.no_grad():
            return mode_actions(self.model, row)


def collect_rusher_demos(
    n_matches: int,
    *,
    split: str = "train",
    seed: int = 0,
    seconds: float = 60.0,
    batch_size: int = 8,
    maps=None,
    verbose: bool = False,
    with_enemy_offset: bool = False,
) -> Demonstrations:
    """Play the rusher against a static bot and record the rusher's observation and action.

    Only team A is recorded. The static bot never leaves its spawn. `with_enemy_offset`
    appends the rusher's private enemy position; the fair recording leaves it off.
    """
    if n_matches < 1:
        raise ValueError("n_matches must be positive")
    maps = list(maps) if maps is not None else load_split(split)
    duration = legal_duration(seconds)
    obs_parts, discrete_parts, continuous_parts, match_parts = [], [], [], []
    done_matches = 0
    batch_index = 0
    while done_matches < n_matches:
        n = min(batch_size, n_matches - done_matches)
        cfg = EngineConfig(team_size=1, match=MatchConfig(duration=duration))
        env = MatchEnv(n, team_size=1, split=split, seed=seed + batch_index * 1000, cfg=cfg, maps=maps)
        obs = env.reset()
        rusher = RusherBot()
        static = StaticTargetBot()
        rng = np.random.default_rng(seed + batch_index)
        rusher_ctx = PolicyContext(env, team_slots(env, 0), rng)
        static_ctx = PolicyContext(env, team_slots(env, 1), rng)
        finished = np.zeros(n, dtype=bool)
        step_obs, step_discrete, step_continuous = [], [], []
        while not finished.all():
            slots = rusher_ctx.slots
            seen = obs[slots[:, 0], slots[:, 1]]
            discrete, continuous = rusher.act(seen, rusher_ctx)
            row = np.asarray(seen, dtype=np.float32)
            if with_enemy_offset:
                row = np.concatenate([row, true_enemy_offset(rusher_ctx)], axis=-1)
            other_d, other_c = static.act(obs[static_ctx.slots[:, 0], static_ctx.slots[:, 1]], static_ctx)
            full_d = np.zeros((n, env.n_agents, len(env.action_nvec)), dtype=np.int64)
            full_c = np.zeros((n, env.n_agents, env.action_cont_dim), dtype=np.float32)
            full_d[slots[:, 0], slots[:, 1]] = discrete
            full_c[slots[:, 0], slots[:, 1]] = continuous
            full_d[static_ctx.slots[:, 0], static_ctx.slots[:, 1]] = other_d
            full_c[static_ctx.slots[:, 0], static_ctx.slots[:, 1]] = other_c
            step_obs.append(np.asarray(row, dtype=np.float32).copy())
            step_discrete.append(np.asarray(discrete, dtype=np.int64).copy())
            step_continuous.append(np.asarray(continuous, dtype=np.float32).copy())
            result = env.step(full_d, full_c)
            finished |= result.done
            obs = result.obs
        obs_parts.append(np.concatenate(step_obs, axis=0))
        discrete_parts.append(np.concatenate(step_discrete, axis=0))
        continuous_parts.append(np.concatenate(step_continuous, axis=0))
        steps = len(step_obs)
        # Rows are stored step by step, and each step holds the matches in order.
        match_parts.append(np.tile(np.arange(done_matches, done_matches + n), steps))
        done_matches += n
        batch_index += 1
        if verbose:
            print(f"recorded {done_matches}/{n_matches} matches", flush=True)
    return Demonstrations(
        np.concatenate(obs_parts, axis=0),
        np.concatenate(discrete_parts, axis=0),
        np.concatenate(continuous_parts, axis=0),
        np.concatenate(match_parts, axis=0).astype(np.int64),
    )


def split_demonstrations(
    demos: Demonstrations,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Demonstrations, Demonstrations]:
    """Hold out whole matches. Adjacent steps of one life stay on the same side."""
    matches = np.unique(demos.match_id)
    if len(matches) < 2 or val_fraction <= 0:
        empty = demos.select(np.zeros(len(demos), dtype=bool))
        return demos, empty
    order = np.random.default_rng(seed).permutation(matches)
    n_val = min(max(1, int(round(len(matches) * val_fraction))), len(matches) - 1)
    val_ids = set(order[:n_val].tolist())
    val_mask = np.isin(demos.match_id, list(val_ids))
    return demos.select(~val_mask), demos.select(val_mask)


def step_in_match(match_id: np.ndarray) -> np.ndarray:
    """How many earlier rows share this row's match. 0 is the first decision."""
    match_id = np.asarray(match_id)
    if len(match_id) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(match_id, kind="stable")
    sorted_ids = match_id[order]
    starts = np.flatnonzero(np.diff(sorted_ids, prepend=sorted_ids[0] - 1))
    counts = np.diff(np.append(starts, len(match_id)))
    ranks = np.empty(len(match_id), dtype=np.int64)
    ranks[order] = np.arange(len(match_id)) - np.repeat(starts, counts)
    return ranks


def opening_weights(
    match_id: np.ndarray,
    opening_steps: int = 30,
    opening_weight: float = 10.0,
) -> np.ndarray:
    """Weight the first `opening_steps` of each match by `opening_weight`.

    Thirty steps is two seconds. The rusher leaves the spawn in that window, and those
    rows are about one in thirty of a minute-long match, so an equal loss leaves them
    as the less likely action.
    """
    if opening_steps < 1:
        raise ValueError("opening_steps must be positive")
    if opening_weight < 1:
        raise ValueError("opening_weight must be at least 1")
    weights = np.ones(len(match_id), dtype=np.float32)
    weights[step_in_match(match_id) < opening_steps] = np.float32(opening_weight)
    return weights


def _mean_log_prob(model: ActorCritic, demos: Demonstrations, batch_size: int) -> float:
    if len(demos) == 0:
        return float("nan")
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(demos), batch_size):
            end = min(start + batch_size, len(demos))
            total += float(action_log_prob(
                model, demos.obs[start:end], demos.discrete[start:end], demos.continuous[start:end]
            ).sum().item())
    return total / len(demos)


def fit_clone(
    model: ActorCritic,
    train: Demonstrations,
    val: Demonstrations | None = None,
    *,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 4096,
    patience: int = 3,
    seed: int = 0,
    grad_clip: float = 0.5,
    sample_weight: np.ndarray | None = None,
) -> CloneFit:
    """Fit the actor to the demonstrations. The critic is left for a later PPO run.

    `sample_weight` changes the training loss only. The held-out log-probability stays
    an unweighted mean, so a heavier opening does not inflate the number we stop on.
    Stops when that held-out log-probability has not risen for `patience` epochs, and
    restores the weights from the best of those epochs.
    """
    if len(train) == 0:
        raise ValueError("no demonstrations to fit")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if sample_weight is None:
        sample_weight = np.ones(len(train), dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != (len(train),):
            raise ValueError("sample_weight must have one value per training row")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")
    val = val or Demonstrations(
        train.obs[:0], train.discrete[:0], train.continuous[:0], train.match_id[:0]
    )
    params = (
        list(model.actor_backbone.parameters())
        + list(model.action_heads.parameters())
        + list(model.aim_mean.parameters())
        + [model.aim_log_std]
    )
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_epoch = -1
    best_score = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stalled = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        counts = []
        for start in range(0, len(train), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            log_prob = action_log_prob(
                model, train.obs[batch], train.discrete[batch], train.continuous[batch]
            )
            batch_weight = torch.as_tensor(sample_weight[batch], dtype=torch.float32, device=log_prob.device)
            loss = -(log_prob * batch_weight).sum() / batch_weight.sum().clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            counts.append(float(batch_weight.sum().item()))
        train_nll = float(np.average(losses, weights=counts))
        model.eval()
        val_lp = _mean_log_prob(model, val, batch_size) if len(val) else None
        history.append({"epoch": epoch, "train_nll": train_nll, "val_log_prob": val_lp})
        score = val_lp if val_lp is not None else -train_nll
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if len(val) and stalled >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return CloneFit(
        epochs_ran=len(history),
        best_epoch=best_epoch,
        best_val_log_prob=None if len(val) == 0 else best_score,
        history=history,
    )
