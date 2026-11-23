"""Fit the actor to saved human controls, without throwing away the policy it already has.

The newest match is weighted more heavily, so the next fight can show the person who
just played. Older matches stay in the file and in the loss, so the next player does
not erase the previous ones. The weights that are kept are the ones that make the
newest match's actions most likely. The weights from before the lesson are one of
the candidates, so a lesson that does not fit is not shipped. A small anchor pulls
each step back toward those starting weights.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl_donut.rl.clone import Demonstrations, action_log_prob
from rl_donut.rl.model import ActorCritic


@dataclass
class LearnReport:
    rows: int
    latest_rows: int
    epochs: int
    logprob_before: float
    logprob_after: float
    logprob_all_before: float
    logprob_all_after: float
    improved: bool

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "latest_rows": self.latest_rows,
            "epochs": self.epochs,
            "logprob_before": self.logprob_before,
            "logprob_after": self.logprob_after,
            "logprob_all_before": self.logprob_all_before,
            "logprob_all_after": self.logprob_all_after,
            "improved": self.improved,
        }


def row_weights(match_id: np.ndarray, latest_weight: float) -> np.ndarray:
    """The newest match counts `latest_weight` times. Earlier matches count once."""
    if latest_weight < 1:
        raise ValueError("latest_weight must be at least 1")
    weights = np.ones(len(match_id), dtype=np.float32)
    if len(match_id):
        weights[np.asarray(match_id) == np.asarray(match_id).max()] = np.float32(latest_weight)
    return weights


def _focus(demos: Demonstrations, max_rows: int, seed: int) -> Demonstrations:
    """Keep the newest match, then fill any remaining room with older rows."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if len(demos) <= max_rows:
        return demos
    rng = np.random.default_rng(seed)
    latest = demos.match_id == demos.match_id.max()
    latest_idx = np.flatnonzero(latest)
    older_idx = np.flatnonzero(~latest)
    if len(latest_idx) >= max_rows:
        pick = rng.choice(latest_idx, size=max_rows, replace=False)
    else:
        need = max_rows - len(latest_idx)
        older_pick = rng.choice(older_idx, size=min(need, len(older_idx)), replace=False)
        pick = np.concatenate([latest_idx, older_pick])
    mask = np.zeros(len(demos), dtype=bool)
    mask[pick] = True
    return demos.select(mask)


def _mean_log_prob(model: ActorCritic, demos: Demonstrations, batch_size: int) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(demos), batch_size):
            end = min(start + batch_size, len(demos))
            total += float(action_log_prob(
                model, demos.obs[start:end], demos.discrete[start:end], demos.continuous[start:end]
            ).sum().item())
    return total / len(demos)


def _actor_params(model: ActorCritic) -> list[nn.Parameter]:
    params = (
        list(model.actor_backbone.parameters())
        + list(model.action_heads.parameters())
        + list(model.aim_mean.parameters())
    )
    if model.cont_dim:
        params.append(model.aim_log_std)
    return params


def learn_from_demos(
    model: ActorCritic,
    demos: Demonstrations,
    *,
    epochs: int = 4,
    lr: float = 1e-4,
    batch_size: int = 512,
    anchor_coef: float = 0.25,
    latest_weight: float = 4.0,
    max_rows: int = 8192,
    seed: int = 0,
    grad_clip: float = 0.5,
) -> LearnReport:
    """Move the actor toward the demonstrated controls. The critic is left alone.

    `logprob_before` and `logprob_after` are the mean log-probability of the newest
    match. `logprob_all_*` is the same mean over every saved row.
    """
    if len(demos) == 0:
        raise ValueError("no demonstrations to learn from")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if anchor_coef < 0:
        raise ValueError("anchor_coef must be non-negative")
    train = _focus(demos, max_rows, seed)
    weights = row_weights(train.match_id, latest_weight)
    latest = demos.select(demos.match_id == demos.match_id.max())
    params = _actor_params(model)
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    anchors = [p.detach().clone() for p in params]
    rng = np.random.default_rng(seed)

    model.eval()
    before_latest = _mean_log_prob(model, latest, batch_size)
    before_all = _mean_log_prob(model, demos, batch_size)
    best_score = before_latest
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(train), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            log_prob = action_log_prob(
                model, train.obs[batch], train.discrete[batch], train.continuous[batch]
            )
            batch_weight = torch.as_tensor(weights[batch], dtype=torch.float32, device=log_prob.device)
            loss = -(log_prob * batch_weight).sum() / batch_weight.sum().clamp_min(1e-8)
            if anchor_coef:
                pull = torch.stack([((p - a) ** 2).mean() for p, a in zip(params, anchors)]).mean()
                loss = loss + anchor_coef * pull
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
        model.eval()
        score = _mean_log_prob(model, latest, batch_size)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    after_latest = _mean_log_prob(model, latest, batch_size)
    after_all = _mean_log_prob(model, demos, batch_size)
    return LearnReport(
        rows=len(demos),
        latest_rows=len(latest),
        epochs=epochs,
        logprob_before=float(before_latest),
        logprob_after=float(after_latest),
        logprob_all_before=float(before_all),
        logprob_all_after=float(after_all),
        improved=bool(after_latest > before_latest + 1e-4),
    )
