"""Self-play on the duel caves. The opponent is the policy itself, one update behind.

Both fighters move and shoot. There is no standing target. Human rows are an extra
lesson only after the rock filter, and the aim is pulled onto an enemy the agent
can actually see, which is the line a hitscan rifle has to follow.
"""

import numpy as np
import torch
import torch.nn.functional as F

from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.play.physics_data import corrected_aim, engage, legal_demonstrations, seen_aim_target
from rl_donut.rl.clone import Demonstrations, action_log_prob
from rl_donut.play.learn import LearnReport
from rl_donut.rl.evaluate import legal_duration
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner


class FightingPolicy(LearnedPolicy):
    """The learned body, with the gun turned onto an enemy it can see."""

    def act(self, obs, ctx=None):
        discrete, continuous = super().act(obs, ctx)
        return engage(obs, discrete, continuous)


class SelfOpponent:
    """A frozen copy of the learner. `pull` makes it the policy from the start of this update."""

    name = "self"

    def __init__(self, model: ActorCritic):
        self.model = _clone(model)
        self.policy = FightingPolicy(self.model, name="self")

    def pull(self, live: ActorCritic) -> None:
        self.model.load_state_dict(live.state_dict())
        self.policy.roamer = type(self.policy.roamer)()

    def act(self, obs, ctx):
        return self.policy.act(obs, ctx)


def _clone(model: ActorCritic) -> ActorCritic:
    hidden = model.actor_backbone[0].out_features
    copy = ActorCritic(model.obs_dim, model.action_nvec, hidden_dim=hidden, cont_dim=model.cont_dim)
    copy.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    copy.eval()
    return copy


def aim_and_fire(model: ActorCritic, optimizer, obs: np.ndarray, coef: float = 1.0) -> float:
    """Point the aim mean at a seen enemy, and fire once that aim is already on the body."""
    target, seen = seen_aim_target(obs)
    if int(np.count_nonzero(seen)) < 4 or coef == 0:
        return 0.0
    device = next(model.parameters()).device
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[seen], device=device)
    target_t = torch.as_tensor(target[seen], dtype=torch.float32, device=device).unsqueeze(-1)
    model.train()
    latent = model.actor_backbone(obs_t)
    mean = model.aim_mean(latent)
    error = torch.atan2(torch.sin(mean - target_t), torch.cos(mean - target_t)).squeeze(-1)
    loss = error.pow(2).mean()
    close = error.detach().abs() < 0.25
    if int(close.sum()) >= 4:
        logits = model.action_heads[2](latent)
        fired = torch.ones(int(close.sum()), dtype=torch.long, device=device)
        loss = loss + 0.5 * F.cross_entropy(logits[close], fired)
    loss = coef * loss
    kept = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    if not _finite_weights(model):
        model.load_state_dict(kept)
        model.eval()
        return 0.0
    model.eval()
    return float(loss.detach())


def imitate_legal(
    model: ActorCritic,
    optimizer,
    demos: Demonstrations,
    *,
    steps: int = 1,
    batch_size: int = 256,
    seed: int = 0,
) -> int:
    """One step on air-legal buttons. Aim is a squared angle, not the narrow Gaussian.

    The Gaussian log-probability divides by the square of a 6° spread, so a human
    aim that is off by a radian produces a gradient large enough to wipe the policy.
    """
    legal = legal_demonstrations(demos)
    if len(legal) < 8 or steps < 1:
        return 0
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    kept = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.train()
    for _ in range(steps):
        pick = rng.choice(len(legal), size=min(batch_size, len(legal)), replace=False)
        obs = legal.obs[pick]
        discrete = legal.discrete[pick]
        continuous = corrected_aim(obs, legal.continuous[pick])
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        discrete_t = torch.as_tensor(discrete, dtype=torch.long, device=device)
        latent = model.actor_backbone(obs_t)
        dists = [torch.distributions.Categorical(logits=head(latent)) for head in model.action_heads]
        discrete_loss = -torch.stack(
            [dist.log_prob(discrete_t[:, i]) for i, dist in enumerate(dists)], dim=-1
        ).sum(dim=-1).mean()
        mean = model.aim_mean(latent)
        target = torch.as_tensor(continuous[:, :1], dtype=torch.float32, device=device)
        aim_loss = torch.atan2(torch.sin(mean - target), torch.cos(mean - target)).pow(2).mean()
        loss = 0.05 * discrete_loss + 0.05 * aim_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    if not _finite_weights(model):
        model.load_state_dict(kept)
    model.eval()
    return len(legal)


def _finite_weights(model: ActorCritic) -> bool:
    return all(torch.isfinite(value).all() for value in model.parameters())


def self_play_updates(
    model: ActorCritic,
    maps,
    updates: int,
    *,
    n_matches: int = 4,
    n_steps: int = 64,
    seconds: float = 20.0,
    seed: int = 0,
    lr: float = 3e-4,
    aim_coef: float = 1.0,
    demos: Demonstrations | None = None,
    trainer: PPOTrainer | None = None,
    on_update=None,
) -> list[dict]:
    """PPO against a frozen copy of `model`. The copy is refreshed at every update."""
    if updates < 1:
        return []
    if len(maps) < n_matches:
        raise ValueError(f"need at least {n_matches} maps, found {len(maps)}")
    cfg = EngineConfig(team_size=1, match=MatchConfig(duration=legal_duration(seconds)))
    env = MatchEnv(n_matches, team_size=1, seed=seed, cfg=cfg, maps=list(maps)[: max(n_matches, 8)])
    if trainer is None:
        trainer = PPOTrainer(model, PPOConfig(lr=lr))
    opponent = SelfOpponent(model)
    runner = Runner(
        env, model, opponent, trainer,
        n_steps=n_steps,
        total_training_steps=updates * n_steps * n_matches,
    )
    history = []
    for step in range(updates):
        if not _finite_weights(model):
            break
        kept = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        opponent.pull(model)
        try:
            metrics = runner.step_rollout()
        except ValueError:
            model.load_state_dict(kept)
            break
        if not _finite_weights(model):
            model.load_state_dict(kept)
            break
        flat = runner.last_obs.reshape(-1, model.obs_dim)
        aim_loss = aim_and_fire(model, trainer.optimizer, flat, coef=aim_coef)
        if demos is not None and len(demos):
            aim_loss += aim_and_fire(model, trainer.optimizer, demos.obs, coef=aim_coef)
        used = imitate_legal(model, trainer.optimizer, demos, seed=seed + step) if demos is not None and step % 5 == 0 else 0
        row = {
            "update": step + 1,
            "mean_return": metrics.mean_return,
            "entropy": metrics.entropy,
            "aim_loss": aim_loss,
            "human_rows": used,
        }
        history.append(row)
        if on_update is not None:
            on_update(model, row)
    model.eval()
    return history


def _mean_log_prob(model: ActorCritic, demos: Demonstrations) -> float:
    if len(demos) == 0:
        return float("nan")
    with torch.no_grad():
        total = 0.0
        for start in range(0, len(demos), 512):
            end = min(start + 512, len(demos))
            total += float(action_log_prob(
                model, demos.obs[start:end], demos.discrete[start:end], demos.continuous[start:end]
            ).sum())
    return total / len(demos)


def practice_round(
    model: ActorCritic,
    demos: Demonstrations,
    maps,
    *,
    updates: int = 8,
    seconds: float = 20.0,
    seed: int = 0,
    n_matches: int = 4,
) -> LearnReport:
    """Self-play, then a short pass on the physics-legal part of the saved matches."""
    legal = legal_demonstrations(demos)
    before = _mean_log_prob(model, legal)
    if updates > 0 and maps:
        self_play_updates(
            model, maps, updates,
            n_matches=min(n_matches, len(maps)),
            seconds=seconds,
            seed=seed,
            demos=legal if len(legal) else None,
        )
    after = _mean_log_prob(model, legal)
    latest_id = int(demos.match_id.max()) if len(demos) else 0
    latest_rows = int(np.count_nonzero(legal.match_id == latest_id)) if len(legal) else 0
    improved = bool(np.isfinite(before) and np.isfinite(after) and after > before + 1e-4)
    return LearnReport(
        rows=len(demos),
        latest_rows=latest_rows,
        epochs=updates,
        logprob_before=float(before) if np.isfinite(before) else 0.0,
        logprob_after=float(after) if np.isfinite(after) else 0.0,
        logprob_all_before=float(before) if np.isfinite(before) else 0.0,
        logprob_all_after=float(after) if np.isfinite(after) else 0.0,
        improved=improved,
    )
