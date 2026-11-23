"""Self-play on the duel caves. The opponent is the policy itself, one update behind.

Both fighters move and shoot. There is no standing target. Human rows are an extra
lesson only after the rock filter, and the aim is pulled onto an enemy the agent
can actually see, which is the line a hitscan rifle has to follow.
"""

import numpy as np
import torch

from rl_donut.env.config import EngineConfig, MatchConfig
from rl_donut.env.match_env import MatchEnv
from rl_donut.play.physics_data import corrected_aim, legal_demonstrations, seen_aim_target
from rl_donut.rl.clone import Demonstrations, action_log_prob
from rl_donut.play.learn import LearnReport
from rl_donut.rl.evaluate import legal_duration
from rl_donut.rl.model import ActorCritic
from rl_donut.rl.ppo import LearnedPolicy, PPOConfig, PPOTrainer
from rl_donut.rl.runner import Runner


class SelfOpponent:
    """A frozen copy of the learner. `pull` makes it the policy from the start of this update."""

    name = "self"

    def __init__(self, model: ActorCritic):
        self.model = _clone(model)
        self.policy = LearnedPolicy(self.model, name="self")

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
    loss = coef * error.pow(2).mean()
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


def median_aim_error(model: ActorCritic, obs: np.ndarray) -> float:
    """Median degrees between the network's aim and a seen enemy. NaN if nobody is seen."""
    from rl_donut.play.physics_data import aim_error_degrees

    _, seen = seen_aim_target(obs)
    if int(np.count_nonzero(seen)) < 4:
        return float("nan")
    device = next(model.parameters()).device
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[seen], device=device)
    with torch.no_grad():
        mean = model.aim_mean(model.actor_backbone(obs_t)).squeeze(-1).detach().cpu().numpy()
    full = np.zeros(len(obs), dtype=np.float64)
    full[seen] = mean
    return aim_error_degrees(full, obs)


def gather_observations(model: ActorCritic, maps, *, seconds: float = 6.0, n_matches: int = 4, seed: int = 0) -> np.ndarray:
    """States from a short match of the policy against its own copy."""
    from rl_donut.env.policy import PolicyContext, team_slots

    cfg = EngineConfig(team_size=1, match=MatchConfig(duration=legal_duration(seconds)))
    env = MatchEnv(n_matches, team_size=1, seed=seed, cfg=cfg, maps=list(maps)[: max(n_matches, 4)])
    policy = LearnedPolicy(model, name="gather")
    opponent = SelfOpponent(model)
    opponent.pull(model)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    rows = [obs.reshape(-1, model.obs_dim)]
    steps = max(1, int(seconds * env.world.cfg.tick_rate / env.world.cfg.match.decision_ticks))
    for _ in range(steps):
        discrete = np.zeros((n_matches, env.n_agents, 3), dtype=np.int64)
        continuous = np.zeros((n_matches, env.n_agents, 1), dtype=np.float32)
        for team, actor in ((0, policy), (1, opponent)):
            slots = team_slots(env, team)
            chosen, aim = actor.act(obs[slots[:, 0], slots[:, 1]], PolicyContext(env, slots, rng))
            discrete[slots[:, 0], slots[:, 1]] = chosen
            continuous[slots[:, 0], slots[:, 1]] = aim
        result = env.step(discrete, continuous, restart_done=False)
        obs = result.obs
        rows.append(obs.reshape(-1, model.obs_dim))
        if result.done.all():
            break
    return np.concatenate(rows, axis=0)


def pretrain_aim(model: ActorCritic, maps, *, rounds: int = 80, lr: float = 3e-3, seed: int = 0) -> float:
    """Fit the aim mean on a fixed set of states where an enemy is already visible.

    One pass over a moving policy is not enough for the network to learn the angle.
    The same states are reused until the median miss is under 8 degrees, or the
    step budget runs out.
    """
    chunks = [gather_observations(model, maps, seed=seed + i) for i in range(3)]
    obs = np.concatenate(chunks, axis=0)
    _, seen = seen_aim_target(obs)
    obs = obs[seen]
    if len(obs) < 32:
        print("Not enough visible enemies to learn an aim.", flush=True)
        return float("nan")
    params = list(model.actor_backbone.parameters()) + list(model.aim_mean.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    rng = np.random.default_rng(seed)
    error = float("nan")
    for step in range(rounds):
        pick = rng.choice(len(obs), size=min(512, len(obs)), replace=False)
        aim_and_fire(model, optimizer, obs[pick], coef=1.0)
        if step % 10 == 0 or step + 1 == rounds:
            error = median_aim_error(model, obs[:1024])
            print(f"aim step {step + 1:3d}  median error {error:.1f} deg  on {len(obs)} states", flush=True)
        if np.isfinite(error) and error < 8:
            break
    model.eval()
    return error


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
