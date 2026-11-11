"""Policy evaluation harness with side-swapping and Wilson score confidence intervals (Phase 5 Step 5.8)."""

from dataclasses import dataclass
import math
import numpy as np

from rl_donut.rl.stand_in import Policy, PolicyContext, StandInEnv


@dataclass
class EvalReport:
    n_matches: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    wilson_ci_lower: float
    wilson_ci_upper: float
    mean_kills: float
    mean_deaths: float
    mean_damage_dealt: float
    mean_damage_taken: float


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.95996  # for 95% confidence
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_policy(
    learner: Policy,
    opponent: Policy,
    n_matches: int = 50,
    split: str = "val",
    team_size: int = 1,
    seed: int = 10_000,
    match_duration_seconds: float = 60.0,
    map_filter=None,
    custom_maps=None,
    cfg=None,
) -> EvalReport:
    """Evaluate learner vs opponent over n_matches, swapping sides halfway."""
    # Ensure even number of matches for equal side representation
    if n_matches % 2 != 0:
        n_matches += 1

    half = n_matches // 2
    wins = 0
    losses = 0
    draws = 0
    total_kills = 0
    total_deaths = 0
    total_dmg_dealt = 0.0
    total_dmg_taken = 0.0

    rng = np.random.default_rng(seed)

    # First half: learner is Team A (slot 0..team_size-1), opponent is Team B
    # Second half: learner is Team B, opponent is Team A
    for half_idx, learner_team in enumerate([0, 1]):
        env = StandInEnv(
            n_matches=half,
            team_size=team_size,
            split=split,
            seed=seed + half_idx * 1000,
            map_filter=map_filter,
            match_duration_seconds=match_duration_seconds,
            custom_maps=custom_maps,
            cfg=cfg,
        )
        obs = env.reset()
        M, A = env.n_matches, env.n_agents

        opp_team = 1 - learner_team
        l_slots = np.array([(m, a) for m in range(M) for a in range(A) if env.team[a] == learner_team])
        o_slots = np.array([(m, a) for m in range(M) for a in range(A) if env.team[a] == opp_team])

        ctx_l = PolicyContext(env=env, slots=l_slots, rng=rng)
        ctx_o = PolicyContext(env=env, slots=o_slots, rng=rng)

        # Run until match ends
        match_done = np.zeros(M, dtype=bool)
        while not match_done.all():
            l_obs = obs[l_slots[:, 0], l_slots[:, 1]]
            o_obs = obs[o_slots[:, 0], o_slots[:, 1]]

            l_actions = learner.act(l_obs, ctx_l)
            o_actions = opponent.act(o_obs, ctx_o)

            full_actions = np.zeros((M, A, len(env.action_nvec)), dtype=np.int64)
            full_actions[l_slots[:, 0], l_slots[:, 1]] = l_actions
            full_actions[o_slots[:, 0], o_slots[:, 1]] = o_actions

            res = env.step(full_actions)
            match_done |= res.done

            total_kills += int(res.events.kills[l_slots[:, 0], l_slots[:, 1]].sum())
            total_deaths += int(res.events.deaths[l_slots[:, 0], l_slots[:, 1]].sum())
            total_dmg_dealt += float(res.events.damage_dealt[l_slots[:, 0], l_slots[:, 1]].sum())
            total_dmg_taken += float(res.events.damage_taken[l_slots[:, 0], l_slots[:, 1]].sum())

            # Count completed match outcomes
            if res.done.any():
                for m in np.nonzero(res.done)[0]:
                    out = res.events.outcome[m, learner_team]
                    if out > 0:
                        wins += 1
                    elif out < 0:
                        losses += 1
                    else:
                        draws += 1

            obs = res.obs

    win_rate = wins / n_matches
    loss_rate = losses / n_matches
    draw_rate = draws / n_matches
    ci_low, ci_high = wilson_score_interval(wins, n_matches)

    return EvalReport(
        n_matches=n_matches,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        loss_rate=loss_rate,
        draw_rate=draw_rate,
        wilson_ci_lower=ci_low,
        wilson_ci_upper=ci_high,
        mean_kills=total_kills / n_matches,
        mean_deaths=total_deaths / n_matches,
        mean_damage_dealt=total_dmg_dealt / n_matches,
        mean_damage_taken=total_dmg_taken / n_matches,
    )

