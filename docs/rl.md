# RL-Donut — Reinforcement Learning Guide & Decision Log (Phases 5–6)

This document is Team B's step-by-step development guide and decision table for
**Phase 5 (rewards, PPO, curriculum)** and **Phase 6 (self-play, opponent pool, Elo, 2v2)**.

At the end of Phase 4, the decisions in this document will be moved into the root
[`DESIGN.md`](../DESIGN.md) during the branch merge.

| Related document | Role |
|---|---|
| [`docs/handoff_phase5_6.md`](handoff_phase5_6.md) | Contract interface, steps, and merge checklist between Team A and Team B |
| [`docs/PLAN.md`](PLAN.md) | Whole-project plan and milestones |
| [`DESIGN.md`](../DESIGN.md) | Main project decision log (Phases 0–2) |

---

## Steps

| Step | Status | Scope |
|---|---|---|
| **5.1** | **done** | Tools, PyTorch pin, and Team B guide (`docs/rl.md`) |
| **5.2** | **done** | Stand-in environment (`StandInEnv`) and contract test (`tests/test_env_contract.py`) |
| **5.3** | **done** | Reward function and shaping (`rl_donut.rl.rewards`) |
| **5.4** | next | PPO implementation (`rl_donut.rl.ppo`) |
| **5.5** | planned | Training runner with opponents (`scripts/train.py`, `rl_donut.rl.runner`) |
| **5.6** | planned | Logging, exploit watchlist, checkpoint manager |
| **5.7** | planned | Curriculum (static target → turret → rusher → pits) |
| **5.8** | planned | Evaluation against fixed bots (`scripts/eval_bots.py`) |
| **6.1** | planned | Checkpoint snapshot pool (`rl_donut.rl.pool`) |
| **6.2** | planned | Opponent sampling (PFSP + latest self + bots) |
| **6.3** | planned | Offline Elo round-robin (`rl_donut.rl.elo`) |
| **6.4** | planned | 1v1 self-play training run |
| **6.5** | planned | 2v2 cooperative-competitive training |
| **6.6** | planned | Final format evaluation (1v1 vs 2v2) |

---

## Decision Log

| # | Decision | Choice | Why |
|---|---|---|---|
| R1 | Deep learning framework & device | PyTorch `2.14.1`, default to `cpu` with auto-detect for `cuda`/`mps` | PyTorch is named in the project brief. For small 2 × 256 MLPs with batch sizes of 16–64 environments, CPU execution avoids GPU/MPS kernel dispatch and synchronization latency, and ensures bitwise reproducibility across runs. |
| R2 | Experiment tracking | Pure local `runs/<id>/metrics.jsonl` | Self-contained, zero cloud credentials or third-party service dependencies, works offline, and directly powers the Phase 9 dashboard. Optional lightweight terminal summary each update. |
| R3 | Stand-in fidelity | Reuse real `World`, `physics_tick`, and `line_of_sight`; minimal health & timer rules | Gives accurate flight and collision mechanics without waiting for Team A's respawn/match loops. The stand-in is purely a scaffold for training pipeline development. |
| R4 | Action space flexibility | `action_nvec = (3, 2, A, 2)` with configurable aim bins `A` (default 16) | PPO multi-head actor handles any discrete action dimensions dynamically, enabling experimentation with finer aim angles or enemy-relative targeting. |
| R5 | Shaping weights & scale | Kill +0.3, death -0.3, dmg ±0.003, step penalty -0.0005 | Keeps dense shaping over an entire match bounded around ±1.0, properly balancing the sparse ±1.0 win/loss signal. |
| R6 | Shaping schedule | Linear decay of $\alpha$ from 1.0 to 0.1 over the first 60% of training | Promotes dense reward-driven exploration early while transitioning to pure win/loss optimization as policies mature. |
| R7 | Fall penalty | Suicide / falling into a pit incurs an extra -0.3 penalty | Teaches fuel preservation and pit avoidance early before combat competence develops. |

---

## Step 5.1: Tools and Environment Setup

### Framework & Dependencies
PyTorch `2.14.1` is pinned in `requirements.txt`. It is verified compatible with `numpy==2.5.4` on Python 3.12/3.13 without disturbing any random stream generators used by `mapgen`.

### Device Strategy
```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```
For our standard MLP architecture ($2 \times 256$), CPU inference and backprop take under 2 ms per batch, eliminating device transfer overhead.

### Metrics Log Format
Every PPO update writes one line of compact JSON to `runs/<id>/metrics.jsonl`:
```json
{
  "update": 1,
  "env_steps": 16384,
  "wall_time": 12.4,
  "loss_policy": 0.045,
  "loss_value": 0.12,
  "entropy": 1.38,
  "approx_kl": 0.008,
  "clip_fraction": 0.06,
  "explained_variance": 0.42,
  "mean_return": 0.15,
  "mean_length": 840,
  "watchlist": {
    "draw_rate": 0.05,
    "mean_engagement_dist": 14.2,
    "airborne_seconds": 18.5,
    "hover_seconds": 4.1,
    "shots_per_hit": 2.8,
    "spawn_kills": 0
  }
}
```
This file format is human-readable, append-only, and trivial to parse for plotting.


---

## Step 5.2: Stand-In Environment & Shared Contract

### `StandInEnv` Architecture
Built in `src/rl_donut/rl/stand_in.py`, the environment provides the exact batched API defined in `docs/handoff_phase5_6.md` §4:
- Batched across $M$ matches and $A = 2 \times \text{team\_size}$ agents.
- `reset()` draws random maps from the specified split (`train` or `val`), returning `(M, A, obs_dim)` float32.
- `step(actions)` executes a 2-tick decision step:
  - Updates agent kinematics via `physics_tick`.
  - Simulates hitscan shots using `line_of_sight`.
  - Tracks damage, kills, and deaths.
  - Automatically resets expired matches onto new maps while recording pre-reset observations in `final_obs`.
- Emits raw `Events` (`damage_dealt`, `damage_taken`, `kills`, `deaths`, `fell`, `shots`, `hits`, `outcome`).

### Shared Contract Test
`tests/test_env_contract.py` tests any environment factory for:
1. Expected tensor shapes and dtypes (`reset` and `step`).
2. `spec_version` tracking.
3. Seeding reproducibility (identical trajectories given identical seeds).
4. Auto-reset behavior preserving `final_obs`.
5. Combat event consistency (hits $\le$ shots, non-negative damage).
6. Non-zero outcomes only occurring upon match completion.


---

## Step 5.3: Reward Function & Shaping

### Formulation
The total reward for an agent on decision step $t$ is:
$$r_t = r_{\text{sparse}} + \alpha \cdot r_{\text{shaping}}$$

1. **Sparse Outcome ($r_{\text{sparse}}$)**:
   - Evaluated only when a match completes: $+1.0$ for winning team, $-1.0$ for losing team, $0.0$ for draw.
   - Zero during intermediate steps.
2. **Dense Shaping ($r_{\text{shaping}}$)**:
   - Kill bonus: $+0.3$
   - Death penalty: $-0.3$
   - Damage dealt: $+0.003$ per health point ($100\,\text{dmg} = +0.3$)
   - Damage taken: $-0.003$ per health point
   - Step penalty: $-0.0005$ per decision step (discourages stalling)
   - Fall penalty: $-0.3$ when agent dies by falling into a bottom pit.
3. **Annealing ($\alpha$)**:
   - $\alpha$ decays linearly from $1.0$ down to $0.1$ over the first 60% of training steps.
