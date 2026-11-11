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
| **5.4** | **done** | PPO implementation (`rl_donut.rl.ppo`) |
| **5.5** | **done** | Training runner with opponents (`scripts/train.py`, `rl_donut.rl.runner`) |
| **5.6** | **done** | Logging, exploit watchlist, checkpoint manager |
| **5.7** | **done** | Curriculum (static target → random bot → pits) |
| **5.8** | **done** | Evaluation against fixed bots (`scripts/eval_bots.py`, `evaluate.py`) |
| **6.1** | **done** | Checkpoint snapshot pool (`rl_donut.rl.pool`) |
| **6.2** | **done** | Opponent sampling (PFSP + latest self + bots) |
| **6.3** | **done** | Offline Elo round-robin (`rl_donut.rl.elo`, `scripts/elo_round_robin.py`) |
| **6.4** | **done** | 1v1 self-play training run (`scripts/train_selfplay.py`) |
| **6.5** | **done** | 2v2 cooperative-competitive training (`rl_donut.rl.mappo`) |
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
| R8 | Actor-Critic architecture | $2 \times 256$ MLP, orthogonal init (gain $\sqrt{2}$, actor heads 0.01), categorical heads | Separate value backbone prevents policy updates from destabilizing value baselines; independent action heads factorize MultiDiscrete action space cleanly. |
| R9 | GAE & PPO parameters | $\gamma=0.99, \lambda=0.95$, clip $\epsilon=0.2$, normalized advantages, value clipping | Standard CleanRL-style hyperparameter set with proven stability on continuous/discrete hybrid control tasks. |
| R10 | Value bootstrapping | Use `final_obs` on time-limit truncation | Match time limits are environment truncations, not agent terminations; bootstrapping from $V(\text{final\_obs})$ removes truncation bias. |
| R11 | Opponent batching | Call `Policy.act()` once for all opponent slots in the batch | Minimizes Python function call overhead and leverages batched array operations for bots. |
| R12 | Single-process rollout | Batched numpy rollouts in the main training process | Eliminates IPC serialization overhead and worker synchronization bottlenecks on loaded multi-core machines. |
| R13 | Exploit alarm thresholds | Draw > 35%, hover > 50%, engagement dist < 3.0 tiles, shots/hit > 8, spawn kills > 40% | Detects passive camping, ceiling stalling, spawn entrapment, and blind shooting early before degenerate policies consolidate. |
| R14 | Checkpoint retention | Keep latest 5 checkpoints plus milestone snapshots every 100 updates | Strict automatic pruning avoids filling the host machine's remaining disk space while preserving milestones for the opponent pool. |
| R15 | Curriculum progression | Stage 1 (static target, 90% win), Stage 2 (random bot no pits, 80%), Stage 3 (random bot with pits, 75%) | Progressively introduces targeting, moving targets, and fall hazard navigation before self-play. Turret/Rusher gates run after Phase 4 merge. |
| R16 | Pit curriculum filtering | Filter frozen maps on `len(m.pits) == 0` | Avoids regenerating maps and preserves exact frozen seed splits committed in `data/maps/`. |
| R17 | Evaluation protocol | Half-split side swapping on held-out `val` seeds with 95% Wilson intervals | Guarantees zero side bias and reports statistically rigorous confidence intervals on benchmark win rates. |
| S1 | Snapshot pool admission | Win rate $\ge 55\%$ vs latest snapshot over evaluation matches | Prevents inferior or degenerate policy versions from polluting the opponent pool. |
| S2 | Pool capacity & pruning | Max 30 snapshots; prune oldest non-permanent snapshots | Enforces disk budget while maintaining diverse historical play styles. |
| S3 | Opponent sampling | 50% latest self, 30% PFSP historical pool, 20% permanent bots | 50% latest pushes the frontier; 30% PFSP prevents catastrophic forgetting of hard styles; 20% bots prevents strategy cycling. |
| S4 | Elo anchoring | Anchor `RandomBot` at 1200.0 | Guarantees Elo ratings are comparable and grounded across independent training runs. |
| S5 | 2v2 Team rewards | $0.5 \times \text{own} + 0.5 \times \text{team\_avg}$ | Balances individual combat responsibility with team victory. |
| T1 | Combat style archetypes | Aggressive ($2\times$ dmg dealt), Defensive ($2\times$ dmg taken penalty), Sniper (bonus for $>15$ tiles) | Prevents strategy cycling and creates specialized agents for pool diversity. |
| T2 | Adaptive matchmaking | Dynamically shifts selection bias $b \in [-1, 1]$ based on rolling 20-match win rate | Keeps learner win rate within optimal 50%–60% zone to maximize sample efficiency. |
| T3 | CLI integration | Parameterized via `--style` and `--adaptive` in `scripts/train_selfplay.py` | Allows launching styled self-play runs with zero code modification. |
| U1 | Three-tier evaluation | Train seeds (0–9,999), Val seeds (10,000–10,999), OOD (physics $\pm 25\%$ & seeds 30,000+) | Comprehensive evaluation spanning in-distribution generalization and physical robustness. |
| U2 | Generalization gaps | $\Delta_{\text{val}} = \text{win}_{\text{train}} - \text{win}_{\text{val}}$, $\Delta_{\text{ood}} = \text{win}_{\text{train}} - \text{mean}(\text{win}_{\text{ood}})$ | Quantitative headline metrics measuring over-fitting and domain shift degradation. |
| U3 | On-the-fly OOD generation | Seeds 30,000+ generated dynamically via `generate_map(seed)` | Tests unseen procedural terrain without modifying frozen cache files in `data/maps/`. |

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


---

## Step 5.4: PPO Implementation

### Multi-Head Actor-Critic Architecture (`model.py`)
- **Backbone**: Separate 2-layer MLPs ($2 \times 256$) for actor and critic with $\tanh$ activations.
- **Actor Heads**: Dedicated linear heads for each dimension in `action_nvec`:
  - Move: 3 classes (left, none, right)
  - Jetpack: 2 classes (off, on)
  - Aim: 16 classes (discrete direction bins)
  - Shoot: 2 classes (no, yes)
- **Log Probability**: $\log \pi(a|s) = \sum_{i} \log \pi_i(a_i|s)$
- **Entropy Bonus**: $H(\pi) = \sum_{i} H(\pi_i)$

### Optimization Core (`ppo.py`)
- CleanRL-style rollout buffer processing.
- Generalized Advantage Estimation ($\text{GAE}(\gamma=0.99, \lambda=0.95)$).
- Advantage normalization per minibatch.
- Clipped surrogate policy objective ($L^{\text{CLIP}}$) with $\epsilon = 0.2$.
- Clipped value function objective ($L^{\text{VF}}$).
- Gradient clipping with maximum norm $0.5$.
- Model checkpoint saving and loading with full metadata and architecture tracking.


---

## Step 5.5: Training Runner & Opponent Coordination

### Runner Architecture (`runner.py`)
- Coordinates the learner policy vs opponent policy in a unified batched loop.
- **Side Alternation**: In match $m$, the learner plays Team $m \pmod 2$ and the opponent plays Team $1 - (m \pmod 2)$. This eliminates left/right side bias in the learned policy.
- Batched rollout collection for $N$ decision steps.
- GAE computation over completed and truncated trajectories with terminal value bootstrapping.
- Clean logging of policy loss, value loss, entropy, explained variance, and return curves.

### Training CLI (`scripts/train.py`)
- Full CLI supporting `--n-matches`, `--n-steps`, `--total-updates`, `--opponent`, `--lr`, `--split`.
- Emits structured append-only logs in `runs/<id>/metrics.jsonl`.
- Saves run metadata in `runs/<id>/config.json`.
- Periodically saves model checkpoints in `runs/<id>/ckpt_<update>.pt`.


---

## Step 5.6: Logging, Exploit Watchlist & Checkpoint Management

### Exploit Watchlist (`watchlist.py`)
Automatically monitors training episodes for degenerate RL strategies:
- **`draw_rate`**: Alarms if $> 35\%$ of completed matches end in a tie (indicating passive mutual evasion).
- **`hover_fraction`**: Alarms if agents spend $> 50\%$ of time in air with $|v_y| < 2.0\,\text{tiles/s}$ (ceiling camping).
- **`mean_engagement_dist`**: Alarms if average hit distance drops below $3.0$ tiles (face-hugging / spawn pinning).
- **`shots_per_hit`**: Alarms if $> 8.0$ (wasteful blind firing).
- **`spawn_kill_fraction`**: Alarms if $> 40\%$ of kills occur within $8$ tiles of an enemy spawn point.

### Checkpoint Manager (`checkpoints.py`)
- Automatically saves PyTorch model and optimizer states.
- Preserves all milestone checkpoints (`update % 100 == 0`).
- Prunes intermediate checkpoints to keep at most the last 5 updates, protecting host storage space.


---

## Step 5.7: Curriculum Progression

### Stages & Gates (`curriculum.py`)
1. **Stage 1 (Static Target, No Pits)**:
   - Opponent: `StaticTargetBot`
   - Map filter: `len(m.pits) == 0` (flat bottom maps from `train`)
   - Gate: $\ge 90\%$ win rate over 20 evaluation matches.
2. **Stage 2 (Random Motion, No Pits)**:
   - Opponent: `RandomBot`
   - Map filter: `len(m.pits) == 0`
   - Gate: $\ge 80\%$ win rate.
3. **Stage 3 (Random Motion, All Maps / Bottom Pits)**:
   - Opponent: `RandomBot`
   - Map filter: `None` (includes maps with 1 or 2 bottom pits)
   - Gate: $\ge 75\%$ win rate.
4. *(Stages 4 & 5 vs Turret and Rusher bots will unlock following Phase 4 merge).*

---

## Step 5.8: Evaluation Against Bots

### Evaluation Harness (`evaluate.py`)
- Evaluates any two policies over $N$ matches.
- Swaps sides halfway ($N/2$ matches with learner as Team A, $N/2$ as Team B) to ensure zero side bias.
- Reports win, loss, and draw counts along with 95% Wilson score confidence intervals.
- Computes mean kills, deaths, damage dealt, and damage taken per match.

### Evaluation CLI (`scripts/eval_bots.py`)
Command-line tool for evaluating trained checkpoints:
```bash
python scripts/eval_bots.py --checkpoint runs/<id>/ckpt_0050.pt --opponent static --n-matches 20
```
Outputs detailed combat and survival metrics with statistical confidence intervals.


---

## Phase 6: Self-Play, Opponent Pool, Elo & 2v2

### Opponent Pool & PFSP (`pool.py`)
- Maintains up to 30 policy snapshots plus permanent benchmark bots (`StaticTargetBot`, `RandomBot`).
- **Admission Gate**: Candidates are evaluated against the current latest snapshot over 16 evaluation matches. A candidate must achieve $\ge 55\%$ win rate to be admitted.
- **Prioritized Fictitious Self-Play (PFSP)**:
  - 50% chance of sampling latest self (frontier exploration).
  - 30% chance of sampling an older snapshot from the pool, weighted by $(1 - w_i)^2$ where $w_i$ is the learner's win rate against snapshot $i$.
  - 20% chance of sampling a fixed baseline bot (anchoring).

### Elo Rating System (`elo.py`, `scripts/elo_round_robin.py`)
- Logistic Elo rating model with iterative Bradley-Terry refinement over tournament matrices.
- Anchored to `RandomBot` at fixed rating $1200.0$ so ratings are comparable across training runs.
- Runs offline round-robin tournaments on held-out `val` maps with side-swapping.
- Outputs structured ratings report to `runs/<id>/elo.json`.

### 2v2 Multi-Agent PPO (`mappo.py`)
- **Centralized Critic**: Evaluates global environment state (`env.get_state()`) across all agents.
- **Team Reward Blending**: Blends individual agent performance with team average ($0.5 \times \text{own} + 0.5 \times \text{team\_avg}$).
- **Parameter Sharing**: Teammates share the actor policy network while conditioning on their specific agent observation slot.


---

## Phase 7: Adaptive Opponents and Combat Styles

### Behavioral Archetypes (`styles.py`)
Defines specialized asymmetric reward configurations to diversify the opponent pool and prevent rock-paper-scissors strategy cycling:
- **`AGGRESSIVE`**: $2\times$ damage dealt bonus ($+0.006$), higher kill bonus ($+0.5$), $2\times$ step penalty ($-0.0010$) to force proactive engagement and pursuit.
- **`DEFENSIVE`**: $2\times$ damage taken penalty ($-0.006$), higher death penalty ($-0.5$), minimal step penalty ($-0.0002$) to encourage patient spacing, cover utilization, and retreat.
- **`SNIPER`**: Flat bonus ($+0.004$) for landed shots exceeding 15 tiles distance, fostering long-range rifle precision and vertical control.

### Dynamic Adaptive Matchmaking (`adaptive.py`)
- **Target Win Rate Zone**: Actively regulates training difficulty so the learner maintains a 50%–60% win rate.
- **Difficulty Bias ($b \in [-1, +1]$)**: When the learner dominates ($>60\%$), bias increases towards sampling the latest self snapshot ($p_{\text{latest}} \to 0.80$). When struggling ($<50\%$), bias shifts towards baseline bots ($p_{\text{bots}} \to 0.50$).
- Integrated via `--adaptive` in `scripts/train_selfplay.py`.



---

## Phase 8: Generalization Evaluation

### Evaluation Protocol (generalization.py)
Evaluates policy models across three tiers with 50/50 side swapping and 95% Wilson confidence intervals:
1. **In-Distribution Train**: Seeds 0–9,999 (baseline training distribution).
2. **In-Distribution Held-Out (al)**: Seeds 10,000–10,999 (unseen maps with standard physics).
3. **Out-of-Distribution (OOD)**:
   - **Physics Parameter Shifts**: Gravity $\pm 25\%$ (18.75 to 31.25 tiles/s²), Jetpack Thrust $\pm 25\%$ (30.0 to 50.0 tiles/s²).
   - **Unseen OOD Cave Maps**: Seeds 30,000+ generated on the fly via generate_map(seed).

### Headline Generalization Gaps
- **Validation Gap**: $\Delta_{\text{val}} = \text{win\_rate}_{\text{train}} - \text{win\_rate}_{\text{val}}$
- **OOD Shift Gap**: $\Delta_{\text{ood}} = \text{win\_rate}_{\text{train}} - \text{mean\_ood\_win\_rate}$

### Evaluation CLI (scripts/eval_generalization.py)
CLI command evaluating any checkpoint across all tiers:
\\ash
python scripts/eval_generalization.py --checkpoint runs/<id>/ckpt_0050.pt --opponent random --n-matches 20
\Outputs tabular per-condition win rates with Wilson intervals and saves 
uns/<id>/generalization.json.


---

## Phase 7: Adaptive Opponents and Combat Styles

### Behavioral Archetypes (`styles.py`)
Defines specialized asymmetric reward configurations to diversify the opponent pool and prevent rock-paper-scissors strategy cycling:
- **`AGGRESSIVE`**: $2\times$ damage dealt bonus ($+0.006$), higher kill bonus ($+0.5$), $2\times$ step penalty ($-0.0010$) to force proactive engagement and pursuit.
- **`DEFENSIVE`**: $2\times$ damage taken penalty ($-0.006$), higher death penalty ($-0.5$), minimal step penalty ($-0.0002$) to encourage patient spacing, cover utilization, and retreat.
- **`SNIPER`**: Flat bonus ($+0.004$) for landed shots exceeding 15 tiles distance, fostering long-range rifle precision and vertical control.

### Dynamic Adaptive Matchmaking (`adaptive.py`)
- **Target Win Rate Zone**: Actively regulates training difficulty so the learner maintains a 50%–60% win rate.
- **Difficulty Bias ($b \in [-1, +1]$)**: When the learner dominates ($>60\%$), bias increases towards sampling the latest self snapshot ($p_{\text{latest}} \to 0.80$). When struggling ($<50\%$), bias shifts towards baseline bots ($p_{\text{bots}} \to 0.50$).
- Integrated via `--adaptive` in `scripts/train_selfplay.py`.

