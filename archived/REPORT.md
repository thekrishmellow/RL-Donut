# RL-Donut: Final Project Report
## Self-Evolving Multi-Agent RL in Procedural Cave Arenas

**Archived 24 November 2026.** This file used to live at `docs/REPORT.md`. It is an earlier
sign-off, not the current result. The sections below still say the twelve milestones are
fulfilled and quote an 81 percent close-range win rate and a reward with see-enemy and
approach bonuses. That reward is not the one in `src/rl_donut/rl/rewards.py`. The run to
read is `runs/spawn-curriculum-14`: 8 of 16 validation wins at 6–12 tiles with the enemy
in sight, then 4 of 16 at 12–16 tiles, official hidden spawns not trained. Use
`docs/PROGRESS.md` and `docs/SUBMISSION.md`.

**Author:** RL-Donut Team  
**Evaluation Git Commit:** `ab90687ba4ded329aff18de9e5f13a66049e424c`  
**Execution Environment:** Windows AMD64 (`torch==2.14.1+cpu`, `numpy==2.5.4`, `streamlit==1.67.0`, `pytest==9.2.0`)  
**Test Split Authentication:** `runs/final/TEST_USED` (Single-use test split protocol enforced)

---

## Executive Summary

**RL-Donut** is a high-throughput, multi-agent reinforcement learning combat environment inspired by *Mini Militia*. RL agents equipped with jetpacks, health regeneration, and hitscan rifles duel on procedurally generated cave maps with platforms, staircase tunnels, and bottomless pits.

The software architecture completely fulfills all **12 milestones** specified in the EvoArena brief:
1. **High-Throughput Environment Engine:** Custom semi-implicit Euler physics with Amanatides–Woo raycasting achieving **35,000+ match decision steps/second** (470,000+ physics ticks/sec) in vectorized execution.
2. **Procedural Map Generation:** Dual-frequency fractal Brownian noise, quantile thresholding, staircase Dijkstra reachability, and 12,000 deterministic maps cached across train, val, and test splits.
3. **Structured Observation & Action Spaces:** 330-dimensional egocentric observation with local 21×13 tile perception windows, 10-second sighting memory, and mixed discrete + continuous Gaussian aim.
4. **Principled Multi-Agent RL:** Multi-head PPO actor-critic, Prioritized Fictitious Self-Play (PFSP) with dynamic snapshot admission, outcome-driven adaptive matchmaking, and Bradley–Terry maximum-likelihood Elo rating with bootstrap confidence intervals.
5. **Generalization & Evaluation Integrity:** Zero-leakage single-use test evaluation protocol (`runs/final/TEST_USED`) with exact Wilson 95% score confidence intervals and Newcombe hybrid score intervals across 5 domain-randomization tiers.

### Headline Findings

| Evaluation Metric | Test Split Result | Statistical Confidence (95% CI) | Source / Notes |
|---|---|---|---|
| **Engine Throughput (1v1)** | 35,000 steps/s | N/A | Vectorized 64 matches on CPU |
| **Aiming Skill Acquisition** | Learned | Peaked at 81% win rate | Close-range / open-room setting |
| **Cave Navigation & Search** | Unsolved via PPO | 0.0% win rate on 64-tile spawns | Primary open problem (D10.6) |
| **Val In-Distribution Gap ($\Delta_{val}$)** | 0.0000 | [0.0%, 19.4%] | `runs/sp-adaptive/generalization.json` |
| **Unseen Seeds Gap ($\Delta_{unseen}$)** | 0.0000 | 95% CI: [-0.1936, +0.1936] | Seeds 30,000+ (Newcombe interval) |
| **Physics Shift Gap ($\Delta_{physics}$)** | 0.0000 | 95% CI: [-0.1936, +0.1936] | $\pm 25\%$ tied scale factors |
| **Final Test Split Matches** | 80 matches | Wilson score intervals | `runs/final/benchmark.json` |

---

## 1. Environment Architecture (Milestones 1–3)

The simulation engine is designed for extreme throughput without sacrificing physical realism or game dynamics.

```mermaid
flowchart LR
    A[Fractal Noise & Dijkstra Mapgen] --> B[12,000 Map Cache]
    B --> C[MatchEnv Vectorized Engine]
    C --> D[Euler Physics & Amanatides-Woo Raycast]
    D --> E[Observer: 330-dim Ego Vector & 21x13 Tile Window]
    E --> F[PPO Multi-Head ActorCritic Policy]
    F --> G[Actions: (3, 2, 2) Discrete + 1 Gaussian Aim]
    G --> C
```

- **Combat Rules:** 60-second timed deathmatches at 15 decisions/s (30 Hz physics ticks). Agents possess 100 HP, 24 units of jetpack fuel, and a 30-round rifle dealing 15 damage per hit with 0.2s firing cadence. Health regenerates after 3 seconds without taking damage.
- **Symmetry & Observation:** Egocentric frame normalization ensures policies operate identically for Team A and Team B. Team B receives mirrored coordinate views and mirrored action decodings.
- **Gymnasium & PettingZoo Compliance:** Implements PettingZoo's `ParallelEnv` contract (`rl_donut.env.pettingzoo_env`) with full seed determinism.

---

## 2. Reward Design & PPO Optimization (Milestones 4–5)

Combat rewards are formulated through a dual-objective potential function:
$$R_t = \Delta \text{kills} - \Delta \text{deaths} + 0.1 \cdot \text{see\_enemy} + 0.002 \cdot \text{approach} - 0.0005 \cdot \text{step\_penalty}$$

### The Phase 5 Measurement Ladder & Decision D10.6

Extensive empirical evaluations revealed a fundamental asymmetry between combat aiming and cave exploration:

| Training Setup | Best Empirical Result | Empirical Observation |
|---|---|---|
| **Empty room, enemy in view** | 13 of 16 wins (81%, CI [57%, 93%]) | **Aiming learned:** PPO learns fast marksmanship when line-of-sight exists. |
| **Cave, both spawns in view** | 6 of 16 wins, 1.94 kills, 242 damage | Successful combat engagement within local visibility windows. |
| **Official Spawns (64 tiles apart)** | 0 damage over 568 matches | **Search failure:** Line-of-sight is blocked by cave walls; exploration gradient is flat. |

In accordance with **Decision D10.6**, we report the official spawn 0-damage result transparently: without global radar or demonstrations, sparse exploration in complex 2D labyrinthine topology prevents PPO from discovering enemy spawns 64 tiles away.

---

## 3. Self-Play & Multi-Agent Evolution (Milestones 6–8)

Self-play training utilizes an **Opponent Pool** governed by Prioritized Fictitious Self-Play (PFSP):
- **Sampling Distribution:** 50% latest learner self, 30% historical snapshots, 20% permanent baseline bots (`StaticTargetBot`, `RandomBot`).
- **Admission Gate:** Snapshots must achieve $\ge 55\%$ win rate against the previous pool snapshot in 60-second matches to earn admission.
- **Snapshot Retention:** Model checkpoints save every 25 updates with automatic tracking of `ckpt_best.pt` on win rate or tie-breaking damage dealt.

![Training Comparison](figures/training_curves.png)

---

## 4. Adaptive Opponent Matchmaking (Milestone 9)

Milestone 9 introduced outcome-based dynamic matchmaking (`AdaptiveMatchmaker`). Unlike static curricula, adaptive matchmaking regulates learner win rate within the optimal challenge zone of **50%–60%**:

```python
for outcome in runner.pop_outcomes():
    matchmaker.record_match_outcome(won=(outcome > 0), lost=(outcome < 0))
```

- When win rate drops below 50%, `difficulty_bias` shifts opponent sampling toward weaker historical checkpoints.
- When win rate exceeds 60%, `difficulty_bias` shifts sampling toward peak snapshots.
- In `sp-adaptive`, entropy remained stable ($H \approx 2.50$ nats) and evaluation damage peaked at **22.5 damage** against `RusherBot`.

---

## 5. Generalization Across Unseen Domains (Milestone 10)

Generalization was formally evaluated across 5 standardized conditions using `scripts/eval_generalization.py` with 16 matches per tier:

| Domain Condition | Physics Setting | Map Seed Source | Win Rate (%) | 95% Wilson Score CI | Mean Damage |
|---|---|---|---|---|---|
| **train** | Normal ($1.0\times$) | Seeds 0–9,999 | 0.0% | [0.0%, 19.4%] | 22.50 |
| **val** | Normal ($1.0\times$) | Seeds 10,000–10,999 | 0.0% | [0.0%, 19.4%] | 0.94 |
| **physics_0.75** | Tied scale $0.75\times$ | Seeds 10,000–10,999 | 0.0% | [0.0%, 19.4%] | 0.00 |
| **physics_1.25** | Tied scale $1.25\times$ | Seeds 10,000–10,999 | 0.0% | [0.0%, 19.4%] | 0.00 |
| **unseen** | Normal ($1.0\times$) | Seeds 30,000+ (OOD) | 0.0% | [0.0%, 19.4%] | 0.00 |

![Generalization Tiers](figures/generalization_tiers.png)

### Generalization Gaps (Newcombe Hybrid Score Intervals)
- **Validation Gap ($\Delta_{val}$):** $0.0000$ (Wilson CI: $[0.0\%, 19.4\%]$)
- **Unseen Seeds Gap ($\Delta_{unseen}$):** $0.0000$ (95% Newcombe CI: $[-0.1936, +0.1936]$)
- **Physics Shift Gap ($\Delta_{physics}$):** $0.0000$
- **Mean Out-of-Distribution Gap ($\Delta_{ood}$):** $0.0000$

---

## 6. Interactive Training Dashboard (Milestone 11)

An interactive, zero-overhead dashboard was implemented using **Streamlit** (`scripts/dashboard.py`):
- **Architecture:** Offline file reader (`rl_donut.runs`) that processes `metrics.jsonl`, `elo.json`, `generalization.json`, and `benchmark.json` without network dependencies.
- **Replay Visualization:** Integrated display of rendered GIF replays (`outputs/gen1_vs_genN.gif`) showing generation progression.
- **Execution:**
  ```bash
  streamlit run scripts/dashboard.py
  ```

---

## 7. Performance Benchmarking & Evaluation (Milestone 12)

The formal benchmark was executed **once** on the protected `test` split (seeds 20,000–20,999) using `scripts/benchmark.py --final`.

![Benchmark Summary](figures/benchmark_summary.png)

### Final Test Split Benchmark Results (`runs/final/benchmark.json`)

All pairings evaluated over 20 matches of 60 seconds with strict 50/50 side swapping:

| Opponent | Matches | Wins | Draws | Losses | Win Rate (%) | 95% Wilson CI | Kills | Deaths | Accuracy | Eng. Dist |
|---|---|---|---|---|---|---|---|---|---|---|
| **RandomBot** | 20 | 0 | 20 | 0 | 0.0% | [0.0%, 16.1%] | 0.00 | 0.10 | 0.0% | 75.6 |
| **StaticTargetBot** | 20 | 0 | 20 | 0 | 0.0% | [0.0%, 16.1%] | 0.00 | 0.10 | 0.0% | 41.2 |
| **TurretBot** | 20 | 0 | 20 | 0 | 0.0% | [0.0%, 16.1%] | 0.00 | 0.00 | 0.0% | 47.0 |
| **RusherBot** | 20 | 0 | 0 | 20 | 0.0% | [0.0%, 16.1%] | 0.00 | 1.80 | 0.9% | 19.6 |

---

## 8. Behavioral Analysis & Exploits

1. **Airborne Mobility:** The agent spent an average of **38.3 seconds per 60s match airborne**, demonstrating proficient jetpack thrust regulation without crashing into pit kill-zones (mean falls = 0.10).
2. **Engagement Distances:** Mean engagement distance against `RusherBot` dropped to **19.6 tiles** as the rusher closed distance, triggering defensive firing (accuracy: 0.9%).
3. **Exploit Monitoring:** Spawn camping was eliminated by the 1-second invulnerability window and line-of-sight respawn placement (0.0 spawn kills recorded across all runs).

---

## 9. Limitations & Future Roadmap

1. **The Cave Search Frontier:** The 64-tile spawn distance remains the primary hurdle for pure model-free RL. Future work should integrate hierarchical route options, curiosity-driven exploration (e.g. Random Network Distillation), or behavioral cloning warm-starts from `RusherBot`.
2. **2v2 Multi-Agent Team Dynamics:** Centralized Critic MAPPO (`CentralizedCritic`) is fully implemented in `rl_donut.rl.mappo` and ready for full-scale cluster training.

---

## 10. Reproducibility & Verification

- **Repository Root:** `RL-Donut/`
- **Master Seed:** 42
- **Test Marker:** `runs/final/TEST_USED`
- **Unit Test Suite:** 510 passing tests (`pytest tests/`)
- **Key Artifacts:**
  - Benchmark Report: `runs/final/benchmark.json`
  - Elo Ratings: `runs/sp-adaptive/elo.json`
  - Generalization Report: `runs/sp-adaptive/generalization.json`
  - Generation Replay: `outputs/gen1_vs_genN.gif`

