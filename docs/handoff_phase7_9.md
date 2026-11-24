# Phases 7–9: what is left, who writes it, how it merges

**Status note, 24 November 2026.** This note was written when the dashboard file did not
exist and some Phase 5 edits were only on one machine. `scripts/dashboard.py` is on `main`
and reads `runs/*/metrics.jsonl`. The learning code is on `main`. Where the table below
says the dashboard has no file, or that Phase 5 is not on `origin/main`, that sentence is
the old state. Current measurements are in `docs/PROGRESS.md`.

Share this file. Your friend builds it on a branch from `origin/main` and opens three pull requests. You merge those into `main`. Phase 5 stays yours.

Repo: [github.com/thekrishmellow/RL-Donut](https://github.com/thekrishmellow/RL-Donut)

```bash
git fetch origin
git checkout -b phase7-9 origin/main
.venv/bin/python -m pytest
```

Use `.venv/bin/python` for every command. Do not start from a dirty checkout. Local Phase 5 edits on this machine are not on `origin/main`.

---

## Already on main (do not rewrite)

| Piece | Where | State |
|---|---|---|
| Trainer on `MatchEnv`, Gaussian aim | `scripts/train.py`, `rl/model.py`, `rl/runner.py` | Runs. Phase 5 owns further training. |
| Self-play, pool, Elo | `scripts/train_selfplay.py`, `rl/pool.py`, `rl/elo.py` | On main. |
| Style table (aggressive, defensive, sniper) | `rl/styles.py` | Weights exist. Sniper bonus is never applied. |
| Adaptive matchmaker | `rl/adaptive.py` | Class exists. It is fed the wrong signal. |
| Generalization eval | `rl/generalization.py`, `scripts/eval_generalization.py` | Runs. Docstring describes a different physics setup than the code. |
| Dashboard | — | No file. |

---

## Not done

1. **Sniper bonus.** `compute_styled_reward` adds 0.004 for a hit at 15 tiles or more, but only if the caller passes hit distances. `Runner` calls `compute_reward`, so `--style sniper` pays nothing extra.
2. **Adaptive win rate.** `train_selfplay.py` sets win/loss from `mean_return > 0`. A 60-second match pays −0.45 in step penalty before any fight, so a real win can look like a loss and the difficulty shift goes the wrong way.
3. **OOD definition (M49).** Unseen seeds 30,000+ use the same generator settings as train. Physics is one tied scale, 0.75 and 1.25, on gravity and thrust together. The script text still says gravity and thrust move independently by ±25%. Shifted rock fraction and wider pits are not built. `data/maps/test` is for Phase 10 only.
4. **A real generalization report.** The script has not been run on a `MatchEnv` checkpoint and checked in as a result. A gap of zero, because both splits won nothing, is an acceptable result.
5. **Dashboard.** No app, no dependency pin, no test.

Do not train style agents in these pull requests. On official spawns the current policy deals 0 damage, because the enemy starts out of sight. Wiring and tests can merge. A style run waits until Phase 5 changes that.

---

## Code your friend writes

### PR 1 — Phase 7

| File | Change |
|---|---|
| `src/rl_donut/rl/runner.py` | One optional argument, `reward_fn`, defaulting to `compute_reward`. No other edit. |
| `src/rl_donut/rl/styles.py` | Sniper path reads hit distance from the events object (`shot_log` on `MatchEvents`). A 20-tile hit pays +0.004. A 10-tile hit pays 0. |
| `scripts/train_selfplay.py` | `--style sniper` passes that function as `reward_fn`. `--adaptive` records win, loss, and draw from `events.outcome` when the match ends, not from the sign of the return. |
| `tests/rl/test_styles.py` | The 20-vs-10 tile case above. |
| `tests/rl/test_adaptive.py` | Twenty wins raise `difficulty_bias`. Twenty losses lower it. Draws stay 0.5. |

Aggressive and defensive already work: they only swap `RewardWeights`. Leave those weights alone.

### PR 2 — Phase 8

| File | Change |
|---|---|
| `src/rl_donut/rl/generalization.py` | Report two tiers with honest names: **unseen** (seeds 30,000+, same generator settings) and **physics** (tied scale 0.75, tied scale 1.25). Do not split gravity from thrust. Do not rebuild `data/maps/`. Do not bump `GENERATOR_VERSION`. |
| `scripts/eval_generalization.py` | Docstring and printed names match that. Writes JSON next to the checkpoint. |
| `tests/rl/test_generalization.py` | Keep the existing tied-scale test. Add `compute_generalization_gap(0.8, 0.5) == 0.3`. |

Run the script once on any real `ckpt_*.pt` and put the JSON path in the PR body. Do not read the `test` split.

Out of this PR: rock fraction 30–35% or 60–65%, wider pits, asymmetric maps. Those need a generator change and a separate decision.

### PR 3 — Phase 9

| File | Change |
|---|---|
| `scripts/dashboard.py` | New. Streamlit. `streamlit run scripts/dashboard.py`. |
| `requirements.txt` | Pin the Streamlit version you install. |
| `tests/rl/test_dashboard.py` | Write a five-line `metrics.jsonl` in a temp directory. The loader returns five rows with `mean_return` and `entropy`. |

The app only reads files. It does not train and does not import the trainer.

Pages:

1. List `runs/*/metrics.jsonl`. Show that run's `config.json`.
2. Curves of `mean_return`, `entropy`, `loss_policy`, `alpha` against `update`.
3. If `generalization.json` is in the run directory, show train win rate, val win rate, and the two gaps.
4. Show an existing `outputs/*.png` or `*.gif`. Do not rewrite `scripts/render_replay.py`.

`metrics.jsonl` fields today: `update`, `env_steps`, `wall_time`, `loss_policy`, `loss_value`, `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`, `mean_return`, `mean_length`, `alpha`, `completed_episodes`. Self-play also has `opponent` and `pool_size`. A missing key is blank, not a crash. An empty `runs/` folder shows the train command, not a traceback.

Leave out of this PR: human vs checkpoint, generation 1 vs generation N, watchlist charts. `watchlist.py` is not called from the runner, and its hover threshold (2 tiles/s) disagrees with the env (1 tile/s). That is Phase 5.

---

## Files your friend does not touch

`src/rl_donut/rl/rewards.py`, `src/rl_donut/rl/practice.py`, `scripts/train.py`, `src/rl_donut/env/`, `data/maps/`, `DESIGN.md`.

`runs/` and `outputs/` stay untracked.

If `runner.py` conflicts on merge, keep `main`'s training loop and re-apply only the `reward_fn` argument.

---

## How you merge

Each PR rebases onto `origin/main`, then:

```bash
git push -u origin HEAD
gh pr create --title "Phase 7: score adaptive opponents from match outcomes" --body "$(cat <<'EOF'
## Summary
- Adaptive matchmaking uses win, loss, and draw from the match.
- The sniper bonus pays on a long hit.

## Test plan
- [ ] .venv/bin/python -m pytest tests/rl/test_adaptive.py tests/rl/test_styles.py
- [ ] .venv/bin/python -m pytest

EOF
)"
```

Use the same shape for Phase 8 and Phase 9. You review and merge into `main`. Do not force-push `main`. Do not mark Phases 7–9 approved in `DESIGN.md`.

Before merging, check:

- [ ] Friend's branch came from `origin/main`.
- [ ] The forbidden files above are absent from the diff.
- [ ] `.venv/bin/python -m pytest` passes.
- [ ] Phase 8 PR links one real `generalization.json`.
- [ ] Phase 9 PR pins Streamlit and the fixture test passes.
- [ ] No checkpoint, metric, or replay file is in the diff.
