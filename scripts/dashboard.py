"""Interactive training and evaluation dashboard for RL-Donut (Phase 9).

Usage:
    .venv/bin/python -m streamlit run scripts/dashboard.py

Reads offline run logs from runs/<run_id>/ and replay outputs from outputs/.
Does not import PyTorch, training loops, or game environments.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from rl_donut.runs import list_runs, load_config, load_json, load_metrics


st.set_page_config(page_title="RL-Donut Dashboard", page_icon="🍩", layout="wide")


@st.cache_data
def get_cached_metrics(run_dir_str: str, mtime: float):
    return load_metrics(Path(run_dir_str))


@st.cache_data
def get_cached_config(run_dir_str: str, mtime: float):
    return load_config(Path(run_dir_str))


st.title("🍩 RL-Donut Dashboard")

runs_root = Path("runs")
runs = list_runs(runs_root)

if not runs:
    st.info(
        "No training runs found in `runs/`.\n\n"
        "Start a training run with:\n"
        "```bash\n"
        ".venv/bin/python scripts/train.py --run-id demo --total-updates 10\n"
        "```"
    )
else:
    st.sidebar.header("Run Selection")
    run_names = [r.name for r in runs]
    selected_run_name = st.sidebar.selectbox("Select Run", run_names)
    selected_run_dir = runs_root / selected_run_name

    metrics_file = selected_run_dir / "metrics.jsonl"
    mtime = metrics_file.stat().st_mtime if metrics_file.exists() else 0.0
    config_file = selected_run_dir / "config.json"
    cfg_mtime = config_file.stat().st_mtime if config_file.exists() else 0.0

    config = get_cached_config(str(selected_run_dir), cfg_mtime)
    metrics = get_cached_metrics(str(selected_run_dir), mtime)

    # 1. Configuration Overview
    with st.expander("⚙️ Run Configuration", expanded=False):
        if config:
            cfg_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No `config.json` found in run folder.")

    # 2. Training Metrics
    st.subheader(f"📈 Training Progress: {selected_run_name}")
    if metrics:
        df = pd.DataFrame(metrics)
        if "update" in df.columns:
            df.set_index("update", inplace=True)

        # Overview Key Stats
        latest = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Updates", int(df.index[-1]))
        col2.metric("Env Steps", int(latest.get("env_steps", 0)))
        col3.metric("Mean Return", f"{latest.get('mean_return', 0.0):.2f}")
        col4.metric("Entropy", f"{latest.get('entropy', 0.0):.3f}")
        col5.metric("Policy Loss", f"{latest.get('loss_policy', 0.0):.3f}")

        # Training Curves & Stats Tabs
        tab_perf, tab_loss, tab_selfplay, tab_eval = st.tabs(
            ["Performance", "Losses & Optimization", "Self-Play Stats", "Evaluation"]
        )

        with tab_perf:
            perf_cols = [c for c in ["mean_return", "mean_length", "alpha"] if c in df.columns]
            if perf_cols:
                st.line_chart(df[perf_cols])

        with tab_loss:
            loss_cols = [c for c in ["loss_policy", "loss_value", "entropy", "approx_kl", "explained_variance"] if c in df.columns]
            if loss_cols:
                st.line_chart(df[loss_cols])

        with tab_selfplay:
            sp_cols = [c for c in ["pool_size", "difficulty_bias", "wins", "losses", "draws"] if c in df.columns]
            if sp_cols:
                st.line_chart(df[sp_cols])
            if "opponent" in df.columns:
                st.caption("Matches per Opponent:")
                st.bar_chart(df["opponent"].value_counts())
            elif not sp_cols:
                st.caption("No self-play metrics recorded for this run.")

        with tab_eval:
            eval_cols = [c for c in ["eval_win_rate", "eval_kills", "eval_damage"] if c in df.columns]
            has_eval = False
            if eval_cols:
                eval_df = df[eval_cols].dropna(how="all")
                if not eval_df.empty:
                    has_eval = True
                    st.line_chart(eval_df)
                    if "eval_win_rate" in eval_df.columns:
                        wr_series = eval_df["eval_win_rate"].dropna()
                        if not wr_series.empty:
                            best_upd = wr_series.idxmax()
                            st.success(f"Best Evaluation Win Rate: **{wr_series.loc[best_upd]:.1%}** at update **{best_upd}**")
            if not has_eval:
                st.caption("No evaluation metrics (`eval_win_rate`) recorded for this run.")
    else:
        st.warning("`metrics.jsonl` is empty or has no valid rows yet.")

    # 3. Generalization Evaluation
    gen_data = load_json(selected_run_dir, "generalization.json")
    if gen_data:
        st.subheader("🌐 Generalization Benchmark")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Win Rate", f"{gen_data.get('train_win_rate', 0.0):.1%}")
        c2.metric("Val Win Rate", f"{gen_data.get('val_win_rate', 0.0):.1%}")
        c3.metric("Val Gap (Δ_val)", f"{gen_data.get('generalization_gap_val', 0.0):+.4f}")
        c4.metric("OOD Gap (Δ_ood)", f"{gen_data.get('generalization_gap_ood', 0.0):+.4f}")

        # Tier breakdown table
        if "conditions" in gen_data:
            conds_df = pd.DataFrame(gen_data["conditions"]).T
            display_cols = [c for c in ["win_rate", "wilson_ci_lower", "wilson_ci_upper", "mean_kills", "mean_deaths"] if c in conds_df.columns]
            st.dataframe(conds_df[display_cols], use_container_width=True)

        # Unseen and Physics Gap metrics
        unseen_gap = gen_data.get("gap_unseen")
        phys_gap = gen_data.get("gap_physics")
        unseen_ci = gen_data.get("gap_unseen_ci")
        if unseen_gap is not None or phys_gap is not None:
            g1, g2 = st.columns(2)
            if unseen_gap is not None:
                ci_str = f" [{unseen_ci[0]:+.4f}, {unseen_ci[1]:+.4f}]" if unseen_ci else ""
                g1.metric("Unseen Map Gap (Δ_unseen)", f"{unseen_gap:+.4f}", help=f"95% CI:{ci_str}")
            if phys_gap is not None:
                g2.metric("Physics Shift Gap (Δ_physics)", f"{phys_gap:+.4f}")

    # 4. Elo Ratings
    elo_data = load_json(selected_run_dir, "elo.json")
    if elo_data and "ratings" in elo_data:
        st.subheader("🏆 Elo Tournament Ratings")
        ratings_df = pd.DataFrame(list(elo_data["ratings"].items()), columns=["Participant", "Elo Rating"])
        st.dataframe(ratings_df.sort_values(by="Elo Rating", ascending=False), use_container_width=True, hide_index=True)

    # 5. Benchmark & Behaviour (Phase 10)
    bench_data = load_json(selected_run_dir, "benchmark.json")
    if bench_data:
        st.subheader("📊 Final Benchmark & Behaviour")
        if "opponents" in bench_data:
            st.json(bench_data["opponents"])
        if "behaviour" in bench_data:
            st.json(bench_data["behaviour"])

# 5. Visualizations & Replays
st.subheader("🎥 Visualizations & Replays")
outputs_dir = Path("outputs")
if outputs_dir.exists():
    media_files = sorted(
        [f for f in outputs_dir.iterdir() if f.suffix.lower() in [".gif", ".png", ".jpg", ".jpeg"]],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if media_files:
        chosen_media = st.selectbox("Select Preview / Replay", [f.name for f in media_files])
        st.image(str(outputs_dir / chosen_media), caption=chosen_media)
    else:
        st.caption("No previews or replay animations found in `outputs/`.")
else:
    st.caption("Directory `outputs/` does not exist.")

