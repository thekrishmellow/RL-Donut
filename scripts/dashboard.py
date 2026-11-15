"""Interactive training and evaluation dashboard for RL-Donut (Phase 9).

Usage:
    .venv/bin/python -m streamlit run scripts/dashboard.py

Reads offline run logs from runs/<run_id>/ and replay outputs from outputs/.
Refreshes every few seconds, so a training run or a newly written preview shows up
without a restart. Does not import PyTorch, training loops, or game environments.
"""

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st

from rl_donut.runs import list_runs, load_config, load_json, load_metrics


st.set_page_config(page_title="RL-Donut", layout="wide")


@st.cache_data
def get_cached_metrics(run_dir_str: str, mtime: float):
    return load_metrics(Path(run_dir_str))


@st.cache_data
def get_cached_config(run_dir_str: str, mtime: float):
    return load_config(Path(run_dir_str))


def _finite(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return value
    except TypeError:
        return default


def draw_runs():
    runs_root = Path("runs")
    runs = list_runs(runs_root)
    if not runs:
        st.info(
            "No training runs found in `runs/`.\n\n"
            "Start one with:\n"
            "```bash\n"
            ".venv/bin/python scripts/train_spawn_curriculum.py --updates 50\n"
            "```"
        )
        return

    st.sidebar.header("Run")
    run_names = [r.name for r in runs]
    selected_run_name = st.sidebar.selectbox("Select run", run_names)
    selected_run_dir = runs_root / selected_run_name

    metrics_file = selected_run_dir / "metrics.jsonl"
    mtime = metrics_file.stat().st_mtime if metrics_file.exists() else 0.0
    config_file = selected_run_dir / "config.json"
    cfg_mtime = config_file.stat().st_mtime if config_file.exists() else 0.0

    config = get_cached_config(str(selected_run_dir), cfg_mtime)
    metrics = get_cached_metrics(str(selected_run_dir), mtime)
    curriculum = load_json(selected_run_dir, "curriculum.json")
    clone = load_json(selected_run_dir, "clone.json")

    st.subheader(selected_run_name)
    if mtime:
        st.caption(f"Log updated {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S}")

    with st.expander("Run configuration", expanded=False):
        shown = dict(config)
        if curriculum:
            shown["curriculum"] = curriculum
        if clone:
            shown["clone"] = {k: clone[k] for k in ("observation", "best_epoch", "best_val_log_prob", "eval") if k in clone}
        if shown:
            st.json(shown)
        else:
            st.caption("No config.json, curriculum.json, or clone.json in this run.")

    if not metrics:
        st.warning("`metrics.jsonl` is empty or has no valid rows yet.")
        return

    df = pd.DataFrame(metrics)
    if "update" in df.columns:
        df = df.set_index("update")
    latest = df.iloc[-1]

    columns = st.columns(5)
    columns[0].metric("Updates", int(df.index[-1]) if "update" in metrics[0] or df.index.name == "update" else len(df))
    if "rung" in df.columns and pd.notna(latest.get("rung")):
        columns[1].metric("Rung", str(latest["rung"]))
    elif "env_steps" in df.columns:
        columns[1].metric("Env steps", int(_finite(latest.get("env_steps"))))
    else:
        columns[1].metric("Rows", len(df))
    columns[2].metric("Mean return", f"{_finite(latest.get('mean_return')):.2f}")
    columns[3].metric("Entropy", f"{_finite(latest.get('entropy')):.3f}")
    if "eval_win_rate" in df.columns and df["eval_win_rate"].notna().any():
        best = df["eval_win_rate"].max()
        columns[4].metric("Best val win rate", f"{best:.0%}")
    else:
        columns[4].metric("Policy loss", f"{_finite(latest.get('loss_policy')):.3f}")

    if curriculum:
        st.caption(
            f"Curriculum stopped on `{curriculum.get('final_rung', '?')}` "
            f"after {curriculum.get('updates', '?')} updates. "
            f"Gate {float(curriculum.get('gate', 0)):.0%}."
        )

    tab_perf, tab_loss, tab_eval, tab_other = st.tabs(
        ["Training", "Losses", "Validation checks", "Other reports"]
    )

    with tab_perf:
        perf_cols = [c for c in ["mean_return", "mean_length", "alpha", "entropy"] if c in df.columns]
        if perf_cols:
            st.line_chart(df[perf_cols])
        if "rung" in df.columns:
            st.caption("Rung on each logged update: " + " → ".join(dict.fromkeys(df["rung"].dropna().astype(str))))

    with tab_loss:
        loss_cols = [c for c in ["loss_policy", "loss_value", "approx_kl", "explained_variance", "clip_fraction"] if c in df.columns]
        if loss_cols:
            st.line_chart(df[loss_cols])
        else:
            st.caption("This run did not log optimizer losses.")

    with tab_eval:
        eval_cols = [c for c in ["rung", "eval_wins", "eval_matches", "eval_win_rate", "eval_kills", "eval_damage"] if c in df.columns]
        if "eval_win_rate" in df.columns:
            eval_df = df[eval_cols].dropna(subset=["eval_win_rate"])
            if not eval_df.empty:
                best_upd = eval_df["eval_win_rate"].idxmax()
                st.success(
                    f"Best validation win rate {eval_df.loc[best_upd, 'eval_win_rate']:.0%} "
                    f"at update {best_upd}"
                    + (f" on {eval_df.loc[best_upd, 'rung']}" if "rung" in eval_df.columns else "")
                )
                st.dataframe(eval_df.reset_index(), hide_index=True)
                st.line_chart(eval_df[[c for c in ["eval_win_rate", "eval_kills", "eval_damage"] if c in eval_df.columns]])
            else:
                st.caption("No validation rows yet. They appear every eval interval.")
        else:
            st.caption("This run has no `eval_win_rate` column.")

    with tab_other:
        gen_data = load_json(selected_run_dir, "generalization.json")
        if gen_data and "conditions" in gen_data:
            st.markdown("**Generalization**")
            st.dataframe(pd.DataFrame(gen_data["conditions"]).T)
        elo_data = load_json(selected_run_dir, "elo.json")
        if elo_data and "ratings" in elo_data:
            st.markdown("**Elo**")
            ratings = pd.DataFrame(list(elo_data["ratings"].items()), columns=["Participant", "Elo"])
            st.dataframe(ratings.sort_values("Elo", ascending=False), hide_index=True)
        bench = load_json(selected_run_dir, "benchmark.json")
        if bench:
            st.markdown("**Benchmark**")
            st.json(bench)
        if not gen_data and not (elo_data and "ratings" in elo_data) and not bench:
            st.caption("No generalization, Elo, or benchmark file in this run.")


def draw_outputs():
    st.subheader("Generated pictures")
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        st.caption("`outputs/` does not exist yet. Preview scripts write maps, matches, and replays there.")
        return
    media_files = sorted(
        [f for f in outputs_dir.iterdir() if f.suffix.lower() in {".gif", ".png", ".jpg", ".jpeg"}],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not media_files:
        st.caption("No png, jpg, or gif files in `outputs/` yet.")
        return
    labels = [
        f"{path.name}  ·  {datetime.fromtimestamp(path.stat().st_mtime):%H:%M:%S}"
        for path in media_files
    ]
    chosen = st.selectbox("Preview", labels)
    path = media_files[labels.index(chosen)]
    st.image(str(path), caption=path.name)


@st.fragment(run_every=timedelta(seconds=3))
def live():
    draw_runs()
    draw_outputs()


st.title("RL-Donut")
st.caption("Training logs in `runs/` and pictures in `outputs/`. This page refreshes every 3 seconds.")
live()
