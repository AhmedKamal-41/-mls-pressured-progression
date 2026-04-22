"""Page 1 — Inter Miami build-up + post-regain profile.

Loads persisted marts (`data/marts/*`) only. Charts render in-process; no network.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from components.inter_miami_plots import (
    fig_post_regain_bar,
    fig_regain_pitch_hex,
    fig_regain_sankey,
    fig_time_to_shot_hist,
)
from components.style import caveat_box, dense_cols, metric_card

ROOT = Path(__file__).resolve().parents[2]
MARTS = ROOT / "data" / "marts"

REGAIN_PATH = MARTS / "regain_events.parquet"


def _load_tables() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]:
    team_bf = pd.read_csv(MARTS / "team_buildup_failure.csv")
    im_row = team_bf.loc[team_bf["team_name"] == "Inter Miami"].iloc[0]
    team_pr = pd.read_csv(MARTS / "team_post_regain.csv")
    baseline_path = MARTS / "asa_mls_2023_baseline.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    return im_row, team_pr, baseline


im_row, team_pr, baseline = _load_tables()
n_matches = int(im_row["n_matches"])
n_chains = int(im_row["n_chains_qualified"])
n_regains = int(team_pr.iloc[0]["n_regains"])
fail_rate = float(im_row["failure_rate"])
fail_lo = float(im_row["ci_lo"])
fail_hi = float(im_row["ci_hi"])

st.markdown("# Inter Miami 2023 — Build-Up and Post-Regain Profile")
st.markdown(
    "*Six StatsBomb Open Data matches (Aug–Oct 2023); the event-level MLS slice "
    "available without paid licensing — patterns are associational, not causal.*"
)

c1, c2, c3, c4 = dense_cols(4)
with c1:
    metric_card(str(n_matches), "Matches", tone="violet")
with c2:
    metric_card(str(n_chains), "Labeled possessions", tone="cyan")
with c3:
    metric_card(str(n_regains), "Regains", tone="violet")
with c4:
    metric_card(
        f"{fail_rate:.0%}",
        "Build-up failure rate",
        f"95% CI [{fail_lo:.0%}, {fail_hi:.0%}]",
        tone="cyan",
    )

regains = None if not REGAIN_PATH.exists() else pd.read_parquet(REGAIN_PATH)

if regains is None:
    st.warning(
        "`data/marts/regain_events.parquet` not found — run the post-regain pipeline "
        "to materialize regain events. Pitch map, Sankey, and time-to-shot charts are hidden."
    )
else:
    st.caption("Regain locations (StatsBomb coordinates; Inter Miami attacking right →)")
    fig_pitch = fig_regain_pitch_hex(regains)
    st.pyplot(fig_pitch, clear_figure=True)
    plt.close(fig_pitch)

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.caption("Post-regain metrics (95% bootstrap CI, match-resampled)")
        fig_bar = fig_post_regain_bar(team_pr, baseline)
        st.pyplot(fig_bar, clear_figure=True)
        plt.close(fig_bar)
    with col_right:
        st.caption("Regain outcome flow")
        fig_sankey = fig_regain_sankey(regains)
        st.plotly_chart(fig_sankey, use_container_width=True)

    st.caption("Time-to-shot among regains with a shot inside 15s")
    fig_hist = fig_time_to_shot_hist(regains)
    st.pyplot(fig_hist, clear_figure=True)
    plt.close(fig_hist)

if regains is None:
    # Still show Module B bars from CSV when parquet missing
    st.caption("Post-regain metrics (95% bootstrap CI, match-resampled)")
    fig_bar_only = fig_post_regain_bar(team_pr, baseline)
    st.pyplot(fig_bar_only, clear_figure=True)
    plt.close(fig_bar_only)

caveat_box(
    source=(
        "StatsBomb Open Data (event-level). Match catalog lists 360 availability; "
        "MLS 2023 360 JSON requests may 404 — see `docs/data_reality.md` §1 addendum."
    ),
    scope=(
        "Inter Miami 2023 only. n = 73 qualifying chains, 6 matches, 353 regains. "
        "Cross-team MLS comparison unavailable in Open Data."
    ),
)
