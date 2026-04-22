"""Page 3 — MLS League Context (ASA live + on-disk fallback).

Pulls team-season aggregates from ASA (xGoals, goals-added, xPass) with 24h
caching and a JSON snapshot under `data/marts/cache/`. Unlike static case-study
pages, the season selector defaults to the current calendar year.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.asa_refresh import (
    goals_added_rank_table,
    inter_miami_team_ids,
    load_team_lookup,
    merge_league_frames,
    resolve_league_bundle,
    summarize,
)
from components.style import BG_BASE, BG_PANEL, CYAN, VIOLET, caveat_box, dense_cols, metric_card

ROOT = Path(__file__).resolve().parents[2]
MARTS = ROOT / "data" / "marts"
LOOKUP_PATH = MARTS / "asa_team_lookup.csv"


st.markdown("# MLS League Context — Current Season")
st.markdown(
    "Team-season aggregates from American Soccer Analysis (ASA): expected goals, "
    "goals-added stacks, and passing model summaries. Data refreshes from ASA at "
    "most once every 24 hours (`st.cache_data`), with an optional on-disk snapshot "
    "when the API is unreachable."
)


season = st.sidebar.number_input(
    "Season",
    min_value=2013,
    max_value=dt.date.today().year,
    value=dt.date.today().year,
    step=1,
)

try:
    bundle, used_fallback = resolve_league_bundle(int(season))
except RuntimeError as e:
    st.error(str(e))
    caveat_box(
        source="ASA v1 (all endpoints failed; no cache).",
        scope=f"Attempted season {int(season)} — no data loaded.",
    )
    st.stop()

if used_fallback:
    st.warning(
        f"League context unavailable from ASA; showing last cached data from "
        f"`{bundle['fetched_at']}` (UTC)."
    )

if int(bundle["season"]) != int(season):
    st.warning(
        f"On-disk snapshot is for season **{bundle['season']}**, not the selected "
        f"**{int(season)}**. Switch season or refresh when online."
    )

df = merge_league_frames(bundle)
fetched_at = bundle.get("fetched_at")

if df.empty:
    st.warning(f"ASA returned no xGoals rows for season {int(season)}.")
    caveat_box(source="ASA v1", scope=f"Season {int(season)} empty xGoals response.")
    st.stop()

stats = summarize(df)
lookup = load_team_lookup(LOOKUP_PATH)
miami_highlight_ids = inter_miami_team_ids(lookup)

st.caption(
    f"Last fetched: `{fetched_at}` · live pull cached up to 24h · disk fallback when offline"
)

ga_rank = goals_added_rank_table(bundle.get("goals_added") or [])

c1, c2, c3, c4 = dense_cols(4)
with c1:
    metric_card(f"{stats['n_teams']}", "Teams reporting", tone="violet")
with c2:
    metric_card(
        f"{stats['xg_per_shot_mean']:.3f}",
        "League xG / shot",
        f"σ = {stats['xg_per_shot_std']:.3f}",
        tone="cyan",
    )
with c3:
    metric_card(f"{stats['points_mean']:.1f}", "Points (mean)", tone="violet")
with c4:
    metric_card(f"{int(df['count_games'].sum()):,}", "Team-games (xGoals)", tone="cyan")

if "xpass_pxe_diff_mean" in stats:
    x1, x2 = dense_cols(2)
    with x1:
        metric_card(
            f"{stats['xpass_pxe_diff_mean']:.2f}",
            "League mean — passes completed over expected (diff)",
            tone="violet",
        )
    with x2:
        vert = stats.get("xpass_vert_diff_mean")
        if vert is not None:
            metric_card(f"{vert:.3f}", "League mean — avg vertical distance (diff)", tone="cyan")

st.markdown("## xGoals for vs xGoals against")
plot_df = df.merge(lookup, on="team_id", how="left") if not lookup.empty else df.copy()
if "team_name" not in plot_df.columns:
    plot_df["team_name"] = pd.NA
hover_label = plot_df["team_name"].where(
    plot_df["team_name"].notna(),
    plot_df["team_id"].astype(str),
)
tid_str = plot_df["team_id"].astype(str)
colors = [VIOLET if str(tid) in miami_highlight_ids else "#6B7280" for tid in tid_str]
fig = go.Figure(
    data=[
        go.Scatter(
            x=plot_df["xgoals_for"],
            y=plot_df["xgoals_against"],
            mode="markers",
            marker=dict(size=11, color=colors, line=dict(width=0.5, color=BG_PANEL)),
            text=hover_label,
            hovertemplate="%{text}<br>xGF %{x:.2f}<br>xGA %{y:.2f}<extra></extra>",
        )
    ]
)
x_bar = float(plot_df["xgoals_for"].mean())
y_bar = float(plot_df["xgoals_against"].mean())
fig.add_vline(x=x_bar, line_dash="dash", line_color=CYAN, opacity=0.55, annotation_text="")
fig.add_hline(y=y_bar, line_dash="dash", line_color=CYAN, opacity=0.55)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG_BASE,
    plot_bgcolor=BG_PANEL,
    font=dict(color="#E5E7EB"),
    margin=dict(l=48, r=24, t=48, b=48),
    xaxis_title="xGoals for",
    yaxis_title="xGoals against",
    showlegend=False,
    height=520,
)
fig.update_xaxes(showgrid=True, gridcolor="#2A3148")
fig.update_yaxes(showgrid=True, gridcolor="#2A3148")
st.plotly_chart(fig, use_container_width=True)

if not miami_highlight_ids:
    st.caption(
        "Point highlight for Inter Miami is disabled until `team_id` is listed for "
        "that club in `data/marts/asa_team_lookup.csv`. ASA JSON does not ship "
        "display names—do not fabricate IDs."
    )

st.markdown("## Goals-added ranking (sum of goals_added_for)")
display_ga = ga_rank[
    ["rank", "team_id", "ga_total_goals_added_for", "ga_total_goals_added_against"]
].rename(
    columns={
        "ga_total_goals_added_for": "total_ga_for (sum)",
        "ga_total_goals_added_against": "total_ga_against (sum)",
    }
)
st.dataframe(
    display_ga.style.format(
        {
            "total_ga_for (sum)": "{:.2f}",
            "total_ga_against (sum)": "{:.2f}",
        }
    ),
    width="stretch",
    height=420,
)

st.markdown("## Narrative context — no statistical claims")
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown(
        """
<div class="pp-narrative pp-panel">
  <div class="pp-narrative-label">Philadelphia Union — narrative</div>
  <p class="pp-narrative-body">
    Coverage in the <em>Philadelphia Inquirer</em> has framed the Union’s 2026
    roster and coaching arc as a deliberate reset after prior peaks. Treat this
    as editorial context only — not event-level evidence for this app’s 2023
    StatsBomb measures.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
with n2:
    st.markdown(
        """
<div class="pp-narrative pp-panel">
  <div class="pp-narrative-label">Columbus Crew — narrative</div>
  <p class="pp-narrative-body">
    Columbus’s 2023 MLS Cup run under Wilfried Nancy appears here as an internal
    MLS contrast case in project prose. ASA aggregates on this page are not
    substitutes for that season’s event-level film or model pipeline.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
with n3:
    st.markdown(
        """
<div class="pp-narrative pp-panel">
  <div class="pp-narrative-label">San Diego FC — narrative</div>
  <p class="pp-narrative-body">
    English-language reporting (e.g. <em>The Guardian</em>) has documented San
    Diego FC’s 2025 expansion story as an organizational thread. No team-season
    ASA figures on this page are presented as statistical claims about 2025 or 2026
    performance — narrative cameo only.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Full merged team-season table (xGoals + GA totals + xPass columns)"):
    show = df.sort_values("points", ascending=False)
    st.dataframe(show, width="stretch", height=480)

with st.expander("MLS 2023 baseline (from `data/marts/asa_mls_2023_baseline.csv`)"):
    baseline_path = MARTS / "asa_mls_2023_baseline.csv"
    if baseline_path.exists():
        st.dataframe(pd.read_csv(baseline_path), width="stretch")
    else:
        st.info(
            "Baseline not yet materialized. Run "
            "`python -m pressured_progression.ingest.asa_league_baseline` "
            "from the repo root to populate it."
        )

caveat_box(
    source=(
        "American Soccer Analysis v1 API: `/mls/teams/xgoals`, `/mls/teams/goals-added`, "
        "`/mls/teams/xpass`. 24h Streamlit cache; optional JSON snapshot in "
        "`data/marts/cache/asa_league_bundle.json`."
    ),
    scope=(
        f"Season {int(season)} (bundle season {bundle['season']}) team-season aggregates — "
        "league summary only, not event-level chains."
    ),
    text=(
        f"Associational league snapshot. Last refresh timestamp: `{fetched_at}` UTC. "
        "ASA reports team-season aggregates; they do not replace event-level possession "
        "or post-regain reconstruction used elsewhere in this project. "
        "Use for situational context, not causal claims."
    ),
)
