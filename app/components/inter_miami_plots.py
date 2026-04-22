"""Matplotlib / Plotly charts for Inter Miami page — reads only pre-aggregated inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from components.style import BG_BASE, BG_PANEL, CYAN, MUTED, VIOLET, pitch_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure

METRIC_ORDER = [
    "xg_per_regain",
    "time_to_shot_median",
    "rushed_shot_rate",
    "regain_to_final_third_rate",
    "regain_to_loss_rate",
    "patience_composite",
]

INK = "#E5E7EB"
GRAY_LINK = "#64748B"


def fig_post_regain_bar(
    team_pr: pd.DataFrame,
    baseline: pd.DataFrame | None,
) -> Figure:
    """Horizontal bars for six post-regain metrics with 95% CIs; optional ASA xG/shot ref line."""
    df = team_pr.set_index("metric").loc[METRIC_ORDER].reset_index()
    labels = df["metric"].tolist()
    vals = df["point_estimate"].to_numpy(dtype=float)
    lo_err = vals - df["ci_lo"].to_numpy(dtype=float)
    hi_err = df["ci_hi"].to_numpy(dtype=float) - vals

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    ax.barh(labels, vals, color=VIOLET, xerr=[lo_err, hi_err], ecolor=MUTED, capsize=3)
    if baseline is not None and not baseline.empty:
        xp_ref = baseline.loc[baseline["metric"] == "xg_per_shot", "league_mean"]
        if not xp_ref.empty:
            ax.axvline(
                float(xp_ref.iloc[0]),
                color=CYAN,
                linestyle="--",
                linewidth=1.2,
                label="ASA MLS 2023 xG/shot league mean",
            )
            ax.legend(loc="lower right", fontsize=8, labelcolor=INK, facecolor=BG_PANEL)
    ax.set_xlabel("value (metric-specific units)", color=MUTED, fontsize=10)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=INK)
    for spine in ax.spines.values():
        spine.set_color(BG_BASE)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_regain_pitch_hex(regains: pd.DataFrame) -> Figure:
    """StatsBomb pitch + hexbin density for regain locations."""
    xs = regains["regain_x"].dropna().astype(float)
    ys = regains["regain_y"].dropna().astype(float)
    fig, ax, pitch = pitch_plot(figsize=(10.5, 6.8))
    ax.set_facecolor(BG_PANEL)
    if len(xs) > 0:
        hb = pitch.hexbin(
            xs,
            ys,
            ax=ax,
            gridsize=(18, 12),
            cmap="Purples",
            mincnt=1,
            edgecolors="none",
        )
        cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("count", color=MUTED)
        cb.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=MUTED)
    fig.tight_layout()
    return fig


def fig_time_to_shot_hist(regains: pd.DataFrame) -> Figure:
    """Histogram of seconds to next shot (≤15 s window)."""
    v = regains["next_shot_seconds"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    if not v.empty:
        ax.hist(v, bins=15, range=(0, 15), color=VIOLET, edgecolor=BG_PANEL, linewidth=0.5)
        med = float(v.median())
        ax.axvline(med, color=CYAN, linestyle="--", linewidth=2, label=f"median = {med:.1f}s")
        ax.legend(loc="upper right", fontsize=9, labelcolor=INK, facecolor=BG_PANEL)
    ax.set_xlim(0, 15)
    ax.set_xlabel("seconds from regain to next shot (≤15)", color=MUTED)
    ax.set_ylabel("regains", color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(BG_PANEL)
    fig.tight_layout()
    return fig


def fig_regain_sankey(regains: pd.DataFrame) -> go.Figure:
    """Sankey: regain → four outcome buckets (same logic as pipeline)."""
    shot = regains["next_shot_seconds"].notna()
    lost = regains["lost_within_4_actions"] & ~shot
    ff_no_shot = regains["reached_final_third"] & ~shot & ~lost
    other = ~(shot | lost | ff_no_shot)

    counts = [int(shot.sum()), int(ff_no_shot.sum()), int(lost.sum()), int(other.sum())]
    labels = [
        "regains",
        "shot ≤15s",
        "final third (no shot)",
        "lost ≤4 actions",
        "other",
    ]
    sources = [0, 0, 0, 0]
    targets = [1, 2, 3, 4]
    link_colors = [VIOLET, CYAN, GRAY_LINK, "#94A3B8"]
    node_colors = ["#334155", VIOLET, CYAN, GRAY_LINK, "#64748B"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, color=node_colors, pad=14, thickness=14),
                link=dict(
                    source=sources,
                    target=targets,
                    value=counts,
                    color=[c + "99" for c in link_colors],
                ),
            )
        ]
    )
    fig.update_layout(
        font=dict(color=INK, size=11, family="Inter, sans-serif"),
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    return fig
