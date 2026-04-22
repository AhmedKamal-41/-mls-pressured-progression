"""Charts for Leverkusen Streamlit page — inputs are DataFrames read from data/marts/."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from components.style import BG_BASE, BG_PANEL, CYAN, MUTED, VIOLET

if TYPE_CHECKING:
    from matplotlib.figure import Figure

INK = "#E5E7EB"


def fig_forest_sorted(
    df: pd.DataFrame,
    *,
    delta_col: str,
    ci_lo_col: str,
    ci_hi_col: str,
    metric_col: str = "metric",
    sort_abs: bool = True,
    xlabel: str = "Δ (metric-specific units)",
) -> Figure:
    """Horizontal forest plot of deltas with symmetric error bars."""
    plot_df = df.copy()
    if sort_abs:
        plot_df["_abs"] = plot_df[delta_col].abs()
        plot_df = plot_df.sort_values("_abs", ascending=True)
    metrics = plot_df[metric_col].tolist()
    vals = plot_df[delta_col].to_numpy(dtype=float)
    lo = plot_df[ci_lo_col].to_numpy(dtype=float)
    hi = plot_df[ci_hi_col].to_numpy(dtype=float)
    err = np.vstack([vals - lo, hi - vals])
    ci_ex = (lo > 0) | (hi < 0)
    colors = np.where(ci_ex, CYAN, MUTED)

    y_pos = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.38 * len(vals))))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    ax.barh(y_pos, vals, xerr=err, color=colors, ecolor=MUTED, capsize=3, height=0.65)
    ax.set_yticks(y_pos, metrics)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(BG_PANEL)
    fig.tight_layout()
    return fig


def fig_forest_from_overlay(overlay: pd.DataFrame) -> Figure:
    """Forest of Leverkusen 23/24 − Inter Miami (cross-case overlay)."""
    df = overlay.copy()
    df = df.sort_values("diff_leverkusen_minus_miami", key=lambda s: s.abs(), ascending=True)
    metrics = df["metric"].tolist()
    vals = df["diff_leverkusen_minus_miami"].to_numpy(dtype=float)
    lo = df["diff_ci_lo"].to_numpy(dtype=float)
    hi = df["diff_ci_hi"].to_numpy(dtype=float)
    err = np.vstack([vals - lo, hi - vals])
    ci_ex = (lo > 0) | (hi < 0)
    colors = np.where(ci_ex, CYAN, MUTED)
    y_pos = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.38 * len(vals))))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    ax.barh(y_pos, vals, xerr=err, color=colors, ecolor=MUTED, capsize=3, height=0.65)
    ax.set_yticks(y_pos, metrics)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Leverkusen 23/24 − Inter Miami 2023 (95% CI)", color=MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig


def fig_single_metric_compare(
    title: str,
    pre_mean: float | None,
    pre_lo: float | None,
    pre_hi: float | None,
    post_mean: float | None,
    post_lo: float | None,
    post_hi: float | None,
    xlabel: str,
) -> Figure | None:
    """One or two horizontal bars: optional 22/23 plus 23/24 from Open Data mart."""
    if post_mean is None or (isinstance(post_mean, float) and np.isnan(post_mean)):
        return None
    labels: list[str] = []
    means: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    if pre_mean is not None and not (isinstance(pre_mean, float) and np.isnan(pre_mean)):
        labels.append("Leverkusen 22/23")
        means.append(float(pre_mean))
        pl = float(pre_lo) if pre_lo is not None else float(pre_mean)
        ph = float(pre_hi) if pre_hi is not None else float(pre_mean)
        err_lo.append(float(pre_mean) - pl)
        err_hi.append(ph - float(pre_mean))
    labels.append("Leverkusen 23/24")
    means.append(float(post_mean))
    plp = float(post_lo) if post_lo is not None else float(post_mean)
    php = float(post_hi) if post_hi is not None else float(post_mean)
    err_lo.append(float(post_mean) - plp)
    err_hi.append(php - float(post_mean))

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    y = np.arange(len(labels))
    cols = [MUTED if "22/23" in lb else CYAN for lb in labels]
    ax.barh(y, means, xerr=[err_lo, err_hi], color=cols, capsize=3, ecolor=MUTED)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig


def fig_miami_overlay_bars(overlay: pd.DataFrame) -> Figure:
    """Leverkusen 23/24 vs Inter Miami per metric (paired CIs)."""
    df = overlay.copy()
    metrics = df["metric"].tolist()
    lv = df["leverkusen_mean"].to_numpy(dtype=float)
    lm = df["inter_miami_mean"].to_numpy(dtype=float)
    lv_e = np.vstack(
        [lv - df["leverkusen_ci_lo"].to_numpy(), df["leverkusen_ci_hi"].to_numpy() - lv]
    )
    im_e = np.vstack(
        [
            lm - df["inter_miami_ci_lo"].to_numpy(),
            df["inter_miami_ci_hi"].to_numpy() - lm,
        ]
    )

    n = len(metrics)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.45 * n)))
    fig.patch.set_facecolor(BG_BASE)
    ax.set_facecolor(BG_PANEL)
    y = np.arange(n)
    h = 0.35
    ax.barh(
        y - h / 2,
        lv,
        height=h,
        xerr=lv_e,
        color=CYAN,
        capsize=2,
        ecolor=MUTED,
        label="LEV 23/24",
    )
    ax.barh(
        y + h / 2,
        lm,
        height=h,
        xerr=im_e,
        color=VIOLET,
        capsize=2,
        ecolor=MUTED,
        label="MIA 23",
    )
    ax.set_yticks(y, metrics)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK, facecolor=BG_PANEL)
    ax.set_xlabel("value (metric-specific units)", color=MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig
