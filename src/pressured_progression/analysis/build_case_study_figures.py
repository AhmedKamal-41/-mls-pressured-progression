"""Figures for notebook 06_inter_miami_case_study.ipynb.

Each builder function:
  - loads only from data/marts/*
  - returns a matplotlib Figure (or writes PNG directly for plotly Sankey)
  - accepts `save: bool = True` to persist to docs/figures/case_study_*.png

CLI (`python -m pressured_progression.analysis.build_case_study_figures`) runs
all seven in order and writes the PNGs. The notebook imports and calls these
same functions so its code cells are three-liners.

Palette: violet #7C3AED primary, cyan #06B6D4 secondary. Inter UI / JetBrains
Mono numerals are requested via rcParams with graceful fallback when the
fonts aren't installed.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MARTS = ROOT / "data" / "marts"
FIG_DIR = ROOT / "docs" / "figures"

VIOLET = "#7C3AED"
CYAN = "#06B6D4"
INK = "#111827"
MUTED = "#6B7280"
BG = "#FFFFFF"


def set_style() -> None:
    """Set global matplotlib style for the case study. Falls back silently
    to sans-serif / monospace defaults if Inter / JetBrains Mono are absent."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.monospace"] = [
        "JetBrains Mono",
        "DejaVu Sans Mono",
        "Consolas",
        "monospace",
    ]
    plt.rcParams["axes.edgecolor"] = INK
    plt.rcParams["axes.labelcolor"] = INK
    plt.rcParams["xtick.color"] = INK
    plt.rcParams["ytick.color"] = INK
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG


def _save(fig, name: str, save: bool) -> None:
    if not save:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight", facecolor=BG)


def _mono(ax, *, size: int = 12):
    """Apply monospace family to tick labels (for numerals)."""
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(size)


# -------------------------------------------------------------- Section 1


def hook_snapshot(save: bool = True):
    """Four-tile data snapshot: matches / chains / regains / failure rate."""
    set_style()
    feats = pd.read_parquet(MARTS / "buildup_features.parquet")
    regains = pd.read_parquet(MARTS / "regain_events.parquet")
    team_bf = pd.read_csv(MARTS / "team_buildup_failure.csv")
    im = team_bf[team_bf["team_name"] == "Inter Miami"].iloc[0]

    n_matches = int(feats["match_id"].nunique())
    n_chains = int(len(feats))
    n_regains = int(len(regains))
    rate = float(im["failure_rate"])

    tiles = [
        ("Matches", f"{n_matches}", "Aug–Oct 2023 (Messi era)"),
        ("Build-up chains", f"{n_chains}", "qualifying: own half + pressure"),
        ("Regains", f"{n_regains}", "defensive actions that flipped"),
        ("Failure rate", f"{rate:.0%}", "of qualifying build-ups"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (label, number, sub) in zip(axes, tiles, strict=False):
        ax.set_axis_off()
        ax.add_patch(
            plt.Rectangle((0.02, 0.05), 0.96, 0.9, linewidth=0, facecolor="#F5F3FF", alpha=0.8)
        )
        ax.text(
            0.5,
            0.72,
            number,
            ha="center",
            va="center",
            fontsize=44,
            fontweight="bold",
            color=VIOLET,
            family="monospace",
        )
        ax.text(0.5, 0.38, label, ha="center", va="center", fontsize=13, color=INK)
        ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=9, color=MUTED)
    fig.suptitle(
        "Inter Miami 2023 — the Open Data corpus",
        fontsize=14,
        y=1.02,
        color=INK,
    )
    fig.tight_layout()
    _save(fig, "case_study_01_hook.png", save)
    return fig


# -------------------------------------------------------------- Section 2


def pressure_exposure(save: bool = True):
    """Hexbin of Inter Miami regain locations — a proxy for 'pressure resolved.'
    Violet colormap; defensive third (x<40) highlighted because that's where
    pressure is faced by a team defending deep."""
    set_style()
    regains = pd.read_parquet(MARTS / "regain_events.parquet")
    xs = regains["regain_x"].dropna().astype(float)
    ys = regains["regain_y"].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Violet-tinted colormap by sampling the violet + fading to white.
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("violet_fade", ["#F5F3FF", VIOLET], N=256)
    hb = ax.hexbin(xs, ys, gridsize=20, cmap=cmap, extent=(0, 120, 0, 80))
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("regains", color=INK)
    cb.ax.tick_params(labelcolor=INK)

    # Shade own-third to flag "pressure faced and resolved."
    ax.axvspan(0, 40, color=CYAN, alpha=0.08, zorder=0)
    ax.axvline(40, color=CYAN, linewidth=1, linestyle="--", alpha=0.9)
    ax.axvline(80, color=MUTED, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(
        20,
        74,
        "own defensive third\n(pressure faced)",
        ha="center",
        color=CYAN,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.set_xlabel("attacking right →", color=INK)
    ax.set_ylabel("y", color=INK)
    ax.set_title(
        "Where Inter Miami won the ball back",
        fontsize=13,
        pad=12,
    )
    _mono(ax)
    fig.tight_layout()
    _save(fig, "case_study_02_pressure_exposure.png", save)
    return fig


# -------------------------------------------------------------- Section 3


def buildup_failure_rate(save: bool = True):
    """Single horizontal bar with 95% bootstrap CI — Inter Miami's build-up
    failure rate with an honest 'small n' note on the chart."""
    set_style()
    team_bf = pd.read_csv(MARTS / "team_buildup_failure.csv")
    im = team_bf[team_bf["team_name"] == "Inter Miami"].iloc[0]
    point = float(im["failure_rate"])
    lo = float(im["ci_lo"])
    hi = float(im["ci_hi"])
    n_chains = int(im["n_chains_qualified"])
    n_matches = int(im["n_matches"])

    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.barh(
        [0],
        [point],
        color=VIOLET,
        height=0.38,
        xerr=[[point - lo], [hi - point]],
        ecolor=INK,
        capsize=6,
    )
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("share of qualifying build-ups that failed", color=INK)
    ax.set_title(
        "Build-up failure rate — Inter Miami 2023",
        fontsize=13,
        pad=12,
    )
    ax.text(
        point + 0.015,
        0,
        f"{point:.0%}  [{lo:.0%}, {hi:.0%}]",
        va="center",
        fontsize=14,
        color=INK,
        family="monospace",
        fontweight="bold",
    )
    ax.text(
        0,
        -0.75,
        f"n = {n_chains} qualifying chains across {n_matches} matches. "
        "95% CI via match-resampled bootstrap. "
        "Treat magnitude with caution — six matches is a small sample.",
        fontsize=9,
        color=MUTED,
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _mono(ax)
    fig.tight_layout()
    _save(fig, "case_study_03_buildup_failure_rate.png", save)
    return fig


# -------------------------------------------------------------- Section 4


def top_shap_features(save: bool = True):
    """Top 5 features by mean |SHAP| — horizontal bar."""
    set_style()
    ranking = pd.read_csv(MARTS / "shap_feature_ranking.csv")
    top = ranking.head(5).iloc[::-1]  # reverse so largest on top when we plot horizontal

    pretty = {
        "first_receiver_betweenness": "first receiver's passing-network centrality",
        "defthird_pressure_density": "defensive-third pressure density",
        "opp_press_height": "opponent press height (prior 30s)",
        "sd_zero": "game state: tied (0–0 diff)",
        "first_receiver_pressure_seconds": "time-to-pressure on first receiver",
        "recent_pass_reach": "recent pass reach (prior 3 passes)",
    }
    labels = [pretty.get(f, f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, top["mean_abs_shap"], color=VIOLET, height=0.55)
    for y, v in zip(labels, top["mean_abs_shap"], strict=False):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=10, color=INK, family="monospace")
    ax.set_xlabel("mean |SHAP| — contribution to predicted failure", color=INK)
    ax.set_title(
        "What the model used to flag a build-up as likely to fail",
        fontsize=13,
        pad=12,
    )
    ax.margins(x=0.12)
    _mono(ax)
    fig.tight_layout()
    _save(fig, "case_study_04_top_shap.png", save)
    return fig


# -------------------------------------------------------------- Section 5


def post_regain_metrics(save: bool = True):
    """Six Module B metrics — horizontal bar with CIs."""
    set_style()
    pr = pd.read_csv(MARTS / "team_post_regain.csv")
    order = [
        "xg_per_regain",
        "time_to_shot_median",
        "rushed_shot_rate",
        "regain_to_final_third_rate",
        "regain_to_loss_rate",
        "patience_composite",
    ]
    pretty = {
        "xg_per_regain": "xG per regain",
        "time_to_shot_median": "time-to-shot median (s)",
        "rushed_shot_rate": "rushed-shot rate",
        "regain_to_final_third_rate": "final-third entry rate",
        "regain_to_loss_rate": "lost within 4 actions",
        "patience_composite": "patience composite (z-score)",
    }
    df = pr.set_index("metric").loc[order]
    labels = [pretty[k] for k in order]
    vals = df["point_estimate"].to_numpy()
    lo_err = vals - df["ci_lo"].to_numpy()
    hi_err = df["ci_hi"].to_numpy() - vals

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        labels,
        vals,
        color=VIOLET,
        height=0.55,
        xerr=[lo_err, hi_err],
        ecolor=INK,
        capsize=3,
    )
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.invert_yaxis()
    ax.set_title(
        "Post-regain decision quality — 6 metrics, 95% bootstrap CI",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("value (each metric in its own units)", color=INK)
    _mono(ax)
    fig.tight_layout()
    _save(fig, "case_study_05_post_regain_metrics.png", save)
    return fig


# -------------------------------------------------------------- Section 6


def regain_outcome_sankey(save: bool = True):
    """Regain → outcome Sankey via plotly+kaleido."""
    import plotly.graph_objects as go

    regains = pd.read_parquet(MARTS / "regain_events.parquet")
    shot = regains["next_shot_seconds"].notna()
    lost = regains["lost_within_4_actions"] & ~shot
    ff_no_shot = regains["reached_final_third"] & ~shot & ~lost
    other = ~(shot | lost | ff_no_shot)
    counts = [int(shot.sum()), int(ff_no_shot.sum()), int(lost.sum()), int(other.sum())]
    labels = [
        "regains",
        "shot within 15s",
        "final-third entry (no shot)",
        "possession lost ≤4 actions",
        "other (foul, half end, long dwell)",
    ]
    colors_link = [VIOLET, CYAN, "#C4B5FD", "#E5E7EB"]
    colors_node = [INK, VIOLET, CYAN, "#C4B5FD", "#9CA3AF"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, color=colors_node, pad=20, thickness=20),
                link=dict(
                    source=[0, 0, 0, 0],
                    target=[1, 2, 3, 4],
                    value=counts,
                    color=colors_link,
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Inter Miami 2023 — what happened after a regain",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        width=1000,
        height=520,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
    )
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(FIG_DIR / "case_study_06_sankey.png"), format="png", scale=2)
    return fig


# -------------------------------------------------------------- Section 7


def case_summary(save: bool = True):
    """One-glance profile: 4 tiles with the headline numbers, plus a tagline."""
    set_style()
    team_bf = pd.read_csv(MARTS / "team_buildup_failure.csv")
    pr = pd.read_csv(MARTS / "team_post_regain.csv")
    regains = pd.read_parquet(MARTS / "regain_events.parquet")

    im = team_bf[team_bf["team_name"] == "Inter Miami"].iloc[0]
    fail_rate = float(im["failure_rate"])
    ff_rate = float(pr.loc[pr["metric"] == "regain_to_final_third_rate", "point_estimate"].iloc[0])
    xg_pr = float(pr.loc[pr["metric"] == "xg_per_regain", "point_estimate"].iloc[0])
    waste = float(regains["next_shot_seconds"].isna().mean())

    tiles = [
        ("Build-up failure", f"{fail_rate:.0%}", "under pressure in own half"),
        ("Reach final third", f"{ff_rate:.0%}", "of regains progress that far"),
        ("xG per regain", f"{xg_pr:.3f}", "what a regain is worth, on avg"),
        ("No-shot waste", f"{waste:.0%}", "regains with no shot in 15s"),
    ]

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1])
    for i, (label, number, sub) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        ax.add_patch(
            plt.Rectangle(
                (0.02, 0.05),
                0.96,
                0.9,
                linewidth=0,
                facecolor="#F5F3FF" if i % 2 == 0 else "#ECFEFF",
            )
        )
        ax.text(
            0.5,
            0.70,
            number,
            ha="center",
            va="center",
            fontsize=38,
            fontweight="bold",
            color=VIOLET if i % 2 == 0 else CYAN,
            family="monospace",
        )
        ax.text(0.5, 0.38, label, ha="center", va="center", fontsize=12, color=INK)
        ax.text(0.5, 0.22, sub, ha="center", va="center", fontsize=9, color=MUTED)

    tagline_ax = fig.add_subplot(gs[1, :])
    tagline_ax.set_axis_off()
    tagline_ax.text(
        0.5,
        0.6,
        "A team that reaches the final third often after winning the ball, "
        "yet rarely converts that territory into a shot — and fails one in three "
        "qualifying build-ups when pressed in its own half.",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        wrap=True,
    )
    tagline_ax.text(
        0.5,
        0.12,
        "Associational, not causal. 6-match corpus; small-n caveats apply.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
        style="italic",
    )
    fig.suptitle("Inter Miami 2023 — the profile, in four numbers", fontsize=14, y=0.98, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "case_study_07_summary.png", save)
    return fig


# -------------------------------------------------------------- CLI


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    builders = [
        ("hook_snapshot", hook_snapshot),
        ("pressure_exposure", pressure_exposure),
        ("buildup_failure_rate", buildup_failure_rate),
        ("top_shap_features", top_shap_features),
        ("post_regain_metrics", post_regain_metrics),
        ("regain_outcome_sankey", regain_outcome_sankey),
        ("case_summary", case_summary),
    ]
    for name, fn in builders:
        logger.info("Rendering %s ...", name)
        fn(save=True)
    logger.info("All 7 case-study figures written to %s.", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
