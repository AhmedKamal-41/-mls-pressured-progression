"""Generate `docs/executive_summary.pdf` — one page, A4 landscape, recruiter-facing.

Matplotlib-native PDF (no HTML conversion). Loads only from `data/marts/`.
Palette: violet #7C3AED + cyan #06B6D4 on white. Inter UI / JetBrains Mono
numerals with graceful fallback.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MARTS = ROOT / "data" / "marts"
OUT_PDF = ROOT / "docs" / "executive_summary.pdf"

VIOLET = "#7C3AED"
CYAN = "#06B6D4"
INK = "#111827"
MUTED = "#6B7280"
PANEL = "#F5F3FF"
ACCENT_BG = "#ECFEFF"
BG = "#FFFFFF"

GITHUB_PLACEHOLDER = "github.com/pressured-progression"
ARTICLE_PLACEHOLDER = "Article forthcoming"

THESIS_ONE_SENTENCE = (
    "Two MLS failure modes — build-up collapse under pressure and post-regain "
    "waste — measured on Inter Miami 2023 (Messi's half-season, 6 matches) and "
    "benchmarked against Bayer Leverkusen 2023/24 (Alonso's unbeaten season, "
    "34 matches) using StatsBomb Open Data and ASA."
)

# Display labels kept short because the charts are narrow (~1/3 page wide).
METRIC_LABEL = {
    "raw_buildup_failure_rate": "build-up fail rate",
    "xg_per_regain": "xG / regain",
    "time_to_shot_median": "time-to-shot (med s)",
    "rushed_shot_rate": "rushed-shot rate",
    "regain_to_final_third_rate": "final-3rd entry rate",
    "regain_to_loss_rate": "lost in ≤4 actions",
    "patience_composite": "patience composite (z)",
}


def set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.monospace"] = [
        "JetBrains Mono",
        "DejaVu Sans Mono",
        "Consolas",
        "monospace",
    ]
    plt.rcParams["axes.edgecolor"] = MUTED
    plt.rcParams["axes.labelcolor"] = INK
    plt.rcParams["xtick.color"] = INK
    plt.rcParams["ytick.color"] = INK
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42  # TrueType in PDF


def _mono_ticks(ax) -> None:
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily("monospace")


# ------------------------------ chart builders ------------------------------


def draw_im_module_b(ax) -> None:
    """Chart 1: Inter Miami's six Module B metrics with 95% CIs."""
    pr = pd.read_csv(MARTS / "team_post_regain.csv").set_index("metric")
    order = [
        "xg_per_regain",
        "time_to_shot_median",
        "rushed_shot_rate",
        "regain_to_final_third_rate",
        "regain_to_loss_rate",
        "patience_composite",
    ]
    df = pr.loc[order]
    labels = [METRIC_LABEL[k] for k in order]
    vals = df["point_estimate"].to_numpy()
    lo_err = vals - df["ci_lo"].to_numpy()
    hi_err = df["ci_hi"].to_numpy() - vals

    ax.barh(
        labels,
        vals,
        color=VIOLET,
        height=0.55,
        xerr=[lo_err, hi_err],
        ecolor=INK,
        capsize=2.5,
    )
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.5)
    ax.set_title(
        "Inter Miami 2023 — post-regain profile",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
    _mono_ticks(ax)


def draw_forest(ax) -> None:
    """Chart 2: Leverkusen 23/24 − Inter Miami 2023 delta forest plot.

    Note: the original Module C called for a Leverkusen pre/post forest
    (22/23 → 23/24); 22/23 is not in Open Data, so this is the overlay
    delta we actually have — honest framing in the title.
    """
    overlay = pd.read_csv(MARTS / "leverkusen_overlay.csv")
    order = [
        "raw_buildup_failure_rate",
        "regain_to_final_third_rate",
        "xg_per_regain",
        "patience_composite",
        "rushed_shot_rate",
        "time_to_shot_median",
        "regain_to_loss_rate",
    ]
    df = overlay.set_index("metric").loc[order]
    labels = [METRIC_LABEL[k] for k in order]
    vals = df["diff_leverkusen_minus_miami"].to_numpy()
    lo = df["diff_ci_lo"].to_numpy()
    hi = df["diff_ci_hi"].to_numpy()
    excludes_zero = df["ci_excludes_zero"].to_numpy()
    colors = [CYAN if e else "#D1D5DB" for e in excludes_zero]

    y = range(len(labels))
    for i, (v, lo_i, hi_i, c) in enumerate(zip(vals, lo, hi, colors, strict=False)):
        ax.plot([lo_i, hi_i], [i, i], color=c, linewidth=1.8)
        ax.plot([v], [i], "o", color=c, markersize=6)
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(
        "Leverkusen − Inter Miami (overlay Δ, 95% CI)",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=7)
    _mono_ticks(ax)


def draw_profile_overlay(ax) -> None:
    """Chart 3: Inter Miami overlaid on Leverkusen's Module-B profile.

    Normalized dumbbell: each metric scaled to [0, 1] using the combined
    CI envelope. Actual values shown as small monospace annotations.
    """
    overlay = pd.read_csv(MARTS / "leverkusen_overlay.csv")
    metrics = [
        "xg_per_regain",
        "time_to_shot_median",
        "rushed_shot_rate",
        "regain_to_final_third_rate",
        "regain_to_loss_rate",
        "patience_composite",
    ]
    df = overlay.set_index("metric").loc[metrics]
    lev_v = df["leverkusen_mean"].to_numpy()
    im_v = df["inter_miami_mean"].to_numpy()
    lo = df[["leverkusen_ci_lo", "inter_miami_ci_lo"]].min(axis=1).to_numpy()
    hi = df[["leverkusen_ci_hi", "inter_miami_ci_hi"]].max(axis=1).to_numpy()
    span = hi - lo
    span = [s if s > 0 else 1.0 for s in span]
    lev_n = [(v - lo_i) / s for v, lo_i, s in zip(lev_v, lo, span, strict=False)]
    im_n = [(v - lo_i) / s for v, lo_i, s in zip(im_v, lo, span, strict=False)]
    labels = [METRIC_LABEL[k] for k in metrics]

    y = list(range(len(labels)))
    for i, (a, b) in enumerate(zip(im_n, lev_n, strict=False)):
        ax.plot([a, b], [i, i], color="#D1D5DB", linewidth=1.2, zorder=1)
    ax.scatter(im_n, y, color=VIOLET, s=45, zorder=3, label="Inter Miami 2023")
    ax.scatter(lev_n, y, color=CYAN, s=45, zorder=3, label="Leverkusen 23/24")

    # Actual values as small in-chart annotations sitting at the top of each row.
    for i, (im, lv) in enumerate(zip(im_v, lev_v, strict=False)):
        ax.text(
            0.5,
            i - 0.35,
            f"IM {im:.3f}   ·   Lev {lv:.3f}",
            va="center",
            ha="center",
            fontsize=6.5,
            color=MUTED,
            family="monospace",
        )

    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_title(
        "Profile overlay — CI-normalized",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)


# ------------------------------ text blocks ------------------------------


def draw_findings(ax) -> None:
    ax.set_axis_off()
    # Each bullet = (lead_bold, body). Lead rendered bold on line 1;
    # body rendered regular on line 2.
    bullets = [
        (
            "Build-up failure under pressure",
            "Inter Miami 34% [CI 25–45%]; Leverkusen 21% [CI 16–25%]; Δ = −14 pp (CI excl. 0).",
        ),
        (
            "Final-third entry after regain",
            "Leverkusen 81% [77–84%] vs Inter Miami 67% [57–77%]; Δ = +14 pp (CI excl. 0).",
        ),
        (
            "Post-regain waste",
            "90% of Inter Miami regains do not produce a shot within 15 s; a sizable "
            "regain-to-threat gap vs Leverkusen's attacking output.",
        ),
        (
            "Model utility modest at small n",
            "CV ROC-AUC 0.54 ± 0.08 on 73 chains; LogReg baseline marginally better. Honest "
            "small-sample band.",
        ),
        (
            "4 of 7 cross-team deltas separable from noise",
            "at 95% CI; three (time-to-shot, rushed rate, ≤4-action loss) are not.",
        ),
    ]
    ax.text(
        0.0,
        1.0,
        "Findings",
        fontsize=10,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
        va="top",
    )
    y = 0.88
    line_gap = 0.18
    for lead, body in bullets:
        ax.text(
            0.0,
            y,
            f"•  {lead}",
            fontsize=8.2,
            fontweight="bold",
            color=INK,
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            0.025,
            y - 0.055,
            body,
            fontsize=7.8,
            color="#374151",
            transform=ax.transAxes,
            va="top",
            wrap=True,
        )
        y -= line_gap


def draw_methods(ax) -> None:
    ax.set_axis_off()
    # Background panel
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            facecolor=PANEL,
            linewidth=0,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.08,
        0.94,
        "Methods",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
        va="top",
    )
    lines = [
        ("Sources", "StatsBomb Open Data (events + 360); American Soccer Analysis v1."),
        (
            "Base years",
            "MLS 2023 (6 Inter Miami matches); Bundesliga 2023/24 (34 Leverkusen matches).",
        ),
        (
            "Scope limit",
            "Only Inter Miami appears in MLS 2023 Open Data; no league-wide event ranking.",
        ),
        ("Uncertainty", "95% bootstrap CIs, matches resampled (not possessions)."),
        (
            "Framing",
            "Associational, not causal. No attribution to Alonso, Messi, Nancy, or any individual.",
        ),
    ]
    y = 0.85
    for head, body in lines:
        ax.text(
            0.08,
            y,
            head,
            fontsize=7.5,
            fontweight="bold",
            color=VIOLET,
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            0.08,
            y - 0.04,
            body,
            fontsize=7.0,
            color="#374151",
            transform=ax.transAxes,
            va="top",
            wrap=True,
        )
        y -= 0.17


def draw_footer(ax) -> None:
    ax.set_axis_off()
    today = date.today().isoformat()
    ax.text(
        0.0,
        0.5,
        f"{GITHUB_PLACEHOLDER}  ·  {ARTICLE_PLACEHOLDER}  ·  {today}",
        fontsize=7.5,
        color=MUTED,
        transform=ax.transAxes,
        va="center",
        family="monospace",
    )
    ax.text(
        1.0,
        0.5,
        "Pressured Progression",
        fontsize=7.5,
        color=MUTED,
        transform=ax.transAxes,
        va="center",
        ha="right",
    )


# ------------------------------ assemble ------------------------------


def build_figure() -> plt.Figure:
    set_style()
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=12,
        height_ratios=[0.95, 3.3, 3.2, 0.45],
        hspace=0.65,
        wspace=2.6,
        left=0.055,
        right=0.975,
        top=0.95,
        bottom=0.045,
    )

    # --- Title + subtitle ---
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.78,
        "Pressured Progression",
        fontsize=22,
        fontweight="bold",
        color=INK,
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0,
        0.10,
        THESIS_ONE_SENTENCE,
        fontsize=9.5,
        color="#374151",
        transform=title_ax.transAxes,
        wrap=True,
    )

    # --- Charts row ---
    ax1 = fig.add_subplot(gs[1, 0:4])
    ax2 = fig.add_subplot(gs[1, 4:8])
    ax3 = fig.add_subplot(gs[1, 8:12])
    draw_im_module_b(ax1)
    draw_forest(ax2)
    draw_profile_overlay(ax3)

    # --- Findings + Methods ---
    findings_ax = fig.add_subplot(gs[2, 0:8])
    methods_ax = fig.add_subplot(gs[2, 8:12])
    draw_findings(findings_ax)
    draw_methods(methods_ax)

    # --- Footer ---
    footer_ax = fig.add_subplot(gs[3, :])
    draw_footer(footer_ax)
    return fig


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, format="pdf", facecolor=BG)
    plt.close(fig)
    logger.info("Wrote %s (%d bytes).", OUT_PDF, OUT_PDF.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
