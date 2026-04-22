"""Module C — Leverkusen 23/24 × Inter Miami 2023 overlay.

Per spec §4 Module C (post-Phase-5 patch): compute raw build-up failure
rate (Module A primary) and the six Module B post-regain metrics on
Leverkusen 23/24, then overlay side-by-side against Inter Miami 2023's
already-persisted aggregates. The pre/post-vs-22/23 step was dropped —
Bundesliga 22/23 is absent from StatsBomb Open Data.

Outputs:
    data/marts/leverkusen_2324_chains.parquet
    data/marts/leverkusen_2324_regains.parquet
    data/marts/leverkusen_2324_buildup_labels.parquet
    data/marts/leverkusen_overlay.csv
    docs/figures/leverkusen_overlay_bar.png
    docs/figures/leverkusen_diff_forest.png
    docs/figures/leverkusen_vs_miami_regain_heatmaps.png

Associational language only. Any cross-team difference IS NOT a causal
claim about Alonso, Nancy, or any individual — differences reflect team,
league, roster, and schedule variance on observational data.

Run:
    python -m pressured_progression.ingest.leverkusen_ingest
    python -m pressured_progression.analysis.leverkusen_overlay
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from pressured_progression.features.post_regain import (  # noqa: E402
    aggregate_team_season,
    compute_regain_events,
)
from pressured_progression.labeling.buildup_failure import label_buildup_failures  # noqa: E402
from pressured_progression.sequences.possession_chain import (  # noqa: E402
    build_possession_chains,
)

logger = logging.getLogger(__name__)

CAVEAT = (
    "Associational, not causal. Cross-team, cross-league overlay on "
    "observational data. Differences between Leverkusen 23/24 and Inter Miami "
    "2023 reflect team, roster, schedule, and league variance — not the "
    "effect of any individual coach or player."
)

ROOT = Path(__file__).resolve().parents[3]
LEV_EVENTS_DIR = ROOT / "data" / "raw" / "statsbomb" / "leverkusen_2324"
MARTS_DIR = ROOT / "data" / "marts"
FIGURES_DIR = ROOT / "docs" / "figures"

# Palette per spec notebook conventions.
VIOLET = "#8A2BE2"  # Inter Miami
CYAN = "#00BFFF"  # Leverkusen
GRAY = "#777777"

N_BOOT = 1000
SEED = 20260421

METRIC_ORDER = [
    "raw_buildup_failure_rate",
    "xg_per_regain",
    "time_to_shot_median",
    "rushed_shot_rate",
    "regain_to_final_third_rate",
    "regain_to_loss_rate",
    "patience_composite",
]


# ------------------------- loading -------------------------


def _load_events(events_dir: Path) -> pd.DataFrame:
    files = sorted(events_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No event parquets in {events_dir}")
    con = duckdb.connect()
    list_sql = "[" + ",".join(f"'{p}'" for p in files) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()
    return df


def _determine_focal_team_id(events: pd.DataFrame, team_name: str) -> int:
    """Determine Leverkusen's team_id: the team that appears in every match."""
    per_match = events[["match_id", "team_id"]].drop_duplicates().groupby("team_id").size()
    n_matches = events["match_id"].nunique()
    candidates = per_match[per_match == n_matches]
    if candidates.empty:
        raise RuntimeError(
            f"No team appears in every Leverkusen match; check ingest ({team_name})."
        )
    return int(candidates.index[0])


# ------------------------- pipeline -------------------------


def build_leverkusen_marts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Run chain/regain/label on Leverkusen 23/24 events.

    Returns (chains, regains, labels, focal_id).
    """
    events = _load_events(LEV_EVENTS_DIR)
    focal_id = _determine_focal_team_id(events, "Bayer Leverkusen")
    logger.info(
        "Leverkusen 23/24: %d events across %d matches. focal_team_id=%d.",
        len(events),
        events["match_id"].nunique(),
        focal_id,
    )

    all_chains = build_possession_chains(events)
    lev_chains = all_chains[all_chains["team_id"] == focal_id].reset_index(drop=True)

    # Label buildup failures on Leverkusen-owned chains only.
    labels = label_buildup_failures(lev_chains, events)

    regains = compute_regain_events(events, focal_team_id=focal_id)

    lev_chains.to_parquet(MARTS_DIR / "leverkusen_2324_chains.parquet", index=False)
    regains.to_parquet(MARTS_DIR / "leverkusen_2324_regains.parquet", index=False)
    labels.to_parquet(MARTS_DIR / "leverkusen_2324_buildup_labels.parquet", index=False)
    logger.info(
        "Wrote Leverkusen marts: %d chains (all), %d labeled (qualified), %d regains.",
        len(lev_chains),
        len(labels),
        len(regains),
    )
    return lev_chains, regains, labels, focal_id


# ------------------------- metrics w/ CIs -------------------------


def _raw_failure_rate(labels_df: pd.DataFrame) -> float:
    if labels_df.empty:
        return float("nan")
    return float(labels_df["is_failure"].mean())


def _bootstrap_raw_rate(
    labels_df: pd.DataFrame, n_boot: int = N_BOOT, ci: float = 0.95, seed: int = SEED
) -> tuple[float, float, float]:
    """Match-level bootstrap on raw failure rate. Returns (point, lo, hi)."""
    if labels_df.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    match_ids = labels_df["match_id"].unique()
    by_match = {
        mid: labels_df[labels_df["match_id"] == mid]["is_failure"].astype(int).to_numpy()
        for mid in match_ids
    }
    pooled = labels_df["is_failure"].astype(int).to_numpy()
    point = float(pooled.mean()) if pooled.size else float("nan")
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(match_ids, size=len(match_ids), replace=True)
        pool = np.concatenate([by_match[m] for m in sample])
        vals[i] = pool.mean() if pool.size else np.nan
    lo, hi = np.nanpercentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, float(lo), float(hi)


def _leverkusen_metrics_with_ci(
    labels: pd.DataFrame, regains: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    """One row per metric: metric, point, ci_lo, ci_hi, n."""
    rows: list[dict] = []
    point, lo, hi = _bootstrap_raw_rate(labels)
    rows.append(
        {
            "metric": "raw_buildup_failure_rate",
            "point_estimate": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_regains": int(labels["match_id"].size),
        }
    )
    module_b = aggregate_team_season(regains, baseline, n_boot=N_BOOT, seed=SEED)
    rows.extend(module_b.to_dict("records"))
    return pd.DataFrame(rows)


def _inter_miami_metrics_with_ci() -> pd.DataFrame:
    """Pull Inter Miami 2023 metrics from already-persisted marts."""
    # Raw rate from team_buildup_failure.csv
    team_bf = pd.read_csv(MARTS_DIR / "team_buildup_failure.csv")
    im = team_bf[team_bf["team_name"] == "Inter Miami"].iloc[0]
    rows = [
        {
            "metric": "raw_buildup_failure_rate",
            "point_estimate": float(im["failure_rate"]),
            "ci_lo": float(im["ci_lo"]),
            "ci_hi": float(im["ci_hi"]),
            "n_regains": int(im["n_chains_qualified"]),
        }
    ]
    # Module B from team_post_regain.csv
    pr = pd.read_csv(MARTS_DIR / "team_post_regain.csv")
    rows.extend(pr.to_dict("records"))
    return pd.DataFrame(rows)


def _diff_bootstrap(
    lev_labels: pd.DataFrame,
    miami_labels: pd.DataFrame,
    lev_regains: pd.DataFrame,
    miami_regains: pd.DataFrame,
    baseline: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Independent bootstrap of (leverkusen_metric - miami_metric) per metric.

    Each side's match IDs resampled with replacement independently; differences
    computed per iteration; 95% percentile CI returned.
    """
    rng = np.random.default_rng(seed)

    # Pre-index by match for both sides.
    def _group_labels(df):
        return {m: df[df["match_id"] == m] for m in df["match_id"].unique()}

    def _group_regains(df):
        return {m: df[df["match_id"] == m] for m in df["match_id"].unique()}

    lev_L = _group_labels(lev_labels)
    mi_L = _group_labels(miami_labels)
    lev_R = _group_regains(lev_regains)
    mi_R = _group_regains(miami_regains)
    lev_ids = np.array(list(lev_L.keys()))
    mi_ids = np.array(list(mi_L.keys()))

    from pressured_progression.features.post_regain import _apply_all

    def _metrics_for(labels_sample, regains_sample):
        metrics = _apply_all(regains_sample, baseline)
        metrics["raw_buildup_failure_rate"] = (
            float(labels_sample["is_failure"].mean()) if not labels_sample.empty else float("nan")
        )
        return metrics

    boots: dict[str, list[float]] = {m: [] for m in METRIC_ORDER}
    for _ in range(n_boot):
        lev_sample = rng.choice(lev_ids, size=len(lev_ids), replace=True)
        mi_sample = rng.choice(mi_ids, size=len(mi_ids), replace=True)
        lev_lab = pd.concat([lev_L[m] for m in lev_sample], ignore_index=True)
        mi_lab = pd.concat([mi_L[m] for m in mi_sample], ignore_index=True)
        lev_reg = pd.concat([lev_R[m] for m in lev_sample], ignore_index=True)
        mi_reg = pd.concat([mi_R[m] for m in mi_sample], ignore_index=True)
        lev_m = _metrics_for(lev_lab, lev_reg)
        mi_m = _metrics_for(mi_lab, mi_reg)
        for k in METRIC_ORDER:
            lv = lev_m.get(k, float("nan"))
            mv = mi_m.get(k, float("nan"))
            if pd.isna(lv) or pd.isna(mv):
                boots[k].append(np.nan)
            else:
                boots[k].append(float(lv - mv))

    result: dict[str, tuple[float, float]] = {}
    for k, arr in boots.items():
        a = np.asarray(arr, dtype=float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        result[k] = (float(lo), float(hi))
    return result


# ------------------------- overlay assembly -------------------------


def _load_inter_miami_labels_regains() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Inter Miami buildup labels (per-match) and regain_events (all)."""
    im_labels_dir = ROOT / "data" / "core" / "buildup_labels" / "Inter_Miami"
    label_files = sorted(im_labels_dir.glob("*.parquet"))
    if not label_files:
        raise RuntimeError(f"Inter Miami labels missing — expected under {im_labels_dir}.")
    con = duckdb.connect()
    list_sql = "[" + ",".join(f"'{p}'" for p in label_files) + "]"
    labels = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()

    regain_path = MARTS_DIR / "regain_events.parquet"
    if not regain_path.exists():
        raise RuntimeError(f"{regain_path} missing — run post_regain.py first.")
    regains = pd.read_parquet(regain_path)
    return labels, regains


def _assemble_overlay(
    lev_metrics: pd.DataFrame,
    miami_metrics: pd.DataFrame,
    diff_cis: dict[str, tuple[float, float]],
    lev_labels: pd.DataFrame,
    miami_labels: pd.DataFrame,
    lev_regains: pd.DataFrame,
    miami_regains: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """One row per metric with both sides + diff + CI excludes zero flag."""
    lev = lev_metrics.set_index("metric")
    mi = miami_metrics.set_index("metric")

    from pressured_progression.features.post_regain import _apply_all

    def _point_diff(metric: str) -> float:
        if metric == "raw_buildup_failure_rate":
            lv = float(lev_labels["is_failure"].mean()) if not lev_labels.empty else float("nan")
            mv = (
                float(miami_labels["is_failure"].mean()) if not miami_labels.empty else float("nan")
            )
            return lv - mv if not (pd.isna(lv) or pd.isna(mv)) else float("nan")
        lev_m = _apply_all(lev_regains, baseline).get(metric, float("nan"))
        mi_m = _apply_all(miami_regains, baseline).get(metric, float("nan"))
        return lev_m - mi_m if not (pd.isna(lev_m) or pd.isna(mi_m)) else float("nan")

    rows: list[dict] = []
    for metric in METRIC_ORDER:
        lo_d, hi_d = diff_cis.get(metric, (float("nan"), float("nan")))
        point_diff = _point_diff(metric)
        excludes_zero = not (pd.isna(lo_d) or pd.isna(hi_d)) and (lo_d > 0 or hi_d < 0)
        rows.append(
            {
                "metric": metric,
                "leverkusen_mean": float(lev.loc[metric, "point_estimate"])
                if metric in lev.index
                else float("nan"),
                "leverkusen_ci_lo": float(lev.loc[metric, "ci_lo"])
                if metric in lev.index
                else float("nan"),
                "leverkusen_ci_hi": float(lev.loc[metric, "ci_hi"])
                if metric in lev.index
                else float("nan"),
                "inter_miami_mean": float(mi.loc[metric, "point_estimate"])
                if metric in mi.index
                else float("nan"),
                "inter_miami_ci_lo": float(mi.loc[metric, "ci_lo"])
                if metric in mi.index
                else float("nan"),
                "inter_miami_ci_hi": float(mi.loc[metric, "ci_hi"])
                if metric in mi.index
                else float("nan"),
                "diff_leverkusen_minus_miami": point_diff,
                "diff_ci_lo": lo_d,
                "diff_ci_hi": hi_d,
                "ci_excludes_zero": excludes_zero,
                "caveat": CAVEAT,
            }
        )
    return pd.DataFrame(rows)


# ------------------------- figures -------------------------


def fig_side_by_side(overlay: pd.DataFrame, out_path: Path) -> None:
    df = overlay.set_index("metric").loc[METRIC_ORDER]
    labels = df.index.tolist()
    y = np.arange(len(labels))

    lev_vals = df["leverkusen_mean"].to_numpy()
    lev_err_lo = lev_vals - df["leverkusen_ci_lo"].to_numpy()
    lev_err_hi = df["leverkusen_ci_hi"].to_numpy() - lev_vals
    mi_vals = df["inter_miami_mean"].to_numpy()
    mi_err_lo = mi_vals - df["inter_miami_ci_lo"].to_numpy()
    mi_err_hi = df["inter_miami_ci_hi"].to_numpy() - mi_vals

    fig, ax = plt.subplots(figsize=(10, 6))
    h = 0.38
    ax.barh(
        y - h / 2,
        lev_vals,
        height=h,
        color=CYAN,
        label="Leverkusen 23/24",
        xerr=[lev_err_lo, lev_err_hi],
        ecolor="black",
        capsize=3,
    )
    ax.barh(
        y + h / 2,
        mi_vals,
        height=h,
        color=VIOLET,
        label="Inter Miami 2023",
        xerr=[mi_err_lo, mi_err_hi],
        ecolor="black",
        capsize=3,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("value (metric-specific units)")
    ax.set_title(
        "Module A + B overlay — Leverkusen 2023/24 vs Inter Miami 2023\n"
        "95% bootstrap CIs, match-resampled. Associational — NOT causal."
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_diff_forest(overlay: pd.DataFrame, out_path: Path) -> None:
    df = overlay.set_index("metric").loc[METRIC_ORDER].copy()
    df = df.sort_values("diff_leverkusen_minus_miami", key=lambda s: s.abs(), ascending=True)
    labels = df.index.tolist()
    y = np.arange(len(labels))
    vals = df["diff_leverkusen_minus_miami"].to_numpy()
    lo = df["diff_ci_lo"].to_numpy()
    hi = df["diff_ci_hi"].to_numpy()
    excl = df["ci_excludes_zero"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CYAN if e else GRAY for e in excl]
    for i, (v, l_lo, l_hi, c) in enumerate(zip(vals, lo, hi, colors, strict=False)):
        ax.plot([l_lo, l_hi], [i, i], color=c, linewidth=2)
        ax.plot([v], [i], "o", color=c, markersize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Leverkusen 23/24 − Inter Miami 2023 (95% CI)")
    ax.set_title(
        "Forest plot — cross-team differences\n"
        "Cyan dots: 95% CI excludes 0; gray: CI overlaps 0. Associational — NOT causal."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_regain_heatmap_two_panel(
    miami_regains: pd.DataFrame, lev_regains: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (label, df, cmap) in zip(
        axes,
        [
            ("Inter Miami 2023", miami_regains, "Purples"),
            ("Leverkusen 2023/24", lev_regains, "Blues"),
        ],
        strict=False,
    ):
        xs = df["regain_x"].dropna().astype(float)
        ys = df["regain_y"].dropna().astype(float)
        if not xs.empty:
            hb = ax.hexbin(xs, ys, gridsize=18, cmap=cmap, extent=(0, 120, 0, 80))
            cb = fig.colorbar(hb, ax=ax, shrink=0.75)
            cb.set_label("regains")
        ax.axvline(40, color=GRAY, linewidth=0.5)
        ax.axvline(80, color=GRAY, linewidth=0.5)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.set_aspect("equal")
        ax.set_xlabel("x (attacking right →)")
        ax.set_title(f"{label} — n={len(df)} regains")
    axes[0].set_ylabel("y")
    fig.suptitle("Regain location hexbins. Associational — NOT causal.", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------- main -------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(MARTS_DIR / "asa_mls_2023_baseline.csv")

    # Leverkusen pipeline
    _, lev_regains, lev_labels, _ = build_leverkusen_marts()
    lev_metrics = _leverkusen_metrics_with_ci(lev_labels, lev_regains, baseline)

    # Inter Miami — already persisted
    mi_metrics = _inter_miami_metrics_with_ci()
    mi_labels, mi_regains = _load_inter_miami_labels_regains()

    # Differences via independent match-level bootstrap
    diff_cis = _diff_bootstrap(
        lev_labels=lev_labels,
        miami_labels=mi_labels,
        lev_regains=lev_regains,
        miami_regains=mi_regains,
        baseline=baseline,
        n_boot=N_BOOT,
        seed=SEED,
    )

    overlay = _assemble_overlay(
        lev_metrics,
        mi_metrics,
        diff_cis,
        lev_labels,
        mi_labels,
        lev_regains,
        mi_regains,
        baseline,
    )
    overlay.to_csv(MARTS_DIR / "leverkusen_overlay.csv", index=False)
    logger.info("Wrote %s (%d metrics).", MARTS_DIR / "leverkusen_overlay.csv", len(overlay))

    # Figures
    fig_side_by_side(overlay, FIGURES_DIR / "leverkusen_overlay_bar.png")
    fig_diff_forest(overlay, FIGURES_DIR / "leverkusen_diff_forest.png")
    fig_regain_heatmap_two_panel(
        mi_regains, lev_regains, FIGURES_DIR / "leverkusen_vs_miami_regain_heatmaps.png"
    )

    # Report
    print("\n=== Leverkusen 23/24 × Inter Miami 2023 overlay ===")
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(
            overlay.drop(columns=["caveat"]).to_string(
                index=False, float_format=lambda v: f"{v: .4f}"
            )
        )
    print(f"\nCaveat: {CAVEAT}")
    print(
        f"\nLeverkusen matches: {lev_labels['match_id'].nunique()} "
        f"(labeled chains: {len(lev_labels)})"
    )
    print(
        f"Inter Miami matches: {mi_labels['match_id'].nunique()} (labeled chains: {len(mi_labels)})"
    )
    print(f"Leverkusen regains: {len(lev_regains)}")
    print(f"Inter Miami regains: {len(mi_regains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
