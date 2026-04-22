"""Module B: post-regain waste metrics (spec §4).

Per-regain enrichment builds on `sequences.regain.detect_regains` by attaching
subsequent-chain outcome fields (`reached_final_third`, `lost_within_4_actions`,
`chain_end_type`). Team-season aggregates then wrap those in 95% bootstrap
CIs resampling matches (not possessions), per spec.

The `patience_composite` z-scores against the MLS 2023 ASA league baseline
from `data/marts/asa_mls_2023_baseline.csv`. Because ASA team-season
aggregates lack a direct rushed-shot proxy, the composite degrades to
`z(xg_per_regain)` only when the rushed baseline row is NaN.

Associational language only in downstream outputs (spec §9).
"""

from __future__ import annotations

import logging
import uuid
import warnings
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pressured_progression.core.schemas import EventRow, validate_columns
from pressured_progression.sequences.possession_chain import _abs_seconds
from pressured_progression.sequences.regain import detect_regains

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

FINAL_THIRD_X = 80.0
LOST_WITHIN_ACTIONS = 4
RUSHED_XG_MAX = 0.05
RUSHED_SECONDS_MAX = 8.0

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DIR = ROOT / "data" / "raw" / "events"
MARTS_DIR = ROOT / "data" / "marts"
FIGURES_DIR = ROOT / "docs" / "figures"
LABELS_DIR = ROOT / "data" / "core" / "buildup_labels"

# violet / cyan palette per spec §notebook; Inter Miami (the focal MLS team) plotted in violet.
VIOLET = "#8A2BE2"
CYAN = "#00BFFF"
GRAY = "#777777"


# ----------------------- per-regain enrichment -----------------------


def _is_turnover(row: pd.Series) -> bool:
    et = str(row.get("event_type", ""))
    outc = str(row.get("outcome") or "").lower()
    if et in {"Dispossessed", "Miscontrol"}:
        return True
    if et == "Ball Receipt*" and outc and outc != "complete":
        return True
    return et == "Pass" and bool(outc) and outc not in {"complete", ""}


def _chain_end_type(last: pd.Series) -> str:
    et = str(last.get("event_type", ""))
    if et == "Shot":
        return "shot"
    if _is_turnover(last) or et == "Clearance":
        return "loss"
    if et == "Foul Won":
        return "foul_won"
    if et == "Half End":
        return "halt"
    return "other"


def _ensure_t(events: pd.DataFrame) -> pd.DataFrame:
    if "_t" in events.columns:
        return events
    df = events.copy()
    df["_t"] = df.apply(
        lambda r: _abs_seconds(int(r["period"]), int(r["minute"]), int(r["second"])), axis=1
    )
    return df


def compute_regain_events(events: pd.DataFrame, focal_team_id: int) -> pd.DataFrame:
    """Detect regains, then enrich each with subsequent-chain outcome fields.

    Output columns:
        regain_id, match_id, regain_time_seconds, regain_x, regain_y, regain_zone,
        next_shot_seconds, next_shot_xg,
        reached_final_third, lost_within_4_actions, chain_end_type,
        subsequent_possession_id
    """
    validate_columns(events, EventRow)
    regains = detect_regains(events, focal_team_id=focal_team_id)
    if regains.empty:
        return pd.DataFrame()

    df = _ensure_t(events)

    enriched: list[dict] = []
    for _, r in regains.iterrows():
        mid = int(r["match_id"])
        sub_pid = int(r["subsequent_possession_id"])
        chain = df[
            (df["match_id"] == mid)
            & (df["possession_id"] == sub_pid)
            & (df["team_id"] == focal_team_id)
        ].sort_values("_t")
        if chain.empty:
            reached = False
            lost_within_4 = False
            end_type = "other"
        else:
            reached = bool((chain["location_x"].dropna().astype(float) >= FINAL_THIRD_X).any())
            last = chain.iloc[-1]
            lost_within_4 = bool(len(chain) <= LOST_WITHIN_ACTIONS and _is_turnover(last))
            end_type = _chain_end_type(last)

        enriched.append(
            {
                "regain_id": str(uuid.uuid4()),
                "match_id": mid,
                "regain_time_seconds": float(r["time_seconds"]),
                "regain_x": r.get("regain_x"),
                "regain_y": r.get("regain_y"),
                "regain_zone": r["regain_zone"],
                "subsequent_possession_id": sub_pid,
                "next_shot_seconds": r.get("next_shot_seconds"),
                "next_shot_xg": r.get("next_shot_xg"),
                "reached_final_third": reached,
                "lost_within_4_actions": lost_within_4,
                "chain_end_type": end_type,
            }
        )

    return pd.DataFrame(enriched)


# ----------------------- metrics -----------------------


def _xg_per_regain(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df["next_shot_xg"].fillna(0.0).sum() / len(df))


def _time_to_shot_median(df: pd.DataFrame) -> float:
    v = df["next_shot_seconds"].dropna()
    if v.empty:
        return float("nan")
    return float(v.median())


def _rushed_shot_rate(df: pd.DataFrame) -> float:
    shots = df[df["next_shot_seconds"].notna()]
    if shots.empty:
        return float("nan")
    rushed = (shots["next_shot_xg"].fillna(1.0) < RUSHED_XG_MAX) & (
        shots["next_shot_seconds"] < RUSHED_SECONDS_MAX
    )
    return float(rushed.sum() / len(shots))


def _final_third_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df["reached_final_third"].mean())


def _loss_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df["lost_within_4_actions"].mean())


def _patience_composite(df: pd.DataFrame, baseline: pd.DataFrame) -> float:
    xp = _xg_per_regain(df)
    rp = _rushed_shot_rate(df)
    xp_row = baseline[baseline["metric"] == "xg_per_shot"]
    if xp_row.empty or pd.isna(xp_row["league_mean"].iloc[0]) or xp_row["league_std"].iloc[0] == 0:
        return float("nan")
    xp_mean = float(xp_row["league_mean"].iloc[0])
    xp_std = float(xp_row["league_std"].iloc[0])
    z_x = (xp - xp_mean) / xp_std

    rp_row = baseline[baseline["metric"] == "rushed_shot_rate"]
    if rp_row.empty or pd.isna(rp_row["league_mean"].iloc[0]):
        # Degrade to z(xg_per_regain) only — documented in spec §4 Module B.
        return z_x
    rp_mean = float(rp_row["league_mean"].iloc[0])
    rp_std = float(rp_row["league_std"].iloc[0])
    if rp_std == 0 or pd.isna(rp_std):
        return z_x
    z_r = (rp - rp_mean) / rp_std
    return z_x - z_r


def _apply_all(df: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    return {
        "xg_per_regain": _xg_per_regain(df),
        "time_to_shot_median": _time_to_shot_median(df),
        "rushed_shot_rate": _rushed_shot_rate(df),
        "regain_to_final_third_rate": _final_third_rate(df),
        "regain_to_loss_rate": _loss_rate(df),
        "patience_composite": _patience_composite(df, baseline),
    }


# ----------------------- bootstrap -----------------------


def _bootstrap_metric(
    df: pd.DataFrame,
    metric_fn,
    baseline: pd.DataFrame | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 20260421,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    if match_ids.size == 0:
        return float("nan"), float("nan")
    by_match = {mid: df[df["match_id"] == mid] for mid in match_ids}

    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(match_ids, size=len(match_ids), replace=True)
        pool = pd.concat([by_match[m] for m in sample], ignore_index=True)
        if baseline is None:
            vals[i] = metric_fn(pool)
        else:
            vals[i] = metric_fn(pool, baseline)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    lo, hi = np.nanpercentile(vals, [lo_pct, hi_pct])
    return float(lo), float(hi)


def aggregate_team_season(
    regain_events: pd.DataFrame,
    baseline: pd.DataFrame,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 20260421,
) -> pd.DataFrame:
    """Return one row per metric with point estimate + 95% bootstrap CI."""
    n_regains = int(len(regain_events))
    points = _apply_all(regain_events, baseline)

    rows: list[dict] = []
    metric_to_fn = {
        "xg_per_regain": (_xg_per_regain, None),
        "time_to_shot_median": (_time_to_shot_median, None),
        "rushed_shot_rate": (_rushed_shot_rate, None),
        "regain_to_final_third_rate": (_final_third_rate, None),
        "regain_to_loss_rate": (_loss_rate, None),
        "patience_composite": (_patience_composite, baseline),
    }

    for m, (fn, aux) in metric_to_fn.items():
        lo, hi = _bootstrap_metric(regain_events, fn, baseline=aux, n_boot=n_boot, ci=ci, seed=seed)
        rows.append(
            {
                "metric": m,
                "point_estimate": float(points[m]),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_regains": n_regains,
            }
        )
    return pd.DataFrame(rows)


# ----------------------- sanity checks -----------------------


def log_sanity(regain_events: pd.DataFrame) -> dict:
    n = len(regain_events)
    no_shot = regain_events["next_shot_seconds"].isna().sum()
    zone_dist = regain_events["regain_zone"].value_counts(normalize=True).to_dict() if n else {}
    small_sample = n < 300

    logger.info("Total regains: %d", n)
    if n:
        logger.info(
            "Regains without a shot within 15s: %d / %d = %.1f%% (the waste signal).",
            no_shot,
            n,
            100 * no_shot / n,
        )
        logger.info("Regain zone distribution: %s", zone_dist)
    if small_sample:
        logger.warning("Small-sample warning: total regains %d < 300.", n)

    return {
        "n_regains": n,
        "no_subsequent_shot_pct": (float(no_shot) / n) if n else float("nan"),
        "zone_distribution": zone_dist,
        "small_sample_warning": small_sample,
    }


# ----------------------- figures -----------------------


def fig_metrics_bar(team_table: pd.DataFrame, baseline: pd.DataFrame, out_path: Path) -> None:
    df = team_table.copy().set_index("metric")
    order = [
        "xg_per_regain",
        "time_to_shot_median",
        "rushed_shot_rate",
        "regain_to_final_third_rate",
        "regain_to_loss_rate",
        "patience_composite",
    ]
    df = df.loc[order]
    labels = df.index.tolist()
    vals = df["point_estimate"].to_numpy()
    lo_err = vals - df["ci_lo"].to_numpy()
    hi_err = df["ci_hi"].to_numpy() - vals

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, vals, color=VIOLET, xerr=[lo_err, hi_err], ecolor="black", capsize=4)
    # League-mean reference where comparable.
    xp_ref = baseline.loc[baseline["metric"] == "xg_per_shot", "league_mean"]
    if not xp_ref.empty:
        ax.axvline(
            float(xp_ref.iloc[0]),
            color=GRAY,
            linestyle="--",
            linewidth=1,
            label="ASA MLS 2023 xG/shot league mean",
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("value (metric-specific units)")
    ax.set_title("Inter Miami 2023 — post-regain metrics (95% bootstrap CI, match-resampled)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_time_to_shot_hist(regain_events: pd.DataFrame, out_path: Path) -> None:
    v = regain_events["next_shot_seconds"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not v.empty:
        ax.hist(v, bins=15, color=VIOLET, edgecolor="white")
        med = v.median()
        ax.axvline(med, color=CYAN, linestyle="--", linewidth=2, label=f"median = {med:.1f}s")
        ax.legend()
    ax.set_xlim(0, 15)
    ax.set_xlabel("seconds from regain to next Inter Miami shot (capped at 15)")
    ax.set_ylabel("regains")
    ax.set_title("Inter Miami 2023 — time-to-shot distribution for regains with a shot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_regain_heatmap(regain_events: pd.DataFrame, out_path: Path) -> None:
    xs = regain_events["regain_x"].dropna().astype(float)
    ys = regain_events["regain_y"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    if not xs.empty:
        hb = ax.hexbin(xs, ys, gridsize=18, cmap="Purples", extent=(0, 120, 0, 80))
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("regains")
    # Pitch reference lines
    ax.axvline(40, color=GRAY, linewidth=0.5)
    ax.axvline(80, color=GRAY, linewidth=0.5)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Inter Miami attacking right →)")
    ax.set_ylabel("y")
    ax.set_title("Inter Miami 2023 — regain location hexbin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_post_regain_sankey(regain_events: pd.DataFrame, out_path: Path) -> None:
    """Plotly Sankey: regain → {shot within 15s, final-third entry without shot,
    possession loss within 4 actions, other}."""
    import plotly.graph_objects as go

    shot = regain_events["next_shot_seconds"].notna()
    lost = regain_events["lost_within_4_actions"] & ~shot
    ff_no_shot = regain_events["reached_final_third"] & ~shot & ~lost
    other = ~(shot | lost | ff_no_shot)

    counts = [int(shot.sum()), int(ff_no_shot.sum()), int(lost.sum()), int(other.sum())]
    labels = [
        "regains",
        "shot within 15s",
        "final third entry (no shot)",
        "possession lost ≤4 actions",
        "other",
    ]
    sources = [0, 0, 0, 0]
    targets = [1, 2, 3, 4]
    colors_link = [VIOLET, CYAN, GRAY, "#C0C0C0"]
    colors_node = ["#444444", VIOLET, CYAN, GRAY, "#C0C0C0"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, color=colors_node, pad=18, thickness=18),
                link=dict(source=sources, target=targets, value=counts, color=colors_link),
            )
        ]
    )
    fig.update_layout(
        title_text="Inter Miami 2023 — post-regain outcome flow",
        font_size=12,
        width=1000,
        height=520,
    )
    fig.write_image(str(out_path), format="png", scale=2)


# ----------------------- CLI -----------------------


def _load_inter_miami_events() -> tuple[pd.DataFrame, int]:
    """Load Inter Miami match events via DuckDB and determine the focal team_id."""
    team_dir = LABELS_DIR / "Inter_Miami"
    label_files = sorted(team_dir.glob("*.parquet"))
    if not label_files:
        raise RuntimeError(
            f"No Inter Miami label parquet files under {team_dir}. "
            "Run smoke_buildup_failure.py first to materialize per-match events."
        )
    match_ids = [int(p.stem) for p in label_files]

    event_files = [EVENTS_DIR / f"{m}.parquet" for m in match_ids]
    event_files = [p for p in event_files if p.exists()]
    if not event_files:
        raise RuntimeError(f"No event parquet files under {EVENTS_DIR}.")

    con = duckdb.connect()
    list_sql = "[" + ",".join(f"'{p}'" for p in event_files) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()

    # Determine Inter Miami team_id from any home/away appearance.
    inter_miami_ids = df.loc[df["team_id"].notna(), "team_id"].unique().tolist()
    # Inter Miami appears in every match; pick the id that occurs in all matches.
    counts_per_team = (
        df[["match_id", "team_id"]]
        .drop_duplicates()
        .groupby("team_id")
        .size()
        .sort_values(ascending=False)
    )
    focal = int(counts_per_team.index[0])
    logger.info(
        "Loaded %d events across %d matches. Focal team_id=%d (Inter Miami).",
        len(df),
        len(event_files),
        focal,
    )
    logger.debug("Team ids seen: %s", inter_miami_ids)
    return df, focal


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = MARTS_DIR / "asa_mls_2023_baseline.csv"
    if not baseline_path.exists():
        raise RuntimeError(
            f"Missing {baseline_path}. Run "
            "`python -m pressured_progression.ingest.asa_league_baseline` first."
        )
    baseline = pd.read_csv(baseline_path)

    events, focal_team_id = _load_inter_miami_events()

    regain_events = compute_regain_events(events, focal_team_id)
    regain_events.to_parquet(MARTS_DIR / "regain_events.parquet", index=False)
    logger.info("Wrote %s (%d regains).", MARTS_DIR / "regain_events.parquet", len(regain_events))

    sanity = log_sanity(regain_events)

    team_table = aggregate_team_season(regain_events, baseline, n_boot=1000, seed=20260421)
    team_table.to_csv(MARTS_DIR / "team_post_regain.csv", index=False)
    logger.info("Wrote %s", MARTS_DIR / "team_post_regain.csv")

    # Figures
    fig_metrics_bar(team_table, baseline, FIGURES_DIR / "post_regain_metrics_bar.png")
    fig_time_to_shot_hist(regain_events, FIGURES_DIR / "time_to_shot_hist.png")
    fig_regain_heatmap(regain_events, FIGURES_DIR / "regain_zones_heatmap.png")
    try:
        fig_post_regain_sankey(regain_events, FIGURES_DIR / "post_regain_sankey.png")
    except Exception as e:
        logger.warning("Sankey render failed (%s); continuing without it.", e)

    # Report
    print("\n=== Post-regain summary (Inter Miami 2023) ===")
    print(team_table.to_string(index=False))
    print(
        f"\nSanity: regains={sanity['n_regains']}, "
        f"no-subsequent-shot={sanity['no_subsequent_shot_pct']:.1%}, "
        f"zones={sanity['zone_distribution']}, "
        f"small_sample_flag={sanity['small_sample_warning']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
