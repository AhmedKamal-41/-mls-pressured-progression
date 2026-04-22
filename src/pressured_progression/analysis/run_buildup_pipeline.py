"""End-to-end pipeline for Module A: features → training → SHAP → persistence.

Consumes per-match event parquet under data/raw/events/ (written by
smoke_buildup_failure.py) and per-match label parquet under
data/core/buildup_labels/<team>/. For each team with Open Data coverage,
extracts features, trains the classifier, emits the artifacts listed in
spec §4.5, and writes team-season failure rates with bootstrap CIs.

Flags honored: cannot fabricate data for teams with 0 matches; for those
we emit a row with NaN metrics and a `note` field rather than a silent skip.

Run:
    python -m pressured_progression.analysis.run_buildup_pipeline
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

warnings.filterwarnings("ignore")

from pressured_progression.features.buildup_features import (  # noqa: E402
    FEATURE_COLUMNS,
    extract_buildup_features,
)
from pressured_progression.models.buildup_failure_xgb import (  # noqa: E402
    bootstrap_match_rate_ci,
    save_calibration_curve,
    train_buildup_failure,
)
from pressured_progression.sequences.possession_chain import (  # noqa: E402
    build_possession_chains,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DIR = ROOT / "data" / "raw" / "events"
LABELS_DIR = ROOT / "data" / "core" / "buildup_labels"
FEATURES_PATH = ROOT / "data" / "marts" / "buildup_features.parquet"
FIGURES_DIR = ROOT / "docs" / "figures"
MARTS_DIR = ROOT / "data" / "marts"
MODELS_DIR = MARTS_DIR / "models"

SPEC_TEAMS = ["Philadelphia Union", "Columbus Crew", "Inter Miami"]


def _ensure_dirs() -> None:
    for d in (FIGURES_DIR, MARTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _read_events_for_matches(match_ids: list[int]) -> pd.DataFrame:
    con = duckdb.connect()
    paths = [str(EVENTS_DIR / f"{m}.parquet") for m in match_ids]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        con.close()
        return pd.DataFrame()
    list_sql = "[" + ",".join(f"'{p}'" for p in paths) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()
    return df


def _read_labels_for_team(team_name: str) -> pd.DataFrame:
    team_dir = LABELS_DIR / team_name.replace(" ", "_")
    if not team_dir.exists():
        return pd.DataFrame()
    parquet_files = sorted(team_dir.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    con = duckdb.connect()
    list_sql = "[" + ",".join(f"'{p}'" for p in parquet_files) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()
    return df


def _build_feature_frame(team_name: str) -> pd.DataFrame:
    labels = _read_labels_for_team(team_name)
    if labels.empty:
        return pd.DataFrame()

    match_ids = sorted(labels["match_id"].unique().tolist())
    events = _read_events_for_matches(match_ids)
    if events.empty:
        return pd.DataFrame()

    # Rebuild chains for this team from events (filtered to qualifying ones for the team).
    all_chains = build_possession_chains(events)
    # Team id: pick from labels via events. Labels are keyed by (match_id, possession_id);
    # join with all_chains to recover team_id consistently.
    merged = labels.merge(
        all_chains[["match_id", "possession_id", "team_id"]],
        on=["match_id", "possession_id"],
        how="inner",
    )
    team_id = int(merged["team_id"].mode().iloc[0])
    qualifying_chains = all_chains.merge(
        labels[["match_id", "possession_id"]], on=["match_id", "possession_id"], how="inner"
    )
    qualifying_chains = qualifying_chains[qualifying_chains["team_id"] == team_id]

    feats = extract_buildup_features(qualifying_chains, events)
    if feats.empty:
        return feats

    # Join labels for y.
    feats = feats.merge(
        labels[["match_id", "possession_id", "is_failure"]],
        on=["match_id", "possession_id"],
        how="inner",
    )
    feats["team_name"] = team_name
    return feats


def _save_shap(
    raw_xgb, X: pd.DataFrame, team_name: str
) -> tuple[list[tuple[str, float]], pd.DataFrame, pd.DataFrame]:
    """Compute SHAP on raw XGB. Emit summary + top-3 dependency plots.
    Return (top5_features_by_mean_abs_shap, team_shap_profile_df, feature_ranking_df)."""
    if X.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    explainer = shap.TreeExplainer(raw_xgb)
    shap_values = explainer.shap_values(X)
    # shap_values shape: (n_samples, n_features) for binary XGB
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]

    mean_abs = np.abs(shap_arr).mean(axis=0)
    mean_signed = shap_arr.mean(axis=0)
    ranking = (
        pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "mean_abs_shap": mean_abs.tolist(),
                "mean_signed_shap": mean_signed.tolist(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1

    top5_pairs = list(
        zip(
            ranking["feature"].tolist()[:5],
            ranking["mean_abs_shap"].tolist()[:5],
            strict=False,
        )
    )

    # Summary plot.
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_arr, X, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dependency plots for top 3.
    for i, feat_name in enumerate(ranking["feature"].tolist()[:3], start=1):
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feat_name, shap_arr, X, show=False, interaction_index=None)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_dep_{i}.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning("Dependency plot for %s failed: %s", feat_name, e)
            plt.close()

    # Team-level mean SHAP per feature (here only one team on the MLS side).
    profile = pd.DataFrame(shap_arr, columns=X.columns)
    profile["team_name"] = team_name
    per_team = profile.groupby("team_name").mean().reset_index()
    return top5_pairs, per_team, ranking


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _ensure_dirs()

    teams_with_data: list[str] = []
    feature_frames: list[pd.DataFrame] = []
    team_row_reports: list[dict] = []

    for team_name in SPEC_TEAMS:
        feats = _build_feature_frame(team_name)
        if feats.empty:
            team_row_reports.append(
                {
                    "team_name": team_name,
                    "n_matches": 0,
                    "n_chains_qualified": 0,
                    "n_failures": 0,
                    "failure_rate": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "note": "no StatsBomb Open Data coverage",
                }
            )
            continue
        teams_with_data.append(team_name)
        feature_frames.append(feats)

    if not feature_frames:
        logger.error("No team has usable data. Pipeline cannot train — halting.")
        pd.DataFrame(team_row_reports).to_csv(MARTS_DIR / "team_buildup_failure.csv", index=False)
        return 2

    all_feats = pd.concat(feature_frames, ignore_index=True)
    all_feats.to_parquet(FEATURES_PATH, index=False)
    logger.info("Wrote %s rows to %s (teams: %s)", len(all_feats), FEATURES_PATH, teams_with_data)

    # Per-team bootstrap CI using the original labels.
    for team_name in teams_with_data:
        team_feats = all_feats[all_feats["team_name"] == team_name]
        labels = team_feats[["match_id", "possession_id", "is_failure"]]
        point, lo, hi = bootstrap_match_rate_ci(labels)
        team_row_reports.append(
            {
                "team_name": team_name,
                "n_matches": int(team_feats["match_id"].nunique()),
                "n_chains_qualified": int(len(team_feats)),
                "n_failures": int(team_feats["is_failure"].sum()),
                "failure_rate": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "note": (
                    "SUBSTITUTE smoke target — NOT a spec case study"
                    if team_name == "Inter Miami"
                    else "spec-named case-study team"
                ),
            }
        )

    team_table = pd.DataFrame(team_row_reports)
    team_table.to_csv(MARTS_DIR / "team_buildup_failure.csv", index=False)
    logger.info("Wrote %s", MARTS_DIR / "team_buildup_failure.csv")

    # Training: pool all teams' chains (here only Inter Miami has rows).
    X = all_feats[FEATURE_COLUMNS].copy()
    # XGBoost requires numeric/bool/category dtype — coerce any object columns
    # (e.g. support_density_ff when freeze-frame data is absent and the column is all None).
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        elif X[col].dtype == bool:
            X[col] = X[col].astype(int)
    y = all_feats["is_failure"].astype(int).to_numpy()
    groups = all_feats["match_id"].to_numpy()

    if len(np.unique(groups)) < 3:
        logger.error(
            "Only %d match(es) available — cannot train a cross-validated model. Halting "
            "before producing noise numbers.",
            len(np.unique(groups)),
        )
        return 3

    result = train_buildup_failure(X, y, groups, n_iter=30)

    # Persist artifacts.
    joblib.dump(
        result.best_estimator_calibrated,
        MODELS_DIR / "buildup_failure_xgb_calibrated.joblib",
    )
    joblib.dump(
        result.best_estimator_raw,
        MODELS_DIR / "buildup_failure_xgb_raw.joblib",
    )
    logger.info("Persisted models.")

    # Feature importance.
    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": result.best_estimator_raw.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(MARTS_DIR / "buildup_failure_importance.csv", index=False)
    logger.info("Wrote feature importance.")

    # Calibration curve.
    save_calibration_curve(result.oof_predictions, FIGURES_DIR / "calibration_curve.png")

    # Per-fold CV metrics across baseline + XGB (n, n_pos per side).
    cv_records = pd.concat(
        [result.baseline_cv.records_df(), result.xgb_cv.records_df()], ignore_index=True
    )
    cv_records.to_csv(MARTS_DIR / "cv_metrics.csv", index=False)
    logger.info("Wrote cv_metrics.csv (%d fold rows).", len(cv_records))

    # OOF predictions with keys joined from the source feature frame (same row order).
    oof_keys = all_feats[["match_id", "possession_id"]].reset_index(drop=True)
    oof_out = pd.concat(
        [oof_keys, result.oof_predictions[["y_true", "y_proba", "fold"]].reset_index(drop=True)],
        axis=1,
    ).rename(columns={"y_proba": "y_pred_proba"})
    oof_out.to_parquet(MARTS_DIR / "oof_predictions.parquet", index=False)
    logger.info("Wrote oof_predictions.parquet (%d rows).", len(oof_out))

    # SHAP outputs.
    top5, team_shap, shap_ranking = _save_shap(result.best_estimator_raw, X, teams_with_data[0])
    team_shap.to_csv(MARTS_DIR / "team_shap_profile.csv", index=False)
    shap_ranking.to_csv(MARTS_DIR / "shap_feature_ranking.csv", index=False)

    # Report summary block.
    print("\n=== RUN SUMMARY ===")
    print(f"teams_with_data: {teams_with_data}")
    print(f"total chains: {len(all_feats)} across {int(all_feats['match_id'].nunique())} matches")
    baseline = result.baseline_cv.summary()
    xgbm = result.xgb_cv.summary()
    print(
        "baseline LogReg: "
        f"ROC-AUC {baseline['roc_auc'][0]:.3f}±{baseline['roc_auc'][1]:.3f}, "
        f"PR-AUC {baseline['pr_auc'][0]:.3f}±{baseline['pr_auc'][1]:.3f}, "
        f"Brier {baseline['brier'][0]:.3f}±{baseline['brier'][1]:.3f}"
    )
    print(
        "XGBoost: "
        f"ROC-AUC {xgbm['roc_auc'][0]:.3f}±{xgbm['roc_auc'][1]:.3f}, "
        f"PR-AUC {xgbm['pr_auc'][0]:.3f}±{xgbm['pr_auc'][1]:.3f}, "
        f"Brier {xgbm['brier'][0]:.3f}±{xgbm['brier'][1]:.3f}"
    )
    print(
        f"leakage holdout: match={result.holdout_match_id} "
        f"AUC={result.holdout_auc:.3f} "
        f"cv_mean={result.cv_mean_auc:.3f} "
        f"pass={result.leakage_pass}"
    )
    print(f"best XGB params: {result.best_params}")
    print("top 5 features by mean |SHAP|:")
    for name, val in top5:
        print(f"  {name:40s} {val:.4f}")
    print("\nteam_buildup_failure.csv:")
    with pd.option_context("display.width", 160, "display.max_colwidth", 80):
        print(team_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
