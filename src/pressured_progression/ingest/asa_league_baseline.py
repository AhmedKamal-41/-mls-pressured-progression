"""Pull MLS 2023 team-season aggregates from ASA and derive league moments.

Used as the z-score reference for Module B's `patience_composite` metric.
`xg_per_shot` is the directly-derivable proxy for event-level
`xg_per_regain`. `rushed_shot_rate` has no clean proxy in ASA team-season
aggregates (no time-of-possession fields), so its baseline row is emitted
with NaN and a documented note. `patience_composite` degrades to
`z(xg_per_regain)` only when the rushed baseline is missing.

Output:
    data/marts/asa_mls_2023_baseline.csv

Run:
    python -m pressured_progression.ingest.asa_league_baseline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ASA_URL = "https://app.americansocceranalysis.com/api/v1/mls/teams/xgoals"
SEASON = 2023
ROOT = Path(__file__).resolve().parents[3]
OUT_CSV = ROOT / "data" / "marts" / "asa_mls_2023_baseline.csv"
TIMEOUT = 30


def fetch_team_xgoals(season: int = SEASON) -> pd.DataFrame:
    r = requests.get(ASA_URL, params={"season_name": season}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list from ASA, got {type(data)}")
    return pd.DataFrame(data)


def compute_baseline(xgoals_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # xG per shot — directly computable
    df = xgoals_df.copy()
    df = df[(df["shots_for"] > 0)]
    df["xg_per_shot"] = df["xgoals_for"] / df["shots_for"]
    rows.append(
        {
            "metric": "xg_per_shot",
            "league_mean": float(df["xg_per_shot"].mean()),
            "league_std": float(df["xg_per_shot"].std(ddof=0)),
            "n_teams": int(len(df)),
            "source_endpoint": "teams/xgoals",
            "note": (
                "xgoals_for / shots_for per team, MLS 2023. Used as league reference for "
                "Module B xg_per_regain z-score."
            ),
        }
    )

    # Rushed-shot rate: no direct proxy in ASA team-season aggregates.
    rows.append(
        {
            "metric": "rushed_shot_rate",
            "league_mean": float("nan"),
            "league_std": float("nan"),
            "n_teams": 0,
            "source_endpoint": "teams/xgoals",
            "note": (
                "ASA team-season aggregates do not expose time-since-possession-start, "
                "so 'shots in first 8s after turnover / total shots' is not derivable "
                "from this endpoint. patience_composite falls back to z(xg_per_regain) "
                "only when this row is NaN."
            ),
        }
    )

    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    xgoals_df = fetch_team_xgoals(SEASON)
    logger.info("Pulled %d team-season rows from ASA teams/xgoals for %d.", len(xgoals_df), SEASON)
    baseline = compute_baseline(xgoals_df)
    baseline.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s", OUT_CSV)

    print("\n=== MLS 2023 ASA baseline ===")
    with pd.option_context("display.width", 160, "display.max_colwidth", 100):
        print(baseline.to_string(index=False))

    # Confidence sanity: is xG/shot mean in the right neighborhood for elite-soccer?
    xgps = baseline.loc[baseline["metric"] == "xg_per_shot", "league_mean"].iloc[0]
    if not (0.05 < xgps < 0.2):
        logger.warning(
            "xG/shot league mean %.3f is outside the plausible 0.05-0.20 range; "
            "check endpoint fields.",
            xgps,
        )
    else:
        logger.info("xG/shot league mean %.3f is in the plausible range.", xgps)

    # Also surface useful intermediate stats not saved to disk yet but helpful for the run log.
    # (No mart written here beyond the baseline CSV.)
    _ = np.nanmean(xgoals_df["xgoals_for"])  # informational; suppress unused warning
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
