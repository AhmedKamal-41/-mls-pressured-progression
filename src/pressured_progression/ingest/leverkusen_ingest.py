"""Pull Leverkusen 2023/24 matches + events from StatsBomb Open Data.

Bundesliga 2022/23 is absent from Open Data (see `docs/data_reality.md`
Phase 5 appendix), so only 2023/24 is ingested. All 34 matches are
Leverkusen per the Open Data distribution; we filter defensively anyway.

Outputs:
    data/raw/statsbomb/leverkusen_2324/<match_id>.parquet  (adapted events)
    data/raw/statsbomb/leverkusen_2324/_manifest.csv       (match metadata)

Run:
    python -m pressured_progression.ingest.leverkusen_ingest
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from pressured_progression.ingest.events_adapter import adapt_statsbomb_events  # noqa: E402

logger = logging.getLogger(__name__)

COMPETITION_ID = 9  # 1. Bundesliga
SEASON_ID_2324 = 281  # 2023/2024
TEAM_NAME = "Bayer Leverkusen"

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "raw" / "statsbomb" / "leverkusen_2324"


def _import_sb():
    from statsbombpy import sb

    return sb


def fetch_matches(sb, competition_id: int, season_id: int) -> pd.DataFrame:
    m = sb.matches(competition_id=competition_id, season_id=season_id)
    if not isinstance(m, pd.DataFrame):
        m = pd.DataFrame(m)
    return m


def filter_team(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    mask = (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    return matches[mask].copy()


def fetch_and_write_events(sb, match_id: int, out_dir: Path) -> Path:
    out = out_dir / f"{match_id}.parquet"
    if out.exists():
        return out
    raw = sb.events(match_id=match_id)
    adapted = adapt_statsbomb_events(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapted.to_parquet(out, index=False)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sb = _import_sb()
    matches = fetch_matches(sb, COMPETITION_ID, SEASON_ID_2324)
    lev = filter_team(matches, TEAM_NAME)
    n_matches = len(lev)
    n_360 = (
        int((lev["match_status_360"].astype(str) == "available").sum())
        if "match_status_360" in lev.columns
        else 0
    )
    logger.info("Bundesliga 23/24: %d Leverkusen matches (%d with 360).", n_matches, n_360)
    if n_matches == 0:
        logger.error("No Leverkusen 23/24 matches in Open Data — halting ingest.")
        return 2

    manifest_rows: list[dict] = []
    for _, m in lev.iterrows():
        match_id = int(m["match_id"])
        fetch_and_write_events(sb, match_id, OUT_DIR)
        manifest_rows.append(
            {
                "match_id": match_id,
                "match_date": m.get("match_date"),
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
                "match_status_360": m.get("match_status_360"),
            }
        )
    manifest = pd.DataFrame(manifest_rows).sort_values("match_date").reset_index(drop=True)
    manifest.to_csv(OUT_DIR / "_manifest.csv", index=False)
    logger.info("Wrote %d per-match event parquets + manifest to %s.", n_matches, OUT_DIR)

    print("\n=== Leverkusen 23/24 ingest summary ===")
    print(f"matches={n_matches}, with_360={n_360}")
    print(
        manifest[["match_date", "home_team", "away_team", "home_score", "away_score"]].to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
