"""StatsBomb Open Data catalog audit.

Pulls the full competitions/seasons catalog via statsbombpy, then counts matches
per season for MLS and the four named European top-5 competitions. Writes a
consolidated catalog CSV to data/raw/statsbomb_catalog.csv.

Run:
    python -m pressured_progression.ingest.statsbomb
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COMPETITIONS: dict[str, list[str]] = {
    "MLS": ["Major League Soccer"],
    "Premier League": ["Premier League"],
    "La Liga": ["La Liga"],
    "Serie A": ["Serie A"],
    "Bundesliga": ["1. Bundesliga", "Bundesliga"],
}

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
OUT_CSV = RAW_DIR / "statsbomb_catalog.csv"


def _import_sb():
    try:
        from statsbombpy import sb  # noqa: PLC0415

        return sb
    except ImportError as e:
        raise RuntimeError(
            "statsbombpy is not installed. Install with: pip install statsbombpy"
        ) from e


def fetch_competitions() -> pd.DataFrame:
    sb = _import_sb()
    comps = sb.competitions()
    if not isinstance(comps, pd.DataFrame):
        comps = pd.DataFrame(comps)
    return comps


def filter_target_rows(comps: pd.DataFrame) -> pd.DataFrame:
    wanted = {alias for names in TARGET_COMPETITIONS.values() for alias in names}
    mask = comps["competition_name"].isin(wanted)
    return comps.loc[mask].copy()


def count_matches(sb, competition_id: int, season_id: int) -> tuple[int, int]:
    """Return (total_matches, matches_with_360)."""
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        logger.warning(
            "matches() failed for competition=%s season=%s: %s",
            competition_id,
            season_id,
            e,
        )
        return (0, 0)
    if not isinstance(matches, pd.DataFrame):
        matches = pd.DataFrame(matches)
    total = len(matches)
    # Column is `match_status_360` in current statsbombpy; value "available" means 360 data present.
    for col in ("match_status_360", "match_available_360", "available_360"):
        if col in matches.columns:
            with_360 = int((matches[col].astype(str) == "available").sum())
            return total, with_360
    return total, 0


def build_catalog() -> pd.DataFrame:
    sb = _import_sb()
    comps = fetch_competitions()
    logger.info("Fetched %d competition-season rows.", len(comps))
    target = filter_target_rows(comps)
    logger.info("Filtered to %d target rows.", len(target))

    rows: list[dict] = []
    for _, r in target.iterrows():
        total, with_360 = count_matches(sb, int(r["competition_id"]), int(r["season_id"]))
        rows.append(
            {
                "competition_id": int(r["competition_id"]),
                "competition_name": r["competition_name"],
                "country_name": r.get("country_name", ""),
                "season_id": int(r["season_id"]),
                "season_name": r["season_name"],
                "matches_total": total,
                "matches_with_360": with_360,
            }
        )
    out = pd.DataFrame(rows).sort_values(["competition_name", "season_name"])
    return out


def print_summary(catalog: pd.DataFrame) -> None:
    if catalog.empty:
        print("No target competitions found in StatsBomb Open Data catalog.")
        return
    print("\n=== StatsBomb catalog: matches per competition-season ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(catalog.to_string(index=False))
    print("\n=== Totals by competition ===")
    totals = (
        catalog.groupby("competition_name")[["matches_total", "matches_with_360"]]
        .sum()
        .sort_index()
    )
    print(totals.to_string())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    catalog.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV, len(catalog))
    print_summary(catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
