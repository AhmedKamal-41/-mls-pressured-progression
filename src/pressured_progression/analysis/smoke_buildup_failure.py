"""Smoke test: run build-up failure labeler on MLS 2023 matches.

Spec asked for Philadelphia Union and Columbus Crew 2023 team-season failure
rates. Neither team is in StatsBomb Open Data (see docs/data_reality.md §1).
We document that directly here (0 matches, 0 rate) and add Inter Miami 2023 —
the only MLS 2023 team with Open Data coverage (6 matches, all with 360) — as
a substitute smoke-test substrate, clearly labeled.

Flow:
  1. Fetch MLS 2023 matches catalog.
  2. For each focal team: fetch its matches' events, adapt schema, write per-match
     parquet under data/raw/events/.
  3. Aggregate labels via DuckDB over the parquet corpus (per-match work stays out
     of full-season pandas memory).
  4. Print failure rate + 95% bootstrap CI per team-season.

Run:
    python -m pressured_progression.analysis.smoke_buildup_failure
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # quiet statsbombpy NoAuthWarning spam in output

from pressured_progression.ingest.events_adapter import adapt_statsbomb_events  # noqa: E402
from pressured_progression.labeling.buildup_failure import label_buildup_failures  # noqa: E402
from pressured_progression.sequences.possession_chain import (  # noqa: E402
    build_possession_chains,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DIR = ROOT / "data" / "raw" / "events"
LABELS_DIR = ROOT / "data" / "core" / "buildup_labels"
MLS_COMPETITION_ID = 44
MLS_2023_SEASON_ID = 107

FOCAL_TEAMS = [
    "Philadelphia Union",
    "Columbus Crew",
    "Inter Miami",
]

BOOT_N = 2000
BOOT_CI = 0.95
BOOT_SEED = 20260421


def _import_sb():
    from statsbombpy import sb

    return sb


def team_matches(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    mask = (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    return matches[mask].copy()


def ensure_events_parquet(sb, match_id: int) -> Path:
    out = EVENTS_DIR / f"{match_id}.parquet"
    if out.exists():
        return out
    raw = sb.events(match_id=match_id)
    adapted = adapt_statsbomb_events(raw)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    adapted.to_parquet(out, index=False)
    return out


def label_one_match(match_parquet: Path, focal_team_id: int) -> pd.DataFrame:
    con = duckdb.connect()
    events = con.execute(
        "SELECT * FROM read_parquet(?)",
        [str(match_parquet)],
    ).df()
    con.close()
    # Only label chains owned by the focal team — opponent buildups aren't our target.
    chains = build_possession_chains(events)
    chains = chains[chains["team_id"] == focal_team_id]
    if chains.empty:
        return pd.DataFrame(columns=["match_id", "possession_id", "is_failure", "failure_type"])
    return label_buildup_failures(chains, events)


def bootstrap_ci(values: np.ndarray, n_boot: int, ci: float, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = sample.mean()
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    lo, hi = np.percentile(boots, [lo_pct, hi_pct])
    return float(lo), float(hi)


def _team_id_for(matches: pd.DataFrame, team_name: str) -> int | None:
    sub = matches[matches["home_team"] == team_name]
    if not sub.empty:
        return int(sub.iloc[0]["home_team_id"])
    sub = matches[matches["away_team"] == team_name]
    if not sub.empty:
        return int(sub.iloc[0]["away_team_id"])
    return None


def _aggregate_labels_with_duckdb(labels_dir: Path) -> pd.DataFrame:
    """Union per-match label parquet via DuckDB (keeps full-season aggregation off pandas)."""
    files = sorted(Path(labels_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["match_id", "possession_id", "is_failure", "failure_type"])
    con = duckdb.connect()
    list_sql = "[" + ",".join(f"'{p}'" for p in files) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({list_sql})").df()
    con.close()
    return df


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sb = _import_sb()
    matches = sb.matches(competition_id=MLS_COMPETITION_ID, season_id=MLS_2023_SEASON_ID)
    logger.info("MLS 2023 Open Data: %d matches total", len(matches))

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []

    for team_name in FOCAL_TEAMS:
        team_labels_dir = LABELS_DIR / team_name.replace(" ", "_")
        team_labels_dir.mkdir(parents=True, exist_ok=True)

        team_m = team_matches(matches, team_name)
        n_matches = len(team_m)
        team_id = _team_id_for(team_m, team_name) if n_matches else None

        print(f"\n=== {team_name} -- 2023 MLS ===")
        print(f"  matches in StatsBomb Open Data: {n_matches}")
        if n_matches == 0:
            report.append(
                {
                    "team": team_name,
                    "matches": 0,
                    "chains_qualified": 0,
                    "failures": 0,
                    "failure_rate": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "note": "no StatsBomb Open Data coverage (see docs/data_reality.md sec 1)",
                }
            )
            print("  -> coverage gap documented (see docs/data_reality.md sec 1). Skipping.")
            continue

        for _, m in team_m.iterrows():
            match_id = int(m["match_id"])
            parquet = ensure_events_parquet(sb, match_id)
            labels = label_one_match(parquet, focal_team_id=team_id)
            out = team_labels_dir / f"{match_id}.parquet"
            labels.to_parquet(out, index=False)

        agg = _aggregate_labels_with_duckdb(team_labels_dir)
        qualified = len(agg)
        failures = int(agg["is_failure"].sum()) if qualified else 0
        rate = failures / qualified if qualified else float("nan")
        values = agg["is_failure"].astype(int).to_numpy() if qualified else np.array([])
        lo, hi = bootstrap_ci(values, BOOT_N, BOOT_CI, BOOT_SEED)

        breakdown = (
            agg[agg["is_failure"]]["failure_type"].value_counts().to_dict() if qualified else {}
        )

        report.append(
            {
                "team": team_name,
                "matches": n_matches,
                "chains_qualified": qualified,
                "failures": failures,
                "failure_rate": rate,
                "ci_lo": lo,
                "ci_hi": hi,
                "note": (
                    "SUBSTITUTE smoke target — not named in spec case studies"
                    if team_name == "Inter Miami"
                    else "spec-named case-study team"
                ),
                "breakdown": breakdown,
            }
        )

        print(f"  qualifying chains (start_x<40 + pressure in first 3 actions): {qualified}")
        print(f"  failures: {failures}")
        if qualified:
            print(f"  failure_rate: {rate:.3f}  95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
            print(f"  breakdown by failure_type: {breakdown}")

    print("\n=== Smoke-test summary ===")
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "breakdown"} for r in report])
    with pd.option_context("display.width", 160, "display.max_colwidth", 80):
        print(summary.to_string(index=False))

    print(
        "\nNote: Philadelphia Union and Columbus Crew 2023 have zero StatsBomb Open Data"
        " coverage.\nInter Miami 2023 is reported as a SUBSTITUTE smoke-test substrate,"
        " not as a spec case study."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
