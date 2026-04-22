"""Live ASA league-context refresh with 24h caching and on-disk fallback.

Orchestrates `/mls/teams/xgoals`, `/mls/teams/goals-added`, and `/mls/teams/xpass`.
After a successful bundle fetch, an atomic JSON snapshot is written under
`data/marts/cache/asa_league_bundle.json` for outage recovery.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ASA_BASE = "https://app.americansocceranalysis.com/api/v1/mls/teams"
ASA_URL_XGOALS = f"{ASA_BASE}/xgoals"
ASA_URL_GOALS_ADDED = f"{ASA_BASE}/goals-added"
ASA_URL_XPASS = f"{ASA_BASE}/xpass"
TIMEOUT = 30

_ROOT = Path(__file__).resolve().parents[2]
DISK_BUNDLE_PATH = _ROOT / "data" / "marts" / "cache" / "asa_league_bundle.json"


def _get_json_list(url: str, season: int) -> list:
    r = requests.get(url, params={"season_name": season}, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list from ASA at {url}, got {type(payload).__name__}")
    return payload


def write_disk_bundle(bundle: dict) -> None:
    """Atomically persist bundle JSON for outage fallback."""
    DISK_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(bundle, indent=2, default=str)
    tmp = DISK_BUNDLE_PATH.with_suffix(".tmp.json")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(DISK_BUNDLE_PATH)


def load_cached_bundle() -> dict | None:
    """Load last successful bundle from disk; returns None if missing or invalid."""
    if not DISK_BUNDLE_PATH.exists():
        return None
    try:
        data = json.loads(DISK_BUNDLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    required = {"season", "fetched_at", "xgoals", "goals_added", "xpass"}
    if not required <= set(data.keys()):
        return None
    if not all(isinstance(data[k], list) for k in ("xgoals", "goals_added", "xpass")):
        return None
    return data


@st.cache_data(ttl=24 * 3600, show_spinner="Pulling ASA live league data…")
def fetch_league_bundle_network(season: int) -> dict:
    """Fetch all three ASA endpoints and persist a disk snapshot on success."""
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    bundle = {
        "fetched_at": fetched_at,
        "season": int(season),
        "xgoals": _get_json_list(ASA_URL_XGOALS, season),
        "goals_added": _get_json_list(ASA_URL_GOALS_ADDED, season),
        "xpass": _get_json_list(ASA_URL_XPASS, season),
    }
    write_disk_bundle(bundle)
    return bundle


def resolve_league_bundle(season: int) -> tuple[dict, bool]:
    """Return (bundle, used_disk_fallback). Raises if live fetch fails and disk empty."""
    try:
        return fetch_league_bundle_network(int(season)), False
    except Exception:
        disk = load_cached_bundle()
        if disk is None:
            raise RuntimeError(
                "ASA endpoints are unavailable and no on-disk snapshot exists at "
                f"{DISK_BUNDLE_PATH}. Try again when online, or restore a prior JSON snapshot."
            ) from None
        return disk, True


def goals_added_totals_df(goals_added: list[dict]) -> pd.DataFrame:
    """One row per team: sum of goals_added_for / goals_added_against across action types."""
    rows: list[dict] = []
    for t in goals_added:
        tid = t.get("team_id")
        if tid is None:
            continue
        blocks = t.get("data") or []
        total_for = sum(float(b.get("goals_added_for", 0)) for b in blocks)
        total_against = sum(float(b.get("goals_added_against", 0)) for b in blocks)
        rows.append(
            {
                "team_id": tid,
                "ga_total_goals_added_for": total_for,
                "ga_total_goals_added_against": total_against,
            }
        )
    return pd.DataFrame(rows)


def xpass_summary_columns(xpass: list[dict]) -> pd.DataFrame:
    """Subset xPass columns merged into the main table (prefixed)."""
    df = pd.DataFrame(xpass)
    if df.empty:
        return df
    want = [
        "passes_completed_over_expected_difference",
        "avg_vertical_distance_difference",
        "passes_completed_over_expected_p100_for",
    ]
    keep = ["team_id"] + [c for c in want if c in df.columns]
    out = df[keep].copy()
    rename = {c: f"xpass_{c}" for c in want if c in out.columns}
    return out.rename(columns=rename)


def _xgoals_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["xg_per_shot"] = out["xgoals_for"] / out["shots_for"].replace(0, pd.NA)
    out["xg_conceded_per_shot"] = out["xgoals_against"] / out["shots_against"].replace(0, pd.NA)
    out["goal_diff"] = out["goals_for"] - out["goals_against"]
    out["xgoal_diff"] = out["xgoals_for"] - out["xgoals_against"]
    return out


def to_dataframe(bundle: dict) -> pd.DataFrame:
    """Build xGoals team-season DataFrame (supports legacy `teams` or bundle `xgoals`)."""
    rows = bundle.get("teams") or bundle.get("xgoals") or []
    df = pd.DataFrame(rows)
    return _xgoals_derived(df)


def merge_league_frames(bundle: dict) -> pd.DataFrame:
    """Join xGoals, goals-added totals, and selected xPass fields on team_id."""
    xg = _xgoals_derived(pd.DataFrame(bundle.get("xgoals") or []))
    if xg.empty:
        return xg
    ga = goals_added_totals_df(bundle.get("goals_added") or [])
    xp = xpass_summary_columns(bundle.get("xpass") or [])
    merged = xg.merge(ga, on="team_id", how="left")
    if not xp.empty:
        merged = merged.merge(xp, on="team_id", how="left")
    return merged


def load_team_lookup(path: Path) -> pd.DataFrame:
    """Optional ASA team_id → display name mapping (no fabricated IDs)."""
    if not path.exists():
        return pd.DataFrame(columns=["team_id", "team_name"])
    return pd.read_csv(path, dtype={"team_id": str})


def inter_miami_team_ids(lookup: pd.DataFrame) -> set[str]:
    """Return team_id values whose name matches Inter Miami, if lookup rows exist."""
    if lookup.empty or "team_name" not in lookup.columns:
        return set()
    mask = lookup["team_name"].astype(str).str.contains("Inter Miami", case=False, na=False)
    return set(lookup.loc[mask, "team_id"].astype(str))


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    """League moments for KPI strip."""
    if df.empty:
        return {"n_teams": 0, "xg_per_shot_mean": float("nan"), "xg_per_shot_std": float("nan")}
    out: dict[str, float | int] = {
        "n_teams": int(len(df)),
        "xg_per_shot_mean": float(df["xg_per_shot"].mean()),
        "xg_per_shot_std": float(df["xg_per_shot"].std(ddof=0)),
        "points_mean": float(df["points"].mean()),
        "points_std": float(df["points"].std(ddof=0)),
    }
    col = "xpass_passes_completed_over_expected_difference"
    if col in df.columns and df[col].notna().any():
        out["xpass_pxe_diff_mean"] = float(df[col].mean())
    colv = "xpass_avg_vertical_distance_difference"
    if colv in df.columns and df[colv].notna().any():
        out["xpass_vert_diff_mean"] = float(df[colv].mean())
    return out


def goals_added_rank_table(goals_added: list[dict]) -> pd.DataFrame:
    """Goals-added stack rank: offensive sum of goals_added_for, descending."""
    df = goals_added_totals_df(goals_added)
    if df.empty:
        return df
    df = df.sort_values("ga_total_goals_added_for", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
