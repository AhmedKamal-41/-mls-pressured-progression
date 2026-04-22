"""Regain detection.

A regain = defensive action by the focal team that transitions possession
opp → us. We look for focal-team events whose type is in DEFENSIVE_ACTIONS and
whose possession block is the focal team's (i.e., the block they just started).
For each regain we measure time to the focal team's next shot, capped at 15s,
and that shot's xG (from `shot_statsbomb_xg` when present).
"""

from __future__ import annotations

import pandas as pd

from pressured_progression.core.schemas import EventRow, validate_columns
from pressured_progression.sequences.possession_chain import (
    _abs_seconds,
    _derive_possession_team,
    _nullable_float,
)

DEFENSIVE_ACTIONS: set[str] = {"Interception", "Ball Recovery", "Tackle", "Duel", "Block"}
SHOT_WINDOW_S: float = 15.0


def detect_regains(events: pd.DataFrame, focal_team_id: int) -> pd.DataFrame:
    """Return one row per regain by focal team."""
    validate_columns(events, EventRow)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "time_seconds",
                "regain_x",
                "regain_y",
                "regain_zone",
                "regaining_team_id",
                "subsequent_possession_id",
                "next_shot_seconds",
                "next_shot_xg",
            ]
        )

    df = events.copy().sort_values(["match_id", "period", "minute", "second"], kind="stable")
    df = df.reset_index(drop=True)
    df["possession_team_id"] = _derive_possession_team(df)
    df["_t"] = df.apply(
        lambda r: _abs_seconds(int(r["period"]), int(r["minute"]), int(r["second"])), axis=1
    )

    focal_defensive = (
        df["event_type"].isin(DEFENSIVE_ACTIONS)
        & (df["team_id"] == focal_team_id)
        & (df["possession_team_id"] == focal_team_id)
    )
    candidates = df[focal_defensive]

    rows: list[dict] = []
    for _, ev in candidates.iterrows():
        t = float(ev["_t"])
        match_id = int(ev["match_id"])
        zone = _zone_for_x(ev["location_x"])
        next_shot_s, next_shot_xg = _time_to_next_shot(df, match_id, focal_team_id, t)
        rows.append(
            {
                "match_id": match_id,
                "time_seconds": t,
                "regain_x": _nullable_float(ev.get("location_x")),
                "regain_y": _nullable_float(ev.get("location_y")),
                "regain_zone": zone,
                "regaining_team_id": focal_team_id,
                "subsequent_possession_id": int(ev["possession_id"]),
                "next_shot_seconds": next_shot_s,
                "next_shot_xg": next_shot_xg,
            }
        )
    return pd.DataFrame(rows)


def _zone_for_x(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
    fx = float(x)
    if fx < 40:
        return "defensive"
    if fx < 80:
        return "middle"
    return "attacking"


def _time_to_next_shot(
    df: pd.DataFrame, match_id: int, focal_team_id: int, t0: float
) -> tuple[float | None, float | None]:
    window = df[
        (df["match_id"] == match_id)
        & (df["team_id"] == focal_team_id)
        & (df["event_type"] == "Shot")
        & (df["_t"] >= t0)
        & (df["_t"] <= t0 + SHOT_WINDOW_S)
    ]
    if window.empty:
        return None, None
    first = window.iloc[0]
    dt = float(first["_t"] - t0)
    xg = first.get("shot_statsbomb_xg")
    try:
        xg_val = float(xg) if xg is not None and not pd.isna(xg) else None
    except (TypeError, ValueError):
        xg_val = None
    return dt, xg_val
