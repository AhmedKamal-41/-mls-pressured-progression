"""Project statsbombpy's wide events DataFrame onto the EventRow contract.

statsbombpy returns ~90 columns per event with nested locations and outcome
columns split per event type. We flatten locations, consolidate outcome fields,
and rename `type`→`event_type`, `possession`→`possession_id`. Extra columns
(e.g., shot_statsbomb_xg, pass_recipient_id) are preserved because downstream
code reads them by name.
"""

from __future__ import annotations

import pandas as pd


def _split_xy(df: pd.DataFrame, src: str, x_col: str, y_col: str) -> None:
    if src not in df.columns:
        df[x_col] = None
        df[y_col] = None
        return
    col = df[src]
    df[x_col] = col.map(lambda v: v[0] if isinstance(v, list | tuple) and len(v) >= 2 else None)
    df[y_col] = col.map(lambda v: v[1] if isinstance(v, list | tuple) and len(v) >= 2 else None)


def _consolidate_outcome(df: pd.DataFrame) -> pd.Series:
    """Pick the event-type-specific outcome column into a single `outcome` field."""
    pass_o = df.get("pass_outcome")
    ball_o = df.get("ball_receipt_outcome")
    shot_o = df.get("shot_outcome")
    dribble_o = df.get("dribble_outcome")
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    if pass_o is not None:
        out = out.where(out.notna(), pass_o)
    if ball_o is not None:
        out = out.where(out.notna(), ball_o)
    if shot_o is not None:
        out = out.where(out.notna(), shot_o)
    if dribble_o is not None:
        out = out.where(out.notna(), dribble_o)
    return out


def adapt_statsbomb_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame matching the EventRow column contract (plus extras)."""
    if raw.empty:
        return raw

    df = raw.copy()
    _split_xy(df, "location", "location_x", "location_y")
    _split_xy(df, "pass_end_location", "pass_end_x", "pass_end_y")

    df["event_type"] = df["type"].astype(str)
    df["possession_id"] = df["possession"].astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(bool)
    df["outcome"] = _consolidate_outcome(df)

    # Ensure required scalar columns exist and typed.
    for col in ("match_id", "period", "minute", "second", "team_id"):
        df[col] = df[col].astype(int)
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    keep = [
        "match_id",
        "period",
        "minute",
        "second",
        "team_id",
        "player_id",
        "event_type",
        "location_x",
        "location_y",
        "outcome",
        "under_pressure",
        "pass_end_x",
        "pass_end_y",
        "possession_id",
        # Extras consumed downstream:
        "possession_team_id",
        "pass_recipient_id",
        "shot_statsbomb_xg",
        "pass_length",
        "duration",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
