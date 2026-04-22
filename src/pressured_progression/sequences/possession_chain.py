"""Possession-chain construction.

A chain is the sequence of events owned by a single team within one StatsBomb
possession block. We group events by (match_id, possession_id), derive the
possessing team as the team of the first event in that block, then keep only
events whose team_id matches. That excludes interleaved defensive actions by
the opposing team (e.g., a Pressure event recorded during the attacker's
possession) while preserving the attacker's own actions.
"""

from __future__ import annotations

import pandas as pd

from pressured_progression.core.schemas import EventRow, validate_columns

_SORT_KEYS = ["match_id", "period", "minute", "second"]


def _abs_seconds(period: int, minute: int, second: int) -> float:
    """Monotonic seconds within a match. Approximate — good enough for ordering."""
    return float((period - 1) * 45 * 60 + minute * 60 + second)


def _derive_possession_team(events: pd.DataFrame) -> pd.Series:
    """team_id of the first event in each (match_id, possession_id) block."""
    first_idx = events.groupby(["match_id", "possession_id"], sort=False).head(1).index
    first_teams = events.loc[first_idx, ["match_id", "possession_id", "team_id"]]
    key = list(zip(events["match_id"], events["possession_id"], strict=False))
    lookup = {
        (m, p): t
        for m, p, t in zip(
            first_teams["match_id"],
            first_teams["possession_id"],
            first_teams["team_id"],
            strict=False,
        )
    }
    return pd.Series([lookup[k] for k in key], index=events.index, name="possession_team_id")


def _terminal_end_type(row: pd.Series) -> str:
    et = str(row.get("event_type", ""))
    outcome = row.get("outcome")
    if et == "Shot":
        if isinstance(outcome, str) and outcome.lower() == "goal":
            return "goal"
        return "shot"
    if et in {"Dispossessed", "Miscontrol"}:
        return "loss"
    if et == "Clearance":
        return "clearance"
    if et == "Pass" and isinstance(outcome, str) and outcome.lower() not in {"", "complete"}:
        return "loss"
    if et == "Ball Receipt*" and isinstance(outcome, str) and outcome.lower() != "complete":
        return "loss"
    if et == "Foul Won":
        return "foul_won"
    if et == "Half End":
        return "half_end"
    return et.lower() if et else "unknown"


def build_possession_chains(events: pd.DataFrame) -> pd.DataFrame:
    """Construct one row per (match_id, possession_id) chain.

    Input: events DataFrame with columns per EventRow. Rows need not be sorted.
    Output columns: per PossessionChain schema.
    """
    validate_columns(events, EventRow)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "possession_id",
                "team_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "end_type",
                "action_count",
                "under_pressure_count",
                "terminal_outcome",
                "first_receiver_id",
            ]
        )

    df = events.copy().sort_values([*_SORT_KEYS, "possession_id"], kind="stable")
    df = df.reset_index(drop=True)
    df["possession_team_id"] = _derive_possession_team(df)
    owned = df[df["team_id"] == df["possession_team_id"]].copy()

    rows: list[dict] = []
    for (match_id, possession_id), grp in owned.groupby(["match_id", "possession_id"], sort=False):
        grp_sorted = grp.sort_values(_SORT_KEYS, kind="stable").reset_index(drop=True)
        first = grp_sorted.iloc[0]
        last = grp_sorted.iloc[-1]
        first_receiver_id = None
        if "pass_recipient_id" in grp_sorted.columns:
            recips = grp_sorted["pass_recipient_id"].dropna()
            if len(recips):
                first_receiver_id = int(recips.iloc[0])
        rows.append(
            {
                "match_id": int(match_id),
                "possession_id": int(possession_id),
                "team_id": int(first["team_id"]),
                "start_x": _nullable_float(first.get("location_x")),
                "start_y": _nullable_float(first.get("location_y")),
                "end_x": _nullable_float(last.get("location_x")),
                "end_y": _nullable_float(last.get("location_y")),
                "end_type": _terminal_end_type(last),
                "action_count": int(len(grp_sorted)),
                "under_pressure_count": int(grp_sorted["under_pressure"].fillna(False).sum()),
                "terminal_outcome": last.get("outcome"),
                "first_receiver_id": first_receiver_id,
            }
        )
    return pd.DataFrame(rows).sort_values(["match_id", "possession_id"]).reset_index(drop=True)


def _nullable_float(v) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(fv):
        return None
    return fv
