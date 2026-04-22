"""Shared test helpers for sequence + labeling tests."""

from __future__ import annotations

import pandas as pd

FOCAL = 100
OPP = 200


def ev(**kw) -> dict:
    """Build one event row with sensible defaults. Override any field via kwargs."""
    base = {
        "match_id": 1,
        "period": 1,
        "minute": 0,
        "second": 0,
        "team_id": FOCAL,
        "player_id": 1,
        "event_type": "Pass",
        "location_x": 30.0,
        "location_y": 40.0,
        "outcome": None,
        "under_pressure": False,
        "pass_end_x": 35.0,
        "pass_end_y": 40.0,
        "possession_id": 1,
    }
    base.update(kw)
    return base


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
