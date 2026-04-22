"""Goals-added aggregation helpers (ASA-shaped payloads)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from components import asa_refresh  # noqa: E402 — app package not on PYTHONPATH by default

_SAMPLE_GOALS_ADDED = [
    {
        "team_id": "tidA",
        "data": [
            {"goals_added_for": 10.5, "goals_added_against": 5.0},
            {"goals_added_for": 3.25, "goals_added_against": 2.5},
        ],
    }
]


def test_goals_added_totals_sum_across_actions() -> None:
    df = asa_refresh.goals_added_totals_df(_SAMPLE_GOALS_ADDED)
    assert len(df) == 1
    assert df.iloc[0]["ga_total_goals_added_for"] == 13.75
    assert df.iloc[0]["ga_total_goals_added_against"] == 7.5


def test_goals_added_rank_table_orders_desc() -> None:
    payload = [
        {"team_id": "low", "data": [{"goals_added_for": 1.0, "goals_added_against": 0.0}]},
        {"team_id": "high", "data": [{"goals_added_for": 9.0, "goals_added_against": 1.0}]},
    ]
    ranked = asa_refresh.goals_added_rank_table(payload)
    assert ranked.iloc[0]["team_id"] == "high"
    assert ranked.iloc[0]["rank"] == 1
