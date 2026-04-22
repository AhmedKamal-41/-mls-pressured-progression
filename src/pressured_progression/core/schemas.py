"""Core Pydantic schemas for the sequence + labeling pipeline.

These describe column contracts — not per-row validators at scale. Functions
receive pandas DataFrames whose columns match these field names; use
`validate_columns` to check shape at boundaries.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class EventRow(BaseModel):
    """One StatsBomb event, projected to the fields we need downstream."""

    model_config = ConfigDict(extra="allow")

    match_id: int
    period: int
    minute: int
    second: int
    team_id: int
    player_id: int | None = None
    event_type: str
    location_x: float | None = None
    location_y: float | None = None
    outcome: str | None = None
    under_pressure: bool = False
    pass_end_x: float | None = None
    pass_end_y: float | None = None
    possession_id: int


class PossessionChain(BaseModel):
    match_id: int
    possession_id: int
    team_id: int
    start_x: float | None
    start_y: float | None
    end_x: float | None
    end_y: float | None
    end_type: str
    action_count: int
    under_pressure_count: int
    terminal_outcome: str | None
    first_receiver_id: int | None


class RegainEvent(BaseModel):
    match_id: int
    time_seconds: float
    regain_x: float | None
    regain_y: float | None
    regain_zone: str  # "defensive" | "middle" | "attacking"
    regaining_team_id: int
    subsequent_possession_id: int | None
    next_shot_seconds: float | None
    next_shot_xg: float | None


class FailureType(StrEnum):
    TURNOVER_OWN_HALF = "turnover_own_half"
    FORCED_LONG_BALL = "forced_long_ball"
    OPP_SHOT_WITHIN_10S = "opp_shot_within_10s"
    BACKWARD_RESET_TURNOVER = "backward_reset_turnover"
    NONE = "none"


class BuildUpFailureLabel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    match_id: int
    possession_id: int
    is_failure: bool
    failure_type: FailureType


def validate_columns(df, schema: type[BaseModel]) -> None:
    """Raise if any schema field is missing from df's columns. Extra columns allowed."""
    required = set(schema.model_fields.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {schema.__name__}: {sorted(missing)}")
