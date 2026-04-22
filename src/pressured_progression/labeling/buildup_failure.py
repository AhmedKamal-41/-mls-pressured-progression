"""Build-up failure labeling (spec §4 Module A).

A possession qualifies if its origin is in the defensive third (start_x < 40)
and opponent pressure appears within the first 3 actions. Qualifying chains
are labeled with the first matching failure type in spec order:

1. turnover_own_half — terminal event is a possession loss (dispossession,
   miscontrol, failed pass/ball receipt) with x < 60.
2. forced_long_ball — a pass with length > 40m is made and possession is
   lost within the next 2 actions.
3. opp_shot_within_10s — after the chain ends in possession loss, the
   opposing team shoots within 10 seconds.
4. backward_reset_turnover — a backward pass (pass_end_x < start_x - 5)
   appears in the chain and possession is lost within 5 subsequent actions.

Chains that don't match any → `none` (no failure).
"""

from __future__ import annotations

import pandas as pd

from pressured_progression.core.schemas import EventRow, FailureType, validate_columns
from pressured_progression.sequences.possession_chain import (
    _abs_seconds,
    _derive_possession_team,
)

LONG_BALL_METERS = 40.0
BACKWARD_X_DELTA = 5.0
TURNOVER_X_MAX = 60.0
OPP_SHOT_WINDOW_S = 10.0
SORT_KEYS = ["match_id", "period", "minute", "second"]


def label_buildup_failures(chains: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per qualifying chain with is_failure + failure_type."""
    validate_columns(events, EventRow)
    if chains.empty or events.empty:
        return pd.DataFrame(columns=["match_id", "possession_id", "is_failure", "failure_type"])

    df = events.copy().sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)
    df["possession_team_id"] = _derive_possession_team(df)
    df["_t"] = df.apply(
        lambda r: _abs_seconds(int(r["period"]), int(r["minute"]), int(r["second"])), axis=1
    )

    # Build possession -> owning team map per match, plus ordered possession ids per match.
    owner = (
        df.groupby(["match_id", "possession_id"], sort=False)["possession_team_id"]
        .first()
        .to_dict()
    )
    ordered_pids: dict[int, list[int]] = {}
    for mid, grp in df.groupby("match_id", sort=False):
        ordered_pids[int(mid)] = sorted(int(p) for p in grp["possession_id"].unique())

    rows: list[dict] = []
    for _, ch in chains.iterrows():
        mid = int(ch["match_id"])
        pid = int(ch["possession_id"])
        tid = int(ch["team_id"])

        chain_events = df[
            (df["match_id"] == mid) & (df["possession_id"] == pid) & (df["team_id"] == tid)
        ].reset_index(drop=True)
        if chain_events.empty:
            continue

        if not _qualifies(chain_events):
            continue

        ft = _classify(chain_events, df, mid, pid, tid, owner, ordered_pids[mid])
        rows.append(
            {
                "match_id": mid,
                "possession_id": pid,
                "is_failure": ft != FailureType.NONE,
                "failure_type": ft.value,
            }
        )

    return pd.DataFrame(rows)


def _qualifies(chain_events: pd.DataFrame) -> bool:
    start_x = chain_events["location_x"].iloc[0]
    if start_x is None or pd.isna(start_x) or float(start_x) >= 40.0:
        return False
    first_3 = chain_events.head(3)
    return bool(first_3["under_pressure"].fillna(False).any())


def _we_lost(
    mid: int, pid: int, tid: int, owner: dict, ordered: list[int]
) -> tuple[bool, int | None]:
    """Return (possession_lost_to_opponent, next_possession_id)."""
    try:
        idx = ordered.index(pid)
    except ValueError:
        return False, None
    if idx + 1 >= len(ordered):
        return False, None
    next_pid = ordered[idx + 1]
    next_team = owner.get((mid, next_pid))
    return (next_team is not None and int(next_team) != tid), next_pid


def _terminal_is_turnover(last: pd.Series) -> bool:
    et = str(last.get("event_type", ""))
    outc = str(last.get("outcome") or "").lower()
    if et in {"Dispossessed", "Miscontrol"}:
        return True
    if et == "Ball Receipt*" and outc and outc != "complete":
        return True
    return et == "Pass" and bool(outc) and outc not in {"complete", ""}


def _pass_length(row: pd.Series) -> float | None:
    sx, sy = row.get("location_x"), row.get("location_y")
    ex, ey = row.get("pass_end_x"), row.get("pass_end_y")
    if None in (sx, sy, ex, ey) or any(pd.isna(v) for v in (sx, sy, ex, ey)):
        return None
    return ((float(ex) - float(sx)) ** 2 + (float(ey) - float(sy)) ** 2) ** 0.5


def _classify(
    chain_events: pd.DataFrame,
    all_events: pd.DataFrame,
    mid: int,
    pid: int,
    tid: int,
    owner: dict,
    ordered: list[int],
) -> FailureType:
    last = chain_events.iloc[-1]
    we_lost, _ = _we_lost(mid, pid, tid, owner, ordered)

    # 1. turnover_own_half
    if we_lost and _terminal_is_turnover(last):
        lx = last.get("location_x")
        if lx is not None and not pd.isna(lx) and float(lx) < TURNOVER_X_MAX:
            return FailureType.TURNOVER_OWN_HALF

    # 2. forced_long_ball: any pass with length > 40m, possession lost within 2 actions after.
    if we_lost:
        n = len(chain_events)
        for i in range(n):
            ev = chain_events.iloc[i]
            if str(ev.get("event_type")) != "Pass":
                continue
            length = _pass_length(ev)
            if length is not None and length > LONG_BALL_METERS:
                remaining_after = n - i - 1
                if remaining_after <= 2:
                    return FailureType.FORCED_LONG_BALL

    # 3. opp_shot_within_10s
    if we_lost:
        t_end = float(last["_t"])
        opp_shots = all_events[
            (all_events["match_id"] == mid)
            & (all_events["team_id"] != tid)
            & (all_events["event_type"] == "Shot")
            & (all_events["_t"] > t_end)
            & (all_events["_t"] <= t_end + OPP_SHOT_WINDOW_S)
        ]
        if not opp_shots.empty:
            return FailureType.OPP_SHOT_WITHIN_10S

    # 4. backward_reset_turnover
    if we_lost:
        n = len(chain_events)
        for i in range(n):
            ev = chain_events.iloc[i]
            if str(ev.get("event_type")) != "Pass":
                continue
            sx, ex = ev.get("location_x"), ev.get("pass_end_x")
            if sx is None or ex is None or pd.isna(sx) or pd.isna(ex):
                continue
            if float(ex) < float(sx) - BACKWARD_X_DELTA:
                remaining_after = n - i - 1
                if remaining_after <= 5:
                    return FailureType.BACKWARD_RESET_TURNOVER

    return FailureType.NONE
