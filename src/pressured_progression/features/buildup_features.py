"""Possession-level features for the build-up failure classifier (spec §4).

One row per qualifying chain. Functions take already-qualified chains
(output of sequences.possession_chain filtered upstream) plus the full
events frame for the relevant matches, and return a DataFrame keyed by
(match_id, possession_id, team_id).

Coordinate convention: StatsBomb 0–120 x / 0–80 y, each event recorded in
the acting team's attacking frame (attacking right). `opp_press_height` is
reported in the opposing team's own frame — higher value = opponent pressing
closer to our goal. That choice is documented here and pinned in
`docs/project_spec.md` §9.

`support_density_ff` DROPPED from the feature set (spec §9). The feature
required StatsBomb 360 freeze-frame data, but all 6 MLS 2023 360 JSON files
return HTTP 404 from the Open Data repo despite catalog-claimed availability
(see `docs/data_reality.md` §1 addendum). Restore when 360 data lands or the
project acquires it via paid API.

Score-diff computation credits both `Shot` events with `outcome == "Goal"`
and `Own Goal For` events (StatsBomb emits the latter against the team that
benefits), so own-goal scoring contributes to `sd_*` buckets correctly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from pressured_progression.core.schemas import EventRow, validate_columns

logger = logging.getLogger(__name__)

PRESSURE_EVENT = "Pressure"
DEFENSIVE_EVENTS = {"Pressure", "Interception", "Tackle", "Ball Recovery", "Duel", "Block"}
PRESS_DENSITY_WINDOW_S = 30.0
PRESS_PRIOR_WINDOW_S = 30.0
FR_PRESSURE_BINARY_S = 2.0
FR_PRESSURE_LOOKAHEAD_S = 5.0

SCORE_BUCKETS = ["minus2_or_worse", "minus1", "zero", "plus1", "plus2_or_better"]
MINUTE_BUCKETS = ["0_15", "16_30", "31_45plus", "46_60", "61_75", "76_90plus"]


def _abs_t(period: int, minute: int, second: int) -> float:
    return float((period - 1) * 45 * 60 + minute * 60 + second)


def _bucket_score(diff: int) -> str:
    if diff <= -2:
        return "minus2_or_worse"
    if diff == -1:
        return "minus1"
    if diff == 0:
        return "zero"
    if diff == 1:
        return "plus1"
    return "plus2_or_better"


def _bucket_minute(minute: int) -> str:
    if minute <= 15:
        return "0_15"
    if minute <= 30:
        return "16_30"
    if minute <= 45:
        return "31_45plus"
    if minute <= 60:
        return "46_60"
    if minute <= 75:
        return "61_75"
    return "76_90plus"


def _pair_xy(row, x_key: str, y_key: str) -> tuple[float, float] | None:
    x = row.get(x_key)
    y = row.get(y_key)
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _pass_length(row) -> float | None:
    s = _pair_xy(row, "location_x", "location_y")
    e = _pair_xy(row, "pass_end_x", "pass_end_y")
    if s is None or e is None:
        return None
    return float(np.hypot(e[0] - s[0], e[1] - s[1]))


def _is_successful_pass(row) -> bool:
    if str(row.get("event_type")) != "Pass":
        return False
    outc = row.get("outcome")
    return outc is None or (isinstance(outc, float) and pd.isna(outc))


def _passing_network(match_events: pd.DataFrame, focal_team_id: int) -> nx.DiGraph:
    passes = match_events[
        (match_events["team_id"] == focal_team_id) & (match_events["event_type"] == "Pass")
    ]
    g = nx.DiGraph()
    if "pass_recipient_id" not in passes.columns:
        return g
    for _, p in passes.iterrows():
        if not _is_successful_pass(p):
            continue
        src = p.get("player_id")
        dst = p.get("pass_recipient_id")
        if src is None or dst is None or pd.isna(src) or pd.isna(dst):
            continue
        src_i, dst_i = int(src), int(dst)
        if g.has_edge(src_i, dst_i):
            g[src_i][dst_i]["weight"] += 1
        else:
            g.add_edge(src_i, dst_i, weight=1)
    return g


def _betweenness_for(g: nx.DiGraph, player_id: int | None) -> float:
    if player_id is None or player_id not in g:
        return 0.0
    bc = nx.betweenness_centrality(g, weight="weight")
    return float(bc.get(player_id, 0.0))


def _cumulative_score_at(
    match_events: pd.DataFrame, focal_team_id: int, t: float
) -> tuple[int, int]:
    """Return (score_for, score_against) as of time t. Includes own goals.

    StatsBomb emits one `Own Goal For` event credited to the benefiting team
    (alongside a sibling `Own Goal Against` on the team that scored on
    themselves). We credit OG using the `Own Goal For` row only, mirroring
    how `Shot` goals are credited to the scoring team_id.
    """
    prior = match_events[match_events["_t"] < t]
    shot_goals = prior[(prior["event_type"] == "Shot") & (prior["outcome"].astype(str) == "Goal")]
    og_for = prior[prior["event_type"] == "Own Goal For"]
    sf = int((shot_goals["team_id"] == focal_team_id).sum()) + int(
        (og_for["team_id"] == focal_team_id).sum()
    )
    sa = int((shot_goals["team_id"] != focal_team_id).sum()) + int(
        (og_for["team_id"] != focal_team_id).sum()
    )
    return sf, sa


def _first_receipt_time(chain_events: pd.DataFrame, first_receiver_id: int | None) -> float | None:
    if first_receiver_id is None:
        return None
    rec = chain_events[
        (chain_events["event_type"] == "Ball Receipt*")
        & (chain_events["player_id"].astype("Int64") == int(first_receiver_id))
    ]
    if rec.empty:
        return None
    return float(rec.iloc[0]["_t"])


def _fr_pressure_features(
    match_events: pd.DataFrame,
    focal_team_id: int,
    receipt_t: float | None,
) -> tuple[bool, float | None]:
    if receipt_t is None:
        return False, None
    window = match_events[
        (match_events["event_type"] == PRESSURE_EVENT)
        & (match_events["team_id"] != focal_team_id)
        & (match_events["_t"] >= receipt_t)
        & (match_events["_t"] <= receipt_t + FR_PRESSURE_LOOKAHEAD_S)
    ].sort_values("_t")
    if window.empty:
        return False, None
    gap = float(window.iloc[0]["_t"] - receipt_t)
    return (gap <= FR_PRESSURE_BINARY_S), gap


def _recent_pass_reach(
    match_events: pd.DataFrame, focal_team_id: int, chain_start_t: float
) -> float:
    prior_passes = match_events[
        (match_events["_t"] < chain_start_t)
        & (match_events["team_id"] == focal_team_id)
        & (match_events["event_type"] == "Pass")
    ].tail(3)
    if len(prior_passes) < 3:
        return 0.0
    lengths = [_pass_length(p) for _, p in prior_passes.iterrows()]
    lengths = [x for x in lengths if x is not None]
    if len(lengths) < 3:
        return 0.0
    return float(np.mean(lengths))


def _opp_press_height(
    match_events: pd.DataFrame, focal_team_id: int, chain_start_t: float
) -> float | None:
    window = match_events[
        (match_events["team_id"] != focal_team_id)
        & (match_events["event_type"].isin(DEFENSIVE_EVENTS))
        & (match_events["_t"] >= chain_start_t - PRESS_PRIOR_WINDOW_S)
        & (match_events["_t"] < chain_start_t)
    ]
    if window.empty:
        return None
    xs = window["location_x"].dropna().astype(float)
    if xs.empty:
        return None
    return float(xs.mean())


def _defthird_pressure_density(
    match_events: pd.DataFrame,
    chain_events: pd.DataFrame,
    focal_team_id: int,
    chain_start_t: float,
) -> float:
    window_end = chain_start_t + PRESS_DENSITY_WINDOW_S
    own_in_window = chain_events[chain_events["_t"] <= window_end]
    denom = max(len(own_in_window), 1)
    opp_pressures = match_events[
        (match_events["event_type"] == PRESSURE_EVENT)
        & (match_events["team_id"] != focal_team_id)
        & (match_events["_t"] >= chain_start_t)
        & (match_events["_t"] <= window_end)
    ]
    return float(len(opp_pressures) / denom)


def _ensure_t(events: pd.DataFrame) -> pd.DataFrame:
    if "_t" in events.columns:
        return events
    df = events.copy()
    df["_t"] = df.apply(
        lambda r: _abs_t(int(r["period"]), int(r["minute"]), int(r["second"])), axis=1
    )
    return df


def _onehot(df: pd.DataFrame, col: str, levels: Iterable[str], prefix: str) -> pd.DataFrame:
    dummies = pd.DataFrame(
        {f"{prefix}_{lvl}": (df[col] == lvl).astype(int) for lvl in levels},
        index=df.index,
    )
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


def extract_buildup_features(
    chains: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Extract per-chain features. Returns a DataFrame keyed by chain."""
    validate_columns(events, EventRow)
    if chains.empty:
        return pd.DataFrame()

    df = _ensure_t(events)

    # Precompute per-match passing networks (betweenness on one node per chain is cheap;
    # recomputing the whole graph per chain is wasteful).
    networks: dict[tuple[int, int], nx.DiGraph] = {}
    for (mid, tid), grp in df.groupby(["match_id", "team_id"], sort=False):
        networks[(int(mid), int(tid))] = _passing_network(grp, int(tid))

    rows: list[dict] = []
    for _, chain in chains.iterrows():
        mid = int(chain["match_id"])
        pid = int(chain["possession_id"])
        tid = int(chain["team_id"])

        match_events = df[df["match_id"] == mid]
        chain_events = match_events[
            (match_events["possession_id"] == pid) & (match_events["team_id"] == tid)
        ].sort_values("_t")
        if chain_events.empty:
            continue

        t0 = float(chain_events.iloc[0]["_t"])
        chain_minute = int(chain_events.iloc[0]["minute"])

        fr_id = chain.get("first_receiver_id")
        fr_id = int(fr_id) if fr_id is not None and not pd.isna(fr_id) else None
        receipt_t = _first_receipt_time(chain_events, fr_id)

        fr_bin, fr_sec = _fr_pressure_features(match_events, tid, receipt_t)
        dens = _defthird_pressure_density(match_events, chain_events, tid, t0)
        reach = _recent_pass_reach(match_events, tid, t0)
        ph = _opp_press_height(match_events, tid, t0)
        btw = _betweenness_for(networks.get((mid, tid), nx.DiGraph()), fr_id)

        sf, sa = _cumulative_score_at(match_events, tid, t0)

        rows.append(
            {
                "match_id": mid,
                "possession_id": pid,
                "team_id": tid,
                "defthird_pressure_density": dens,
                "first_receiver_pressure_binary": bool(fr_bin),
                "first_receiver_pressure_seconds": fr_sec,
                "recent_pass_reach": reach,
                "opp_press_height": ph,
                "first_receiver_betweenness": btw,
                "score_diff_bucket": _bucket_score(sf - sa),
                "minute_bucket": _bucket_minute(chain_minute),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = _onehot(out, "score_diff_bucket", SCORE_BUCKETS, "sd")
    out = _onehot(out, "minute_bucket", MINUTE_BUCKETS, "min")
    return out


FEATURE_COLUMNS = [
    "defthird_pressure_density",
    "first_receiver_pressure_binary",
    "first_receiver_pressure_seconds",
    "recent_pass_reach",
    "opp_press_height",
    "first_receiver_betweenness",
    *[f"sd_{lvl}" for lvl in SCORE_BUCKETS],
    *[f"min_{lvl}" for lvl in MINUTE_BUCKETS],
]
