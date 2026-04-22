from __future__ import annotations

import pandas as pd

from pressured_progression.features.post_regain import (
    aggregate_team_season,
    compute_regain_events,
)
from tests.fixtures import FOCAL, OPP, ev, frame


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "xg_per_shot",
                "league_mean": 0.11,
                "league_std": 0.02,
                "n_teams": 29,
                "source_endpoint": "teams/xgoals",
                "note": "test",
            },
            {
                "metric": "rushed_shot_rate",
                "league_mean": float("nan"),
                "league_std": float("nan"),
                "n_teams": 0,
                "source_endpoint": "teams/xgoals",
                "note": "unavailable",
            },
        ]
    )


def test_enriched_fields_for_regain_to_shot():
    """Regain at x=25, then focal team progresses and shoots within 15s at x=110."""
    events = frame(
        [
            # Opp possession 1
            ev(
                possession_id=1,
                team_id=OPP,
                minute=10,
                second=0,
                event_type="Pass",
                location_x=60.0,
                pass_end_x=30.0,
            ),
            # Focal interception starts possession 2
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=1,
                event_type="Interception",
                location_x=25.0,
                location_y=40.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=3,
                event_type="Pass",
                location_x=25.0,
                location_y=40.0,
                pass_end_x=60.0,
                pass_end_y=40.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=6,
                event_type="Pass",
                location_x=60.0,
                location_y=40.0,
                pass_end_x=95.0,
                pass_end_y=40.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=10,
                event_type="Shot",
                location_x=110.0,
                location_y=40.0,
                shot_statsbomb_xg=0.25,
            ),
        ]
    )
    regains = compute_regain_events(events, focal_team_id=FOCAL)
    assert len(regains) == 1
    r = regains.iloc[0]
    assert r["regain_zone"] == "defensive"
    assert r["regain_x"] == 25.0
    assert r["regain_y"] == 40.0
    assert r["next_shot_seconds"] == 9.0  # 10:01 -> 10:10
    assert r["next_shot_xg"] == 0.25
    assert r["reached_final_third"]  # event at x=95 and x=110
    assert not r["lost_within_4_actions"]
    assert r["chain_end_type"] == "shot"


def test_regain_to_loss_within_four_actions():
    """Regain at x=25, three focal events, fourth is a turnover."""
    events = frame(
        [
            ev(
                possession_id=1,
                team_id=OPP,
                minute=20,
                second=0,
                event_type="Pass",
                location_x=60.0,
                pass_end_x=30.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=1,
                event_type="Ball Recovery",
                location_x=25.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=3,
                event_type="Pass",
                location_x=25.0,
                pass_end_x=30.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=5,
                event_type="Pass",
                location_x=30.0,
                pass_end_x=35.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=7,
                event_type="Dispossessed",
                location_x=35.0,
            ),
        ]
    )
    regains = compute_regain_events(events, focal_team_id=FOCAL)
    assert len(regains) == 1
    r = regains.iloc[0]
    assert r["lost_within_4_actions"]
    assert not r["reached_final_third"]
    assert r["chain_end_type"] == "loss"
    assert pd.isna(r["next_shot_seconds"])
    assert pd.isna(r["next_shot_xg"])


def test_aggregate_and_bootstrap():
    """Two matches each with one regain-to-shot + one regain-to-loss. Check aggregates."""
    ev_list = []
    for mid in (1, 2):
        ev_list.extend(
            [
                # regain-to-shot
                ev(
                    match_id=mid,
                    possession_id=1,
                    team_id=OPP,
                    minute=1,
                    second=0,
                    event_type="Pass",
                    location_x=60.0,
                    pass_end_x=30.0,
                ),
                ev(
                    match_id=mid,
                    possession_id=2,
                    team_id=FOCAL,
                    minute=1,
                    second=1,
                    event_type="Interception",
                    location_x=25.0,
                    location_y=40.0,
                ),
                ev(
                    match_id=mid,
                    possession_id=2,
                    team_id=FOCAL,
                    minute=1,
                    second=5,
                    event_type="Pass",
                    location_x=25.0,
                    pass_end_x=100.0,
                ),
                ev(
                    match_id=mid,
                    possession_id=2,
                    team_id=FOCAL,
                    minute=1,
                    second=8,
                    event_type="Shot",
                    location_x=110.0,
                    shot_statsbomb_xg=0.10,
                ),
                # regain-to-loss
                ev(
                    match_id=mid,
                    possession_id=3,
                    team_id=OPP,
                    minute=5,
                    second=0,
                    event_type="Pass",
                    location_x=60.0,
                    pass_end_x=30.0,
                ),
                ev(
                    match_id=mid,
                    possession_id=4,
                    team_id=FOCAL,
                    minute=5,
                    second=1,
                    event_type="Ball Recovery",
                    location_x=20.0,
                    location_y=40.0,
                ),
                ev(
                    match_id=mid,
                    possession_id=4,
                    team_id=FOCAL,
                    minute=5,
                    second=4,
                    event_type="Dispossessed",
                    location_x=22.0,
                ),
            ]
        )
    events = frame(ev_list)
    regains = compute_regain_events(events, focal_team_id=FOCAL)
    assert len(regains) == 4
    agg = aggregate_team_season(regains, _baseline(), n_boot=200, seed=42)
    metric_map = dict(zip(agg["metric"], agg["point_estimate"], strict=False))
    # 2 shots (xG 0.10 each), 4 regains -> xg_per_regain = 0.05
    assert abs(metric_map["xg_per_regain"] - 0.05) < 1e-9
    # median time-to-shot across the 2 shot regains = 7.0
    assert metric_map["time_to_shot_median"] == 7.0
    # 2 losses / 4 regains -> 0.5
    assert metric_map["regain_to_loss_rate"] == 0.5
    # final-third entry from the shot chains -> 2/4 = 0.5
    assert metric_map["regain_to_final_third_rate"] == 0.5
    # rushed = xG<0.05 AND within 8s; our shots xG=0.10 → not rushed; rate = 0.0
    assert metric_map["rushed_shot_rate"] == 0.0
    # n_regains column
    assert int(agg["n_regains"].iloc[0]) == 4
