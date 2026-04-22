from __future__ import annotations

import math

from pressured_progression.sequences.regain import detect_regains
from tests.fixtures import FOCAL, OPP, ev, frame


def test_tackle_leads_to_shot_within_window():
    events = frame(
        [
            # Opp has possession 1.
            ev(
                possession_id=1,
                team_id=OPP,
                minute=10,
                second=0,
                event_type="Pass",
                location_x=60.0,
                pass_end_x=40.0,
            ),
            # Focal intercepts — starts possession 2 for focal.
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=1,
                event_type="Interception",
                location_x=30.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=4,
                event_type="Pass",
                location_x=35.0,
                pass_end_x=80.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=10,
                second=8,
                event_type="Shot",
                location_x=100.0,
                shot_statsbomb_xg=0.15,
            ),
        ]
    )
    regains = detect_regains(events, focal_team_id=FOCAL)
    assert len(regains) == 1
    r = regains.iloc[0]
    assert r["regain_zone"] == "defensive"
    assert r["regaining_team_id"] == FOCAL
    assert r["subsequent_possession_id"] == 2
    assert math.isclose(r["next_shot_seconds"], 7.0)
    assert math.isclose(r["next_shot_xg"], 0.15)


def test_interception_then_loss_no_shot():
    events = frame(
        [
            ev(
                possession_id=1,
                team_id=OPP,
                minute=5,
                second=0,
                event_type="Pass",
                location_x=60.0,
                pass_end_x=30.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=5,
                second=1,
                event_type="Interception",
                location_x=25.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=5,
                second=3,
                event_type="Pass",
                location_x=25.0,
                pass_end_x=30.0,
                outcome="Incomplete",
            ),
            ev(
                possession_id=3,
                team_id=OPP,
                minute=5,
                second=5,
                event_type="Pass",
                location_x=90.0,
                pass_end_x=80.0,
            ),
        ]
    )
    regains = detect_regains(events, focal_team_id=FOCAL)
    assert len(regains) == 1
    r = regains.iloc[0]
    assert r["next_shot_seconds"] is None
    assert r["next_shot_xg"] is None


def test_ball_recovery_stalled_past_window():
    # Focal recovers, holds ball, eventually shoots 20s later — outside 15s window.
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
                second=2,
                event_type="Ball Recovery",
                location_x=45.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=5,
                event_type="Pass",
                location_x=45.0,
                pass_end_x=60.0,
            ),
            ev(
                possession_id=2,
                team_id=FOCAL,
                minute=20,
                second=25,
                event_type="Shot",
                location_x=100.0,
                shot_statsbomb_xg=0.05,
            ),
        ]
    )
    regains = detect_regains(events, focal_team_id=FOCAL)
    assert len(regains) == 1
    r = regains.iloc[0]
    assert r["regain_zone"] == "middle"
    assert r["next_shot_seconds"] is None
    assert r["next_shot_xg"] is None
