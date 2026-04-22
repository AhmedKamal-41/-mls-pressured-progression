from __future__ import annotations

from pressured_progression.labeling.buildup_failure import label_buildup_failures
from pressured_progression.sequences.possession_chain import build_possession_chains
from tests.fixtures import OPP, ev, frame


def _label(events_df):
    chains = build_possession_chains(events_df)
    return label_buildup_failures(chains, events_df)


def test_turnover_own_half():
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=25.0,
                under_pressure=True,
            ),
            ev(
                possession_id=1,
                second=2,
                event_type="Pass",
                location_x=25.0,
                pass_end_x=28.0,
                under_pressure=True,
            ),
            ev(possession_id=1, second=5, event_type="Dispossessed", location_x=30.0),
            # Opponent possession starts, no shot.
            ev(
                possession_id=2,
                second=7,
                team_id=OPP,
                event_type="Pass",
                location_x=90.0,
                pass_end_x=70.0,
            ),
        ]
    )
    labels = _label(events)
    assert len(labels) == 1
    r = labels.iloc[0]
    assert r["is_failure"]
    assert r["failure_type"] == "turnover_own_half"


def test_forced_long_ball():
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=15.0,
                pass_end_x=20.0,
                location_y=40.0,
                pass_end_y=40.0,
                under_pressure=True,
            ),
            # Long ball: x=20 -> x=85, distance ~65m (> 40)
            ev(
                possession_id=1,
                second=3,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=85.0,
                location_y=40.0,
                pass_end_y=40.0,
            ),
            ev(
                possession_id=1,
                second=6,
                event_type="Ball Receipt*",
                location_x=85.0,
                outcome="Incomplete",
            ),
            # Opp takes over, no shot within 10s.
            ev(
                possession_id=2,
                second=9,
                team_id=OPP,
                event_type="Pass",
                location_x=35.0,
                pass_end_x=60.0,
            ),
        ]
    )
    labels = _label(events)
    assert len(labels) == 1
    assert labels.iloc[0]["failure_type"] == "forced_long_ball"


def test_opp_shot_within_10s():
    # Turnover is OUT of own half (x=70) so turnover_own_half won't match; opp shot within 10s.
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=45.0,
                under_pressure=True,
            ),
            ev(possession_id=1, second=2, event_type="Pass", location_x=45.0, pass_end_x=70.0),
            ev(possession_id=1, second=4, event_type="Dispossessed", location_x=70.0),
            ev(
                possession_id=2,
                second=8,
                team_id=OPP,
                event_type="Shot",
                location_x=110.0,
                shot_statsbomb_xg=0.25,
            ),
        ]
    )
    labels = _label(events)
    assert len(labels) == 1
    assert labels.iloc[0]["failure_type"] == "opp_shot_within_10s"


def test_backward_reset_turnover():
    # Backward pass early, no long balls in chain, chain exits own half before the
    # loss so turnover_own_half can't fire, and no opp shot within 10s.
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=35.0,
                under_pressure=True,
            ),
            # Backward pass: 35 -> 20 (delta = -15 < -5)
            ev(possession_id=1, second=3, event_type="Pass", location_x=35.0, pass_end_x=20.0),
            # Short forward passes — none > 40m.
            ev(possession_id=1, second=6, event_type="Pass", location_x=20.0, pass_end_x=55.0),
            ev(possession_id=1, second=9, event_type="Pass", location_x=55.0, pass_end_x=75.0),
            # Terminal event past midfield (x=75) — turnover_own_half check at x<60 must miss.
            ev(
                possession_id=1,
                second=11,
                event_type="Ball Receipt*",
                location_x=75.0,
                outcome="Incomplete",
            ),
            # Opp takes over; no shot within 10s.
            ev(
                possession_id=2,
                second=13,
                team_id=OPP,
                event_type="Pass",
                location_x=45.0,
                pass_end_x=60.0,
            ),
        ]
    )
    labels = _label(events)
    assert len(labels) == 1
    assert labels.iloc[0]["failure_type"] == "backward_reset_turnover"


def test_happy_path_no_failure():
    # Qualifies (starts x<40, pressure in first 3), progresses to shot — no failure.
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=35.0,
                under_pressure=True,
            ),
            ev(possession_id=1, second=2, event_type="Pass", location_x=35.0, pass_end_x=65.0),
            ev(possession_id=1, second=5, event_type="Pass", location_x=65.0, pass_end_x=95.0),
            ev(
                possession_id=1,
                second=8,
                event_type="Shot",
                location_x=110.0,
                shot_statsbomb_xg=0.10,
            ),
            # Possession flips after shot (GK catch), but that's not a build-up failure.
            ev(
                possession_id=2,
                second=12,
                team_id=OPP,
                event_type="Pass",
                location_x=5.0,
                pass_end_x=30.0,
            ),
        ]
    )
    labels = _label(events)
    assert len(labels) == 1
    r = labels.iloc[0]
    assert not r["is_failure"]
    assert r["failure_type"] == "none"


def test_does_not_qualify_origin_past_halfway():
    # Chain starts at x=50 — never qualifies.
    events = frame(
        [
            ev(
                possession_id=1,
                second=0,
                event_type="Pass",
                location_x=50.0,
                pass_end_x=65.0,
                under_pressure=True,
            ),
            ev(possession_id=1, second=2, event_type="Dispossessed", location_x=55.0),
            ev(
                possession_id=2,
                second=5,
                team_id=OPP,
                event_type="Pass",
                location_x=65.0,
                pass_end_x=40.0,
            ),
        ]
    )
    labels = _label(events)
    assert labels.empty


def test_does_not_qualify_no_pressure():
    events = frame(
        [
            ev(possession_id=1, second=0, event_type="Pass", location_x=20.0, pass_end_x=35.0),
            ev(possession_id=1, second=2, event_type="Pass", location_x=35.0, pass_end_x=55.0),
            ev(possession_id=1, second=5, event_type="Dispossessed", location_x=55.0),
            ev(
                possession_id=2,
                second=7,
                team_id=OPP,
                event_type="Pass",
                location_x=65.0,
                pass_end_x=50.0,
            ),
        ]
    )
    labels = _label(events)
    assert labels.empty
