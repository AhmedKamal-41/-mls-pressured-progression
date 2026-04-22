from __future__ import annotations

from pressured_progression.sequences.possession_chain import build_possession_chains
from tests.fixtures import FOCAL, OPP, ev, frame


def test_clean_single_chain():
    events = frame(
        [
            ev(second=0, event_type="Pass", location_x=20.0, pass_end_x=40.0),
            ev(second=2, event_type="Pass", location_x=40.0, pass_end_x=70.0),
            ev(second=5, event_type="Shot", location_x=100.0),
        ]
    )
    chains = build_possession_chains(events)
    assert len(chains) == 1
    row = chains.iloc[0]
    assert row["team_id"] == FOCAL
    assert row["action_count"] == 3
    assert row["start_x"] == 20.0
    assert row["end_x"] == 100.0
    assert row["end_type"] == "shot"


def test_broken_by_out_of_play_new_possession():
    # StatsBomb increments possession_id when ball goes out of play, even if same team restarts.
    events = frame(
        [
            ev(possession_id=1, second=0, event_type="Pass"),
            ev(possession_id=1, second=2, event_type="Pass", location_x=40.0, pass_end_x=50.0),
            ev(possession_id=2, second=10, event_type="Pass", location_x=20.0, pass_end_x=35.0),
        ]
    )
    chains = build_possession_chains(events)
    assert len(chains) == 2
    assert chains.iloc[0]["action_count"] == 2
    assert chains.iloc[1]["action_count"] == 1


def test_broken_by_turnover_team_change():
    events = frame(
        [
            ev(possession_id=1, second=0, team_id=FOCAL, event_type="Pass"),
            ev(
                possession_id=1, second=2, team_id=FOCAL, event_type="Dispossessed", location_x=30.0
            ),
            ev(
                possession_id=2,
                second=3,
                team_id=OPP,
                event_type="Pass",
                location_x=90.0,
                pass_end_x=70.0,
            ),
        ]
    )
    chains = build_possession_chains(events)
    assert len(chains) == 2
    teams = chains["team_id"].tolist()
    assert teams == [FOCAL, OPP]
