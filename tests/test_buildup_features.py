from __future__ import annotations

import pandas as pd

from pressured_progression.features.buildup_features import (
    FEATURE_COLUMNS,
    extract_buildup_features,
)
from pressured_progression.sequences.possession_chain import build_possession_chains
from tests.fixtures import FOCAL, OPP, ev, frame


def _mk_chain_match(focal_scores: int = 0, opp_scores: int = 0, start_minute: int = 10):
    """Build a minimal events frame for one qualifying chain:
    pass 1 (pressured), ball receipt, pass 2 -- then opp pass.
    Optional goals scored earlier in the match to exercise game-state.
    """
    evs: list[dict] = []
    # Earlier-match prior own-team passes to feed recent_pass_reach
    for i in range(3):
        evs.append(
            ev(
                possession_id=1,
                minute=start_minute - 2,
                second=i * 2,
                event_type="Pass",
                location_x=20.0 + i * 5,
                location_y=40.0,
                pass_end_x=30.0 + i * 5,
                pass_end_y=40.0,
            )
        )
    # Optional pre-chain goals
    gm = start_minute - 1
    for _ in range(focal_scores):
        evs.append(
            ev(
                possession_id=1,
                minute=gm,
                second=0,
                event_type="Shot",
                outcome="Goal",
                location_x=100.0,
            )
        )
    for _ in range(opp_scores):
        evs.append(
            ev(
                possession_id=1,
                team_id=OPP,
                minute=gm,
                second=1,
                event_type="Shot",
                outcome="Goal",
                location_x=100.0,
            )
        )
    # Opponent defensive action before chain to feed opp_press_height
    evs.append(
        ev(
            possession_id=1,
            team_id=OPP,
            minute=start_minute,
            second=0,
            event_type="Pressure",
            location_x=90.0,  # opp's perspective: deep in our half
            location_y=40.0,
        )
    )
    # The chain itself (possession=2 to ensure clean new chain)
    evs.extend(
        [
            ev(
                possession_id=2,
                player_id=1,
                minute=start_minute,
                second=10,
                event_type="Pass",
                location_x=20.0,
                location_y=40.0,
                pass_end_x=35.0,
                pass_end_y=40.0,
                pass_recipient_id=7,
                under_pressure=True,
            ),
            ev(
                possession_id=2,
                player_id=7,
                minute=start_minute,
                second=11,
                event_type="Ball Receipt*",
                location_x=35.0,
                location_y=40.0,
            ),
            ev(
                possession_id=2,
                player_id=7,
                minute=start_minute,
                second=13,
                event_type="Pass",
                location_x=35.0,
                location_y=40.0,
                pass_end_x=60.0,
                pass_end_y=40.0,
                pass_recipient_id=9,
            ),
            # Opp pressure 1s after receipt (within binary window)
            ev(
                possession_id=2,
                team_id=OPP,
                minute=start_minute,
                second=12,
                event_type="Pressure",
                location_x=85.0,
                location_y=40.0,
            ),
            # Terminal: dispossession in own half so the chain qualifies + is a failure
            ev(
                possession_id=2,
                player_id=9,
                minute=start_minute,
                second=15,
                event_type="Dispossessed",
                location_x=45.0,
                location_y=40.0,
            ),
            # Opp possession
            ev(
                possession_id=3,
                team_id=OPP,
                minute=start_minute,
                second=17,
                event_type="Pass",
                location_x=75.0,
                pass_end_x=60.0,
            ),
        ]
    )
    return pd.DataFrame(evs)


def test_happy_path_all_features_computed():
    events = _mk_chain_match(focal_scores=1, opp_scores=2, start_minute=22)
    chains = build_possession_chains(events)
    # Focal team chain only
    chains = chains[chains["team_id"] == FOCAL]
    feats = extract_buildup_features(chains[chains["possession_id"] == 2], events)

    assert len(feats) == 1
    row = feats.iloc[0]
    # All expected columns present
    for c in FEATURE_COLUMNS:
        assert c in feats.columns, f"missing column {c}"
    # Feature spot-checks
    assert row["defthird_pressure_density"] >= 0
    assert row["first_receiver_pressure_binary"] in (True, False)
    # Opp pressure at second=12, receipt at second=11 → gap 1.0s
    assert row["first_receiver_pressure_seconds"] == 1.0
    assert bool(row["first_receiver_pressure_binary"]) is True
    # recent_pass_reach: three prior passes all length 10 → mean 10
    assert row["recent_pass_reach"] == 10.0
    # opp_press_height: one opp pressure at x=90, within 30s prior
    assert row["opp_press_height"] == 90.0
    # Game state: focal 1 - opp 2 = -1 → minus1
    assert row["sd_minus1"] == 1
    # Minute 22 → min_16_30
    assert row["min_16_30"] == 1
    # support_density_ff dropped from the feature set (spec §9) — column should not exist.
    assert "support_density_ff" not in feats.columns


def test_no_prior_passes_recent_reach_zero():
    # Only the chain itself — no earlier passes → recent_pass_reach == 0
    events = pd.DataFrame(
        [
            ev(
                possession_id=1,
                player_id=1,
                minute=5,
                second=0,
                event_type="Pass",
                location_x=20.0,
                pass_end_x=35.0,
                pass_recipient_id=7,
                under_pressure=True,
            ),
            ev(
                possession_id=1,
                player_id=7,
                minute=5,
                second=1,
                event_type="Ball Receipt*",
                location_x=35.0,
            ),
            ev(
                possession_id=1,
                player_id=7,
                minute=5,
                second=3,
                event_type="Dispossessed",
                location_x=35.0,
            ),
            ev(
                possession_id=2,
                team_id=OPP,
                minute=5,
                second=5,
                event_type="Pass",
                location_x=85.0,
                pass_end_x=70.0,
            ),
        ]
    )
    chains = build_possession_chains(events)
    chains = chains[chains["team_id"] == FOCAL]
    feats = extract_buildup_features(chains, events)
    assert len(feats) == 1
    r = feats.iloc[0]
    assert r["recent_pass_reach"] == 0.0
    # No prior opp defensive actions → opp_press_height null
    assert pd.isna(r["opp_press_height"])
    # No first receiver betweenness edges with single pass → 0.0
    assert r["first_receiver_betweenness"] == 0.0


def test_score_diff_credits_own_goals():
    # Pre-chain: focal benefits from an "Own Goal For" (scored by opp on themselves);
    # opp scores a normal Shot → Goal. Expected: score_for=1, score_against=1, sd_zero fires.
    evs = [
        # Prior own-team passes so recent_pass_reach has material (not the point of this test)
        ev(
            possession_id=1, minute=3, second=0, event_type="Pass", location_x=20.0, pass_end_x=30.0
        ),
        ev(
            possession_id=1, minute=3, second=2, event_type="Pass", location_x=30.0, pass_end_x=40.0
        ),
        ev(
            possession_id=1, minute=3, second=4, event_type="Pass", location_x=40.0, pass_end_x=50.0
        ),
        # Own goal crediting focal (team_id=FOCAL on the Own Goal For row)
        ev(possession_id=1, minute=4, second=0, event_type="Own Goal For", location_x=100.0),
        # Regular goal by opponent
        ev(
            possession_id=1,
            team_id=OPP,
            minute=4,
            second=30,
            event_type="Shot",
            outcome="Goal",
            location_x=100.0,
        ),
        # The chain, starting minute 5 in own half under pressure
        ev(
            possession_id=2,
            player_id=1,
            minute=5,
            second=0,
            event_type="Pass",
            location_x=20.0,
            pass_end_x=35.0,
            pass_recipient_id=7,
            under_pressure=True,
        ),
        ev(
            possession_id=2,
            player_id=7,
            minute=5,
            second=1,
            event_type="Ball Receipt*",
            location_x=35.0,
        ),
        ev(
            possession_id=2,
            player_id=7,
            minute=5,
            second=4,
            event_type="Dispossessed",
            location_x=35.0,
        ),
        ev(
            possession_id=3,
            team_id=OPP,
            minute=5,
            second=7,
            event_type="Pass",
            location_x=90.0,
            pass_end_x=70.0,
        ),
    ]
    events = frame(evs)
    chains = build_possession_chains(events)
    chains = chains[chains["team_id"] == FOCAL]
    feats = extract_buildup_features(chains[chains["possession_id"] == 2], events)
    assert len(feats) == 1
    r = feats.iloc[0]
    # 1 OG For (focal) + 1 Shot Goal (opp) → score 1-1 → sd_zero
    assert r["sd_zero"] == 1
    assert r["sd_plus1"] == 0
    assert r["sd_minus1"] == 0


def test_empty_chains_returns_empty():
    # Need at least one event to pass schema validation for the events param.
    events = pd.DataFrame([ev(possession_id=1, event_type="Pass")])
    empty_chains = pd.DataFrame(
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
    feats = extract_buildup_features(empty_chains, events)
    assert feats.empty
