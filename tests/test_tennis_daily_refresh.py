from __future__ import annotations

import pandas as pd

from update_tennis_data import (
    MATCH_COLUMNS,
    convert_api_fixtures,
    merge_live_matches,
    player_signature,
)


def _blank_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in MATCH_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[MATCH_COLUMNS]


def test_player_signature_handles_provider_name_formats():
    assert player_signature("Tirante T.A.") == ("tirante", "t")
    assert player_signature("T. A. Tirante") == ("tirante", "t")
    assert player_signature("Thiago Agustin Tirante") == ("tirante", "t")
    assert player_signature("Fritz T.") == ("fritz", "t")
    assert player_signature("Taylor Fritz") == ("fritz", "t")
    assert player_signature("Felix Auger-Aliassime") == ("aliassime", "f")
    assert player_signature("Auger-Aliassime F.") == ("aliassime", "f")
    assert player_signature("Juan Manuel Cerundolo") == ("cerundolo", "j")
    assert player_signature("Cerundolo J.M.") == ("cerundolo", "j")


def test_convert_api_fixture_maps_winner_score_and_existing_names():
    fixtures = [{
        "event_date": "2026-08-05",
        "event_first_player": "T. A. Tirante",
        "first_player_key": "101",
        "event_second_player": "T. Fritz",
        "second_player_key": "202",
        "event_winner": "First Player",
        "event_status": "Finished",
        "event_type_type": "Atp Singles",
        "tournament_name": "ATP Montreal, Canada Men Singles",
        "tournament_round": "ATP Montreal, Canada Men Singles - Round of 64",
        "scores": [
            {"score_first": "7", "score_second": "5", "score_set": "1"},
            {"score_first": "6", "score_second": "3", "score_set": "2"},
        ],
        "statistics": [],
    }]
    existing_names = {
        ("tirante", "t"): "Tirante T.A.",
        ("fritz", "t"): "Fritz T.",
    }
    frame = convert_api_fixtures(
        fixtures,
        existing_names=existing_names,
        historical_surfaces={"canadian open": "Hard"},
        ranks_by_key={"101": 50.0, "202": 4.0},
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["winner_name"] == "Tirante T.A."
    assert row["loser_name"] == "Fritz T."
    assert row["score"] == "7-5 6-3"
    assert row["round"] == "R64"
    assert row["surface"] == "Hard"
    assert row["winner_rank"] == 50.0
    assert row["loser_rank"] == 4.0


def test_merge_live_matches_replaces_stale_duplicate():
    baseline = _blank_frame([{
        "tourney_date": "20260805",
        "tourney_name": "Montreal",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R64",
        "winner_name": "Fritz T.",
        "loser_name": "Tirante T.A.",
        "score": "6-4 6-4",
    }])
    live = _blank_frame([{
        "tourney_date": "20260805",
        "tourney_name": "ATP Montreal, Canada Men Singles",
        "surface": "Hard",
        "tourney_level": "A",
        "round": "R64",
        "winner_name": "Tirante T.A.",
        "loser_name": "Fritz T.",
        "score": "7-5 6-3",
    }])
    merged = merge_live_matches(baseline, live)
    assert len(merged) == 1
    assert merged.iloc[0]["winner_name"] == "Tirante T.A."
    assert merged.iloc[0]["score"] == "7-5 6-3"
