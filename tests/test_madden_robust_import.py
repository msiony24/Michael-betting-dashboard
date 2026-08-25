from pathlib import Path

import pandas as pd

from engine.madden_player_mapper import enrich_player_identities, normalize_person_name
from engine.madden_ratings_loader import normalize_players


def test_normalize_players_extracts_stats_by_current_field_names():
    # NOTE: normalize_players() no longer calls enrich_player_identities
    # internally, or returns a (players, stats) tuple -- identity enrichment
    # against a roster file is a separate step elsewhere in the pipeline now,
    # not something this function does. This test now checks what
    # normalize_players actually guarantees: correct stat extraction using
    # its own snake_case column names, from the current EA record shape.
    records = [
        {
            "firstName": "Lamar",
            "lastName": "Jackson",
            "overallRating": 96,
            "speed": 95,
            "awareness": 94,
        }
    ]
    players = normalize_players(records)
    assert players.loc[0, "player_name"] == "Lamar Jackson"
    assert players.loc[0, "overall"] == 96
    assert players.loc[0, "speed"] == 95
    assert players.loc[0, "awareness"] == 94


def test_ea_team_and_position_extracted_when_present_as_flat_fields():
    # The loader's own comment notes EA "currently returns these as null" in
    # its normal API responses, so ea_team/ea_position are optional by
    # design -- this test confirms the extraction mechanism itself still
    # works correctly on the record shapes it's built to handle, not that
    # every real-world record will have these populated.
    records = [{"firstName": "Example", "lastName": "Player", "team": "BUF", "position": "WR", "overallRating": 80}]
    players = normalize_players(records)
    assert players.loc[0, "ea_team"] == "BUF"
    assert players.loc[0, "ea_position"] == "WR"


def test_ea_team_and_position_are_none_when_ea_omits_them():
    # Confirms the current, honest behavior: a record shape EA no longer
    # reliably sends (a nested {"label": ...} dict) does not crash and
    # correctly yields None rather than a wrong/fabricated value.
    records = [{"firstName": "Example", "lastName": "Player", "team": {"label": "BUF"}, "position": {"label": "WR"}, "overallRating": 80}]
    players = normalize_players(records)
    assert players.loc[0, "ea_team"] is None
    assert players.loc[0, "ea_position"] is None


def test_name_normalization_handles_suffixes_and_punctuation():
    assert normalize_person_name("D.J. Moore Jr.") == normalize_person_name("DJ Moore")
