import json
from pathlib import Path

import pandas as pd

from engine import madden_ratings_loader as loader
from engine import madden_team_builder as builder
from engine import nfl_rating_engine
from engine import nfl_ratings_loader


def test_default_paths_point_to_madden_27():
    assert loader.DEFAULT_CSV_PATH.name == "madden_27_players.csv"
    assert loader.DEFAULT_RAW_PATH.name == "madden_27_raw.json"
    assert loader.DEFAULT_METADATA_PATH.name == "madden_27_metadata.json"
    assert builder.DEFAULT_PLAYERS_PATH.name == "madden_27_players.csv"
    assert builder.DEFAULT_OUTPUT_PATH.name == "madden_27_team_ratings.json"
    assert nfl_rating_engine.DEFAULT_MADDEN_PATH.name == "madden_27_players.csv"
    assert nfl_ratings_loader.DEFAULT_MADDEN_RATINGS_PATH.name == "madden_27_team_ratings.json"


def test_builder_labels_source_as_madden_27():
    players = pd.DataFrame([
        {"player_name": "QB One", "team": "Buffalo Bills", "position": "QB", "overall": 95},
        {"player_name": "QB Two", "team": "Buffalo Bills", "position": "QB", "overall": 72},
        {"player_name": "WR One", "team": "Buffalo Bills", "position": "WR", "overall": 91},
        {"player_name": "TE One", "team": "Buffalo Bills", "position": "TE", "overall": 84},
        {"player_name": "LT One", "team": "Buffalo Bills", "position": "LT", "overall": 86},
        {"player_name": "LG One", "team": "Buffalo Bills", "position": "LG", "overall": 82},
        {"player_name": "C One", "team": "Buffalo Bills", "position": "C", "overall": 83},
        {"player_name": "RG One", "team": "Buffalo Bills", "position": "RG", "overall": 81},
        {"player_name": "RT One", "team": "Buffalo Bills", "position": "RT", "overall": 85},
        {"player_name": "CB One", "team": "Buffalo Bills", "position": "CB", "overall": 88},
    ])
    ratings = builder.build_team_ratings(players)
    assert ratings["Buffalo Bills"]["source"] == "Madden NFL 27"
    assert ratings["Buffalo Bills"]["units"]["quarterback"]["starter_grade"] == 95.0


def test_normalize_players_keeps_detailed_attributes(monkeypatch):
    records = [{
        "firstName": "Test",
        "lastName": "Quarterback",
        "overallRating": 90,
        "team": {"label": "Buffalo Bills"},
        "position": {"label": "QB"},
        "stats": {
            "throwPower": {"value": 95},
            "throwAccuracyDeep": {"value": 91},
            "throwUnderPressure": {"value": 89},
        },
    }]
    monkeypatch.setattr(loader, "enrich_player_identities", lambda frame: (frame, {"fully_resolved": len(frame)}))
    frame, _ = loader.normalize_players(records)
    assert frame.iloc[0]["overall"] == 90
    assert frame.iloc[0]["stats_throwPower_value"] == 95
    assert frame.iloc[0]["stats_throwAccuracyDeep_value"] == 91
    assert frame.iloc[0]["stats_throwUnderPressure_value"] == 89
