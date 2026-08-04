from pathlib import Path

import pandas as pd

from engine.madden_player_mapper import enrich_player_identities, normalize_person_name
from engine.madden_ratings_loader import normalize_players


def test_current_ea_stats_value_schema_uses_roster_identity(monkeypatch):
    records = [
        {
            "firstName": "Lamar",
            "lastName": "Jackson",
            "stats": {
                "overall": {"value": 96, "diff": 1},
                "speed": {"value": 95},
                "awareness": {"value": 94},
            },
        }
    ]
    identity = pd.DataFrame([
        {"name_key": normalize_person_name("Lamar Jackson"), "team": "BAL", "position": "QB", "identity_source": "weekly_rosters.csv"}
    ])

    monkeypatch.setattr(
        "engine.madden_ratings_loader.enrich_player_identities",
        lambda frame: enrich_player_identities(frame, identity),
    )
    players, stats = normalize_players(records)
    assert players.loc[0, "player_name"] == "Lamar Jackson"
    assert players.loc[0, "overall"] == 96
    assert players.loc[0, "speed"] == 95
    assert players.loc[0, "team"] == "BAL"
    assert players.loc[0, "position"] == "QB"
    assert stats["fully_resolved"] == 1


def test_ea_team_and_position_are_preserved(monkeypatch):
    records = [{"firstName": "Example", "lastName": "Player", "team": {"label": "BUF"}, "position": {"label": "WR"}, "stats_overall_value": 80}]
    monkeypatch.setattr(
        "engine.madden_ratings_loader.enrich_player_identities",
        lambda frame: enrich_player_identities(frame, pd.DataFrame(columns=["name_key", "team", "position", "identity_source"])),
    )
    players, _ = normalize_players(records)
    assert players.loc[0, "team"] == "BUF"
    assert players.loc[0, "position"] == "WR"


def test_name_normalization_handles_suffixes_and_punctuation():
    assert normalize_person_name("D.J. Moore Jr.") == normalize_person_name("DJ Moore")
