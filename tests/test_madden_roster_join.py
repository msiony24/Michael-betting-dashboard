import pandas as pd

from engine.madden_roster_join import enrich_madden_players, standardize_roster


def test_join_uses_nfl_roster_team_and_position():
    madden = pd.DataFrame([
        {"player_name": "Justin Jefferson", "birthdate": "1999-06-16", "height": 73, "weight": 195, "overall": 99},
        {"player_name": "Patrick Mahomes II", "birthdate": "1995-09-17", "height": 74, "weight": 225, "overall": 95},
    ])
    roster = pd.DataFrame([
        {"full_name": "Justin Jefferson", "team": "MIN", "position": "WR", "birth_date": "1999-06-16", "gsis_id": "00-0036322", "status": "ACT"},
        {"full_name": "Patrick Mahomes", "team": "KC", "position": "QB", "birth_date": "1995-09-17", "gsis_id": "00-0033873", "status": "ACT"},
    ])
    enriched, report = enrich_madden_players(madden, roster)
    assert enriched.loc[0, "team"] == "MIN"
    assert enriched.loc[0, "position"] == "WR"
    assert enriched.loc[1, "team"] == "KC"
    assert enriched.loc[1, "position"] == "QB"
    assert report["matched_players"] == 2


def test_duplicate_name_prefers_birthdate():
    madden = pd.DataFrame([
        {"player_name": "John Smith", "birthdate": "2000-01-01", "height": 72, "weight": 200, "overall": 80},
    ])
    roster = pd.DataFrame([
        {"full_name": "John Smith", "team": "BUF", "position": "WR", "birth_date": "1998-01-01", "status": "ACT"},
        {"full_name": "John Smith", "team": "DAL", "position": "CB", "birth_date": "2000-01-01", "status": "ACT"},
    ])
    enriched, _ = enrich_madden_players(madden, roster)
    assert enriched.loc[0, "team"] == "DAL"
    assert enriched.loc[0, "position"] == "CB"
    assert enriched.loc[0, "roster_match_method"] == "exact_name_birthdate"
