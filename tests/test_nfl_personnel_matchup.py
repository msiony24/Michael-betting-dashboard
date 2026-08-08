import json
from pathlib import Path

from engine.nfl_personnel_matchup import build_personnel_matchup_context


def test_personnel_context_has_six_directional_matchups(tmp_path: Path):
    units_a = {
        "quarterback": {"grade": 82}, "running_backs": {"grade": 78},
        "receiving_weapons": {"grade": 84}, "offensive_line": {"grade": 80},
        "defensive_front": {"grade": 79}, "linebackers": {"grade": 77},
        "secondary": {"grade": 76},
    }
    units_b = {
        "quarterback": {"grade": 88, "source": "40% roster + 60% NFL performance"},
        "running_backs": {"grade": 83}, "receiving_weapons": {"grade": 86},
        "offensive_line": {"grade": 85}, "defensive_front": {"grade": 90},
        "linebackers": {"grade": 82}, "secondary": {"grade": 87},
    }
    path = tmp_path / "team_ratings_auto.json"
    path.write_text(json.dumps({"Away": {"units": units_a}, "Home": {"units": units_b}}))
    result = build_personnel_matchup_context(
        away_team="Away", home_team="Home", week=6,
        team_ratings_path=path, madden_ratings_path=path,
    )
    assert result["available"] is True
    assert len(result["matchups"]) == 6
    assert -1.5 <= result["home_margin_adjustment"] <= 1.5
