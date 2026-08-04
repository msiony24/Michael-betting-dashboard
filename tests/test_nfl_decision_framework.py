from engine.nfl_decision_framework import build_decision_framework
from engine.nfl_team_schema import profile_from_legacy_components


LEGACY = {
    "quarterback": 85,
    "offense": 82,
    "defense": 80,
    "coaching": 84,
    "offensive_line": 81,
    "defensive_line": 83,
    "skill_positions": 82,
    "secondary": 79,
    "special_teams": 75,
    "continuity": 80,
}


def test_legacy_profiles_block_all_eight_questions():
    away = profile_from_legacy_components("Away", LEGACY)
    home = profile_from_legacy_components("Home", LEGACY)
    result = build_decision_framework(away, home)

    assert result["status"] == "blocked_by_data_quality"
    assert result["prediction_influence"] == "disabled"
    assert len(result["questions"]) == 8
    assert all(q["answer"] == "Insufficient current data" for q in result["questions"])
    assert all(q["can_influence_prediction"] is False for q in result["questions"])
