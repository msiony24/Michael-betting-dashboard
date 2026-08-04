from engine.nfl_brain import build_matchup_brain


COMPONENTS = {
    "quarterback": 90,
    "offense": 88,
    "defense": 86,
    "coaching": 84,
    "offensive_line": 87,
    "defensive_line": 85,
    "skill_positions": 89,
    "secondary": 83,
    "special_teams": 78,
    "continuity": 82,
}


def test_provisional_legacy_data_generates_no_matchup_claims():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=COMPONENTS,
        home_components=COMPONENTS,
    )

    assert result["status"] == "blocked_by_data_quality"
    assert result["matchup_leader"] == "Unavailable"
    assert result["qb_gates"] == {}
    assert result["exploits"] == []
    assert result["win_conditions"] == {}
    assert result["conflicts"] == []
    assert result["data_contract"]["allowed_to_influence_prediction"] is False
