from engine.nfl_brain import build_matchup_brain


def _components(**overrides):
    base = {
        "quarterback": 80,
        "offense": 78,
        "defense": 77,
        "coaching": 79,
        "offensive_line": 78,
        "defensive_line": 78,
        "skill_positions": 78,
        "secondary": 77,
        "special_teams": 75,
        "continuity": 76,
    }
    base.update(overrides)
    return base


def test_legacy_data_is_blocked_instead_of_generating_conflicts():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(),
        home_components=_components(),
    )

    assert result["status"] == "blocked_by_data_quality"
    assert result["conflicts"] == []
    assert result["matchup_leader"] == "Unavailable"


def test_legacy_rating_gaps_do_not_create_chain_reactions():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(defensive_line=65, secondary=65, defense=67),
        home_components=_components(quarterback=92, offensive_line=91, offense=90, skill_positions=88),
    )

    assert result["matchup_leader"] == "Unavailable"
    assert result["chain_reactions"] == []
    assert result["data_contract"]["allowed_to_influence_prediction"] is False


def test_brain_exposes_schema_and_strict_decision_framework():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(),
        home_components=_components(),
    )

    assert result["version"] == "NFL Brain v0.3-strict-data-gate"
    assert set(result["team_profiles"]) == {"Away", "Home"}
    assert result["decision_framework"]["status"] == "blocked_by_data_quality"
    assert len(result["decision_framework"]["questions"]) == 8


def test_qb_gate_is_not_claimed_from_provisional_data():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(quarterback=60, offensive_line=61),
        home_components=_components(defensive_line=92, secondary=91, defense=92),
    )

    assert result["qb_gates"] == {}
    assert all(
        question["answer"] == "Insufficient current data"
        for question in result["decision_framework"]["questions"]
    )
