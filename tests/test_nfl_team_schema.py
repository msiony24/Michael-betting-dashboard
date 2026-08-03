from engine.nfl_team_schema import profile_from_legacy_components


def _legacy(**overrides):
    values = {
        "quarterback": 82,
        "offense": 79,
        "defense": 77,
        "coaching": 80,
        "offensive_line": 78,
        "defensive_line": 76,
        "skill_positions": 81,
        "secondary": 75,
        "special_teams": 72,
        "continuity": 74,
    }
    values.update(overrides)
    return values


def test_legacy_adapter_produces_complete_stable_schema():
    profile = profile_from_legacy_components("Example", _legacy())

    assert profile.team == "Example"
    assert profile.offense.quarterback == 82
    assert 0 <= profile.offense.pass_protection <= 100
    assert 0 <= profile.defense.linebacker_coverage <= 100
    assert profile.source == "legacy_components_adapter"
    assert profile.data_quality == "complete_legacy"


def test_future_detailed_fields_are_not_required_from_legacy_source():
    profile = profile_from_legacy_components("Example", _legacy())
    payload = profile.to_dict()

    assert "pass_protection" in payload["offense"]
    assert "run_blocking" in payload["offense"]
    assert "pass_rush" in payload["defense"]
    assert "cornerbacks" in payload["defense"]
    assert "depth" in payload
