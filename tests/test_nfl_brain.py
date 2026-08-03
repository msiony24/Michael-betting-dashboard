from engine.nfl_brain import build_matchup_brain


def _components(**overrides):
    base = {
        "quarterback": 75,
        "offensive_line": 75,
        "defense": 75,
        "offense": 75,
        "recent_form": 75,
        "coaching": 75,
        "defensive_line": 75,
        "secondary": 75,
        "skill_positions": 75,
        "special_teams": 75,
        "continuity": 75,
    }
    base.update(overrides)
    return base


def test_brain_builds_direct_conflicts_for_both_offenses():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(),
        home_components=_components(),
    )

    names = [item["name"] for item in result["conflicts"]]
    assert names.count("Quarterback vs coverage") == 2
    assert names.count("Pass protection vs defensive front") == 2
    assert result["matchup_leader"] == "Even"


def test_protection_and_qb_edges_create_chain_reaction():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(defensive_line=65, secondary=65, defense=67),
        home_components=_components(quarterback=92, offensive_line=91, offense=90, skill_positions=88),
    )

    assert result["matchup_leader"] == "Home"
    assert any(
        chain["team"] == "Home" and "Protection and passing" in chain["trigger"]
        for chain in result["chain_reactions"]
    )


def test_brain_does_not_invent_scheme_or_injury_claims():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(),
        home_components=_components(quarterback=85),
    )

    text = " ".join(
        conflict["explanation"] + " " + conflict["consequence"]
        for conflict in result["conflicts"]
    ).lower()
    assert "blitz rate" not in text
    assert "injury" not in text
    assert result["limitations"]


def test_brain_exposes_schema_exploits_and_paths():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(defensive_line=66, secondary=65, defense=67),
        home_components=_components(
            quarterback=91,
            offensive_line=89,
            offense=88,
            skill_positions=90,
        ),
    )

    assert result["version"] == "NFL Brain v0.2-schema"
    assert set(result["team_profiles"]) == {"Away", "Home"}
    assert set(result["qb_gates"]) == {"Away", "Home"}
    assert result["exploits"]
    assert result["win_conditions"]["Home"]["realism_score"] >= 50
    assert result["failure_conditions"]["Away"]["threat_team"] == "Home"
    assert result["data_contract"]["status"] == "Ready for upgraded ratings"


def test_qb_gate_can_fail_when_environment_is_badly_overmatched():
    result = build_matchup_brain(
        away_team="Away",
        home_team="Home",
        away_components=_components(
            quarterback=60,
            offensive_line=61,
            skill_positions=63,
            offense=62,
        ),
        home_components=_components(
            defensive_line=92,
            secondary=91,
            defense=92,
        ),
    )

    assert result["qb_gates"]["Away"]["verdict"] == "Fail"
    assert "Quarterback environment" in result["failure_conditions"]["Away"]["title"]
