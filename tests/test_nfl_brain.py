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
