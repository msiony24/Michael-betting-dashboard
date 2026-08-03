from engine.nfl_reasoning import challenge_reasoning


EAGLES = {
    "quarterback": 88,
    "offensive_line": 92,
    "skill_positions": 86,
    "offense": 89,
    "defense": 87,
    "defensive_line": 93,
    "secondary": 82,
    "coaching": 86,
}

COWBOYS = {
    "quarterback": 85,
    "offensive_line": 79,
    "skill_positions": 84,
    "offense": 84,
    "defense": 83,
    "defensive_line": 84,
    "secondary": 85,
    "coaching": 80,
}


def test_confirms_specific_trench_reasoning():
    result = challenge_reasoning(
        reasoning="I like the Eagles because their offensive line should control Dallas and keep the pass rush away from the quarterback.",
        selected_team="Philadelphia Eagles",
        opponent="Dallas Cowboys",
        selected_profile=EAGLES,
        opponent_profile=COWBOYS,
        projected_winner="Philadelphia Eagles",
        selected_is_home=True,
    )
    assert result["verdict"] in {"Confirmed", "Mostly confirmed"}
    assert any(point["topic"] == "protection_vs_pass_rush" for point in result["points"])
    assert result["confidence_adjustment"] > 0


def test_pushes_back_when_matchup_disagrees():
    result = challenge_reasoning(
        reasoning="I like Dallas because its offensive line will dominate the Eagles pass rush.",
        selected_team="Dallas Cowboys",
        opponent="Philadelphia Eagles",
        selected_profile=COWBOYS,
        opponent_profile=EAGLES,
        projected_winner="Philadelphia Eagles",
        selected_is_home=False,
    )
    assert result["verdict"] == "Macabets disagrees"
    assert any(point["status"] == "Pushback" for point in result["points"])


def test_flags_unverified_injury_claim():
    result = challenge_reasoning(
        reasoning="I like the Eagles because Dallas is injured and Philadelphia has the better offensive line.",
        selected_team="Philadelphia Eagles",
        opponent="Dallas Cowboys",
        selected_profile=EAGLES,
        opponent_profile=COWBOYS,
        projected_winner="Philadelphia Eagles",
        selected_is_home=True,
    )
    assert result["assumptions"]
    assert "current external data check" in result["assumptions"][0]
