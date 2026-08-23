from __future__ import annotations

import pytest

from engine.nfl import (
    _decisive_factors,
    _market_no_vig_home_probability,
    american_to_probability,
    analyze,
    probability_to_american,
    spread_to_home_probability,
    team_power_score,
)
from engine.nfl_data import NFL_TEAM_RATINGS, TEAM_RATING_WEIGHTS


# --- pure odds math -----------------------------------------------------------

def test_american_to_probability_known_values():
    assert american_to_probability(-150) == pytest.approx(0.6)
    assert american_to_probability(150) == pytest.approx(0.4)
    assert american_to_probability(-100) == pytest.approx(0.5)


def test_american_to_probability_zero_raises():
    with pytest.raises(ValueError):
        american_to_probability(0)


def test_probability_to_american_boundary_at_half():
    assert probability_to_american(0.5) == -100


def test_probability_to_american_round_trips():
    for p in (0.2, 0.4, 0.5, 0.6, 0.8):
        odds = probability_to_american(p)
        recovered = american_to_probability(odds)
        assert recovered == pytest.approx(p, abs=0.01)


def test_probability_to_american_clips_extremes():
    assert isinstance(probability_to_american(0.0), int)
    assert isinstance(probability_to_american(1.0), int)


def test_spread_to_home_probability_even_at_zero_margin():
    assert spread_to_home_probability(0.0) == pytest.approx(0.5)


def test_spread_to_home_probability_three_point_favorite_calibration():
    # NOTE: the inline comment in engine/nfl.py claims "a 3-point favorite is
    # approximately 59%", but with the current divisor (12.0) the actual
    # value is ~56.2%. This pins the real behavior and flags that the
    # comment is stale documentation left over from an earlier calibration.
    assert spread_to_home_probability(3.0) == pytest.approx(0.562, abs=0.005)


def test_spread_to_home_probability_monotonic():
    low = spread_to_home_probability(-7.0)
    mid = spread_to_home_probability(0.0)
    high = spread_to_home_probability(7.0)
    assert low < mid < high


def test_market_no_vig_home_probability_sums_correctly():
    p_home = _market_no_vig_home_probability(-150, 130)
    assert 0.0 < p_home < 1.0


def test_market_no_vig_home_probability_symmetric_market():
    # Equal odds on both sides -> 50/50 after removing the vig.
    assert _market_no_vig_home_probability(-110, -110) == pytest.approx(0.5)


# --- team_power_score -----------------------------------------------------

def test_team_power_score_unknown_team_raises():
    with pytest.raises(ValueError):
        team_power_score("Springfield Isotopes")


def test_team_power_score_clips_overrides_to_0_100():
    team = next(iter(NFL_TEAM_RATINGS))
    score_default, _ = team_power_score(team)
    score_overridden, components = team_power_score(team, {"quarterback": 500.0})
    assert components["quarterback"] == 100.0  # clipped, not 500
    assert score_overridden > score_default  # a maxed-out QB grade should raise the score


def test_team_power_score_overrides_only_touch_known_weight_keys():
    team = next(iter(NFL_TEAM_RATINGS))
    _, components = team_power_score(team, {"not_a_real_component": 999.0})
    assert "not_a_real_component" not in components


# --- _decisive_factors ------------------------------------------------------

def test_decisive_factors_small_gaps_are_excluded():
    away = {k: 70.0 for k in TEAM_RATING_WEIGHTS}
    home = {k: 70.05 for k in TEAM_RATING_WEIGHTS}  # tiny gap everywhere
    factors = _decisive_factors("Away Team", "Home Team", away, home)
    assert factors == []


def test_decisive_factors_identifies_the_leading_side():
    away = {k: 60.0 for k in TEAM_RATING_WEIGHTS}
    home = {k: 60.0 for k in TEAM_RATING_WEIGHTS}
    home["quarterback"] = 90.0  # big, clearly decisive gap
    factors = _decisive_factors("Away Team", "Home Team", away, home)
    assert any(f["leader"] == "Home Team" and f["category"] == "Quarterback" for f in factors)


def test_decisive_factors_capped_at_4_and_sorted_by_impact():
    away = {k: 40.0 for k in TEAM_RATING_WEIGHTS}
    home = {k: 90.0 for k in TEAM_RATING_WEIGHTS}  # every category is decisive
    factors = _decisive_factors("Away Team", "Home Team", away, home)
    assert len(factors) <= 4
    impacts = [f["weighted_impact"] for f in factors]
    assert impacts == sorted(impacts, reverse=True)


# --- analyze(): end-to-end invariants using real team data ------------------

def _two_real_teams() -> tuple[str, str]:
    teams = list(NFL_TEAM_RATINGS.keys())
    return teams[0], teams[1]


def test_analyze_rejects_identical_teams():
    team, _ = _two_real_teams()
    with pytest.raises(ValueError):
        analyze(
            away_team=team, home_team=team,
            market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
            market_total=45.5,
        )


def test_analyze_rejects_non_positive_total():
    away, home = _two_real_teams()
    with pytest.raises(ValueError):
        analyze(
            away_team=away, home_team=home,
            market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
            market_total=0,
        )


def test_analyze_probabilities_sum_to_one():
    away, home = _two_real_teams()
    result = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5,
    )
    assert result["home_win_probability"] + result["away_win_probability"] == pytest.approx(1.0)


def test_analyze_projected_winner_matches_higher_probability_side():
    away, home = _two_real_teams()
    result = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5,
    )
    if result["home_win_probability"] >= result["away_win_probability"]:
        assert result["projected_winner"] == home
    else:
        assert result["projected_winner"] == away


def test_analyze_fair_total_reflects_market_total_when_no_weather_adjustment():
    away, home = _two_real_teams()
    result = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5, weather_context={},
    )
    assert result["projected_away_score"] + result["projected_home_score"] == pytest.approx(45.5, abs=1.0)


def test_analyze_neutral_site_removes_home_field_advantage():
    away, home = _two_real_teams()
    home_field_game = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5, neutral_site=False, home_field_points=3.0,
    )
    neutral_game = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5, neutral_site=True, home_field_points=3.0,
    )
    # Removing home field should not leave the home team better off than
    # with it (all else equal, home field can only help the home side).
    assert home_field_game["home_win_probability"] >= neutral_game["home_win_probability"] - 1e-6


def test_analyze_confidence_is_bounded():
    away, home = _two_real_teams()
    result = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5,
    )
    assert 50.0 <= result["confidence"] <= 78.0


def test_analyze_rating_overrides_change_the_outcome():
    away, home = _two_real_teams()
    baseline = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5,
    )
    boosted = analyze(
        away_team=away, home_team=home,
        market_spread_home=-2.5, market_moneyline_away=110, market_moneyline_home=-130,
        market_total=45.5, away_rating_overrides={"quarterback": 100.0},
    )
    # Boosting the away team's QB grade should not make the away team
    # (relatively) worse off.
    assert boosted["away_win_probability"] >= baseline["away_win_probability"] - 1e-6
