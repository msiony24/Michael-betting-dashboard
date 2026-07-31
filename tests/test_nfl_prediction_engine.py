from __future__ import annotations

from engine.nfl import analyze, team_power_score
from engine.nfl_data import TEAM_RATING_WEIGHTS


def test_team_state_weights_sum_to_one():
    assert round(sum(TEAM_RATING_WEIGHTS.values()), 10) == 1.0
    assert TEAM_RATING_WEIGHTS["quarterback"] == max(TEAM_RATING_WEIGHTS.values())


def test_qb_edge_moves_probability_more_than_equal_small_component_edge():
    neutral = {key: 70.0 for key in TEAM_RATING_WEIGHTS}
    qb_home = dict(neutral)
    qb_home["quarterback"] = 80.0
    special_home = dict(neutral)
    special_home["special_teams"] = 80.0

    qb_result = analyze(
        away_team="Buffalo Bills", home_team="Kansas City Chiefs",
        market_spread_home=0.0, market_moneyline_away=-110, market_moneyline_home=-110,
        market_total=47.0, neutral_site=True,
        away_rating_overrides=neutral, home_rating_overrides=qb_home,
    )
    special_result = analyze(
        away_team="Buffalo Bills", home_team="Kansas City Chiefs",
        market_spread_home=0.0, market_moneyline_away=-110, market_moneyline_home=-110,
        market_total=47.0, neutral_site=True,
        away_rating_overrides=neutral, home_rating_overrides=special_home,
    )
    assert qb_result["home_win_probability"] > special_result["home_win_probability"]


def test_moneyline_is_primary_output_and_spread_is_present():
    result = analyze(
        away_team="Buffalo Bills", home_team="Kansas City Chiefs",
        market_spread_home=-2.5, market_moneyline_away=120, market_moneyline_home=-140,
        market_total=48.0,
    )
    assert result["projected_winner"] in {"Buffalo Bills", "Kansas City Chiefs"}
    assert isinstance(result["fair_moneyline_home"], int)
    assert isinstance(result["fair_moneyline_away"], int)
    assert isinstance(result["fair_spread_home"], float)
    assert "decisive_factors" in result
