from engine.nfl import analyze
from engine.nfl_ratings_loader import load_all_team_ratings


def test_unified_intelligence_connects_model_display_and_brain():
    ratings = load_all_team_ratings()
    result = analyze(
        away_team="Baltimore Ravens",
        home_team="Chicago Bears",
        market_spread_home=3.0,
        market_moneyline_away=-150,
        market_moneyline_home=130,
        market_total=45.5,
        away_rating_overrides=ratings["Baltimore Ravens"],
        home_rating_overrides=ratings["Chicago Bears"],
        week=1,
        season=2026,
    )
    intelligence = result["matchup_intelligence"]
    assert intelligence["available"] is True
    assert intelligence["data_mode"].startswith("Preseason")
    assert len(intelligence["categories"]) >= 8
    assert len(intelligence["questions"]) == 8
    assert result["matchup_brain"]["status"] == "preseason_ready"
    assert result["matchup_brain"]["decision_framework"]["ready_questions"] == 8
    # The final model margin is the unified football edge.
    expected_home_margin = float(intelligence["football_home_edge_points"])
    expected_fair = round((-expected_home_margin) * 2.0) / 2.0
    assert result["fair_spread_home"] == expected_fair


def test_small_matchup_gaps_are_even_not_category_wins():
    ratings = load_all_team_ratings()
    result = analyze(
        away_team="Baltimore Ravens",
        home_team="Chicago Bears",
        market_spread_home=0.0,
        market_moneyline_away=-110,
        market_moneyline_home=-110,
        market_total=45.5,
        neutral_site=True,
        away_rating_overrides=ratings["Baltimore Ravens"],
        home_rating_overrides=ratings["Chicago Bears"],
        week=1,
        season=2026,
    )
    categories = result["matchup_intelligence"]["categories"]
    # The old 10-0 category-count behavior is gone: multiple close conflicts are explicitly even.
    assert sum(row["Advantage"] == "Even" for row in categories) >= 3
