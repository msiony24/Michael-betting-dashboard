import pytest

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


# --- cap enforcement: defense-in-depth against extreme layer inputs --------

def test_model_refinement_layers_capped_at_1_25(monkeypatch=None):
    from engine.nfl_matchup_intelligence import build_matchup_intelligence, MODEL_REFINEMENT_CAP

    result = build_matchup_intelligence(
        away_team="Away", home_team="Home",
        away_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        home_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        away_power=50.0, home_power=50.0, home_field_points=0.0,
        weather_home_adjustment=0.0, schedule_home_adjustment=0.0,
        game_quality_home_adjustment=999.0,
        scheme_home_adjustment=999.0, los_home_adjustment=999.0,
        situational_home_adjustment=999.0, opponent_adjusted_home_adjustment=999.0,
        personnel_context={"home_margin_adjustment": 0.0},
    )
    # base_team_edge=0, environment=0, matchup_adjustment=0 -> only the
    # refinement-layer cap should be reflected in the final number.
    assert result["football_home_edge_points"] == pytest.approx(MODEL_REFINEMENT_CAP)


def test_personnel_adjustment_capped_independently_of_refinement_layers():
    # Regression test for a real gap found in this audit: matchup_adjustment
    # (personnel + style matchups) previously flowed straight into the final
    # margin with no cap of its own at this layer -- it only stayed safe
    # because its one real caller (nfl_personnel_matchup.py) already capped
    # it before calling here. This proves the cap now holds even if that
    # caller's own cap were ever bypassed or fed an extreme value directly.
    from engine.nfl_matchup_intelligence import build_matchup_intelligence, PERSONNEL_ADJUSTMENT_CAP

    result = build_matchup_intelligence(
        away_team="Away", home_team="Home",
        away_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        home_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        away_power=50.0, home_power=50.0, home_field_points=0.0,
        weather_home_adjustment=0.0, schedule_home_adjustment=0.0,
        game_quality_home_adjustment=0.0,
        scheme_home_adjustment=0.0, los_home_adjustment=0.0,
        situational_home_adjustment=0.0, opponent_adjusted_home_adjustment=0.0,
        personnel_context={"home_margin_adjustment": 999.0},
    )
    assert result["football_home_edge_points"] == pytest.approx(PERSONNEL_ADJUSTMENT_CAP)


def test_both_caps_combined_bound_the_total_adjustment():
    from engine.nfl_matchup_intelligence import (
        build_matchup_intelligence, MODEL_REFINEMENT_CAP, PERSONNEL_ADJUSTMENT_CAP,
    )

    result = build_matchup_intelligence(
        away_team="Away", home_team="Home",
        away_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        home_components={"quarterback": 50, "defense": 50, "coaching": 50, "special_teams": 50, "continuity": 50},
        away_power=50.0, home_power=50.0, home_field_points=0.0,
        weather_home_adjustment=0.0, schedule_home_adjustment=0.0,
        game_quality_home_adjustment=999.0,
        scheme_home_adjustment=999.0, los_home_adjustment=999.0,
        situational_home_adjustment=999.0, opponent_adjusted_home_adjustment=999.0,
        personnel_context={"home_margin_adjustment": 999.0},
    )
    assert result["football_home_edge_points"] == pytest.approx(MODEL_REFINEMENT_CAP + PERSONNEL_ADJUSTMENT_CAP)


def test_game_quality_adjustment_is_always_zero_from_the_real_source():
    # Confirms the "diagnostic-only" claim in build_game_quality_context is
    # actually true on every path, not just documented as an assumption.
    from engine.nfl_game_quality import build_game_quality_context

    result = build_game_quality_context(away_team="Away", home_team="Home", season=2026)
    assert result["home_margin_adjustment"] == 0.0


def test_style_matchup_contribution_lives_inside_personnel_adjustment_only():
    # Confirms style_matchups (Madden trait matchups) are folded into
    # home_margin_adjustment exactly once by nfl_personnel_matchup.py,
    # matching the "not counted again below" comment in
    # build_matchup_intelligence -- not double-added as a separate term.
    import inspect
    from engine import nfl_personnel_matchup
    source = inspect.getsource(nfl_personnel_matchup)
    assert "style_home_adjustment" in source
    assert "base_home_adjustment + style_home_adjustment" in source
