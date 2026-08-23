from __future__ import annotations

import pytest

from engine.nfl_simulation import simulate_game


def test_simulate_game_is_deterministic_for_the_same_seed_context():
    kwargs = dict(
        away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0,
        simulations=5000, seed_context="2026:1:2026-09-10",
    )
    result1 = simulate_game(**kwargs)
    result2 = simulate_game(**kwargs)
    assert result1["seed"] == result2["seed"]
    assert result1["home_win_probability"] == result2["home_win_probability"]


def test_simulate_game_different_matchups_get_different_seeds():
    result_a = simulate_game(away_team="A", home_team="B", projected_home_margin=3.0, fair_total=45.0, simulations=2000)
    result_b = simulate_game(away_team="C", home_team="D", projected_home_margin=3.0, fair_total=45.0, simulations=2000)
    assert result_a["seed"] != result_b["seed"]


def test_simulate_game_probabilities_sum_to_one():
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0, simulations=5000)
    assert result["home_win_probability"] + result["away_win_probability"] == pytest.approx(1.0)


def test_simulate_game_mean_scores_track_the_input_margin_and_total():
    # The simulator adds variance around the already-decided center; it must
    # not silently shift that center. With enough draws the sample mean
    # should land close to the analytic center.
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=6.0, fair_total=44.0, simulations=20000)
    assert result["mean_home_score"] - result["mean_away_score"] == pytest.approx(6.0, abs=0.5)
    assert result["mean_home_score"] + result["mean_away_score"] == pytest.approx(44.0, abs=0.5)


def test_simulate_game_larger_home_margin_increases_home_win_probability():
    small_edge = simulate_game(away_team="Away", home_team="Home", projected_home_margin=1.0, fair_total=45.0, simulations=10000)
    big_edge = simulate_game(away_team="Away", home_team="Home", projected_home_margin=10.0, fair_total=45.0, simulations=10000)
    assert big_edge["home_win_probability"] > small_edge["home_win_probability"]


def test_simulate_game_favorite_and_upset_probability_are_consistent():
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=10.0, fair_total=45.0, simulations=10000)
    assert result["favorite"] == "Home"
    assert result["favorite_win_probability"] == pytest.approx(result["home_win_probability"])
    assert result["upset_probability"] == pytest.approx(1.0 - result["favorite_win_probability"])


def test_simulate_game_cover_and_total_probabilities_none_without_market_lines():
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0, simulations=2000)
    assert result["home_cover_probability"] is None
    assert result["over_probability"] is None


def test_simulate_game_cover_probability_present_with_market_spread():
    result = simulate_game(
        away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0,
        market_spread_home=-3.0, simulations=5000,
    )
    assert result["home_cover_probability"] is not None
    assert result["away_cover_probability"] == pytest.approx(1.0 - result["home_cover_probability"])


def test_simulate_game_over_probability_present_with_market_total():
    result = simulate_game(
        away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0,
        market_total=44.0, simulations=5000,
    )
    assert result["over_probability"] is not None
    assert result["under_probability"] == pytest.approx(1.0 - result["over_probability"])


def test_simulate_game_scores_never_negative():
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=30.0, fair_total=10.0, simulations=5000)
    assert result["away_score_range_80"][0] >= 0.0


def test_simulate_game_minimum_simulation_count_enforced():
    result = simulate_game(away_team="Away", home_team="Home", projected_home_margin=3.0, fair_total=45.0, simulations=10)
    assert result["simulations"] >= 2000
