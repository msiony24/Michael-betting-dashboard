from __future__ import annotations

import pytest

from engine.ufc_markets import (
    UFCDerivativeMarketConfig,
    _round_finish_masses,
    _single_price_evaluation,
    _supported_total_lines,
    _verdict,
    american_to_decimal,
    build_derivative_markets,
    evaluate_derivative_market,
    implied_probability,
    probability_to_american,
    total_round_probabilities,
)


# --- odds conversion math (must be exactly right -- this is real money) ----

def test_american_to_decimal_negative_odds():
    assert american_to_decimal(-150) == pytest.approx(1.0 + 100.0 / 150.0)


def test_american_to_decimal_positive_odds():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_to_decimal_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_implied_probability_known_values():
    assert implied_probability(-150) == pytest.approx(0.6, abs=1e-6)
    assert implied_probability(150) == pytest.approx(0.4, abs=1e-6)
    assert implied_probability(-100) == pytest.approx(0.5, abs=1e-6)


def test_probability_to_american_boundary_at_half():
    assert probability_to_american(0.5) == -100


def test_probability_to_american_favorite_is_negative():
    assert probability_to_american(0.75) < 0


def test_probability_to_american_underdog_is_positive():
    assert probability_to_american(0.25) > 0


def test_probability_to_american_round_trips_through_implied_probability():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        odds = probability_to_american(p)
        recovered = implied_probability(odds)
        assert recovered == pytest.approx(p, abs=0.01)


def test_probability_to_american_clips_extreme_inputs():
    # Should not blow up on 0 or 1 -- clipped internally.
    assert isinstance(probability_to_american(0.0), int)
    assert isinstance(probability_to_american(1.0), int)


# --- round total lines and finish-mass splitting ----------------------------

def test_supported_total_lines_3_vs_5_round():
    assert _supported_total_lines(3) == [1.5, 2.5]
    assert _supported_total_lines(5) == [1.5, 2.5, 3.5, 4.5]


def test_round_finish_masses_sum_to_finish_probability():
    sim = {
        "rounds": 3, "finish_probability": 0.6,
        "finish_round_probabilities_given_finish": {"Round 1": 0.5, "Round 2": 0.3, "Round 3": 0.2},
    }
    masses = _round_finish_masses(sim)
    assert sum(masses) == pytest.approx(0.6)
    assert masses[0] == pytest.approx(0.3)  # 0.6 * 0.5


def test_round_finish_masses_zero_when_no_finish_probability():
    sim = {"rounds": 3, "finish_probability": 0.0, "finish_round_probabilities_given_finish": {}}
    assert _round_finish_masses(sim) == [0.0, 0.0, 0.0]


def test_round_finish_masses_renormalizes_when_conditional_probs_dont_sum_to_one():
    # Conditional probabilities sum to 0.5 instead of 1.0 -- the function
    # should rescale so the masses still total finish_probability exactly.
    sim = {
        "rounds": 3, "finish_probability": 0.6,
        "finish_round_probabilities_given_finish": {"Round 1": 0.25, "Round 2": 0.25, "Round 3": 0.0},
    }
    masses = _round_finish_masses(sim)
    assert sum(masses) == pytest.approx(0.6)


def test_total_round_probabilities_over_and_under_sum_to_one():
    sim = {
        "rounds": 3, "finish_probability": 0.6,
        "finish_round_probabilities_given_finish": {"Round 1": 0.4, "Round 2": 0.3, "Round 3": 0.3},
    }
    over, under = total_round_probabilities(sim, 2.5)
    assert over + under == pytest.approx(1.0)


def test_total_round_probabilities_unsupported_line_raises():
    sim = {"rounds": 3, "finish_probability": 0.5, "finish_round_probabilities_given_finish": {}}
    with pytest.raises(ValueError):
        total_round_probabilities(sim, 3.5)  # 3.5 isn't a valid line for a 3-round fight


def test_total_round_probabilities_half_round_split_is_half_of_that_rounds_mass():
    # All finish mass concentrated in round 2 -> the 1.5 line (target_round=2)
    # should assign exactly half of round 2's mass to "under".
    sim = {
        "rounds": 3, "finish_probability": 0.8,
        "finish_round_probabilities_given_finish": {"Round 1": 0.0, "Round 2": 1.0, "Round 3": 0.0},
    }
    over, under = total_round_probabilities(sim, 1.5)
    assert under == pytest.approx(0.4)  # 0.5 * 0.8


# --- build_derivative_markets -----------------------------------------------

def test_build_derivative_markets_unavailable_without_simulation():
    assert build_derivative_markets({}, "Alpha", "Bravo")["available"] is False
    assert build_derivative_markets({"available": False}, "Alpha", "Bravo")["available"] is False


def test_build_derivative_markets_full_shape():
    sim = {
        "available": True, "rounds": 3,
        "a_ko_tko_probability": 0.3, "a_submission_probability": 0.1, "a_decision_probability": 0.2,
        "b_ko_tko_probability": 0.15, "b_submission_probability": 0.05, "b_decision_probability": 0.2,
        "goes_distance_probability": 0.4,
        "finish_round_probabilities_given_finish": {"Round 1": 0.3, "Round 2": 0.3, "Round 3": 0.4},
    }
    result = build_derivative_markets(sim, "Alpha", "Bravo")
    assert result["available"] is True
    assert len(result["method_markets"]) == 6
    assert len(result["round_totals"]) == 2  # 3-round fight -> [1.5, 2.5]
    assert result["distance_market"]["yes_probability"] == pytest.approx(0.4)


def test_build_derivative_markets_5_round_fight_has_4_total_lines():
    sim = {
        "available": True, "rounds": 5,
        "a_ko_tko_probability": 0.2, "a_submission_probability": 0.1, "a_decision_probability": 0.2,
        "b_ko_tko_probability": 0.1, "b_submission_probability": 0.05, "b_decision_probability": 0.35,
        "goes_distance_probability": 0.35,
        "finish_round_probabilities_given_finish": {},
    }
    result = build_derivative_markets(sim, "Alpha", "Bravo")
    assert len(result["round_totals"]) == 4


# --- _verdict: this is the actual BET/WATCH/PASS money decision ------------

def test_verdict_bet_requires_all_three_thresholds():
    config = UFCDerivativeMarketConfig(bet_roi_threshold=0.08, watch_roi_threshold=0.03, min_bet_confidence=60, min_probability_for_bet=0.04)
    assert _verdict(probability=0.5, roi=0.10, confidence=70, config=config) == "BET"


def test_verdict_falls_to_watch_when_confidence_too_low_for_bet():
    config = UFCDerivativeMarketConfig(bet_roi_threshold=0.08, watch_roi_threshold=0.03, min_bet_confidence=60, min_probability_for_bet=0.04)
    result = _verdict(probability=0.5, roi=0.10, confidence=50, config=config)
    assert result == "WATCH"


def test_verdict_falls_to_watch_when_probability_too_low_for_bet():
    config = UFCDerivativeMarketConfig(bet_roi_threshold=0.08, watch_roi_threshold=0.03, min_bet_confidence=60, min_probability_for_bet=0.04)
    result = _verdict(probability=0.02, roi=0.10, confidence=70, config=config)
    assert result == "WATCH"


def test_verdict_pass_when_roi_below_watch_threshold():
    config = UFCDerivativeMarketConfig(bet_roi_threshold=0.08, watch_roi_threshold=0.03, min_bet_confidence=60, min_probability_for_bet=0.04)
    assert _verdict(probability=0.5, roi=0.01, confidence=90, config=config) == "PASS"


def test_verdict_pass_on_negative_roi_regardless_of_confidence():
    config = UFCDerivativeMarketConfig()
    assert _verdict(probability=0.9, roi=-0.05, confidence=99, config=config) == "PASS"


# --- _single_price_evaluation: ROI/edge math --------------------------------

def test_single_price_evaluation_edge_is_probability_minus_implied():
    config = UFCDerivativeMarketConfig()
    result = _single_price_evaluation(0.6, -150, confidence=80, config=config)
    assert result["implied_probability"] == pytest.approx(0.6, abs=1e-6)
    assert result["edge"] == pytest.approx(0.0, abs=1e-6)


def test_single_price_evaluation_positive_roi_when_probability_exceeds_market():
    config = UFCDerivativeMarketConfig()
    # True probability (65%) higher than what -150 (60%) implies -> positive edge/ROI.
    result = _single_price_evaluation(0.65, -150, confidence=80, config=config)
    assert result["edge"] > 0
    assert result["roi"] > 0


def test_single_price_evaluation_negative_roi_when_probability_below_market():
    config = UFCDerivativeMarketConfig()
    result = _single_price_evaluation(0.5, -150, confidence=80, config=config)
    assert result["roi"] < 0


# --- evaluate_derivative_market: routing and no-vig math --------------------

def _sample_markets() -> dict:
    return build_derivative_markets({
        "available": True, "rounds": 3,
        "a_ko_tko_probability": 0.3, "a_submission_probability": 0.1, "a_decision_probability": 0.2,
        "b_ko_tko_probability": 0.15, "b_submission_probability": 0.05, "b_decision_probability": 0.2,
        "goes_distance_probability": 0.4,
        "finish_round_probabilities_given_finish": {"Round 1": 0.3, "Round 2": 0.3, "Round 3": 0.4},
    }, "Alpha", "Bravo")


def test_evaluate_market_unavailable_without_odds():
    result = evaluate_derivative_market(_sample_markets(), "goes_distance", odds_primary=None)
    assert result["available"] is False


def test_evaluate_market_method_has_no_paired_no_vig():
    result = evaluate_derivative_market(_sample_markets(), "a_ko_tko", odds_primary=-120, confidence=70)
    assert result["available"] is True
    assert result["market_type"] == "method"
    assert result["paired_no_vig_available"] is False


def test_evaluate_market_two_way_computes_no_vig_and_hold():
    result = evaluate_derivative_market(
        _sample_markets(), "goes_distance", odds_primary=-110, odds_secondary=-110, confidence=70,
    )
    assert result["available"] is True
    assert result["paired_no_vig_available"] is True
    # -110/-110 is a standard ~4.5% hold market.
    assert result["sportsbook_hold"] == pytest.approx(implied_probability(-110) * 2 - 1.0, abs=1e-6)
    assert result["primary"]["no_vig_probability"] + result["secondary"]["no_vig_probability"] == pytest.approx(1.0)


def test_evaluate_market_total_rounds_requires_a_valid_line():
    result = evaluate_derivative_market(_sample_markets(), "total_rounds", odds_primary=-110, total_line=None)
    assert result["available"] is False
    result_bad_line = evaluate_derivative_market(_sample_markets(), "total_rounds", odds_primary=-110, total_line=9.5)
    assert result_bad_line["available"] is False


def test_evaluate_market_total_rounds_valid_line_works():
    result = evaluate_derivative_market(_sample_markets(), "total_rounds", odds_primary=-110, odds_secondary=-110, total_line=2.5)
    assert result["available"] is True
    assert result["primary_label"] == "Over 2.5 rounds"


def test_evaluate_market_unknown_method_key_unavailable():
    result = evaluate_derivative_market(_sample_markets(), "not_a_real_market", odds_primary=-110)
    assert result["available"] is False
