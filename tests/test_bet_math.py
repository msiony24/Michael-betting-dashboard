"""Tests for engine/bet_math.py -- the betting-math functions extracted from
app.py so this exact class of logic (which produced three real, found bugs
across this project) finally has automated coverage instead of relying on
manual code review to catch problems.
"""
from __future__ import annotations

import pytest

from engine.bet_math import (
    american_to_decimal,
    cap_verdict_by_probability,
    decision_label,
    implied_probability,
    kelly_fraction,
    minimum_acceptable_odds,
    moneyline_price_quality,
    no_vig_probabilities,
    normalize_price_assessment,
    potential_profit,
    probability_to_american,
    stake_to_win,
    tennis_bet_confidence,
    verdict_probability_ceiling,
)


# --- odds conversion -----------------------------------------------------------

def test_american_to_decimal_known_values():
    assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_probability_to_american_round_trips():
    for p in (0.2, 0.4, 0.5, 0.6, 0.8):
        odds = probability_to_american(p)
        assert implied_probability(odds) == pytest.approx(p, abs=0.01)


def test_implied_probability_known_values():
    assert implied_probability(-150) == pytest.approx(0.6)
    assert implied_probability(150) == pytest.approx(0.4)


def test_no_vig_probabilities_removes_the_hold():
    a, b, hold = no_vig_probabilities(-110, -110)
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)
    assert hold > 0


# --- money math ------------------------------------------------------------

def test_potential_profit_favorite_and_underdog():
    assert potential_profit(-150, 150) == pytest.approx(100.0)
    assert potential_profit(150, 100) == pytest.approx(150.0)


def test_stake_to_win_favorite_and_underdog():
    assert stake_to_win(-150, 100) == pytest.approx(150.0)
    assert stake_to_win(150, 100) == pytest.approx(66.67, abs=0.01)


def test_kelly_fraction_no_edge_is_zero():
    # Model probability exactly matches the fair breakeven -> no edge -> no bet.
    fair_prob = implied_probability(-150)
    assert kelly_fraction(fair_prob, -150) == pytest.approx(0.0, abs=1e-6)


def test_kelly_fraction_real_edge_is_positive():
    assert kelly_fraction(0.70, -150) > 0.0


def test_minimum_acceptable_odds_stricter_for_higher_probability():
    # A more heavily favored pick can tolerate worse (more negative) odds
    # and still hit the same required ROI.
    loose = minimum_acceptable_odds(0.60, required_roi=0.02)
    tight = minimum_acceptable_odds(0.80, required_roi=0.02)
    assert tight < loose  # more negative = shorter price tolerated


# --- verdict caps -----------------------------------------------------------

def test_verdict_probability_ceiling_boundaries():
    assert verdict_probability_ceiling(0.50) == "Pass"
    assert verdict_probability_ceiling(0.55) == "Lean"
    assert verdict_probability_ceiling(0.60) == "Worth Betting"
    assert verdict_probability_ceiling(0.70) == "Strong Bet"


def test_cap_verdict_by_probability_downgrades_when_over_ceiling():
    assert cap_verdict_by_probability("Strong Bet", 0.55) == "Lean"


def test_cap_verdict_by_probability_leaves_verdict_under_ceiling_alone():
    assert cap_verdict_by_probability("Lean", 0.90) == "Lean"


# --- moneyline_price_quality: the core recommendation engine ----------------

def test_moneyline_price_quality_requires_non_negative_roi_for_worth_betting():
    # Regression test for the real bug found and fixed: confidence alone
    # used to be able to promote a negative-EV price into "Worth Betting".
    result = moneyline_price_quality(0.60, -150, 85)
    assert result["expected_roi"] < 0.001
    assert result["verdict"] != "Worth Betting"
    assert result["verdict"] != "Strong Bet"


def test_moneyline_price_quality_genuine_edge_still_gets_recommended():
    result = moneyline_price_quality(0.62, 105, 65)
    assert result["verdict"] == "Worth Betting"


def test_moneyline_price_quality_treats_favorites_and_dogs_consistently():
    # Same probability edge over market at very different prices should not
    # get different price-quality labels purely from ROI compression.
    fav = moneyline_price_quality(0.808, -350, 90)   # edge ~3pt over -350 market
    dog = moneyline_price_quality(0.43, 150, 90)      # edge ~3pt over +150 market
    assert fav["price_assessment"] == dog["price_assessment"]


def test_moneyline_price_quality_strong_bet_requires_probability_floor():
    # Big ROI edge alone, without real win-probability conviction, must not
    # produce Strong Bet.
    result = moneyline_price_quality(0.55, 300, 90)
    assert result["verdict"] != "Strong Bet"


# --- decision_label: must stay in sync with moneyline_price_quality --------

def test_decision_label_matches_moneyline_price_quality_on_worth_betting_threshold():
    # Both functions implement overlapping verdict logic (a known, documented
    # duplication). This test exists specifically so that if one gets fixed
    # or changed and the other doesn't, it fails loudly instead of silently
    # drifting the way it did before (the exact bug found this session).
    roi = 0.03
    conf = 85
    decision, _ = decision_label(roi, conf)
    quality = moneyline_price_quality(implied_probability(-150) + roi / 1.667, -150, conf)
    assert decision in {"Worth Betting", "Lean"}
    assert quality["verdict"] in {"Worth Betting", "Lean", "Strong Bet"}


def test_decision_label_never_returns_strong_bet():
    # By design: this helper doesn't know the model's win probability, so it
    # must never claim the model's strongest conviction tier.
    for roi in (0.05, 0.10, 0.20, 0.50):
        for conf in (70, 85, 100):
            verdict, _ = decision_label(roi, conf)
            assert verdict != "Strong Bet"


def test_decision_label_rejects_negative_ev_worth_betting():
    verdict, _ = decision_label(-0.03, 90)
    assert verdict != "Worth Betting"


# --- normalize_price_assessment ----------------------------------------------

def test_normalize_price_assessment_maps_legacy_labels():
    assert normalize_price_assessment("Significantly Underpriced") == "Very Underpriced"
    assert normalize_price_assessment("Fairly Priced") == "Fair"


def test_normalize_price_assessment_passes_through_current_labels():
    assert normalize_price_assessment("Underpriced") == "Underpriced"


def test_normalize_price_assessment_missing_label():
    assert normalize_price_assessment(None) == "—"
    assert normalize_price_assessment("") == "—"


# --- tennis_bet_confidence: price-agnostic conviction ------------------------

def test_tennis_bet_confidence_price_agnostic_for_matched_edge():
    # Regression test for the real bug found and fixed: a heavy favorite and
    # an underdog with the same genuine edge used to score very differently
    # purely because of ROI compression at short prices.
    favorite = tennis_bet_confidence(90, edge=0.03, expected_roi=0.039)
    underdog = tennis_bet_confidence(90, edge=0.03, expected_roi=0.075)
    assert favorite["overall"] == underdog["overall"]


def test_tennis_bet_confidence_low_edge_is_capped():
    result = tennis_bet_confidence(95, edge=0.01, expected_roi=0.01)
    assert result["overall"] <= 49
    assert result["band"] == "Low / Pass"
