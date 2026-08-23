"""Gap-filling coverage for engine/result_settlement.py.

tests/test_result_settlement.py already covers the core name-matching and
grading behavior. This file adds coverage for functions and branches that
file doesn't touch: the raw odds-conversion math, entry-price lookup,
status classification helpers, and a few grading/closing-line branches.
"""
from __future__ import annotations

import pytest

from engine.result_settlement import (
    american_implied_probability,
    clv_metrics,
    consensus_moneyline_close,
    decimal_to_american,
    event_participants_match,
    grade_moneyline_prediction,
    is_finished_status,
    is_provider_exception,
    no_vig_two_way_probability,
    prediction_entry_no_vig_probability,
    prediction_entry_odds,
)


# --- raw odds conversion math: never directly tested elsewhere -------------

def test_american_implied_probability_known_values():
    assert american_implied_probability(-150) == pytest.approx(0.6)
    assert american_implied_probability(150) == pytest.approx(0.4)
    assert american_implied_probability(-100) == pytest.approx(0.5)


def test_american_implied_probability_zero_raises():
    with pytest.raises(ValueError):
        american_implied_probability(0)


def test_decimal_to_american_even_money():
    assert decimal_to_american(2.0) == 100


def test_decimal_to_american_favorite_and_underdog():
    assert decimal_to_american(1.5) == -200
    assert decimal_to_american(3.0) == 200


def test_decimal_to_american_invalid_raises():
    with pytest.raises(ValueError):
        decimal_to_american(1.0)
    with pytest.raises(ValueError):
        decimal_to_american(0.5)


def test_no_vig_two_way_probability_removes_the_hold():
    assert no_vig_two_way_probability(-110, -110) == pytest.approx(0.5)


# --- event_participants_match: order-independence ----------------------------

def test_event_participants_match_handles_reversed_order():
    assert event_participants_match("Alpha", "Beta", "Beta", "Alpha") is True
    assert event_participants_match("Alpha", "Beta", "Alpha", "Beta") is True
    assert event_participants_match("Alpha", "Beta", "Alpha", "Gamma") is False


# --- prediction_entry_odds / prediction_entry_no_vig_probability: untested --

def test_prediction_entry_odds_picks_the_predicted_side():
    row = {"prediction": "Alpha", "participant_a": "Alpha", "participant_b": "Beta",
           "market_odds_a": -150, "market_odds_b": 130}
    assert prediction_entry_odds(row) == -150


def test_prediction_entry_odds_falls_back_to_market_line():
    row = {"prediction": "Unknown Player", "participant_a": "Alpha", "participant_b": "Beta",
           "market_line": -120}
    assert prediction_entry_odds(row) == -120


def test_prediction_entry_odds_zero_is_treated_as_missing():
    # A stored 0 for market_odds_a is not a valid price -- must not be
    # returned as if it were a real entry price.
    row = {"prediction": "Alpha", "participant_a": "Alpha", "participant_b": "Beta",
           "market_odds_a": 0, "market_odds_b": 130}
    assert prediction_entry_odds(row) is None


def test_prediction_entry_no_vig_probability_picks_correct_side():
    row = {"prediction": "Beta", "participant_a": "Alpha", "participant_b": "Beta",
           "market_odds_a": -150, "market_odds_b": 130}
    result = prediction_entry_no_vig_probability(row)
    expected = no_vig_two_way_probability(130, -150)
    assert result == pytest.approx(expected)


def test_prediction_entry_no_vig_probability_none_when_prediction_unmatched():
    row = {"prediction": "Nobody", "participant_a": "Alpha", "participant_b": "Beta",
           "market_odds_a": -150, "market_odds_b": 130}
    assert prediction_entry_no_vig_probability(row) is None


# --- status classification: never directly tested -----------------------------

def test_is_provider_exception_detects_retirement_and_walkover():
    assert is_provider_exception("Retired") is True
    assert is_provider_exception("Walkover") is True
    assert is_provider_exception("Postponed") is True
    assert is_provider_exception("Finished") is False


def test_is_finished_status():
    assert is_finished_status("Finished") is True
    assert is_finished_status("Final") is True
    assert is_finished_status("Scheduled") is False


# --- grading branches not exercised by the existing suite -------------------

def test_grade_moneyline_unrecognized_verdict_is_not_actionable():
    grade = grade_moneyline_prediction(
        prediction="Alpha", actual_winner="Alpha", recommendation="Something Unusual",
    )
    assert grade.value_call_correct is None
    assert grade.value_call_result == "Prediction graded; value verdict not actionable"


def test_grade_moneyline_unfinished_event_stays_pending():
    grade = grade_moneyline_prediction(
        prediction="Alpha", actual_winner="", recommendation="Bet",
        provider_status="Scheduled",
    )
    assert grade.status == "Pending"
    assert grade.prediction_correct is None


def test_grade_moneyline_finished_but_no_trustworthy_winner_needs_review():
    grade = grade_moneyline_prediction(
        prediction="Alpha", actual_winner="", recommendation="Bet",
        provider_status="Finished",
    )
    assert grade.status == "Pending"
    assert "manual review" in grade.value_call_result.lower()


# --- consensus_moneyline_close: unmatched/empty branches ---------------------

def test_consensus_close_returns_none_when_prediction_unmatched():
    snapshots = [{"bookmaker": "BookA", "participant": "Alpha", "american_odds": -150}]
    result = consensus_moneyline_close(
        snapshots, prediction="Nobody", participant_a="Alpha", participant_b="Beta",
    )
    assert result is None


def test_consensus_close_returns_none_with_no_snapshots():
    result = consensus_moneyline_close([], prediction="Alpha", participant_a="Alpha", participant_b="Beta")
    assert result is None


# --- clv_metrics: missing-data branch ------------------------------------------

def test_clv_metrics_none_when_closing_probability_missing():
    row = {"prediction": "Alpha", "participant_a": "Alpha", "participant_b": "Beta",
           "market_odds_a": 150, "market_odds_b": -180, "predicted_probability": 0.45}
    result = clv_metrics(row=row, closing_no_vig_probability=None)
    assert result["clv_probability"] is None
    assert result["closing_no_vig_probability"] is None
