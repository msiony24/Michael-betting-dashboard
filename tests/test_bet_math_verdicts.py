"""Verdict rules after confidence was removed as a gate.

These lock in two properties:
  1. Confidence cannot change a verdict. Same probability and price must
     produce the same verdict at any confidence value, including None.
  2. Market-implied probability is de-vigged when both sides are supplied,
     so edge is not understated on favorites.
"""

from __future__ import annotations

import pytest

from engine.bet_math import (
    implied_probability,
    moneyline_price_quality,
    no_vig_probabilities,
    verdict_probability_ceiling,
)


# --- 1. confidence is inert --------------------------------------------------

@pytest.mark.parametrize("prob, odds", [
    (0.62, 105),
    (0.60, -150),
    (0.808, -350),
    (0.55, 300),
    (0.80, -220),
    (0.45, 180),
])
def test_confidence_cannot_change_the_verdict(prob, odds):
    verdicts = {
        moneyline_price_quality(prob, odds, c)["verdict"]
        for c in (None, 0, 40, 62, 78, 90, 100)
    }
    assert len(verdicts) == 1


def test_confidence_argument_is_optional():
    assert moneyline_price_quality(0.62, 105)["verdict"] == "Worth Betting"


# --- 2. the inverted-gate bug is gone ----------------------------------------

def test_mid_edge_spot_no_longer_falls_through_to_pass():
    """Regression test for the real bug.

    Old thresholds required confidence >= 78 for "Lean" but only >= 62 for
    "Worth Betting". A spot with a small positive edge and moderate confidence
    cleared neither and landed on Pass. It should now be Lean or better.
    """
    result = moneyline_price_quality(0.605, -150, 65)
    assert result["verdict"] in {"Lean", "Worth Betting"}


def test_lean_is_not_harder_to_reach_than_worth_betting():
    """Whatever the thresholds are, the ladder must be monotonic in edge."""
    order = {"Complete Pass": 0, "Pass": 1, "Lean": 2,
             "Worth Betting": 3, "Strong Bet": 4}
    ranks = [
        order[moneyline_price_quality(p, -150)["verdict"]]
        # rising model probability at a fixed price == rising edge
        for p in (0.50, 0.56, 0.60, 0.62, 0.66, 0.72, 0.80)
    ]
    assert ranks == sorted(ranks)


# --- 3. conviction still comes from win probability --------------------------

def test_big_edge_on_a_coinflip_cannot_be_a_strong_bet():
    result = moneyline_price_quality(0.55, 300)
    assert result["verdict"] != "Strong Bet"
    assert result["verdict"] == verdict_probability_ceiling(0.55)


def test_low_probability_pick_is_capped_regardless_of_price():
    result = moneyline_price_quality(0.45, 250)
    assert result["verdict"] == "Pass"


# --- 4. de-vig ----------------------------------------------------------------

def test_no_vig_reduces_market_implied_probability():
    raw = moneyline_price_quality(0.80, -500)
    devig = moneyline_price_quality(0.80, -500, opponent_odds=350)
    assert devig["market_implied_probability"] < raw["market_implied_probability"]
    assert devig["edge"] > raw["edge"]
    assert devig["vig_removed"] is True
    assert raw["vig_removed"] is False


def test_no_vig_matches_the_two_sided_helper():
    expected, _, _ = no_vig_probabilities(-500, 350)
    result = moneyline_price_quality(0.80, -500, opponent_odds=350)
    assert result["market_implied_probability"] == pytest.approx(expected)


def test_raw_implied_is_used_when_opponent_odds_absent():
    result = moneyline_price_quality(0.80, -500)
    assert result["market_implied_probability"] == pytest.approx(
        implied_probability(-500)
    )


def test_vig_hurts_favorites_more_than_underdogs():
    """The reason this matters: single-side implied probability overstates the
    market most where the price is shortest."""
    fav_gap = (
        moneyline_price_quality(0.80, -500, opponent_odds=350)["edge"]
        - moneyline_price_quality(0.80, -500)["edge"]
    )
    dog_gap = (
        moneyline_price_quality(0.40, 180, opponent_odds=-220)["edge"]
        - moneyline_price_quality(0.40, 180)["edge"]
    )
    assert fav_gap > dog_gap


# --- 5. raw numbers are exposed for the log ----------------------------------

def test_report_carries_the_numbers_behind_the_label():
    result = moneyline_price_quality(0.62, 105, opponent_odds=-125)
    for key in ("model_probability", "market_implied_probability",
                "edge", "edge_points", "vig_removed", "expected_roi"):
        assert key in result
    assert result["edge_points"] == pytest.approx(result["edge"] * 100.0)
