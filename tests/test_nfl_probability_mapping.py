"""Pins engine.nfl.spread_to_home_probability to empirical NFL behavior.

These are not regression tests against the model's own output. They are
assertions about how often NFL favorites actually win, so the mapping cannot
drift back toward a compressed curve without a test failing.

Empirical reference points (closing-spread favorites, modern NFL):
    3 points  -> ~59%
    7 points  -> ~70%
    10 points -> ~77%
    14 points -> ~85%
"""

from __future__ import annotations

import pytest

from engine.nfl import NFL_MARGIN_SD, spread_to_home_probability


def test_pickem_is_even():
    assert spread_to_home_probability(0.0) == pytest.approx(0.5)


def test_symmetry_around_zero():
    for margin in (1.5, 3.0, 7.0, 13.5):
        assert spread_to_home_probability(margin) + spread_to_home_probability(
            -margin
        ) == pytest.approx(1.0)


def test_monotonic():
    margins = [-14.0, -7.0, -3.0, 0.0, 3.0, 7.0, 14.0]
    probs = [spread_to_home_probability(m) for m in margins]
    assert probs == sorted(probs)
    assert len(set(probs)) == len(probs)


@pytest.mark.parametrize(
    "margin, expected",
    [
        (3.0, 0.588),
        (6.0, 0.671),
        (7.0, 0.698),
        (10.0, 0.771),
        (14.0, 0.850),
    ],
)
def test_matches_empirical_favorite_win_rates(margin, expected):
    assert spread_to_home_probability(margin) == pytest.approx(expected, abs=0.015)


def test_not_compressed_like_the_old_divisor_12_logistic():
    """Guards against reverting to the Phase 2 audit constant.

    The old mapping put a 7-point favorite at 64.2% and a 14-point favorite at
    76.3%. Both are far below what NFL favorites actually do. If either of
    these assertions fails, the compressed curve is back.
    """
    assert spread_to_home_probability(7.0) > 0.67
    assert spread_to_home_probability(14.0) > 0.82


def test_margin_sd_is_in_a_plausible_range():
    """NFL margin SD is roughly 13-14 points. Anything far outside that is a
    typo or a fitted value, not a measurement."""
    assert 12.0 <= NFL_MARGIN_SD <= 15.0
