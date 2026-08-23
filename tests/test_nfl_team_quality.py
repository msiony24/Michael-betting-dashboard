"""Real tests for engine/nfl_team_quality.py.

The root-level test_nfl_team_quality.py is a print-only smoke script (no
assertions). This file replaces that gap with real coverage of the weighted
rating calculation and the fair-spread comparison.
"""
from __future__ import annotations

import pytest

from engine.nfl_team_quality import (
    WEIGHTS,
    TeamQualityInputs,
    calculate_team_quality,
    compare_team_quality,
)


def _inputs(**overrides) -> TeamQualityInputs:
    base = dict(
        quarterback=70, offense=70, defense=70, coaching=70, offensive_line=70,
        defensive_line=70, skill_positions=70, secondary=70, special_teams=70,
        continuity=70, injury_adjustment=0.0, rookie_adjustment=0.0,
    )
    base.update(overrides)
    return TeamQualityInputs(**base)


def test_weights_sum_to_one():
    # If this ever drifts, every team's base_rating quietly stops being a
    # true 0-100 scale.
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_calculate_team_quality_uniform_inputs_equals_that_value():
    result = calculate_team_quality("Buffalo Bills", _inputs())
    assert result.base_rating == pytest.approx(70.0)
    assert result.final_rating == pytest.approx(70.0)


def test_calculate_team_quality_weighted_average_is_correct():
    result = calculate_team_quality("Buffalo Bills", _inputs(quarterback=100))
    expected = 100 * WEIGHTS["quarterback"] + 70 * (1 - WEIGHTS["quarterback"])
    assert result.base_rating == pytest.approx(expected, abs=0.01)


def test_calculate_team_quality_quarterback_moves_rating_more_than_special_teams():
    # Confirms the weighting actually differentiates -- a change to the
    # highest-weighted component should move the rating more than the same
    # change to the lowest-weighted one.
    qb_boost = calculate_team_quality("A", _inputs(quarterback=90)).base_rating
    st_boost = calculate_team_quality("A", _inputs(special_teams=90)).base_rating
    baseline = calculate_team_quality("A", _inputs()).base_rating
    assert (qb_boost - baseline) > (st_boost - baseline)


def test_calculate_team_quality_applies_injury_and_rookie_adjustments():
    result = calculate_team_quality("Buffalo Bills", _inputs(injury_adjustment=-5.0, rookie_adjustment=2.0))
    assert result.final_rating == pytest.approx(70.0 - 5.0 + 2.0, abs=0.01)


def test_calculate_team_quality_final_rating_clipped_to_0_100():
    high = calculate_team_quality("A", _inputs(quarterback=100, rookie_adjustment=50.0))
    low = calculate_team_quality("A", _inputs(quarterback=0, injury_adjustment=-50.0))
    assert high.final_rating <= 100.0
    assert low.final_rating >= 0.0


def test_calculate_team_quality_rejects_out_of_range_component():
    with pytest.raises(ValueError):
        calculate_team_quality("A", _inputs(quarterback=150))
    with pytest.raises(ValueError):
        calculate_team_quality("A", _inputs(defense=-10))


def test_calculate_team_quality_rejects_non_numeric_component():
    # NOTE: _validate_score's TypeError branch is actually unreachable in
    # practice -- component_scores is built with float(inputs.X) first, which
    # raises ValueError on a non-numeric string before _validate_score's
    # isinstance check ever runs. This pins the real observed behavior.
    bad_inputs = _inputs()
    object.__setattr__(bad_inputs, "quarterback", "not a number")
    with pytest.raises(ValueError):
        calculate_team_quality("A", bad_inputs)


def test_calculate_team_quality_rejects_empty_team_name():
    with pytest.raises(ValueError):
        calculate_team_quality("", _inputs())
    with pytest.raises(ValueError):
        calculate_team_quality("   ", _inputs())


def test_calculate_team_quality_strips_team_name_whitespace():
    result = calculate_team_quality("  Buffalo Bills  ", _inputs())
    assert result.team == "Buffalo Bills"


# --- compare_team_quality -----------------------------------------------------

def test_compare_team_quality_favors_the_stronger_home_team():
    home = calculate_team_quality("Home", _inputs(quarterback=95))
    away = calculate_team_quality("Away", _inputs(quarterback=50))
    result = compare_team_quality(away_team=away, home_team=home)
    assert result["favored_team"] == "Home"
    assert result["fair_spread_home"] > 0


def test_compare_team_quality_favors_the_stronger_away_team_despite_home_field():
    home = calculate_team_quality("Home", _inputs(quarterback=50))
    away = calculate_team_quality("Away", _inputs(quarterback=95))
    result = compare_team_quality(away_team=away, home_team=home, home_field_advantage=1.5)
    assert result["favored_team"] == "Away"
    assert result["fair_spread_home"] < 0


def test_compare_team_quality_exact_tie_after_home_field_is_pickem():
    home = calculate_team_quality("Home", _inputs())
    away = calculate_team_quality("Away", _inputs())
    result = compare_team_quality(away_team=away, home_team=home, home_field_advantage=0.0)
    assert result["favored_team"] == "Pick'em"
    assert result["fair_spread_home"] == pytest.approx(0.0)


def test_compare_team_quality_home_field_advantage_can_flip_a_close_favorite():
    # Away team is very slightly better, but home field should be enough to flip it.
    home = calculate_team_quality("Home", _inputs())
    away = calculate_team_quality("Away", _inputs(quarterback=71))
    result = compare_team_quality(away_team=away, home_team=home, home_field_advantage=5.0)
    assert result["favored_team"] == "Home"


def test_compare_team_quality_scales_by_points_per_rating_point():
    home = calculate_team_quality("Home", _inputs(quarterback=90))
    away = calculate_team_quality("Away", _inputs())
    low_scale = compare_team_quality(away_team=away, home_team=home, points_per_rating_point=0.3)
    high_scale = compare_team_quality(away_team=away, home_team=home, points_per_rating_point=0.9)
    assert high_scale["fair_spread_home"] > low_scale["fair_spread_home"] > 0
