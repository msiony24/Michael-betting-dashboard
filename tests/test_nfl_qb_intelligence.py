"""Tests for engine/nfl_qb_intelligence.py: QB replacement/injury adjustment."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.nfl_qb_intelligence import _experience_credit, apply_qb_replacement_adjustment


def test_experience_credit_veteran_by_starts():
    credit, note = _experience_credit({"career_starts": 30})
    assert credit == 3.0
    assert "veteran" in note


def test_experience_credit_veteran_by_attempts():
    credit, note = _experience_credit({"career_attempts": 800})
    assert credit == 3.0


def test_experience_credit_meaningful_experience():
    credit, note = _experience_credit({"career_starts": 10})
    assert credit == 1.5
    assert "meaningful" in note


def test_experience_credit_limited_experience():
    credit, note = _experience_credit({"career_starts": 0, "career_attempts": 0})
    assert credit == 0.0
    assert "limited" in note


def test_experience_credit_falls_back_to_starts_attempts_alias_keys():
    # "starts"/"attempts" are checked as fallbacks when career_* is absent.
    credit, _ = _experience_credit({"starts": 30})
    assert credit == 3.0


def _healthy(rating=90.0) -> pd.DataFrame:
    return pd.DataFrame([{"player_name": "Starter QB", "macabets_rating": rating}])


def _active(rating=70.0, **overrides) -> pd.DataFrame:
    row = {"player_name": "Backup QB", "macabets_rating": rating, "career_starts": 0, "career_attempts": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def test_apply_adjustment_no_starters_returns_grade_unchanged():
    grade, context = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=pd.DataFrame(), active_starters=_active(), depth={},
    )
    assert grade == 85.0
    assert context == {}


def test_apply_adjustment_no_active_starters_returns_grade_unchanged():
    grade, context = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(), active_starters=pd.DataFrame(), depth={},
    )
    assert grade == 85.0
    assert context == {}


def test_apply_adjustment_none_inputs_returns_grade_unchanged():
    grade, context = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=None, active_starters=None, depth={},
    )
    assert grade == 85.0
    assert context == {}


def test_apply_adjustment_large_drop_is_capped_at_negative_5():
    grade, context = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=95.0), active_starters=_active(rating=40.0), depth={},
    )
    assert context["grade_adjustment"] == -5.0
    assert grade == 80.0
    assert context["severity"] == "Major"


def test_apply_adjustment_experience_credit_reduces_effective_drop():
    novice = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(), active_starters=_active(career_starts=0), depth={},
    )
    veteran = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(), active_starters=_active(career_starts=30), depth={},
    )
    # A veteran backup should soften the penalty relative to a total novice.
    assert veteran[0] > novice[0]
    assert veteran[1]["experience_credit"] == 3.0


def test_apply_adjustment_no_rating_drop_means_no_adjustment():
    grade, context = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=70.0), active_starters=_active(rating=90.0), depth={},
    )
    # Replacement actually graded higher than "starter" -- drop floors at 0.
    assert context["raw_rating_drop"] == 0.0
    assert context["grade_adjustment"] == 0.0
    assert grade == 85.0


def test_apply_adjustment_strong_offensive_support_softens_penalty():
    # Use a modest rating gap so the -5.0 cap doesn't mask the support_factor's effect.
    weak_support = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=80.0), active_starters=_active(rating=70.0),
        depth={"offensive_support": 50.0},
    )
    strong_support = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=80.0), active_starters=_active(rating=70.0),
        depth={"offensive_support": 85.0},
    )
    assert strong_support[0] > weak_support[0]


def test_apply_adjustment_elite_opponent_pressure_worsens_penalty():
    low_pressure = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=90.0), active_starters=_active(rating=75.0),
        depth={"opponent_pressure": 50.0},
    )
    high_pressure = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=90.0), active_starters=_active(rating=75.0),
        depth={"opponent_pressure": 90.0},
    )
    assert high_pressure[0] < low_pressure[0]


def test_apply_adjustment_severity_labels():
    small = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=75.0), active_starters=_active(rating=71.0), depth={},
    )
    moderate = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=85.0), active_starters=_active(rating=75.0), depth={},
    )
    major = apply_qb_replacement_adjustment(
        grade=85.0, healthy_starters=_healthy(rating=95.0), active_starters=_active(rating=75.0), depth={},
    )
    assert small[1]["severity"] == "Small"
    assert moderate[1]["severity"] == "Moderate"
    assert major[1]["severity"] == "Major"
