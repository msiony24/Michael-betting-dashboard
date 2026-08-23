from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.ufc_opponent_adjustment import (
    UFCOpponentAdjustmentConfig,
    _adjust_value,
    _mean,
    _num,
    _quality_label,
    _recent_opponents,
    _skill_scores,
    adjust_fighter_profile,
    build_opponent_adjusted_matchup,
)


# --- small numeric helpers ---------------------------------------------------

def test_num_handles_missing_and_invalid_values():
    assert _num({"x": "42.5"}, "x") == 42.5
    assert _num({"x": None}, "x") is None
    assert _num({}, "x") is None
    assert _num({"x": "not a number"}, "x") is None
    assert _num({"x": float("nan")}, "x") is None


def test_mean_ignores_none_and_nan():
    assert _mean([1.0, None, 3.0]) == 2.0
    assert _mean([None, None]) is None
    assert _mean([]) is None


def test_skill_scores_maps_expected_keys():
    profile = {"sig_accuracy_pct": 60, "sig_diff_per_min_pct": 70, "kd_per15_pct": 50}
    scores = _skill_scores(profile)
    assert scores["striking_offense"] == pytest.approx(60.0)
    assert set(scores.keys()) == {
        "striking_offense", "striking_defense", "wrestling_offense", "wrestling_defense",
        "grappling_offense", "grappling_defense", "power", "durability", "pace",
    }


def test_quality_label_boundaries():
    assert _quality_label(None) == "Unknown"
    assert _quality_label(65) == "Elite"
    assert _quality_label(64.9) == "Strong"
    assert _quality_label(57) == "Strong"
    assert _quality_label(56.9) == "Average"
    assert _quality_label(43.1) == "Average"
    assert _quality_label(43) == "Below average"
    assert _quality_label(35.1) == "Below average"
    assert _quality_label(35) == "Weak"


# --- _adjust_value: the ±8 point cap is the core safety property -----------

def test_adjust_value_returns_base_unchanged_when_no_opponent_data():
    config = UFCOpponentAdjustmentConfig()
    adjusted, movement = _adjust_value(50.0, None, reliability=1.0, config=config)
    assert adjusted == 50.0
    assert movement == 0.0


def test_adjust_value_returns_none_when_base_missing():
    config = UFCOpponentAdjustmentConfig()
    adjusted, movement = _adjust_value(None, 80.0, reliability=1.0, config=config)
    assert adjusted is None
    assert movement == 0.0


def test_adjust_value_capped_at_max_skill_adjustment_even_against_elite_opponents():
    config = UFCOpponentAdjustmentConfig(max_skill_adjustment=8.0, adjustment_strength=0.30)
    # opponent_quality=100 is the most extreme possible input.
    adjusted, movement = _adjust_value(50.0, 100.0, reliability=1.0, config=config)
    assert movement == pytest.approx(8.0)
    assert adjusted == pytest.approx(58.0)


def test_adjust_value_capped_at_negative_max_against_weak_opponents():
    config = UFCOpponentAdjustmentConfig(max_skill_adjustment=8.0, adjustment_strength=0.30)
    adjusted, movement = _adjust_value(50.0, 0.0, reliability=1.0, config=config)
    assert movement == pytest.approx(-8.0)
    assert adjusted == pytest.approx(42.0)


def test_adjust_value_scaled_down_by_reliability():
    config = UFCOpponentAdjustmentConfig()
    # Use a quality level well inside the cap (quality=60 -> raw movement of
    # only 3.0 at full reliability) so the cap doesn't mask the scaling.
    full, _ = _adjust_value(50.0, 60.0, reliability=1.0, config=config)
    half, _ = _adjust_value(50.0, 60.0, reliability=0.5, config=config)
    assert (half - 50.0) == pytest.approx((full - 50.0) / 2, abs=0.01)


def test_adjust_value_clips_to_0_100_bounds():
    config = UFCOpponentAdjustmentConfig(max_skill_adjustment=50.0, adjustment_strength=1.0)
    adjusted, _ = _adjust_value(97.0, 100.0, reliability=1.0, config=config)
    assert adjusted <= 100.0
    adjusted_low, _ = _adjust_value(3.0, 0.0, reliability=1.0, config=config)
    assert adjusted_low >= 0.0


# --- _recent_opponents -------------------------------------------------------

def _fights(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_recent_opponents_orders_most_recent_first():
    fights = _fights([
        {"fighter": "Alpha", "opponent": "Old Foe", "event_date": "2024-01-01"},
        {"fighter": "Alpha", "opponent": "New Foe", "event_date": "2025-06-01"},
    ])
    result = _recent_opponents(fights, "Alpha", limit=8)
    assert result[0] == "New Foe"
    assert result[1] == "Old Foe"


def test_recent_opponents_respects_limit():
    fights = _fights([
        {"fighter": "Alpha", "opponent": f"Foe {i}", "event_date": f"2024-0{i}-01"}
        for i in range(1, 6)
    ])
    result = _recent_opponents(fights, "Alpha", limit=3)
    assert len(result) == 3


def test_recent_opponents_case_insensitive_fighter_match():
    fights = _fights([{"fighter": "alpha fighter", "opponent": "Foe", "event_date": "2024-01-01"}])
    result = _recent_opponents(fights, "Alpha Fighter", limit=8)
    assert result == ["Foe"]


def test_recent_opponents_missing_columns_returns_empty():
    assert _recent_opponents(pd.DataFrame({"fighter": ["Alpha"]}), "Alpha", limit=8) == []
    assert _recent_opponents(pd.DataFrame(), "Alpha", limit=8) == []


# --- adjust_fighter_profile: reliability shrinkage and end-to-end shape ----

def _profile(**overrides) -> dict:
    base = {
        "sig_accuracy_pct": 50.0, "sig_diff_per_min_pct": 50.0, "kd_per15_pct": 50.0,
        "sig_defense_pct": 50.0, "kd_absorbed_per15_pct": 50.0,
        "td_per15_pct": 50.0, "td_accuracy_pct": 50.0, "control_share_pct": 50.0,
        "td_defense_pct": 50.0, "sub_attempts_per15_pct": 50.0,
        "durability_score": 50.0, "pace_score": 50.0,
    }
    base.update(overrides)
    return base


def test_adjust_fighter_profile_reliability_scales_with_opponent_sample():
    config = UFCOpponentAdjustmentConfig(min_opponents_for_full_weight=6, recent_fights=8)
    fights = _fights([
        {"fighter": "Alpha", "opponent": "Foe 1", "event_date": "2025-01-01"},
        {"fighter": "Alpha", "opponent": "Foe 2", "event_date": "2025-02-01"},
        {"fighter": "Alpha", "opponent": "Foe 3", "event_date": "2025-03-01"},
    ])
    lookup = {f"foe {i}": _profile() for i in (1, 2, 3)}
    _, report = adjust_fighter_profile("Alpha", _profile(), fights, lookup, config=config)
    # 3 known opponents out of 6 needed for full weight -> reliability 0.5
    assert report["reliability"] == pytest.approx(0.5)
    assert report["opponent_sample"] == 3
    assert report["available"] is True


def test_adjust_fighter_profile_unavailable_with_no_known_opponents():
    config = UFCOpponentAdjustmentConfig()
    fights = _fights([{"fighter": "Alpha", "opponent": "Total Unknown", "event_date": "2025-01-01"}])
    lookup: dict = {}  # opponent not in the reference table
    adjusted, report = adjust_fighter_profile("Alpha", _profile(), fights, lookup, config=config)
    assert report["available"] is False
    assert report["reliability"] == 0.0
    # With zero reliability, adjustment must be a no-op -- base values pass through.
    assert adjusted["sig_accuracy_pct"] == pytest.approx(50.0)


def test_adjust_fighter_profile_against_elite_opponents_raises_offense_scores():
    config = UFCOpponentAdjustmentConfig(min_opponents_for_full_weight=1)
    fights = _fights([{"fighter": "Alpha", "opponent": "Elite Foe", "event_date": "2025-01-01"}])
    lookup = {"elite foe": _profile(sig_defense_pct=95.0, td_defense_pct=95.0)}
    adjusted, report = adjust_fighter_profile("Alpha", _profile(), fights, lookup, config=config)
    # Alpha's striking offense is scored against the opponent's striking
    # defense; beating a much better defense should raise the adjusted score.
    assert adjusted["striking_score"] > 50.0
    assert report["reliability"] == pytest.approx(1.0)


# --- build_opponent_adjusted_matchup: combined reliability ------------------

def _ratings_frame(fighters: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"fighter": f, "division": "Lightweight", "active_pool": True} for f in fighters])


def test_build_opponent_adjusted_matchup_reliability_is_the_weaker_side():
    config = UFCOpponentAdjustmentConfig(min_opponents_for_full_weight=1, recent_fights=8)
    fights = _fights([
        {"fighter": "Alpha", "opponent": "Known Foe", "event_date": "2025-01-01", "round": 3, "time": "5:00",
         "sig_diff_per_min": 3.0, "sig_accuracy": 0.5, "sig_defense": 0.5, "kd_per15": 0.5,
         "td_per15": 1.0, "td_accuracy": 0.4, "td_defense": 0.6, "sub_attempts_per15": 0.3,
         "control_share": 0.3, "kd_absorbed_per15": 0.3, "finish_loss_rate": 0.0, "sig_attempted_per_min": 6.0},
        {"fighter": "Bravo", "opponent": "Nobody Known", "event_date": "2025-01-01", "round": 3, "time": "5:00",
         "sig_diff_per_min": 2.0, "sig_accuracy": 0.4, "sig_defense": 0.5, "kd_per15": 0.4,
         "td_per15": 0.5, "td_accuracy": 0.3, "td_defense": 0.5, "sub_attempts_per15": 0.2,
         "control_share": 0.2, "kd_absorbed_per15": 0.4, "finish_loss_rate": 0.0, "sig_attempted_per_min": 5.0},
        {"fighter": "Known Foe", "opponent": "Alpha", "event_date": "2025-01-01", "round": 3, "time": "5:00",
         "sig_diff_per_min": 1.0, "sig_accuracy": 0.4, "sig_defense": 0.5, "kd_per15": 0.3,
         "td_per15": 0.4, "td_accuracy": 0.3, "td_defense": 0.5, "sub_attempts_per15": 0.1,
         "control_share": 0.2, "kd_absorbed_per15": 0.5, "finish_loss_rate": 0.1, "sig_attempted_per_min": 4.0},
    ])
    ratings = _ratings_frame(["Alpha", "Bravo", "Known Foe"])
    result = build_opponent_adjusted_matchup(
        "Alpha", "Bravo", _profile(), _profile(), fights, ratings, config=config,
    )
    # Alpha has one recognized opponent (reliability 1.0 at min_opponents=1),
    # Bravo has none (reliability 0.0) -> combined reliability must be the min.
    assert result["reliability"] == pytest.approx(0.0)
    assert result["available"] is False
    assert "version" in result
    assert "±8" in result["guardrail"]
