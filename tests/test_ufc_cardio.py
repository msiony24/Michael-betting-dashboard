from __future__ import annotations

import pandas as pd
import pytest

from engine.ufc_cardio import (
    UFCCardioConfig,
    _retention,
    _round_minutes,
    _safe,
    build_cardio_matchup,
    fighter_cardio_profile,
)


# --- small numeric helpers ---------------------------------------------------

def test_safe_handles_missing_and_invalid():
    assert _safe("3.5") == 3.5
    assert _safe(None) is None
    assert _safe("nope") is None
    assert _safe(float("nan")) is None


def test_round_minutes_full_five_for_rounds_before_the_finish():
    assert _round_minutes(fight_round=3, finish_time="2:15", round_no=1) == 5.0
    assert _round_minutes(fight_round=3, finish_time="2:15", round_no=2) == 5.0


def test_round_minutes_zero_for_rounds_after_the_finish():
    assert _round_minutes(fight_round=2, finish_time="1:00", round_no=3) == 0.0


def test_round_minutes_partial_for_the_finishing_round():
    assert _round_minutes(fight_round=2, finish_time="2:30", round_no=2) == pytest.approx(2.5)


def test_round_minutes_malformed_time_defaults_to_full_round():
    assert _round_minutes(fight_round=2, finish_time="garbage", round_no=2) == 5.0
    assert _round_minutes(fight_round=2, finish_time="", round_no=2) == 5.0


# --- _retention: clip bounds -------------------------------------------------

def test_retention_missing_values_returns_none():
    assert _retention(None, 5.0) is None
    assert _retention(5.0, None) is None


def test_retention_normal_ratio():
    assert _retention(4.0, 5.0) == pytest.approx(0.8)


def test_retention_clips_to_bounds():
    assert _retention(100.0, 1.0) == pytest.approx(1.35)
    assert _retention(0.1, 100.0) == pytest.approx(0.50)


def test_retention_first_zero_non_inverse_returns_none():
    assert _retention(5.0, 0.0) is None


def test_retention_inverse_mode_both_zero_is_perfectly_stable():
    assert _retention(0.0, 0.0, inverse=True) == 1.0


def test_retention_inverse_mode_later_zero_is_improvement():
    assert _retention(0.0, 5.0, inverse=True) == 1.25


def test_retention_inverse_mode_first_zero_is_decline():
    assert _retention(5.0, 0.0, inverse=True) == 0.75


def test_retention_inverse_mode_normal_ratio_is_flipped_and_still_clipped():
    # Absorbing fewer strikes later (2) than in round 1 (4) is an improvement
    # in defense, so the inverse ratio (first/later = 2.0) is > 1 -- but it's
    # still subject to the same [0.50, 1.35] safety clip as every other case.
    result = _retention(2.0, 4.0, inverse=True)
    assert result == pytest.approx(1.35)


# --- fighter_cardio_profile: availability gating -----------------------------

def test_cardio_profile_unavailable_without_round_level_columns():
    fights = pd.DataFrame([{"fighter": "Alpha", "opponent": "Beta", "event_date": "2025-01-01"}])
    profile = fighter_cardio_profile(fights, "Alpha")
    assert profile["available"] is False
    assert "reason" in profile


def test_cardio_profile_unavailable_with_no_fight_history():
    profile = fighter_cardio_profile(pd.DataFrame(), "Alpha")
    assert profile["available"] is False


def _two_sided_fight(fight_url: str, event_date: str, fighter_a: str, fighter_b: str, a_stats: dict, b_stats: dict) -> list[dict]:
    """Build the two paired rows _attach_opponent_round_stats expects."""
    row_a = {"fighter": fighter_a, "opponent": fighter_b, "fight_url": fight_url, "event_date": event_date, **a_stats}
    row_b = {"fighter": fighter_b, "opponent": fighter_a, "fight_url": fight_url, "event_date": event_date, **b_stats}
    return [row_a, row_b]


def _flat_round_stats(round_no: int, sig_landed: float, sig_attempted: float, td_landed: float = 1.0, td_attempted: float = 2.0, control_seconds: float = 60.0) -> dict:
    return {
        f"r{round_no}_sig_str_landed": sig_landed,
        f"r{round_no}_sig_str_attempted": sig_attempted,
        f"r{round_no}_td_landed": td_landed,
        f"r{round_no}_td_attempted": td_attempted,
        f"r{round_no}_control_seconds": control_seconds,
    }


def _steady_fighter_stats(rounds: int = 3) -> dict:
    """A fighter whose output is identical every round (no fade, no improvement)."""
    stats = {"round": rounds, "time": "5:00"}
    for r in range(1, rounds + 1):
        stats.update(_flat_round_stats(r, sig_landed=20.0, sig_attempted=40.0))
    return stats


def _fading_fighter_stats(rounds: int = 3) -> dict:
    """A fighter whose output drops sharply after round 1."""
    stats = {"round": rounds, "time": "5:00"}
    stats.update(_flat_round_stats(1, sig_landed=30.0, sig_attempted=50.0))
    for r in range(2, rounds + 1):
        stats.update(_flat_round_stats(r, sig_landed=8.0, sig_attempted=30.0))
    return stats


def test_cardio_profile_available_with_full_round_data():
    fights = pd.DataFrame(_two_sided_fight(
        "f1", "2025-01-01", "Alpha", "Beta", _steady_fighter_stats(), _steady_fighter_stats(),
    ))
    profile = fighter_cardio_profile(fights, "Alpha")
    assert profile["available"] is True
    assert profile["sample"] == 1
    assert "cardio_score" in profile
    assert 20.0 <= profile["cardio_score"] <= 80.0


def test_cardio_profile_flags_sharp_fade_for_a_fighter_who_fades():
    fights = pd.DataFrame(_two_sided_fight(
        "f1", "2025-01-01", "Alpha", "Beta", _fading_fighter_stats(), _steady_fighter_stats(),
    ))
    profile = fighter_cardio_profile(fights, "Alpha")
    assert profile["available"] is True
    assert profile["retention"] < 1.0
    assert profile["trend"] in {"Moderate fade", "Sharp fade"}


def test_cardio_profile_steady_fighter_trend_is_stable_or_better():
    fights = pd.DataFrame(_two_sided_fight(
        "f1", "2025-01-01", "Alpha", "Beta", _steady_fighter_stats(), _steady_fighter_stats(),
    ))
    profile = fighter_cardio_profile(fights, "Alpha")
    assert profile["trend"] in {"Stable", "Improves / sustains late"}


# --- build_cardio_matchup: capped adjustment ---------------------------------

def test_cardio_matchup_unavailable_when_either_fighter_lacks_data():
    fights = pd.DataFrame([{"fighter": "Alpha", "opponent": "Beta", "event_date": "2025-01-01"}])
    result = build_cardio_matchup(fights, "Alpha", "Beta")
    assert result["available"] is False
    assert result["adjustment_a"] == 0.0


def test_cardio_matchup_adjustment_never_exceeds_cap_for_3_round_fights():
    fights = pd.DataFrame(
        _two_sided_fight("f1", "2025-01-01", "Alpha", "Beta", _steady_fighter_stats(), _fading_fighter_stats())
    )
    config = UFCCardioConfig()
    result = build_cardio_matchup(fights, "Alpha", "Beta", rounds=3, config=config)
    assert result["available"] is True
    assert abs(result["adjustment_a"]) <= config.max_adjustment_3r + 1e-9


def test_cardio_matchup_5_round_cap_is_larger_than_3_round_cap():
    config = UFCCardioConfig()
    assert config.max_adjustment_5r > config.max_adjustment_3r


def test_cardio_matchup_advantage_favors_the_fresher_fighter():
    fights = pd.DataFrame(
        _two_sided_fight("f1", "2025-01-01", "Alpha", "Beta", _steady_fighter_stats(), _fading_fighter_stats())
    )
    result = build_cardio_matchup(fights, "Alpha", "Beta")
    assert result["advantage"] == "Alpha"
    assert result["adjustment_a"] > 0


def test_cardio_matchup_even_when_gap_is_small():
    # Two fighters with identical cardio profiles -> gap of exactly 0.
    fights = pd.DataFrame(
        _two_sided_fight("f1", "2025-01-01", "Alpha", "Beta", _steady_fighter_stats(), _steady_fighter_stats())
    )
    result = build_cardio_matchup(fights, "Alpha", "Beta")
    assert result["advantage"] == "Even"
    assert result["cardio_gap"] == pytest.approx(0.0)
