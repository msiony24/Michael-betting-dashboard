from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.player_profiles import (
    PlayerProfile,
    _find_historical_name,
    _historical_metrics,
    _player_history,
    build_player_profile,
    compare_experience,
    experience_reliability,
)


def _matches() -> pd.DataFrame:
    # A small synthetic history for "Jannik Sinner": wins and losses, mixed
    # surfaces and levels, spanning several dates so date filtering and
    # recency windows are both exercised.
    rows = [
        {"tourney_date": "20240110", "winner_name": "Jannik Sinner", "loser_name": "Novak Djokovic",
         "winner_rank": 4, "loser_rank": 1, "surface": "Hard", "tourney_level": "G"},
        {"tourney_date": "20240301", "winner_name": "Carlos Alcaraz", "loser_name": "Jannik Sinner",
         "winner_rank": 2, "loser_rank": 4, "surface": "Clay", "tourney_level": "M"},
        {"tourney_date": "20240501", "winner_name": "Jannik Sinner", "loser_name": "Random Journeyman",
         "winner_rank": 3, "loser_rank": 180, "surface": "Clay", "tourney_level": "A"},
        {"tourney_date": "20240815", "winner_name": "Jannik Sinner", "loser_name": "Daniil Medvedev",
         "winner_rank": 1, "loser_rank": 5, "surface": "Hard", "tourney_level": "M"},
        # After the event date used in tests below; must never be counted.
        {"tourney_date": "20241001", "winner_name": "Jannik Sinner", "loser_name": "Carlos Alcaraz",
         "winner_rank": 1, "loser_rank": 2, "surface": "Hard", "tourney_level": "F"},
    ]
    return pd.DataFrame(rows)


EVENT_DATE = date(2024, 9, 1)


def test_find_historical_name_exact_match():
    assert _find_historical_name(_matches(), "Jannik Sinner") == "Jannik Sinner"


def test_find_historical_name_matches_via_canonical_key_variants():
    # "J. Sinner" should resolve to the same canonical identity as the
    # dataset's "Jannik Sinner" entries.
    assert _find_historical_name(_matches(), "J. Sinner") == "Jannik Sinner"


def test_find_historical_name_unknown_player_returns_none():
    assert _find_historical_name(_matches(), "Nobody Real") is None


def test_find_historical_name_empty_frame_returns_none():
    assert _find_historical_name(pd.DataFrame(), "Jannik Sinner") is None


def test_player_history_excludes_matches_on_or_after_event_date():
    history = _player_history(_matches(), "Jannik Sinner", EVENT_DATE)
    assert len(history) == 4  # the 2024-10-01 match must be excluded
    assert (pd.to_datetime(history["tourney_date"]) < pd.Timestamp(EVENT_DATE)).all()


def test_historical_metrics_counts_wins_and_losses():
    history = _player_history(_matches(), "Jannik Sinner", EVENT_DATE)
    metrics = _historical_metrics(history, "Jannik Sinner", EVENT_DATE)
    assert metrics["career_matches"] == 4
    assert metrics["career_wins"] == 3
    assert metrics["career_losses"] == 1


def test_historical_metrics_surface_breakdown():
    history = _player_history(_matches(), "Jannik Sinner", EVENT_DATE)
    metrics = _historical_metrics(history, "Jannik Sinner", EVENT_DATE)
    # Fixture (before the 2024-09-01 event date): 2024-01-10 Hard (win),
    # 2024-03-01 Clay (loss), 2024-05-01 Clay (win), 2024-08-15 Hard (win).
    assert metrics["surface_matches"]["Hard"] == 2
    assert metrics["surface_matches"]["Clay"] == 2
    assert metrics["surface_wins"]["Clay"] == 1  # won one clay match, lost one
    assert metrics["surface_wins"]["Hard"] == 2


def test_historical_metrics_top_rank_thresholds():
    history = _player_history(_matches(), "Jannik Sinner", EVENT_DATE)
    metrics = _historical_metrics(history, "Jannik Sinner", EVENT_DATE)
    # Beat #1 Djokovic, lost to #2 Alcaraz, beat #5 Medvedev -> all count
    # toward top-10; the win over #180 does not.
    assert metrics["top_10_matches"] == 3
    assert metrics["top_10_record"] == "2-1"


def test_historical_metrics_empty_history_returns_empty_dict():
    assert _historical_metrics(pd.DataFrame(), "Jannik Sinner", EVENT_DATE) == {}


def test_build_player_profile_flags_unknown_player_without_inventing_stats():
    profile = build_player_profile(
        _matches(), "Nobody Real", EVENT_DATE, include_api=False, use_store=False,
    )
    assert profile.historical_name is None
    assert profile.career_matches == 0
    assert "historical_player_not_found" in profile.data_flags
    assert "very_small_historical_sample" in profile.data_flags


def test_build_player_profile_small_sample_flag_boundary():
    profile = build_player_profile(
        _matches(), "Jannik Sinner", EVENT_DATE, include_api=False, use_store=False,
    )
    # 4 historical matches in the fixture -> "very small", not "small".
    assert profile.career_matches == 4
    assert "very_small_historical_sample" in profile.data_flags
    assert "small_historical_sample" not in profile.data_flags


def test_build_player_profile_skips_api_when_disabled():
    profile = build_player_profile(
        _matches(), "Jannik Sinner", EVENT_DATE, include_api=False, use_store=False,
    )
    assert profile.api_source == "unavailable"
    assert "api_tennis_standings" not in profile.data_sources


# --- experience_reliability / compare_experience ---------------------------

def test_experience_reliability_bounds():
    empty = PlayerProfile(requested_name="Nobody")
    assert 0.0 <= experience_reliability(empty) <= 1.0

    veteran = PlayerProfile(
        requested_name="Veteran",
        career_matches=500,
        surface_matches={"Hard": 200},
        ranking=1,
    )
    assert experience_reliability(veteran) == pytest.approx(1.0, abs=0.01)


def test_experience_reliability_increases_with_sample_size():
    small = PlayerProfile(requested_name="Small", career_matches=5, ranking=50)
    large = PlayerProfile(requested_name="Large", career_matches=200, ranking=50)
    assert experience_reliability(large) > experience_reliability(small)


def test_compare_experience_favors_more_experienced_player():
    veteran = PlayerProfile(
        requested_name="Veteran", career_matches=400, surface_matches={"Hard": 150},
        grand_slam_matches=80, masters_matches=60, top_50_matches=200, ranking=3,
    )
    newcomer = PlayerProfile(
        requested_name="Newcomer", career_matches=8, surface_matches={"Hard": 3},
        grand_slam_matches=1, masters_matches=0, top_50_matches=1, ranking=250,
    )
    result = compare_experience(veteran, newcomer, "Hard")
    assert result["advantage"] == "Veteran"
    assert result["probability_adjustment_a"] > 0


def test_compare_experience_adjustment_is_capped():
    veteran = PlayerProfile(
        requested_name="Veteran", career_matches=2000, surface_matches={"Hard": 1000},
        grand_slam_matches=400, masters_matches=400, top_50_matches=1500, ranking=1,
    )
    newcomer = PlayerProfile(requested_name="Newcomer", career_matches=0, ranking=None)
    result = compare_experience(veteran, newcomer, "Hard", maximum_probability_adjustment=0.04)
    assert abs(result["probability_adjustment_a"]) <= 0.04 + 1e-9


def test_compare_experience_near_equal_profiles_report_even():
    a = PlayerProfile(requested_name="A", career_matches=50, surface_matches={"Hard": 20}, ranking=20)
    b = PlayerProfile(requested_name="B", career_matches=50, surface_matches={"Hard": 20}, ranking=20)
    result = compare_experience(a, b, "Hard")
    assert result["advantage"] == "Even"
    assert abs(result["probability_adjustment_a"]) < 0.005
