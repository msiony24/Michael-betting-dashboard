"""Shape/invariant checks for engine/nfl_data.py.

This module has no functions of its own -- it assembles constants at import
time from already-tested building blocks (nfl_team_state, nfl_foundation,
nfl_availability). These tests lock in the resulting shape rather than
re-testing logic covered elsewhere.
"""
from __future__ import annotations

from engine.nfl_data import (
    NFL_DATA_STATUS,
    NFL_TEAM_RATINGS,
    NFL_TEAMS,
    TEAM_RATING_WEIGHTS,
    VENUE_TYPES,
    WEATHER_OPTIONS,
)


def test_nfl_teams_list_has_32_unique_entries():
    assert len(NFL_TEAMS) == 32
    assert len(set(NFL_TEAMS)) == 32


def test_team_rating_weights_sum_to_one():
    assert sum(TEAM_RATING_WEIGHTS.values()) == 1.0


def test_nfl_team_ratings_only_contains_real_teams():
    assert set(NFL_TEAM_RATINGS.keys()) <= set(NFL_TEAMS)


def test_nfl_team_ratings_components_cover_every_weight_key():
    for team, components in NFL_TEAM_RATINGS.items():
        assert set(TEAM_RATING_WEIGHTS.keys()) <= set(components.keys())


def test_nfl_data_status_has_required_keys():
    required = {
        "available", "teams", "data_source", "season", "through_week",
        "rating_mode", "reason", "foundation_available_datasets",
        "availability_players",
    }
    assert required <= set(NFL_DATA_STATUS.keys())


def test_nfl_data_status_teams_count_matches_ratings():
    assert NFL_DATA_STATUS["teams"] == len(NFL_TEAM_RATINGS)


def test_venue_and_weather_options_are_nonempty():
    assert "Outdoor" in VENUE_TYPES
    assert "Normal" in WEATHER_OPTIONS
