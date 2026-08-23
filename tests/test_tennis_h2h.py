"""Tests for engine/tennis_h2h.py: head-to-head match history summary."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.tennis_h2h import build_head_to_head_summary


def test_empty_matches_returns_empty_summary():
    result = build_head_to_head_summary(pd.DataFrame(), "Player A", "Player B", "Hard")
    assert result["meetings"] == 0
    assert result["wins_a"] == 0
    assert result["last_meeting"] is None


def test_missing_winner_loser_columns_returns_empty_summary():
    matches = pd.DataFrame([{"surface": "Hard"}])
    result = build_head_to_head_summary(matches, "Player A", "Player B", "Hard")
    assert result["meetings"] == 0


def test_no_prior_meetings_returns_empty_but_with_resolution_info():
    matches = pd.DataFrame([
        {"winner_name": "Someone Else", "loser_name": "Another Player", "surface": "Hard", "tourney_date": "20240101"},
    ])
    result = build_head_to_head_summary(matches, "Player A", "Player B", "Hard")
    assert result["meetings"] == 0
    assert result["requested_player_a"] == "Player A"


def _two_meetings() -> pd.DataFrame:
    return pd.DataFrame([
        {"winner_name": "Novak Djokovic", "loser_name": "Carlos Alcaraz", "surface": "Hard",
         "tourney_date": "20240101", "tourney_name": "Test Open", "round": "F", "score": "6-4 6-3"},
        {"winner_name": "Carlos Alcaraz", "loser_name": "Novak Djokovic", "surface": "Clay",
         "tourney_date": "20230601", "tourney_name": "Clay Open", "round": "SF", "score": "7-5 6-2"},
    ])


def test_head_to_head_counts_meetings_and_wins():
    result = build_head_to_head_summary(_two_meetings(), "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["meetings"] == 2
    assert result["wins_a"] == 1
    assert result["wins_b"] == 1


def test_head_to_head_surface_filter_only_counts_matching_surface():
    result = build_head_to_head_summary(_two_meetings(), "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["surface_meetings"] == 1
    assert result["surface_wins_a"] == 1
    assert result["surface_wins_b"] == 0


def test_head_to_head_surface_filter_is_case_insensitive():
    result = build_head_to_head_summary(_two_meetings(), "Novak Djokovic", "Carlos Alcaraz", "hard")
    assert result["surface_meetings"] == 1


def test_head_to_head_last_meeting_is_the_most_recent_by_date():
    result = build_head_to_head_summary(_two_meetings(), "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["last_meeting"]["date"] == "2024-01-01"
    assert result["last_meeting"]["event"] == "Test Open — F"


def test_head_to_head_winner_display_uses_requested_name_not_raw_provider_string():
    # If the requested name is a fuller/different form than the raw historical
    # string, the winner in last_meeting should still show the requested name.
    result = build_head_to_head_summary(_two_meetings(), "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["last_meeting"]["winner"] == "Novak Djokovic"


def test_head_to_head_works_with_alternate_column_names():
    matches = pd.DataFrame([
        {"winner": "Novak Djokovic", "loser": "Carlos Alcaraz", "Surface": "Hard", "date": "2024-01-01"},
    ])
    result = build_head_to_head_summary(matches, "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["meetings"] == 1


def test_head_to_head_reversed_player_order_still_matches():
    result = build_head_to_head_summary(_two_meetings(), "Carlos Alcaraz", "Novak Djokovic", "Hard")
    assert result["meetings"] == 2
    # wins_a now refers to Alcaraz (requested first), wins_b to Djokovic.
    assert result["wins_a"] == 1
    assert result["wins_b"] == 1


def test_head_to_head_numeric_date_format_parses_correctly():
    matches = pd.DataFrame([
        {"winner_name": "Novak Djokovic", "loser_name": "Carlos Alcaraz", "surface": "Hard", "tourney_date": "20240315"},
    ])
    result = build_head_to_head_summary(matches, "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["last_meeting"]["date"] == "2024-03-15"


def test_head_to_head_missing_optional_columns_does_not_crash():
    matches = pd.DataFrame([{"winner_name": "Novak Djokovic", "loser_name": "Carlos Alcaraz"}])
    result = build_head_to_head_summary(matches, "Novak Djokovic", "Carlos Alcaraz", "Hard")
    assert result["meetings"] == 1
    assert result["last_meeting"]["date"] == "Date unavailable"
    assert result["last_meeting"]["event"] == "Event unavailable"
