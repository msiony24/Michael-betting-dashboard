"""Tests for engine/tennis_handedness.py.

Includes the module's core safety property: an ambiguous name (e.g. two
different real players both recorded as "Zverev A.") must fail closed (None)
rather than silently guessing a hand for either of them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import engine.tennis_handedness as th
from engine.tennis_handedness import (
    _add_consistent,
    _exact_name_key,
    _identity_signature,
    handedness_matchup_profile,
    handedness_record_splits,
    normalize_hand,
    player_hand,
)


def test_normalize_hand_variants():
    assert normalize_hand("L") == "Left"
    assert normalize_hand("left-handed") == "Left"
    assert normalize_hand("R") == "Right"
    assert normalize_hand("right handed") == "Right"


def test_normalize_hand_unknown_returns_none():
    assert normalize_hand("ambidextrous") is None
    assert normalize_hand(None) is None
    assert normalize_hand("") is None


def test_exact_name_key_preserves_initials():
    # The whole point of this module: "Zverev A." and "Zverev M." must stay distinct.
    assert _exact_name_key("Zverev A.") != _exact_name_key("Zverev M.")
    assert _exact_name_key("Zverev A.") == "zverev a"


def test_exact_name_key_strips_suffixes():
    assert _exact_name_key("Player Name Jr.") == _exact_name_key("Player Name")


def test_identity_signature_returns_surname_and_initial():
    surname, initial = _identity_signature("Zverev A.")
    assert surname == "zverev"
    assert initial == "a"


def test_add_consistent_agreeing_values_stay_resolved():
    mapping: dict = {}
    _add_consistent(mapping, ("zverev", "a"), "Left")
    _add_consistent(mapping, ("zverev", "a"), "Left")
    assert mapping[("zverev", "a")] == "Left"


def test_add_consistent_conflicting_values_fail_closed_to_none():
    # This is the real-world Zverev case: two different players recorded
    # under the same surname+initial disagreeing on hand.
    mapping: dict = {}
    _add_consistent(mapping, ("zverev", "a"), "Left")
    _add_consistent(mapping, ("zverev", "a"), "Right")
    assert mapping[("zverev", "a")] is None


def test_add_consistent_ignores_empty_key():
    mapping: dict = {}
    _add_consistent(mapping, ("", ""), "Left")
    assert mapping == {}


# --- player_hand: file-backed lookup with manual override ------------------

def test_player_hand_manual_override_takes_precedence(tmp_path: Path):
    path = tmp_path / "hands.csv"
    pd.DataFrame([{"alias": "Rafael Nadal", "resolved_player": "Rafael Nadal", "hand": "Left"}]).to_csv(path, index=False)
    assert player_hand("Rafael Nadal", manual_hand="R", path=path) == "Right"


def test_player_hand_exact_match_from_file(tmp_path: Path):
    path = tmp_path / "hands.csv"
    pd.DataFrame([{"alias": "Rafael Nadal", "resolved_player": "Rafael Nadal", "hand": "Left"}]).to_csv(path, index=False)
    assert player_hand("Rafael Nadal", path=path) == "Left"


def test_player_hand_unknown_player_returns_none(tmp_path: Path):
    path = tmp_path / "hands.csv"
    pd.DataFrame([{"alias": "Rafael Nadal", "resolved_player": "Rafael Nadal", "hand": "Left"}]).to_csv(path, index=False)
    assert player_hand("Totally Unknown Player", path=path) is None


def test_player_hand_missing_file_returns_none(tmp_path: Path):
    assert player_hand("Rafael Nadal", path=tmp_path / "does_not_exist.csv") is None


def test_player_hand_ambiguous_alias_returns_none(tmp_path: Path):
    path = tmp_path / "hands.csv"
    pd.DataFrame([
        {"alias": "Zverev A.", "resolved_player": "Alexander Zverev", "hand": "Right"},
        {"alias": "Zverev A.", "resolved_player": "Anton Zverev", "hand": "Left"},
    ]).to_csv(path, index=False)
    assert player_hand("Zverev A.", path=path) is None


# --- handedness_record_splits / handedness_matchup_profile -----------------

HAND_MAP = {"Righty One": "Right", "Righty Two": "Right", "Lefty One": "Left", "Lefty Two": "Left"}


def _fake_player_hand(player, **kwargs):
    return HAND_MAP.get(player)


def _history(lefty_losses: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(5):
        rows.append({
            "winner_name": "Test Player", "loser_name": f"Righty {['One', 'Two'][i % 2]}",
            "tourney_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "surface": "Hard",
        })
    for i in range(lefty_losses):
        rows.append({
            "winner_name": f"Lefty {['One', 'Two'][i % 2]}", "loser_name": "Test Player",
            "tourney_date": pd.Timestamp("2024-02-01") + pd.Timedelta(days=i), "surface": "Hard",
        })
    frame = pd.DataFrame(rows)
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"])
    return frame


def test_handedness_record_splits_career_record_exact(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    splits = handedness_record_splits(_history(), "Test Player", pd.Timestamp("2024-06-01"), surface="Hard")
    assert splits["career"]["vs_right"] == {"wins": 5, "losses": 0, "matches": 5, "win_rate": 1.0}
    assert splits["career"]["vs_left"] == {"wins": 0, "losses": 3, "matches": 3, "win_rate": 0.0}


def test_handedness_record_splits_coverage_is_1_when_all_hands_known(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    splits = handedness_record_splits(_history(), "Test Player", pd.Timestamp("2024-06-01"), surface="Hard")
    assert splits["coverage"] == pytest.approx(1.0)
    assert splits["total_matches"] == 8


def test_handedness_record_splits_excludes_future_matches(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    # This is a regression test for a real bug found in this audit:
    # requesting splits as of a date before a player's earliest tracked
    # match used to crash with KeyError('opponent_hand') instead of
    # returning empty splits. Fixed in _decorate_with_opponent_hand.
    splits = handedness_record_splits(_history(), "Test Player", pd.Timestamp("2023-01-01"), surface="Hard")
    assert splits["total_matches"] == 0
    assert splits["career"]["vs_left"]["matches"] == 0
    assert splits["career"]["vs_right"]["matches"] == 0


def test_handedness_record_splits_surface_filter(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    splits = handedness_record_splits(_history(), "Test Player", pd.Timestamp("2024-06-01"), surface="Clay")
    # No clay matches in the fixture -> surface_split should be empty.
    assert splits["surface_split"]["vs_right"]["matches"] == 0


def test_handedness_matchup_profile_unavailable_when_opponent_hand_unknown(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    result = handedness_matchup_profile(_history(), "Test Player", "ambidextrous", pd.Timestamp("2024-06-01"), "Hard")
    assert result["available"] is False


def test_handedness_matchup_profile_unavailable_with_zero_relevant_matches(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    # A history that exists but contains no matches for this player at all.
    other_players_history = pd.DataFrame([{
        "winner_name": "Someone Else", "loser_name": "Another Player",
        "tourney_date": pd.Timestamp("2024-01-01"), "surface": "Hard",
    }])
    result = handedness_matchup_profile(other_players_history, "Test Player", "Left", pd.Timestamp("2024-06-01"), "Hard")
    assert result["available"] is False


def test_handedness_matchup_profile_adjustment_is_capped_at_0_02(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    result = handedness_matchup_profile(_history(), "Test Player", "Left", pd.Timestamp("2024-06-01"), "Hard")
    assert result["available"] is True
    assert abs(result["adjustment"]) <= 0.02 + 1e-9


def test_handedness_matchup_profile_perfect_record_favors_that_hand(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    # Test Player is 5-0 vs righties -> adjustment should be positive (favorable).
    result = handedness_matchup_profile(_history(), "Test Player", "Right", pd.Timestamp("2024-06-01"), "Hard")
    assert result["adjustment"] > 0


def test_handedness_matchup_profile_poor_record_disfavors_that_hand(monkeypatch):
    monkeypatch.setattr(th, "player_hand", _fake_player_hand)
    # Use a larger sample (8 losses) so the sample-reliability floor
    # ((n-3)/17, zero at n<=3) doesn't zero out the adjustment entirely.
    history = _history(lefty_losses=8)
    result = handedness_matchup_profile(history, "Test Player", "Left", pd.Timestamp("2024-06-01"), "Hard")
    assert result["available"] is True
    assert result["sample_reliability"] > 0
    assert result["adjustment"] < 0
