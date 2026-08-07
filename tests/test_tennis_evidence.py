from datetime import date

import pandas as pd

from engine.tennis_evidence import build_tennis_evidence_packet


def _matches():
    return pd.DataFrame([
        {"tourney_date": pd.Timestamp("2026-05-29"), "tourney_name": "French Open", "surface": "Clay", "tourney_level": "G", "round": "R32", "winner_name": "Fonseca J.", "loser_name": "Djokovic N.", "winner_rank": 30, "loser_rank": 4, "score": "4-6 4-6 6-3 7-5 7-5"},
        {"tourney_date": pd.Timestamp("2026-05-31"), "tourney_name": "French Open", "surface": "Clay", "tourney_level": "G", "round": "R16", "winner_name": "Fonseca J.", "loser_name": "Ruud C.", "winner_rank": 30, "loser_rank": 16, "score": "7-5 7-6 5-7 6-2"},
        {"tourney_date": pd.Timestamp("2026-07-03"), "tourney_name": "Wimbledon", "surface": "Grass", "tourney_level": "G", "round": "R32", "winner_name": "Safiullin R.", "loser_name": "Fonseca J.", "winner_rank": 132, "loser_rank": 27, "score": "6-3 6-3 6-3"},
        {"tourney_date": pd.Timestamp("2026-07-17"), "tourney_name": "Gstaad", "surface": "Clay", "tourney_level": "A", "round": "QF", "winner_name": "Cerundolo J.M.", "loser_name": "Ruud C.", "winner_rank": 45, "loser_rank": 13, "score": "3-6 7-5 6-2"},
    ])


def test_packet_contains_verified_recent_wins_and_freshness():
    packet = build_tennis_evidence_packet(
        _matches(), "Joao Fonseca", "Casper Ruud", date(2026, 8, 7), "Hard", "Cincinnati"
    )
    assert packet["latest_match_date_in_database"] == "2026-07-17"
    a = packet["player_a"]
    assert a["recent_matches"][0]["opponent"] == "Safiullin R."
    assert any(row["opponent"] == "Djokovic N." for row in a["top_10_wins"])
    assert any(row["opponent"] == "Ruud C." for row in a["top_20_wins"])


def test_future_or_same_day_match_is_not_used_as_prematch_evidence():
    frame = _matches()
    extra = pd.DataFrame([{
        "tourney_date": pd.Timestamp("2026-08-07"), "tourney_name": "Cincinnati", "surface": "Hard", "tourney_level": "M", "round": "R32",
        "winner_name": "Fonseca J.", "loser_name": "Ruud C.", "winner_rank": 20, "loser_rank": 15, "score": "6-4 6-4"
    }])
    packet = build_tennis_evidence_packet(pd.concat([frame, extra], ignore_index=True), "Joao Fonseca", "Casper Ruud", date(2026, 8, 7), "Hard")
    assert all(row["date"] != "2026-08-07" for row in packet["player_a"]["recent_matches"])
