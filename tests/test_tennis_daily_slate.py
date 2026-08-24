from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from engine.tennis_daily_slate import merge_tennis_schedule_with_market

ET = ZoneInfo("America/New_York")


def _schedule():
    return pd.DataFrame([
        {
            "event_id": "101",
            "start_time": datetime(2026, 8, 24, 14, 0, tzinfo=ET),
            "time_et": "2:00 PM",
            "sport": "ATP Winston-Salem",
            "participant_a": "Sebastian Korda",
            "participant_b": "Alex Michelsen",
            "odds_a": -145,
            "odds_b": 125,
            "book_a": "bet365 (API-Tennis)",
            "book_b": "bet365 (API-Tennis)",
        },
        {
            "event_id": "102",
            "start_time": datetime(2026, 8, 24, 16, 0, tzinfo=ET),
            "time_et": "4:00 PM",
            "sport": "ATP Winston-Salem",
            "participant_a": "Player One",
            "participant_b": "Player Two",
            "odds_a": pd.NA,
            "odds_b": pd.NA,
            "book_a": "—",
            "book_b": "—",
        },
    ])


def test_api_tennis_schedule_survives_when_primary_odds_feed_is_empty():
    merged = merge_tennis_schedule_with_market(_schedule(), pd.DataFrame())
    assert len(merged) == 2
    assert "ATP Winston-Salem" in set(merged["sport"])
    assert merged.loc[0, "odds_a"] == -145


def test_primary_us_odds_override_api_tennis_fallback_even_if_player_order_reversed():
    market = pd.DataFrame([
        {
            "event_id": "odds-api-1",
            "start_time": datetime(2026, 8, 24, 14, 5, tzinfo=ET),
            "time_et": "2:05 PM",
            "sport": "ATP Winston Salem Open",
            "participant_a": "Alex Michelsen",
            "participant_b": "Sebastian Korda",
            "odds_a": 135,
            "odds_b": -150,
            "book_a": "DraftKings",
            "book_b": "FanDuel",
        }
    ])
    merged = merge_tennis_schedule_with_market(_schedule(), market)
    korda = merged[merged["participant_a"] == "Sebastian Korda"].iloc[0]
    assert korda["odds_a"] == -150
    assert korda["book_a"] == "FanDuel"
    assert korda["odds_b"] == 135
    assert korda["book_b"] == "DraftKings"


def test_market_only_match_is_retained_when_schedule_provider_misses_it():
    market = pd.DataFrame([
        {
            "event_id": "odds-api-2",
            "start_time": datetime(2026, 8, 24, 18, 0, tzinfo=ET),
            "time_et": "6:00 PM",
            "sport": "WTA Monterrey",
            "participant_a": "Ann Li",
            "participant_b": "Camila Osorio",
            "odds_a": -110,
            "odds_b": 100,
            "book_a": "DraftKings",
            "book_b": "DraftKings",
        }
    ])
    merged = merge_tennis_schedule_with_market(_schedule(), market)
    assert len(merged) == 3
    assert "Ann Li" in set(merged["participant_a"])
