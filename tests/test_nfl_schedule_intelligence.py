from pathlib import Path
import pandas as pd

from engine.nfl_schedule_intelligence import build_schedule_context


def _powers():
    names = [
        "New England Patriots", "Seattle Seahawks", "San Francisco 49ers", "Los Angeles Rams",
        "Buffalo Bills", "Houston Texans", "Baltimore Ravens", "Indianapolis Colts",
    ]
    return {name: float(i) for i, name in enumerate(names, 1)}


def test_schedule_context_finds_2026_game_and_neutral_division():
    power = _powers()
    # Add all teams used by the actual schedule through a neutral lookup-safe fallback.
    from engine.nfl_fetch import TEAM_ABBR_TO_NAME
    for name in TEAM_ABBR_TO_NAME.values():
        power.setdefault(name, 0.0)
    ctx = build_schedule_context(
        away_team="San Francisco 49ers",
        home_team="Los Angeles Rams",
        season=2026,
        team_power=power,
        game_date="2026-09-10",
        week=1,
    )
    assert ctx["available"] is True
    assert ctx["div_game"] is True
    assert ctx["scheduled_neutral"] is True
    assert ctx["confidence_penalty"] > 0


def test_future_schedule_difficulty_does_not_create_sos_side_adjustment():
    power = _powers()
    from engine.nfl_fetch import TEAM_ABBR_TO_NAME
    for name in TEAM_ABBR_TO_NAME.values():
        power.setdefault(name, 0.0)
    ctx = build_schedule_context(
        away_team="New England Patriots",
        home_team="Seattle Seahawks",
        season=2026,
        team_power=power,
        game_date="2026-09-09",
        week=1,
    )
    assert ctx["away"]["full_schedule_games"] == 17
    assert ctx["home"]["full_schedule_games"] == 17
    assert ctx["sos_home_margin_adjustment"] == 0.0
