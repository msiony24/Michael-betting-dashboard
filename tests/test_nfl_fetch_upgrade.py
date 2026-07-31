from __future__ import annotations

import pandas as pd

from engine.nfl_fetch import build_team_snapshot


def _play(game_id, week, posteam, defteam, home, away, home_score, away_score, play_type, epa, yards, success, sack=0, qb_hit=0):
    return {
        "season": 2026,
        "season_type": "REG",
        "game_id": game_id,
        "week": week,
        "posteam": posteam,
        "defteam": defteam,
        "home_team": home,
        "away_team": away,
        "total_home_score": home_score,
        "total_away_score": away_score,
        "play_type": play_type,
        "pass_attempt": int(play_type == "pass"),
        "rush_attempt": int(play_type == "run"),
        "epa": epa,
        "yards_gained": yards,
        "success": success,
        "sack": sack,
        "qb_hit": qb_hit,
        "cpoe": 2.0 if play_type == "pass" else None,
        "interception": 0,
        "fumble_lost": 0,
        "play_id": 1,
    }


def test_snapshot_contains_new_weekly_components():
    rows = []
    # Use real abbreviations so the team-name mapping is exercised.
    for week in (1, 2):
        rows.extend([
            _play(f"g{week}", week, "KC", "BUF", "KC", "BUF", 27, 20, "pass", 0.4, 12, 1),
            _play(f"g{week}", week, "KC", "BUF", "KC", "BUF", 27, 20, "run", 0.2, 6, 1),
            _play(f"g{week}", week, "BUF", "KC", "KC", "BUF", 27, 20, "pass", -0.1, 5, 0, sack=1, qb_hit=1),
            _play(f"g{week}", week, "BUF", "KC", "KC", "BUF", 27, 20, "run", -0.2, 2, 0),
        ])
    frame = pd.DataFrame(rows)
    snapshot = build_team_snapshot(frame, 2026)

    assert {"offensive_line", "defensive_line", "secondary", "recent_form", "through_week"}.issubset(snapshot.columns)
    assert set(snapshot["team"]) == {"Buffalo Bills", "Kansas City Chiefs"}
    assert snapshot["recent_form"].notna().all()
