from __future__ import annotations

import pandas as pd

from engine.nfl_fetch import build_game_quality_snapshot
from engine.nfl_game_quality import build_game_quality_context


def _play(game, week, offense, defense, home, away, home_score, away_score, epa, yards, success, turnover=0):
    return {
        "season": 2026, "season_type": "REG", "game_id": game, "week": week,
        "gameday": f"2026-09-{6+week:02d}", "posteam": offense, "defteam": defense,
        "home_team": home, "away_team": away, "total_home_score": home_score,
        "total_away_score": away_score, "play_type": "pass", "pass_attempt": 1,
        "rush_attempt": 0, "epa": epa, "yards_gained": yards, "success": success,
        "interception": turnover, "fumble_lost": 0, "play_id": week * 100 + 1,
    }


def test_game_quality_prefers_repeatable_play_over_scoreboard_luck(tmp_path):
    rows = []
    for week in range(1, 6):
        # KC consistently controls underlying play.
        rows += [
            _play(f"g{week}", week, "KC", "BUF", "KC", "BUF", 24, 21, 0.55, 11, 1),
            _play(f"g{week}", week, "KC", "BUF", "KC", "BUF", 24, 21, 0.35, 8, 1),
            _play(f"g{week}", week, "BUF", "KC", "KC", "BUF", 24, 21, -0.10, 4, 0),
            _play(f"g{week}", week, "BUF", "KC", "KC", "BUF", 24, 21, 0.05, 5, 0),
        ]
    quality = build_game_quality_snapshot(pd.DataFrame(rows), 2026)
    path = tmp_path / "game_quality.csv"
    quality.to_csv(path, index=False)
    context = build_game_quality_context(
        away_team="Buffalo Bills", home_team="Kansas City Chiefs", season=2026,
        game_date="2026-10-30", quality_path=path,
    )
    assert context["available"]
    assert context["home"]["quality_score"] > context["away"]["quality_score"]
    assert context["home_margin_adjustment"] == 0.0
    assert context["diagnostic_only"] is True


def test_no_current_season_sample_means_no_adjustment(tmp_path):
    pd.DataFrame([{"season": 2025, "team_abbr": "KC", "gameday": "2025-10-01"}]).to_csv(tmp_path / "q.csv", index=False)
    context = build_game_quality_context(
        away_team="Buffalo Bills", home_team="Kansas City Chiefs", season=2026,
        quality_path=tmp_path / "q.csv",
    )
    assert not context["available"]
    assert context["home_margin_adjustment"] == 0.0
