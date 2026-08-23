from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine import nfl_player_weekly


def _frame(season: int, player_id: str = "00-TEST", week: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [player_id],
            "player_display_name": ["Test Player"],
            "position": ["QB"],
            "recent_team": ["BUF"],
            "season": [season],
            "week": [week],
            "season_type": ["REG"],
            "attempts": [30],
            "passing_yards": [250],
            "passing_tds": [2],
            "interceptions": [0],
        }
    )


def test_refresh_player_weekly_falls_back_and_caps_prior_season(monkeypatch, tmp_path: Path):
    def fake_load(season: int):
        if season == 2026:
            raise ValueError("2026 weekly stats not published yet")
        assert season == 2025
        return _frame(2025, week=18)

    monkeypatch.setattr(nfl_player_weekly, "_load_one", fake_load)
    output = tmp_path / "player_weekly_stats.csv"
    metadata = tmp_path / "player_weekly_stats_metadata.json"

    result = nfl_player_weekly.refresh_player_weekly_stats(
        2026, output_path=output, metadata_path=metadata
    )

    saved = pd.read_csv(output)
    meta = json.loads(metadata.read_text())
    assert result["active_season"] == 2025
    assert result["fallback_prior"] is True
    assert result["performance_cap"] == 0.20
    assert saved["macabets_performance_cap"].eq(0.20).all()
    assert saved["macabets_is_fallback"].astype(bool).all()
    assert meta["requested_season"] == 2026
    assert meta["active_season"] == 2025
    assert meta["through_week"] == 18


def test_refresh_player_weekly_switches_to_current_season_when_available(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(nfl_player_weekly, "_load_one", lambda season: _frame(season, week=2))
    output = tmp_path / "player_weekly_stats.csv"
    metadata = tmp_path / "player_weekly_stats_metadata.json"

    result = nfl_player_weekly.refresh_player_weekly_stats(
        2026, output_path=output, metadata_path=metadata
    )

    saved = pd.read_csv(output)
    meta = json.loads(metadata.read_text())
    assert result["active_season"] == 2026
    assert result["fallback_prior"] is False
    assert result["performance_cap"] == 0.80
    assert saved["macabets_performance_cap"].eq(0.80).all()
    assert not saved["macabets_is_fallback"].astype(bool).any()
    assert meta["active_season"] == 2026
    assert meta["fallback_prior"] is False
