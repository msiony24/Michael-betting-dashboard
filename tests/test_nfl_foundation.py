from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import engine.nfl_foundation as foundation
from engine.nfl_fetch import FetchResult


class FakeNFL:
    def load_schedules(self, seasons):
        return pd.DataFrame([{"season": seasons[0], "game_id": "g1"}])

    def load_rosters(self, seasons):
        return pd.DataFrame([{"season": seasons[0], "gsis_id": "p1", "team": "KC"}])

    def load_rosters_weekly(self, seasons):
        return pd.DataFrame([
            {"season": seasons[0], "week": 1, "gsis_id": "p1", "team": "KC", "status": "Active"},
            {"season": seasons[0], "week": 2, "gsis_id": "p1", "team": "KC", "status": "Active"},
        ])

    def load_player_stats(self, seasons, summary_level="week"):
        return pd.DataFrame([{"season": seasons[0], "week": 1, "player_id": "p1"}])

    def load_team_stats(self, seasons, summary_level="week"):
        return pd.DataFrame([{"season": seasons[0], "week": 1, "team": "KC"}])

    def load_snap_counts(self, seasons):
        return pd.DataFrame([{"season": seasons[0], "week": 1, "player": "Player"}])

    def load_injuries(self, seasons):
        return pd.DataFrame([{"season": seasons[0], "week": 1, "gsis_id": "p1"}])

    def load_depth_charts(self, seasons):
        return pd.DataFrame([{"season": seasons[0], "week": 1, "gsis_id": "p1", "team": "KC"}])


def test_refresh_writes_foundation_files_and_manifest(tmp_path, monkeypatch):
    def fake_performance(season, output_path):
        pd.DataFrame([{"team": "Kansas City Chiefs", "season": season}]).to_csv(output_path, index=False)
        return FetchResult(season=season, rows=1, output_path=str(output_path), fetched_at_utc="now")

    monkeypatch.setattr(foundation, "_fetch_performance_with_fallback", fake_performance)
    result = foundation.refresh_nfl_foundation(2026, data_dir=tmp_path, nfl_module=FakeNFL())

    assert result.available_count == 10
    assert (tmp_path / "team_snapshot.csv").exists()
    assert (tmp_path / "player_weekly_stats.csv").exists()
    manifest = json.loads((tmp_path / "foundation_status.json").read_text())
    assert manifest["available_datasets"] == 10
    weekly = pd.read_csv(tmp_path / "weekly_rosters.csv")
    assert weekly["week"].tolist() == [2]


def test_optional_dataset_failure_is_recorded(tmp_path, monkeypatch):
    class PartialNFL(FakeNFL):
        def load_injuries(self, seasons):
            raise RuntimeError("not published yet")

    def fake_performance(season, output_path):
        pd.DataFrame([{"team": "Kansas City Chiefs", "season": season}]).to_csv(output_path, index=False)
        return FetchResult(season=season, rows=1, output_path=str(output_path), fetched_at_utc="now")

    monkeypatch.setattr(foundation, "_fetch_performance_with_fallback", fake_performance)
    result = foundation.refresh_nfl_foundation(2026, data_dir=tmp_path, nfl_module=PartialNFL())
    injuries = next(item for item in result.datasets if item.name == "injuries")
    assert not injuries.available
    assert "not published yet" in injuries.error
    assert result.available_count == 9
