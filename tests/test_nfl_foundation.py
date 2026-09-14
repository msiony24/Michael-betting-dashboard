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


def _write_current_schema_snapshot(season, output_path):
    """Write a team snapshot that already satisfies the current schema guard.

    Tests must not reach the network. The real `_ensure_current_performance_schema`
    calls `fetch_and_build` when the snapshot looks stale, and `fetch_and_build`
    downloads live play-by-play AND writes a sibling `scheme_tendencies.csv` into
    the same directory -- which silently flips `scheme_tendencies` to available and
    makes dataset counts depend on whether nflreadpy is installed and whether the
    real season has started. Stubbing it keeps these tests hermetic.
    """
    row = {"team": "Kansas City Chiefs", "season": season}
    row.update({column: 0.0 for column in foundation.PERFORMANCE_REQUIRED_COLUMNS})
    row["team"] = "Kansas City Chiefs"
    row["season"] = season
    pd.DataFrame([row]).to_csv(output_path, index=False)
    return FetchResult(
        season=season, rows=1, output_path=str(output_path), fetched_at_utc="now"
    )


def _stub_foundation_network(monkeypatch):
    """Point both performance entry points at local stubs."""
    monkeypatch.setattr(foundation, "_fetch_performance_with_fallback", _write_current_schema_snapshot)
    monkeypatch.setattr(foundation, "fetch_and_build", _write_current_schema_snapshot)


# Datasets the foundation is expected to report on. Asserting the name set rather
# than a bare count means adding a dataset does not break these tests, while
# removing one still does.
EXPECTED_DATASETS = {
    "team_performance",
    "scheme_tendencies",
    "schedules",
    "rosters",
    "prior_rosters",
    "weekly_rosters",
    "player_weekly_stats",
    "team_weekly_stats",
    "snap_counts",
    "injuries",
    "depth_charts",
}

# FakeNFL has no load_pbp, so the scheme snapshot cannot be built in tests.
UNAVAILABLE_IN_TESTS = {"scheme_tendencies"}


def test_refresh_writes_foundation_files_and_manifest(tmp_path, monkeypatch):
    _stub_foundation_network(monkeypatch)
    result = foundation.refresh_nfl_foundation(2026, data_dir=tmp_path, nfl_module=FakeNFL())

    reported = {item.name for item in result.datasets}
    assert EXPECTED_DATASETS.issubset(reported)
    unavailable = {item.name for item in result.datasets if not item.available}
    assert unavailable == UNAVAILABLE_IN_TESTS
    assert result.available_count == len(result.datasets) - len(UNAVAILABLE_IN_TESTS)

    assert (tmp_path / "team_snapshot.csv").exists()
    assert (tmp_path / "player_weekly_stats.csv").exists()
    manifest = json.loads((tmp_path / "foundation_status.json").read_text())
    assert manifest["available_datasets"] == result.available_count
    weekly = pd.read_csv(tmp_path / "weekly_rosters.csv")
    assert weekly["week"].tolist() == [2]


def test_optional_dataset_failure_is_recorded(tmp_path, monkeypatch):
    class PartialNFL(FakeNFL):
        def load_injuries(self, seasons):
            raise RuntimeError("not published yet")

    _stub_foundation_network(monkeypatch)
    result = foundation.refresh_nfl_foundation(2026, data_dir=tmp_path, nfl_module=PartialNFL())
    injuries = next(item for item in result.datasets if item.name == "injuries")
    assert not injuries.available
    assert "not published yet" in injuries.error

    unavailable = {item.name for item in result.datasets if not item.available}
    assert unavailable == UNAVAILABLE_IN_TESTS | {"injuries"}
    assert result.available_count == len(result.datasets) - len(unavailable)


def test_depth_chart_refresh_keeps_only_latest_timestamp_snapshot(tmp_path, monkeypatch):
    class SnapshotNFL(FakeNFL):
        def load_depth_charts(self, seasons):
            return pd.DataFrame([
                {"dt": "2026-08-20T10:00:00Z", "team": "BUF", "player_name": "Old QB", "gsis_id": "old", "pos_abb": "QB", "pos_rank": 1},
                {"dt": "2026-08-22T10:00:00Z", "team": "BUF", "player_name": "Current QB", "gsis_id": "new", "pos_abb": "QB", "pos_rank": 1},
                {"dt": "2026-08-22T10:00:00Z", "team": "KC", "player_name": "Current KC QB", "gsis_id": "kc", "pos_abb": "QB", "pos_rank": 1},
            ])

    _stub_foundation_network(monkeypatch)
    foundation.refresh_nfl_foundation(2026, data_dir=tmp_path, nfl_module=SnapshotNFL())

    depth = pd.read_csv(tmp_path / "depth_charts.csv")
    assert len(depth) == 2
    assert set(depth["player_name"]) == {"Current QB", "Current KC QB"}
    assert depth["dt"].nunique() == 1
