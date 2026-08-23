from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.nfl_rating_engine import _resolve_depth_chart_path


def _write_auto_depth_chart(path: Path, teams: list[str]) -> None:
    rows = []
    for team in teams:
        rows.append({
            "dt": "2026-08-20T00:00:00Z", "team": team, "player_name": f"{team} QB1",
            "espn_id": "1", "gsis_id": "1", "pos_grp_id": "1", "pos_grp": "QB",
            "pos_id": "1", "pos_name": "Quarterback", "pos_abb": "QB", "pos_slot": "1", "pos_rank": "1",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_footballguys(path: Path) -> None:
    pd.DataFrame([{
        "Team": "Buffalo Bills", "Unit": "Offense", "Position": "QB",
        "Starter": "Josh Allen", "2nd String": "", "3rd String": "", "4th String": "", "5th String": "",
        "Source URL": "https://www.footballguys.com/depth-charts",
    }]).to_csv(path, index=False)


ALL_32_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS",
]


def test_prefers_auto_depth_chart_when_all_32_teams_present(tmp_path: Path, monkeypatch):
    nfl_dir = tmp_path
    auto_path = nfl_dir / "depth_charts.csv"
    _write_auto_depth_chart(auto_path, ALL_32_TEAMS)
    fb_path = tmp_path / "footballguys_depth_charts.csv"
    _write_footballguys(fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.DEFAULT_DEPTH_CHART_PATH", fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.AUTO_DEPTH_CHART_PATH", tmp_path / "unused_default_auto.csv")

    resolved = _resolve_depth_chart_path(nfl_dir)
    assert resolved == auto_path


def test_falls_back_to_footballguys_when_auto_missing_teams(tmp_path: Path, monkeypatch):
    nfl_dir = tmp_path
    auto_path = nfl_dir / "depth_charts.csv"
    # Only 5 teams -- an obviously incomplete/broken snapshot.
    _write_auto_depth_chart(auto_path, ["BUF", "NYJ", "MIA", "NE", "KC"])
    fb_path = tmp_path / "footballguys_depth_charts.csv"
    _write_footballguys(fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.DEFAULT_DEPTH_CHART_PATH", fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.AUTO_DEPTH_CHART_PATH", tmp_path / "unused_default_auto.csv")

    resolved = _resolve_depth_chart_path(nfl_dir)
    assert resolved == fb_path


def test_falls_back_to_footballguys_when_auto_file_missing(tmp_path: Path, monkeypatch):
    nfl_dir = tmp_path
    fb_path = tmp_path / "footballguys_depth_charts.csv"
    _write_footballguys(fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.DEFAULT_DEPTH_CHART_PATH", fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.AUTO_DEPTH_CHART_PATH", tmp_path / "unused_default_auto.csv")

    resolved = _resolve_depth_chart_path(nfl_dir)
    assert resolved == fb_path


def test_falls_back_to_footballguys_when_auto_file_malformed(tmp_path: Path, monkeypatch):
    nfl_dir = tmp_path
    auto_path = nfl_dir / "depth_charts.csv"
    auto_path.write_text("this is not a valid depth chart csv\n", encoding="utf-8")
    fb_path = tmp_path / "footballguys_depth_charts.csv"
    _write_footballguys(fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.DEFAULT_DEPTH_CHART_PATH", fb_path)
    monkeypatch.setattr("engine.nfl_rating_engine.AUTO_DEPTH_CHART_PATH", tmp_path / "unused_default_auto.csv")

    resolved = _resolve_depth_chart_path(nfl_dir)
    assert resolved == fb_path


def test_returns_default_path_when_nothing_available(tmp_path: Path, monkeypatch):
    nfl_dir = tmp_path
    fb_default = tmp_path / "no_such_footballguys.csv"
    monkeypatch.setattr("engine.nfl_rating_engine.DEFAULT_DEPTH_CHART_PATH", fb_default)
    monkeypatch.setattr("engine.nfl_rating_engine.AUTO_DEPTH_CHART_PATH", tmp_path / "unused_default_auto.csv")

    resolved = _resolve_depth_chart_path(nfl_dir)
    assert resolved == fb_default
