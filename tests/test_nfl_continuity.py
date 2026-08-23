"""Tests for engine/nfl_continuity.py: roster-continuity priors."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_continuity import (
    _continuity_score,
    _norm_name,
    build_continuity_priors,
)


def test_norm_name_strips_accents_suffixes_and_punctuation():
    assert _norm_name("Odell Beckham Jr.") == "odellbeckham"
    assert _norm_name("Félix Auger-Aliassime") == "felixaugeraliassime"


def test_norm_name_case_insensitive():
    assert _norm_name("JOSH ALLEN") == _norm_name("josh allen")


def test_norm_name_empty_input():
    assert _norm_name(None) == ""
    assert _norm_name("") == ""


def test_continuity_score_compressed_range():
    assert _continuity_score(0.0) == 55.0
    assert _continuity_score(1.0) == 85.0
    assert _continuity_score(0.5) == 70.0


def test_continuity_score_clips_out_of_range_input():
    assert _continuity_score(-0.5) == 55.0
    assert _continuity_score(1.5) == 85.0


def _depth_chart_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _roster_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_continuity_priors_missing_files_returns_empty(tmp_path: Path):
    result = build_continuity_priors(
        depth_chart_path=tmp_path / "missing_depth.csv",
        prior_rosters_path=tmp_path / "missing_roster.csv",
    )
    assert result == {}


def test_build_continuity_priors_missing_required_columns_returns_empty(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    _depth_chart_csv(depth_path, [{"Team": "Buffalo Bills"}])  # no Position/Starter
    _roster_csv(roster_path, [{"team": "BUF", "full_name": "Josh Allen"}])
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    assert result == {}


def test_build_continuity_priors_retention_rate_exact(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    # 4 starters this year; 3 of them were on last year's roster.
    _depth_chart_csv(depth_path, [
        {"Team": "Buffalo Bills", "Position": "QB", "Starter": "Josh Allen"},
        {"Team": "Buffalo Bills", "Position": "RB", "Starter": "James Cook"},
        {"Team": "Buffalo Bills", "Position": "WR", "Starter": "Khalil Shakir"},
        {"Team": "Buffalo Bills", "Position": "TE", "Starter": "New Tight End"},
    ])
    _roster_csv(roster_path, [
        {"team": "BUF", "full_name": "Josh Allen"},
        {"team": "BUF", "full_name": "James Cook"},
        {"team": "BUF", "full_name": "Khalil Shakir"},
        {"team": "BUF", "full_name": "Someone Else"},
    ])
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    assert result["Buffalo Bills"]["retained_starters"] == 3
    assert result["Buffalo Bills"]["starter_count"] == 4
    assert result["Buffalo Bills"]["retained_rate"] == pytest.approx(0.75)


def test_build_continuity_priors_excludes_special_teams_positions(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    _depth_chart_csv(depth_path, [
        {"Team": "Buffalo Bills", "Position": "QB", "Starter": "Josh Allen"},
        {"Team": "Buffalo Bills", "Position": "K", "Starter": "Some Kicker"},
        {"Team": "Buffalo Bills", "Position": "P", "Starter": "Some Punter"},
    ])
    _roster_csv(roster_path, [
        {"team": "BUF", "full_name": "Josh Allen"},
    ])
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    # Only the QB should count -- K and P are excluded from the starter set.
    assert result["Buffalo Bills"]["starter_count"] == 1
    assert result["Buffalo Bills"]["retained_starters"] == 1


def test_build_continuity_priors_dedupes_player_on_multiple_rows(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    # A swing tackle listed at both LT and RT should only count once.
    _depth_chart_csv(depth_path, [
        {"Team": "Buffalo Bills", "Position": "LT", "Starter": "Swing Tackle"},
        {"Team": "Buffalo Bills", "Position": "RT", "Starter": "Swing Tackle"},
    ])
    _roster_csv(roster_path, [{"team": "BUF", "full_name": "Swing Tackle"}])
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    assert result["Buffalo Bills"]["starter_count"] == 1


def test_build_continuity_priors_team_with_no_prior_roster_data_is_skipped(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    _depth_chart_csv(depth_path, [{"Team": "Buffalo Bills", "Position": "QB", "Starter": "Josh Allen"}])
    _roster_csv(roster_path, [{"team": "KC", "full_name": "Patrick Mahomes"}])  # different team only
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    assert "Buffalo Bills" not in result


def test_build_continuity_priors_name_matching_ignores_accents_and_suffixes(tmp_path: Path):
    depth_path = tmp_path / "depth.csv"
    roster_path = tmp_path / "roster.csv"
    _depth_chart_csv(depth_path, [{"Team": "Buffalo Bills", "Position": "WR", "Starter": "Odell Beckham Jr."}])
    _roster_csv(roster_path, [{"team": "BUF", "full_name": "Odell Beckham"}])
    result = build_continuity_priors(depth_chart_path=depth_path, prior_rosters_path=roster_path)
    assert result["Buffalo Bills"]["retained_starters"] == 1
