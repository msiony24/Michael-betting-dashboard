from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_scheme_tendencies import (
    _evidence_weight,
    _matchup_edge,
    _num,
    _pct,
    _row,
    _tendency_label,
    build_scheme_matchup_context,
)


def test_num_handles_missing_and_invalid():
    assert _num("0.55") == 0.55
    assert _num(None, default=1.0) == 1.0
    assert _num("bad", default=2.0) == 2.0


def test_pct_formats_or_dashes_missing():
    assert _pct(0.5) == "50.0%"
    assert _pct(None) == "—"


def test_row_returns_none_for_missing_team_or_empty_frame():
    assert _row(pd.DataFrame(), "BUF") is None
    frame = pd.DataFrame([{"team": "NYJ"}])
    assert _row(frame, "BUF") is None


def test_row_returns_the_latest_matching_row():
    frame = pd.DataFrame([{"team": "BUF", "season": 2025}, {"team": "BUF", "season": 2026}])
    row = _row(frame, "BUF")
    assert row["season"] == 2026


def test_tendency_label_boundaries():
    assert _tendency_label(0.45, low=0.50, high=0.62, low_label="Run", mid_label="Balanced", high_label="Pass") == "Run"
    assert _tendency_label(0.55, low=0.50, high=0.62, low_label="Run", mid_label="Balanced", high_label="Pass") == "Balanced"
    assert _tendency_label(0.70, low=0.50, high=0.62, low_label="Run", mid_label="Balanced", high_label="Pass") == "Pass"
    assert _tendency_label(None, low=0.50, high=0.62, low_label="Run", mid_label="Balanced", high_label="Pass") == "Unavailable"


def test_matchup_edge_positive_when_team_is_the_leader():
    personnel = {"matchups": [{"Matchup": "BUF passing attack vs secondary", "Advantage": "BUF", "Edge": 5.0}]}
    assert _matchup_edge(personnel, "BUF", "passing attack vs secondary") == pytest.approx(5.0)


def test_matchup_edge_negative_when_opponent_is_the_leader():
    personnel = {"matchups": [{"Matchup": "BUF passing attack vs secondary", "Advantage": "NYJ", "Edge": 5.0}]}
    assert _matchup_edge(personnel, "BUF", "passing attack vs secondary") == pytest.approx(-5.0)


def test_matchup_edge_zero_when_even_or_not_found():
    personnel = {"matchups": [{"Matchup": "BUF passing attack vs secondary", "Advantage": "Even", "Edge": 5.0}]}
    assert _matchup_edge(personnel, "BUF", "passing attack vs secondary") == 0.0
    assert _matchup_edge({}, "BUF", "passing attack vs secondary") == 0.0


def test_evidence_weight_zero_for_missing_row():
    assert _evidence_weight(None, 2026, 5) == 0.0


def test_evidence_weight_zero_for_future_season_data():
    row = pd.Series({"season": 2027})
    assert _evidence_weight(row, 2026, 5) == 0.0


def test_evidence_weight_prior_season_is_a_capped_prior():
    row = pd.Series({"season": 2025})
    assert _evidence_weight(row, 2026, 1) == pytest.approx(0.35)


def test_evidence_weight_current_season_grows_with_week_and_caps_at_one():
    row_early = pd.Series({"season": 2026, "through_week": 1})
    row_late = pd.Series({"season": 2026, "through_week": 10})
    early = _evidence_weight(row_early, 2026, 1)
    late = _evidence_weight(row_late, 2026, 10)
    assert early < late
    assert late <= 1.0


def _write_scheme_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_scheme_context_unavailable_when_a_team_is_missing(tmp_path: Path):
    path = tmp_path / "scheme.csv"
    _write_scheme_csv(path, [{"team": "BUF", "season": 2026, "through_week": 3, "early_down_pass_rate": 0.6}])
    result = build_scheme_matchup_context(
        away_team="BUF", home_team="NYJ", season=2026, week=3, snapshot_path=path,
    )
    assert result["available"] is False
    assert result["home_margin_adjustment"] == 0.0


def test_build_scheme_context_adjustment_capped_at_0_65(tmp_path: Path):
    path = tmp_path / "scheme.csv"
    _write_scheme_csv(path, [
        {"team": "BUF", "season": 2026, "through_week": 10, "early_down_pass_rate": 0.75, "blitz_rate": 0.10},
        {"team": "NYJ", "season": 2026, "through_week": 10, "early_down_pass_rate": 0.35, "blitz_rate": 0.45},
    ])
    personnel = {"matchups": [
        {"Matchup": "BUF passing attack vs secondary", "Advantage": "BUF", "Edge": 100.0},
        {"Matchup": "BUF run game vs front seven", "Advantage": "BUF", "Edge": 100.0},
        {"Matchup": "BUF pass protection vs defensive front", "Advantage": "BUF", "Edge": 100.0},
    ]}
    result = build_scheme_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10,
        personnel_context=personnel, snapshot_path=path,
    )
    assert result["available"] is True
    assert abs(result["home_margin_adjustment"]) <= 0.65 + 1e-9


def test_build_scheme_context_even_when_adjustment_below_threshold(tmp_path: Path):
    path = tmp_path / "scheme.csv"
    _write_scheme_csv(path, [
        {"team": "BUF", "season": 2026, "through_week": 3, "early_down_pass_rate": 0.56, "blitz_rate": 0.25},
        {"team": "NYJ", "season": 2026, "through_week": 3, "early_down_pass_rate": 0.56, "blitz_rate": 0.25},
    ])
    result = build_scheme_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=3, snapshot_path=path,
    )
    assert result["available"] is True
    assert result["overall_advantage"] == "Even"


def test_build_scheme_context_missing_snapshot_file_is_unavailable(tmp_path: Path):
    result = build_scheme_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=3,
        snapshot_path=tmp_path / "does_not_exist.csv",
    )
    assert result["available"] is False
