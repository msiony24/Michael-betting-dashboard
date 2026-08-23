"""Tests for engine/nfl_los_intelligence.py: line-of-scrimmage matchup layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_los_intelligence import (
    LOS_ADJUSTMENT_CAP,
    _evidence_weight,
    _num,
    _rank,
    build_los_matchup_context,
)


def test_num_handles_missing_and_invalid():
    assert _num("0.05") == 0.05
    import math
    assert math.isnan(_num(None))
    assert _num("bad", default=1.0) == 1.0


def test_rank_missing_column_returns_neutral_50():
    frame = pd.DataFrame({"team": ["A", "B"]})
    result = _rank(frame, "missing_col")
    assert (result == 50.0).all()


def test_rank_lower_is_better_reverses_order():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    result = _rank(frame, "value", higher=False)
    assert result.iloc[0] > result.iloc[2]


def test_evidence_weight_off_season_data_is_weak():
    assert _evidence_weight({"season": 2025}, 2026, 5) == pytest.approx(0.20)


def test_evidence_weight_current_season_grows_with_week():
    early = _evidence_weight({"season": 2026, "through_week": 1}, 2026, 1)
    late = _evidence_weight({"season": 2026, "through_week": 10}, 2026, 10)
    assert early < late <= 1.0


def _row(team, **overrides) -> dict:
    base = {
        "team": team, "season": 2026, "through_week": 10,
        "offense_sack_rate": 0.06, "offense_qb_hit_rate": 0.12,
        "defense_sack_rate": 0.06, "defense_qb_hit_rate": 0.10,
        "rush_success": 0.45, "rush_success_allowed": 0.45,
    }
    base.update(overrides)
    return base


def test_build_los_context_missing_file_unavailable(tmp_path: Path):
    result = build_los_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10,
        snapshot_path=tmp_path / "missing.csv",
    )
    assert result["available"] is False


def test_build_los_context_missing_required_columns_unavailable(tmp_path: Path):
    path = tmp_path / "snap.csv"
    pd.DataFrame([{"team": "BUF"}]).to_csv(path, index=False)
    result = build_los_matchup_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is False


def test_build_los_context_missing_team_unavailable(tmp_path: Path):
    path = tmp_path / "snap.csv"
    pd.DataFrame([_row("BUF")]).to_csv(path, index=False)
    result = build_los_matchup_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is False


def test_build_los_context_adjustment_capped_at_0_45(tmp_path: Path):
    path = tmp_path / "snap.csv"
    rows = [
        _row("BUF", offense_sack_rate=0.03, offense_qb_hit_rate=0.10, defense_sack_rate=0.10,
             defense_qb_hit_rate=0.15, rush_success=0.55, rush_success_allowed=0.35),
        _row("NYJ", offense_sack_rate=0.10, offense_qb_hit_rate=0.20, defense_sack_rate=0.03,
             defense_qb_hit_rate=0.05, rush_success=0.35, rush_success_allowed=0.55),
        _row("KC"), _row("DEN"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_los_matchup_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is True
    assert abs(result["home_margin_adjustment"]) <= LOS_ADJUSTMENT_CAP + 1e-9
    assert result["home_margin_adjustment"] > 0


def test_build_los_context_identical_teams_are_even(tmp_path: Path):
    path = tmp_path / "snap.csv"
    rows = [_row("BUF"), _row("NYJ"), _row("KC"), _row("DEN")]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_los_matchup_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["overall_advantage"] == "Even"


def test_build_los_context_missing_optional_columns_treated_as_neutral(tmp_path: Path):
    # qb_epa_when_disrupted etc. aren't in the strict required set -- missing
    # them should fall back to neutral rather than crash.
    path = tmp_path / "snap.csv"
    rows = [_row("BUF"), _row("NYJ"), _row("KC"), _row("DEN")]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_los_matchup_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is True
