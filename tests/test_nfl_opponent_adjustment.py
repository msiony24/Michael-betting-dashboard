"""Tests for engine/nfl_opponent_adjustment.py: strength-of-schedule refinement layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_opponent_adjustment import (
    OPPONENT_ADJUSTMENT_CAP,
    _evidence_weight,
    _num,
    build_opponent_adjusted_context,
)


def test_num_handles_missing_and_invalid():
    assert _num("0.1") == 0.1
    assert _num("bad", default=5.0) == 5.0


def test_evidence_weight_prior_season_is_weak_prior():
    assert _evidence_weight({"season": 2025}, 2026, 5) == pytest.approx(0.20)


def test_evidence_weight_grows_with_week_and_caps_at_one():
    early = _evidence_weight({"season": 2026, "through_week": 1}, 2026, 1)
    late = _evidence_weight({"season": 2026, "through_week": 20}, 2026, 20)
    assert early < late <= 1.0


REQUIRED = ["team", "season", "through_week", "offense_epa_per_play", "defense_epa_allowed",
            "sos_opponent_offense_epa", "sos_opponent_defense_epa_allowed", "opponent_quality_epa",
            "opponent_adjusted_net_epa"]


def _row(team, **overrides) -> dict:
    base = {c: 0.0 for c in REQUIRED if c not in ("team", "season", "through_week")}
    base.update({"team": team, "season": 2026, "through_week": 10})
    base.update(overrides)
    return base


def test_build_context_missing_file_unavailable(tmp_path: Path):
    result = build_opponent_adjusted_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=tmp_path / "missing.csv",
    )
    assert result["available"] is False


def test_build_context_missing_required_columns_unavailable(tmp_path: Path):
    path = tmp_path / "snap.csv"
    pd.DataFrame([{"team": "BUF"}]).to_csv(path, index=False)
    result = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is False


def test_build_context_missing_team_unavailable(tmp_path: Path):
    path = tmp_path / "snap.csv"
    pd.DataFrame([_row("BUF")]).to_csv(path, index=False)
    result = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is False


def test_build_context_adjustment_capped_at_0_40(tmp_path: Path):
    path = tmp_path / "snap.csv"
    rows = [
        _row("BUF", opponent_quality_epa=0.6),
        _row("NYJ", opponent_quality_epa=-0.6),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["available"] is True
    assert abs(result["home_margin_adjustment"]) <= OPPONENT_ADJUSTMENT_CAP + 1e-9
    assert result["home_margin_adjustment"] > 0
    assert result["overall_advantage"] == "BUF"


def test_build_context_identical_opponent_quality_is_even(tmp_path: Path):
    path = tmp_path / "snap.csv"
    rows = [_row("BUF"), _row("NYJ")]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path)
    assert result["overall_advantage"] == "Even"


def test_build_context_prior_season_data_gets_reduced_weight(tmp_path: Path):
    path_current = tmp_path / "current.csv"
    path_prior = tmp_path / "prior.csv"
    rows = [_row("BUF", season=2026, opponent_quality_epa=0.6), _row("NYJ", season=2026, opponent_quality_epa=-0.6)]
    pd.DataFrame(rows).to_csv(path_current, index=False)
    pd.DataFrame([dict(r, season=2025) for r in rows]).to_csv(path_prior, index=False)

    current = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path_current)
    prior = build_opponent_adjusted_context(away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path_prior)
    assert current["evidence_weight"] > prior["evidence_weight"]
