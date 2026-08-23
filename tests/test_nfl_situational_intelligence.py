from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_situational_intelligence import (
    SITUATIONAL_ADJUSTMENT_CAP,
    _edge_label,
    _evidence_weight,
    _num,
    _rank,
    build_situational_matchup_context,
)


def test_num_handles_missing_and_invalid():
    assert _num("3.5") == 3.5
    assert _num(None, default=1.0) == 1.0
    assert _num("bad", default=2.0) == 2.0


def test_rank_missing_column_returns_neutral_50_everywhere():
    frame = pd.DataFrame({"team": ["A", "B", "C"]})
    result = _rank(frame, "missing_column")
    assert (result == 50.0).all()


def test_rank_higher_is_better_orders_correctly():
    frame = pd.DataFrame({"value": [10, 20, 30]})
    result = _rank(frame, "value", higher=True)
    assert result.iloc[2] > result.iloc[0]


def test_rank_lower_is_better_reverses_order():
    frame = pd.DataFrame({"value": [10, 20, 30]})
    result = _rank(frame, "value", higher=False)
    assert result.iloc[0] > result.iloc[2]


def test_edge_label_boundaries():
    assert _edge_label(2.9) == "Even"
    assert _edge_label(3.0) == "Slight"
    assert _edge_label(8.0) == "Clear"


def test_evidence_weight_off_season_data_is_a_weak_prior():
    row = {"season": 2025, "through_week": 10}
    assert _evidence_weight(row, 2026, 5) == pytest.approx(0.15)


def test_evidence_weight_current_season_grows_with_week_and_caps_at_one():
    early = _evidence_weight({"season": 2026, "through_week": 1}, 2026, 1)
    late = _evidence_weight({"season": 2026, "through_week": 15}, 2026, 15)
    assert early < late
    assert late <= 1.0


REQUIRED_COLUMNS = [
    "team", "third_down_conversion_rate", "third_down_conversion_allowed",
    "red_zone_td_rate", "red_zone_td_rate_allowed", "offense_turnover_rate",
    "defense_takeaway_rate", "offense_explosive_rate", "defense_explosive_allowed",
    "high_leverage_epa", "high_leverage_epa_allowed", "season", "through_week",
]


def _team_row(team: str, **overrides) -> dict:
    base = {col: 0.5 for col in REQUIRED_COLUMNS if col not in ("team", "season", "through_week")}
    base.update({"team": team, "season": 2026, "through_week": 10})
    base.update(overrides)
    return base


def test_build_situational_context_missing_file_is_unavailable(tmp_path: Path):
    result = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10,
        snapshot_path=tmp_path / "does_not_exist.csv",
    )
    assert result["available"] is False


def test_build_situational_context_missing_required_columns_is_unavailable(tmp_path: Path):
    path = tmp_path / "snapshot.csv"
    pd.DataFrame([{"team": "BUF"}]).to_csv(path, index=False)
    result = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path,
    )
    assert result["available"] is False


def test_build_situational_context_missing_team_is_unavailable(tmp_path: Path):
    path = tmp_path / "snapshot.csv"
    pd.DataFrame([_team_row("BUF")]).to_csv(path, index=False)
    result = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path,
    )
    assert result["available"] is False


def test_build_situational_context_adjustment_capped_at_0_35(tmp_path: Path):
    path = tmp_path / "snapshot.csv"
    rows = [
        _team_row("BUF", third_down_conversion_rate=0.60, third_down_conversion_allowed=0.20,
                   red_zone_td_rate=0.80, red_zone_td_rate_allowed=0.20,
                   high_leverage_epa=0.5, high_leverage_epa_allowed=-0.5),
        _team_row("NYJ", third_down_conversion_rate=0.20, third_down_conversion_allowed=0.60,
                   red_zone_td_rate=0.20, red_zone_td_rate_allowed=0.80,
                   high_leverage_epa=-0.5, high_leverage_epa_allowed=0.5),
        _team_row("KC"), _team_row("DEN"), _team_row("MIA"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path,
    )
    assert result["available"] is True
    assert abs(result["home_margin_adjustment"]) <= SITUATIONAL_ADJUSTMENT_CAP + 1e-9
    assert result["home_margin_adjustment"] > 0


def test_build_situational_context_identical_teams_are_even(tmp_path: Path):
    path = tmp_path / "snapshot.csv"
    rows = [_team_row("BUF"), _team_row("NYJ"), _team_row("KC"), _team_row("DEN")]
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path,
    )
    assert result["available"] is True
    assert result["overall_advantage"] == "Even"


def test_build_situational_context_prior_season_data_gets_reduced_weight(tmp_path: Path):
    path_current = tmp_path / "current.csv"
    path_prior = tmp_path / "prior.csv"
    rows_current = [
        _team_row("BUF", season=2026, third_down_conversion_rate=0.60, red_zone_td_rate=0.80),
        _team_row("NYJ", season=2026, third_down_conversion_rate=0.20, red_zone_td_rate=0.20),
        _team_row("KC", season=2026), _team_row("DEN", season=2026),
    ]
    rows_prior = [dict(r, season=2025) for r in rows_current]
    pd.DataFrame(rows_current).to_csv(path_current, index=False)
    pd.DataFrame(rows_prior).to_csv(path_prior, index=False)

    current = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path_current,
    )
    prior = build_situational_matchup_context(
        away_team="NYJ", home_team="BUF", season=2026, week=10, snapshot_path=path_prior,
    )
    assert current["evidence_weight"] > prior["evidence_weight"]
