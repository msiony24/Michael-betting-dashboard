from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.nfl_team_state import TEAM_STATE_WEIGHTS, build_team_state


def _write_priors(path: Path) -> None:
    path.write_text(json.dumps({
        "Test Team": {
            "quarterback": 80,
            "offense": 75,
            "defense": 78,
            "coaching": 82,
            "offensive_line": 76,
            "defensive_line": 77,
            "skill_positions": 74,
            "secondary": 79,
            "special_teams": 70,
            "continuity": 81,
            "injury_adjustment": -1,
            "rookie_adjustment": 0,
        }
    }), encoding="utf-8")


def test_weights_sum_to_one():
    assert round(sum(TEAM_STATE_WEIGHTS.values()), 10) == 1.0


def test_live_snapshot_replaces_available_components(tmp_path: Path):
    priors = tmp_path / "ratings.json"
    snapshot = tmp_path / "snapshot.csv"
    _write_priors(priors)
    pd.DataFrame([{
        "team": "Test Team",
        "season": 2026,
        "through_week": 4,
        "quarterback": 88,
        "offense": 84,
        "defense": 86,
        "offensive_line": 83,
        "defensive_line": 85,
        "secondary": 82,
        "recent_form": 89,
        "special_teams": 73,
        "data_source": "test snapshot",
        "updated_at_utc": "2026-09-30T10:00:00+00:00",
    }]).to_csv(snapshot, index=False)

    state = build_team_state("Test Team", snapshot_path=snapshot, ratings_path=priors)

    assert state.components["quarterback"] == 88
    assert state.components["offensive_line"] == 83
    assert state.components["recent_form"] == 89
    assert state.components["coaching"] == 82
    assert state.component_sources["coaching"] == "manual prior"
    assert state.season == 2026
    assert state.week == 4


def test_missing_snapshot_uses_safe_priors_and_neutral_form(tmp_path: Path):
    priors = tmp_path / "ratings.json"
    _write_priors(priors)

    state = build_team_state(
        "Test Team",
        snapshot_path=tmp_path / "missing.csv",
        ratings_path=priors,
    )

    assert state.components["quarterback"] == 80
    assert state.components["recent_form"] == 67.5
    assert state.data_source == "manual priors"
    assert state.overall_rating < state.base_rating  # stored -1 injury adjustment
