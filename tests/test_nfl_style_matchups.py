from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_style_matchups import (
    STYLE_ADJUSTMENT_CAP,
    _mean,
    _num,
    _starter_rows,
    _strength,
    _style_row,
    _top_mean,
    build_style_matchup_context,
)


# --- small helpers -----------------------------------------------------------

def test_num_handles_missing_and_invalid():
    assert _num("70") == 70.0
    assert _num(None) == 67.5
    assert _num(float("nan")) == 67.5
    assert _num("bad", default=50.0) == 50.0


def test_strength_boundaries():
    assert _strength(1.9) == "Even"
    assert _strength(2.0) == "Slight"
    assert _strength(4.5) == "Clear"
    assert _strength(7.5) == "Strong"


def test_style_row_advantage_follows_sign():
    positive = _style_row("test", "Offense", "Defense", 5.0, "reason")
    negative = _style_row("test", "Offense", "Defense", -5.0, "reason")
    even = _style_row("test", "Offense", "Defense", 0.5, "reason")
    assert positive["Advantage"] == "Offense"
    assert negative["Advantage"] == "Defense"
    assert even["Advantage"] == "Even"


def test_mean_ignores_missing_columns():
    frame = pd.DataFrame({"a": [70.0, 80.0]})
    assert _mean(frame, ["a", "missing"]) == pytest.approx(75.0)


def test_mean_empty_frame_returns_default():
    assert _mean(pd.DataFrame(columns=["a"]), ["a"], default=60.0) == 60.0


def test_top_mean_takes_the_highest_n_values():
    frame = pd.DataFrame({"speed": [70, 90, 60, 85, 50]})
    result = _top_mean(frame, "speed", n=2)
    assert result == pytest.approx((90 + 85) / 2)


def test_top_mean_missing_column_returns_default():
    frame = pd.DataFrame({"other": [1, 2, 3]})
    assert _top_mean(frame, "speed", default=55.0) == 55.0


# --- _starter_rows: only actual starters, matched by normalized name --------

def _raw_madden(rows: list[dict]) -> pd.DataFrame:
    from engine.nfl_depth_chart import normalize_player_name
    frame = pd.DataFrame(rows)
    frame["_name_key"] = frame["player_name"].map(normalize_player_name)
    return frame.drop_duplicates("_name_key", keep="first").set_index("_name_key", drop=False)


def test_starter_rows_only_includes_flagged_starters():
    raw = _raw_madden([
        {"player_name": "Josh Allen", "speed": 85, "position": "QB"},
        {"player_name": "Backup QB", "speed": 70, "position": "QB"},
    ])
    team_data = {"units": {"quarterback": {"top_players": [
        {"name": "Josh Allen", "starter": True},
        {"name": "Backup QB", "starter": False},
    ]}}}
    result = _starter_rows(team_data, "quarterback", raw)
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Josh Allen"


def test_starter_rows_missing_unit_returns_empty_with_same_columns():
    raw = _raw_madden([{"player_name": "Josh Allen", "speed": 85}])
    result = _starter_rows({}, "quarterback", raw)
    assert result.empty
    assert list(result.columns) == list(raw.columns)


# --- build_style_matchup_context: availability and the overall cap ---------

def test_build_style_matchup_unavailable_without_madden_data(tmp_path: Path):
    result = build_style_matchup_context(
        away_team="Away", home_team="Home", away_data={}, home_data={},
        madden_players_path=tmp_path / "does_not_exist.csv",
    )
    assert result["available"] is False
    assert result["home_margin_adjustment"] == 0.0


def _team_data(unit_players: dict[str, list[dict]]) -> dict:
    units = {}
    for unit, players in unit_players.items():
        units[unit] = {"top_players": [{"name": p["player_name"], "starter": True} for p in players]}
    return {"units": units}


def test_build_style_matchup_adjustment_capped_with_extreme_trait_gaps(tmp_path: Path):
    madden_path = tmp_path / "madden.csv"
    home_players = {
        "quarterback": [{"player_name": "Home QB", "throw_accuracy_deep": 99, "throw_power": 99,
                          "speed": 99, "acceleration": 99, "agility": 99, "throw_on_the_run": 99, "break_sack": 99}],
        "receiving_weapons": [{"player_name": f"Home WR{i}", "position": "WR", "speed": 99,
                                "short_route_running": 99, "medium_route_running": 99, "deep_route_running": 99,
                                "release": 99} for i in range(3)],
        "offensive_line": [{"player_name": f"Home OL{i}", "pass_block_finesse": 99, "pass_block_power": 99,
                             "pass_block": 99, "awareness": 99, "run_block": 99, "run_block_power": 99,
                             "run_block_finesse": 99, "impact_blocking": 99} for i in range(5)],
        "running_backs": [{"player_name": "Home RB", "break_tackle": 99, "bc_vision": 99,
                            "change_of_direction": 99, "trucking": 99, "speed": 99, "acceleration": 99}],
        "defensive_front": [{"player_name": f"Home DL{i}", "position": "DE", "finesse_moves": 99, "power_moves": 99,
                              "acceleration": 99, "block_shedding": 99, "play_recognition": 99, "strength": 99,
                              "tackle": 99, "pursuit": 99, "speed": 99} for i in range(4)],
        "linebackers": [{"player_name": f"Home LB{i}", "acceleration": 99, "block_shedding": 99,
                          "play_recognition": 99, "strength": 99, "tackle": 99, "pursuit": 99, "speed": 99} for i in range(3)],
    }
    away_players = {
        "quarterback": [{"player_name": "Away QB", "throw_accuracy_deep": 40, "throw_power": 40,
                          "speed": 40, "acceleration": 40, "agility": 40, "throw_on_the_run": 40, "break_sack": 40}],
        "receiving_weapons": [{"player_name": f"Away WR{i}", "position": "WR", "speed": 40,
                                "short_route_running": 40, "medium_route_running": 40, "deep_route_running": 40,
                                "release": 40} for i in range(3)],
        "offensive_line": [{"player_name": f"Away OL{i}", "pass_block_finesse": 40, "pass_block_power": 40,
                             "pass_block": 40, "awareness": 40, "run_block": 40, "run_block_power": 40,
                             "run_block_finesse": 40, "impact_blocking": 40} for i in range(5)],
        "running_backs": [{"player_name": "Away RB", "break_tackle": 40, "bc_vision": 40,
                            "change_of_direction": 40, "trucking": 40, "speed": 40, "acceleration": 40}],
        "defensive_front": [{"player_name": f"Away DL{i}", "position": "DE", "finesse_moves": 40, "power_moves": 40,
                              "acceleration": 40, "block_shedding": 40, "play_recognition": 40, "strength": 40,
                              "tackle": 40, "pursuit": 40, "speed": 40} for i in range(4)],
        "linebackers": [{"player_name": f"Away LB{i}", "acceleration": 40, "block_shedding": 40,
                          "play_recognition": 40, "strength": 40, "tackle": 40, "pursuit": 40, "speed": 40} for i in range(3)],
    }

    all_rows = []
    for group in (home_players, away_players):
        for players in group.values():
            all_rows.extend(players)
    pd.DataFrame(all_rows).to_csv(madden_path, index=False)

    result = build_style_matchup_context(
        away_team="Away", home_team="Home",
        away_data=_team_data(away_players), home_data=_team_data(home_players),
        madden_players_path=madden_path,
    )
    assert result["available"] is True
    assert abs(result["home_margin_adjustment"]) <= STYLE_ADJUSTMENT_CAP + 1e-9
    assert result["home_margin_adjustment"] > 0


def test_build_style_matchup_overall_strength_even_with_identical_teams(tmp_path: Path):
    madden_path = tmp_path / "madden.csv"
    shared_players = {
        "quarterback": [{"player_name": "Shared QB A", "throw_accuracy_deep": 80, "throw_power": 80,
                          "speed": 80, "acceleration": 80, "agility": 80, "throw_on_the_run": 80, "break_sack": 80}],
    }
    other_qb = {"quarterback": [{"player_name": "Shared QB B", "throw_accuracy_deep": 80, "throw_power": 80,
                                  "speed": 80, "acceleration": 80, "agility": 80, "throw_on_the_run": 80, "break_sack": 80}]}
    pd.DataFrame(shared_players["quarterback"] + other_qb["quarterback"]).to_csv(madden_path, index=False)

    result = build_style_matchup_context(
        away_team="Away", home_team="Home",
        away_data=_team_data(shared_players), home_data=_team_data(other_qb),
        madden_players_path=madden_path,
    )
    assert result["available"] is True
    assert result["overall_advantage"] == "Even"
    assert result["overall_strength"] == "Even"
