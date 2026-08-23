from __future__ import annotations

import pandas as pd
import pytest

from engine.nfl_fetch import _final_game_rows, _numeric, _percentile_score, _weighted_rating, build_team_snapshot


# --- _percentile_score: compressed to [45, 90], not [0, 100] ---------------

def test_percentile_score_compressed_range():
    series = pd.Series([1, 2, 3, 4, 5])
    scores = _percentile_score(series)
    assert scores.min() >= 45.0
    assert scores.max() <= 90.0


def test_percentile_score_higher_is_better_by_default():
    series = pd.Series([1, 2, 3])
    scores = _percentile_score(series)
    assert scores.iloc[2] > scores.iloc[0]


def test_percentile_score_lower_is_better_flag_reverses_order():
    series = pd.Series([1, 2, 3])
    scores = _percentile_score(series, higher_is_better=False)
    assert scores.iloc[0] > scores.iloc[2]


def test_percentile_score_missing_values_get_neutral_midpoint():
    series = pd.Series([1.0, None, 3.0])
    scores = _percentile_score(series)
    assert scores.iloc[1] == pytest.approx(45.0 + 0.5 * 45.0)


# --- _weighted_rating: graceful missing-column fallback ---------------------

def test_weighted_rating_computes_weighted_average():
    frame = pd.DataFrame({"a": [80.0, 60.0], "b": [70.0, 50.0]})
    result = _weighted_rating(frame, [("a", 0.6), ("b", 0.4)])
    assert result.iloc[0] == pytest.approx(80.0 * 0.6 + 70.0 * 0.4)


def test_weighted_rating_missing_column_falls_back_to_neutral_67_5():
    frame = pd.DataFrame({"a": [80.0, 60.0]})
    result = _weighted_rating(frame, [("a", 0.5), ("missing_column", 0.5)])
    assert result.iloc[0] == pytest.approx(80.0)


def test_weighted_rating_no_columns_present_returns_neutral_everywhere():
    frame = pd.DataFrame({"unrelated": [1, 2, 3]})
    result = _weighted_rating(frame, [("a", 0.5), ("b", 0.5)])
    assert (result == 67.5).all()


def test_weighted_rating_nan_values_use_neutral_fill():
    frame = pd.DataFrame({"a": [None, 80.0]})
    result = _weighted_rating(frame, [("a", 1.0)])
    assert result.iloc[0] == pytest.approx(67.5)


# --- _numeric: coercion with default fill -------------------------------------

def test_numeric_missing_column_returns_default_series():
    frame = pd.DataFrame({"other": [1, 2, 3]})
    result = _numeric(frame, "missing", default=5.0)
    assert (result == 5.0).all()
    assert len(result) == 3


def test_numeric_coerces_invalid_strings_to_default():
    frame = pd.DataFrame({"x": ["10", "not a number", "20"]})
    result = _numeric(frame, "x", default=-1.0)
    assert list(result) == [10.0, -1.0, 20.0]


# --- _final_game_rows: extracting final scores from play-by-play data ------

def _pbp_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_final_game_rows_missing_required_columns_returns_empty():
    result = _final_game_rows(pd.DataFrame({"game_id": ["g1"]}))
    assert result.empty


def test_final_game_rows_missing_score_columns_returns_empty():
    frame = _pbp_rows([{"game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1}])
    result = _final_game_rows(frame)
    assert result.empty


def test_final_game_rows_picks_the_last_play_per_game_by_play_id():
    frame = _pbp_rows([
        {"game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1,
         "total_home_score": 0, "total_away_score": 0},
        {"game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 200,
         "total_home_score": 24, "total_away_score": 17},
    ])
    result = _final_game_rows(frame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["home_score"] == 24
    assert row["away_score"] == 17


def test_final_game_rows_prefers_total_score_columns_over_plain_score_columns():
    frame = _pbp_rows([{
        "game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1,
        "total_home_score": 24, "total_away_score": 17,
        "home_score": 999, "away_score": 999,
    }])
    result = _final_game_rows(frame)
    assert result.iloc[0]["home_score"] == 24
    assert result.iloc[0]["away_score"] == 17


def test_final_game_rows_falls_back_to_plain_score_columns_when_total_missing():
    frame = _pbp_rows([{
        "game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1,
        "home_score": 21, "away_score": 14,
    }])
    result = _final_game_rows(frame)
    assert result.iloc[0]["home_score"] == 21
    assert result.iloc[0]["away_score"] == 14


def test_final_game_rows_drops_games_with_unparseable_scores():
    frame = _pbp_rows([{
        "game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1,
        "total_home_score": "not a number", "total_away_score": 17,
    }])
    result = _final_game_rows(frame)
    assert result.empty


def test_final_game_rows_handles_multiple_games_independently():
    frame = _pbp_rows([
        {"game_id": "g1", "home_team": "BUF", "away_team": "NYJ", "play_id": 1,
         "total_home_score": 10, "total_away_score": 7},
        {"game_id": "g2", "home_team": "KC", "away_team": "DEN", "play_id": 1,
         "total_home_score": 30, "total_away_score": 20},
    ])
    result = _final_game_rows(frame)
    assert len(result) == 2
    assert set(result["game_id"]) == {"g1", "g2"}


def test_final_game_rows_without_play_id_falls_back_to_row_order():
    frame = _pbp_rows([
        {"game_id": "g1", "home_team": "BUF", "away_team": "NYJ",
         "total_home_score": 0, "total_away_score": 0},
        {"game_id": "g1", "home_team": "BUF", "away_team": "NYJ",
         "total_home_score": 24, "total_away_score": 17},
    ])
    result = _final_game_rows(frame)
    assert result.iloc[0]["home_score"] == 24


# --- build_team_snapshot: input-validation guard clauses -------------------

def test_build_team_snapshot_missing_required_columns_raises_clear_error():
    frame = pd.DataFrame({"posteam": ["BUF"], "defteam": ["NYJ"]})  # no "epa"
    with pytest.raises(ValueError, match="missing required columns"):
        build_team_snapshot(frame, 2026)


def test_build_team_snapshot_no_usable_plays_raises_clear_error():
    # Has the required columns, but every play gets filtered out (wrong
    # season, non-REG, or non-pass/run) -- must fail loudly, not silently
    # return an empty/garbage snapshot.
    frame = pd.DataFrame({
        "posteam": ["BUF"], "defteam": ["NYJ"], "epa": [0.5],
        "season": [2020], "season_type": ["REG"], "play_type": ["pass"],
    })
    with pytest.raises(ValueError, match="No usable regular-season plays"):
        build_team_snapshot(frame, 2026)  # wrong season filters everything out
