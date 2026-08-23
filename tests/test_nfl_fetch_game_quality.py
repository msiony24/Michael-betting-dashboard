"""Tests for build_game_quality_snapshot() in engine/nfl_fetch.py, using a
small fully-controlled synthetic play-by-play dataset with hand-calculable
per-game team quality metrics.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.nfl_fetch import build_game_quality_snapshot


def _play(game_id, posteam, defteam, epa, play_type, home_team, away_team, *,
          total_home_score=0, total_away_score=0, play_id=1, week=1,
          interception=0, fumble_lost=0, season=2026, season_type="REG") -> dict:
    return dict(
        game_id=game_id, posteam=posteam, defteam=defteam, epa=epa, play_type=play_type,
        season=season, season_type=season_type,
        pass_attempt=1 if play_type == "pass" else 0, rush_attempt=1 if play_type == "run" else 0,
        success=1 if epa > 0 else 0, yards_gained=5, interception=interception, fumble_lost=fumble_lost,
        home_team=home_team, away_team=away_team,
        total_home_score=total_home_score, total_away_score=total_away_score,
        play_id=play_id, week=week,
    )


def _game_rows() -> list[dict]:
    """BUF (home) plays NYJ (away). BUF has better EPA and wins 24-17."""
    rows = []
    for i in range(10):
        rows.append(_play("g1", "BUF", "NYJ", 0.3, "pass" if i % 2 == 0 else "run", "BUF", "NYJ", play_id=i * 2))
    for i in range(10):
        final = i == 9
        rows.append(_play(
            "g1", "NYJ", "BUF", -0.1, "pass" if i % 2 == 0 else "run", "BUF", "NYJ",
            total_home_score=24 if final else 0, total_away_score=17 if final else 0, play_id=i * 2 + 1,
        ))
    return rows


def test_build_game_quality_net_epa_exact():
    pbp = pd.DataFrame(_game_rows())
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "net_epa"] == pytest.approx(0.4)
    assert snap.loc["NYJ", "net_epa"] == pytest.approx(-0.4)


def test_build_game_quality_score_margin_from_final_score():
    pbp = pd.DataFrame(_game_rows())
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "score_margin"] == pytest.approx(7.0)
    assert snap.loc["NYJ", "score_margin"] == pytest.approx(-7.0)


def test_build_game_quality_opponent_correctly_assigned():
    pbp = pd.DataFrame(_game_rows())
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "opponent_abbr"] == "NYJ"
    assert snap.loc["NYJ", "opponent_abbr"] == "BUF"


def test_build_game_quality_better_underlying_play_gets_higher_quality_score():
    pbp = pd.DataFrame(_game_rows())
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "quality_score"] > snap.loc["NYJ", "quality_score"]


def test_build_game_quality_turnover_margin_reflects_takeaways():
    rows = _game_rows()
    # Add one BUF interception thrown (a turnover) and one NYJ fumble lost.
    rows.append(_play("g1", "BUF", "NYJ", -0.5, "pass", "BUF", "NYJ", play_id=99, interception=1))
    pbp = pd.DataFrame(rows)
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    # BUF turned it over once (their own giveaway) -> NYJ gets 1 takeaway,
    # BUF's turnover_margin = takeaways(0) - turnovers(1) = -1.
    assert snap.loc["BUF", "turnover_margin"] == pytest.approx(-1.0)
    assert snap.loc["NYJ", "turnover_margin"] == pytest.approx(1.0)


def test_build_game_quality_missing_required_columns_raises():
    frame = pd.DataFrame({"posteam": ["BUF"], "defteam": ["NYJ"]})  # no game_id/epa
    with pytest.raises(ValueError, match="missing game-quality columns"):
        build_game_quality_snapshot(frame, 2026)


def test_build_game_quality_no_usable_plays_returns_empty_frame():
    # Unlike build_team_snapshot, this one returns an empty frame rather than
    # raising when nothing matches -- confirmed by reading the source.
    pbp = pd.DataFrame(_game_rows())
    result = build_game_quality_snapshot(pbp, 1999)
    assert result.empty


def test_build_game_quality_excludes_wrong_season():
    rows = _game_rows() + [_play("g2", "BUF", "NYJ", 99.0, "pass", "BUF", "NYJ", play_id=200, season=2020)]
    pbp = pd.DataFrame(rows)
    snap = build_game_quality_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "net_epa"] == pytest.approx(0.4)


def test_build_game_quality_row_per_team_per_game():
    pbp = pd.DataFrame(_game_rows())
    snap = build_game_quality_snapshot(pbp, 2026)
    assert len(snap) == 2  # one row for BUF, one for NYJ, for this single game
    assert set(snap["team_abbr"]) == {"BUF", "NYJ"}
