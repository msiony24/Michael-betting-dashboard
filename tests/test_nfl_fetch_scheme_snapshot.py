"""Tests for build_scheme_snapshot() in engine/nfl_fetch.py, using a small
fully-controlled synthetic play-by-play dataset with hand-calculable rates.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.nfl_fetch import build_scheme_snapshot


def _play(game_id, posteam, defteam, play_type, *, sack=0, qb_hit=0,
          yardline_100=50, drive=1, touchdown=0, season=2026, season_type="REG") -> dict:
    return dict(
        game_id=game_id, posteam=posteam, defteam=defteam, play_type=play_type,
        season=season, season_type=season_type, down=1, yards_gained=5,
        pass_attempt=1 if play_type == "pass" else 0, rush_attempt=1 if play_type == "run" else 0,
        sack=sack, qb_hit=qb_hit, success=1, epa=0.1,
        yardline_100=yardline_100, drive=drive, touchdown=touchdown,
    )


def test_build_scheme_snapshot_pass_rate_exact():
    rows = [_play("g1", "BUF", "NYJ", "pass") for _ in range(8)] + [_play("g1", "BUF", "NYJ", "run") for _ in range(2)]
    rows += [_play("g1", "NYJ", "BUF", "pass") for _ in range(2)] + [_play("g1", "NYJ", "BUF", "run") for _ in range(8)]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "pass_rate"] == pytest.approx(0.8)
    assert snap.loc["NYJ", "pass_rate"] == pytest.approx(0.2)
    assert snap.loc["BUF", "offensive_plays"] == 10


def test_build_scheme_snapshot_pressure_rate_exact():
    rows = [
        _play("g1", "BUF", "NYJ", "pass", sack=1),
        _play("g1", "BUF", "NYJ", "pass", qb_hit=1),
        _play("g1", "BUF", "NYJ", "pass"),
        _play("g1", "BUF", "NYJ", "pass"),
        _play("g1", "BUF", "NYJ", "run"),
        _play("g1", "NYJ", "BUF", "run"),
    ]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026).set_index("team_abbr")
    # 2 of 4 BUF pass plays were pressured (sack or qb_hit).
    assert snap.loc["BUF", "pressure_rate_allowed"] == pytest.approx(0.5)


def test_build_scheme_snapshot_red_zone_td_rate_exact():
    rows = [
        _play("g1", "BUF", "NYJ", "run"), _play("g1", "NYJ", "BUF", "run"),
        _play("g1", "BUF", "NYJ", "run", yardline_100=10, drive=5, touchdown=1),
        _play("g1", "NYJ", "BUF", "run", yardline_100=15, drive=6, touchdown=0),
    ]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "red_zone_td_rate"] == pytest.approx(1.0)
    assert snap.loc["NYJ", "red_zone_td_rate"] == pytest.approx(0.0)
    # BUF's defense allowed NYJ's 0% red-zone conversion on that one trip.
    assert snap.loc["BUF", "red_zone_td_rate_allowed"] == pytest.approx(0.0)
    assert snap.loc["NYJ", "red_zone_td_rate_allowed"] == pytest.approx(1.0)


def test_build_scheme_snapshot_optional_charting_columns_default_to_na():
    rows = [_play("g1", "BUF", "NYJ", "pass"), _play("g1", "NYJ", "BUF", "run")]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026)
    for col in ("no_huddle_rate", "motion_rate", "play_action_rate", "rpo_rate", "blitz_rate", "man_rate", "zone_rate"):
        assert col in snap.columns
        assert snap[col].isna().all()


def test_build_scheme_snapshot_excludes_wrong_season():
    rows = [_play("g1", "BUF", "NYJ", "pass", season=2020)]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026)
    assert snap.empty


def test_build_scheme_snapshot_excludes_non_regular_season():
    rows = [_play("g1", "BUF", "NYJ", "pass", season_type="POST")]
    pbp = pd.DataFrame(rows)
    snap = build_scheme_snapshot(pbp, 2026)
    assert snap.empty


def test_build_scheme_snapshot_empty_input_returns_empty_frame():
    assert build_scheme_snapshot(pd.DataFrame({"posteam": [], "defteam": []}), 2026).empty
