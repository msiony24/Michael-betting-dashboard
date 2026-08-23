"""Tests for build_team_snapshot(), the heaviest untested aggregation in
engine/nfl_fetch.py. Uses a small, fully-controlled synthetic play-by-play
dataset (4 teams) where every rate is hand-calculable, so assertions check
exact values rather than just "did it crash."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.nfl_fetch import build_team_snapshot


def _play(game_id, posteam, defteam, epa, play_type, *, down=1, yards=5, qtr=1,
          score_diff=0, third_down_converted=None, yardline_100=50, drive=1,
          touchdown=0, sack=0, qb_hit=0, interception=0, fumble_lost=0, week=1,
          season=2026, season_type="REG") -> dict:
    row = dict(
        game_id=game_id, posteam=posteam, defteam=defteam, epa=epa,
        season=season, season_type=season_type, play_type=play_type,
        success=1 if epa > 0 else 0, yards_gained=yards,
        pass_attempt=1 if play_type == "pass" else 0, rush_attempt=1 if play_type == "run" else 0,
        interception=interception, fumble_lost=fumble_lost, sack=sack, qb_hit=qb_hit,
        cpoe=5.0 if play_type == "pass" else np.nan,
        down=down, qtr=qtr, score_differential=score_diff,
        yardline_100=yardline_100, drive=drive, touchdown=touchdown,
        td_team=posteam if touchdown else "",
        special_teams_play=0, week=week,
    )
    if third_down_converted is not None:
        row["third_down_converted"] = third_down_converted
    return row


def _base_rows() -> list[dict]:
    """BUF (strong offense) vs NYJ (weak offense); KC vs DEN as neutral filler
    games so percentile ranks aren't degenerate with only 2 teams."""
    rows = []
    for i in range(20):
        rows.append(_play("g1", "BUF", "NYJ", 0.3, "pass" if i % 2 == 0 else "run", yards=8))
        rows.append(_play("g1", "NYJ", "BUF", -0.2, "pass" if i % 2 == 0 else "run", yards=2))
    for i in range(20):
        rows.append(_play("g2", "KC", "DEN", 0.05, "pass" if i % 2 == 0 else "run", yards=5))
        rows.append(_play("g2", "DEN", "KC", 0.0, "pass" if i % 2 == 0 else "run", yards=4))
    return rows


def test_build_team_snapshot_offense_epa_matches_exactly():
    pbp = pd.DataFrame(_base_rows())
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "offense_epa_per_play"] == pytest.approx(0.3)
    assert snap.loc["NYJ", "offense_epa_per_play"] == pytest.approx(-0.2)


def test_build_team_snapshot_defense_epa_allowed_matches_the_opponents_offense():
    pbp = pd.DataFrame(_base_rows())
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    # NYJ's defense faced BUF's 0.3 EPA/play offense.
    assert snap.loc["NYJ", "defense_epa_allowed"] == pytest.approx(0.3)
    assert snap.loc["BUF", "defense_epa_allowed"] == pytest.approx(-0.2)


def test_build_team_snapshot_higher_offense_epa_gives_higher_offense_rating():
    pbp = pd.DataFrame(_base_rows())
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "offense"] > snap.loc["KC", "offense"] > snap.loc["DEN", "offense"] > snap.loc["NYJ", "offense"]


def test_build_team_snapshot_third_down_conversion_rate_exact():
    rows = _base_rows()
    for i in range(4):
        rows.append(_play("g1", "BUF", "NYJ", 0.1, "pass", down=3, third_down_converted=1 if i < 3 else 0))
        rows.append(_play("g1", "NYJ", "BUF", -0.1, "pass", down=3, third_down_converted=1 if i < 1 else 0))
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "third_down_conversion_rate"] == pytest.approx(0.75)
    assert snap.loc["NYJ", "third_down_conversion_rate"] == pytest.approx(0.25)
    # NYJ's defense allowed BUF's 3/4 conversion rate.
    assert snap.loc["NYJ", "third_down_conversion_allowed"] == pytest.approx(0.75)


def test_build_team_snapshot_red_zone_td_rate_exact():
    rows = _base_rows()
    # BUF: 1 red-zone trip (drive 5), scores -> 100%.
    rows.append(_play("g1", "BUF", "NYJ", 0.5, "run", yardline_100=10, drive=5, touchdown=1))
    # NYJ: 1 red-zone trip (drive 6), no score -> 0%.
    rows.append(_play("g1", "NYJ", "BUF", -0.3, "run", yardline_100=15, drive=6, touchdown=0))
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "red_zone_td_rate"] == pytest.approx(1.0)
    assert snap.loc["NYJ", "red_zone_td_rate"] == pytest.approx(0.0)
    # BUF's defense (NYJ's opponent) allowed a 0% red-zone rate on that trip.
    assert snap.loc["NYJ", "red_zone_td_rate_allowed"] == pytest.approx(1.0)


def test_build_team_snapshot_high_leverage_epa_only_counts_close_4th_quarter():
    rows = _base_rows()
    # Close 4th-quarter plays: should count.
    for i in range(5):
        rows.append(_play("g1", "BUF", "NYJ", 0.4, "pass", qtr=4, score_diff=3))
        rows.append(_play("g1", "NYJ", "BUF", -0.4, "pass", qtr=4, score_diff=-3))
    # Garbage-time 4th-quarter play (score_diff > 8): should NOT count.
    rows.append(_play("g1", "BUF", "NYJ", 0.9, "pass", qtr=4, score_diff=21))
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "high_leverage_epa"] == pytest.approx(0.4)
    assert snap.loc["NYJ", "high_leverage_epa"] == pytest.approx(-0.4)


def test_build_team_snapshot_excludes_wrong_season():
    rows = _base_rows() + [_play("g3", "BUF", "NYJ", 99.0, "pass", season=2020)]
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    # The absurd 2020 play must not leak into the 2026 aggregate.
    assert snap.loc["BUF", "offense_epa_per_play"] == pytest.approx(0.3)


def test_build_team_snapshot_excludes_non_regular_season():
    rows = _base_rows() + [_play("g3", "BUF", "NYJ", 99.0, "pass", season_type="POST")]
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "offense_epa_per_play"] == pytest.approx(0.3)


def test_build_team_snapshot_excludes_non_pass_run_plays():
    rows = _base_rows() + [_play("g3", "BUF", "NYJ", 99.0, "punt")]
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "offense_epa_per_play"] == pytest.approx(0.3)


def test_build_team_snapshot_turnover_detection_via_interception_or_fumble():
    rows = _base_rows()
    rows.append(_play("g1", "BUF", "NYJ", -0.5, "pass", interception=1))
    rows.append(_play("g1", "BUF", "NYJ", -0.5, "run", fumble_lost=1))
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    # 2 turnovers out of 22 BUF offensive plays now.
    assert snap.loc["BUF", "offense_turnover_rate"] == pytest.approx(2 / 22, abs=1e-6)


def test_build_team_snapshot_disrupted_vs_clean_qb_splits():
    rows = _base_rows()
    # A sacked/hit dropback with poor EPA, and a clean dropback with good EPA.
    rows.append(_play("g1", "BUF", "NYJ", -1.5, "pass", sack=1))
    rows.append(_play("g1", "BUF", "NYJ", 1.0, "pass", sack=0, qb_hit=0))
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    assert snap.loc["BUF", "qb_epa_when_disrupted"] < snap.loc["BUF", "qb_epa_when_clean"]


def test_build_team_snapshot_opponent_quality_reflects_strength_of_schedule():
    pbp = pd.DataFrame(_base_rows())
    snap = build_team_snapshot(pbp, 2026).set_index("team_abbr")
    # NYJ's only opponent (BUF) is a strong offense -> NYJ faced tougher offense
    # than BUF faced (NYJ). opponent_quality_epa = opp_offense - opp_defense_allowed.
    assert snap.loc["NYJ", "opponent_quality_epa"] > snap.loc["BUF", "opponent_quality_epa"]


def test_build_team_snapshot_missing_required_columns_raises():
    frame = pd.DataFrame({"posteam": ["BUF"], "defteam": ["NYJ"]})  # no "epa"
    with pytest.raises(ValueError, match="missing required columns"):
        build_team_snapshot(frame, 2026)


def test_build_team_snapshot_no_usable_plays_raises():
    pbp = pd.DataFrame(_base_rows())
    with pytest.raises(ValueError, match="No usable regular-season plays"):
        build_team_snapshot(pbp, 1999)  # no plays exist for this season


def test_build_team_snapshot_missing_optional_columns_degrades_gracefully():
    # Strip down to only the required columns plus play/pass/rush flags --
    # third-down, red-zone, and high-leverage sections should all be skipped
    # without crashing rather than raising on missing 'down'/'qtr'/etc.
    minimal_rows = []
    for i in range(10):
        minimal_rows.append({
            "posteam": "BUF", "defteam": "NYJ", "epa": 0.2, "play_type": "pass",
            "season": 2026, "season_type": "REG",
        })
        minimal_rows.append({
            "posteam": "NYJ", "defteam": "BUF", "epa": -0.1, "play_type": "run",
            "season": 2026, "season_type": "REG",
        })
    pbp = pd.DataFrame(minimal_rows)
    snap = build_team_snapshot(pbp, 2026)
    assert not snap.empty
    assert "team" in snap.columns


def test_build_team_snapshot_season_and_through_week_metadata():
    rows = _base_rows()
    for row in rows:
        row["week"] = 3
    pbp = pd.DataFrame(rows)
    snap = build_team_snapshot(pbp, 2026)
    assert (snap["season"] == 2026).all()
    assert snap["through_week"].iloc[0] == 3
