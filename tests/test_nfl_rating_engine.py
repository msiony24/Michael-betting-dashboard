import json
from pathlib import Path

import pandas as pd

from engine.nfl_rating_engine import build_player_ratings, build_team_ratings, save_rating_outputs


def _madden_fixture(path: Path):
    rows = []
    for name, pos, ovr in [
        ("Test Quarterback", "QB", 88), ("Backup Quarterback", "QB", 70),
        ("Test Runner", "RB", 84), ("Test Receiver", "WR", 86),
        ("Test Tight End", "TE", 80), ("Left Tackle", "LT", 82),
        ("Left Guard", "LG", 79), ("Center Player", "C", 81),
        ("Right Guard", "RG", 78), ("Right Tackle", "RT", 80),
        ("Edge Player", "DE", 85), ("Tackle Player", "DT", 82),
        ("Linebacker One", "LB", 81), ("Corner One", "CB", 84),
        ("Safety One", "FS", 82), ("Kicker One", "K", 79),
    ]:
        rows.append({"player_name": name, "team": "BUF", "position": pos, "overall": ovr,
                     "speed": 80, "strength": 80, "agility": 80, "awareness": 80,
                     "injury": 90, "change_of_direction": 80})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_builds_players_and_blends_qb_performance(tmp_path):
    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir()
    _madden_fixture(madden)
    pd.DataFrame([{"player_display_name": "Test Quarterback", "recent_team": "BUF", "position": "QB",
                   "attempts": 600, "passing_yards": 5000, "passing_tds": 45, "interceptions": 5,
                   "rushing_yards": 500, "rushing_tds": 5, "sacks": 20}]).to_csv(nfl / "player_weekly_stats.csv", index=False)
    players = build_player_ratings(madden, nfl)
    qb = players.loc[players.player_name.eq("Test Quarterback")].iloc[0]
    assert qb.performance_weight > 0
    assert qb.rating_source == "Madden 27 + nflverse performance"


def test_builds_team_units_and_keeps_prediction_off(tmp_path):
    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir(); _madden_fixture(madden)
    pd.DataFrame([{"team_abbr": "BUF", "quarterback": 90, "offensive_line": 82,
                   "defensive_line": 84, "secondary": 83, "special_teams": 70,
                   "offense": 86, "defense": 84}]).to_csv(nfl / "team_snapshot.csv", index=False)
    players = build_player_ratings(madden, nfl)
    teams = build_team_ratings(players, nfl / "team_snapshot.csv")
    bills = teams["Buffalo Bills"]
    assert bills["overall_rating"] > 0
    assert bills["prediction_influence_enabled"] is False
    assert "quarterback" in bills["units"]


def test_saves_status_and_history(tmp_path):
    players = pd.DataFrame([{"player_name": "A", "performance_weight": 0.2}])
    teams = {"Buffalo Bills": {"overall_rating": 80.0}}
    status = save_rating_outputs(players, teams, nfl_dir=tmp_path)
    assert status["players_rated"] == 1
    assert json.loads((tmp_path / "rating_status.json").read_text())["teams_rated"] == 1
    assert (tmp_path / "rating_history.jsonl").read_text().strip()


def test_automatic_nflverse_depth_chart_is_normalized_and_preferred(tmp_path):
    from engine.nfl_depth_chart import TEAM_TO_ABBR, load_depth_charts
    from engine.nfl_rating_engine import _resolve_depth_chart_path

    rows = []
    for team_abbr in TEAM_TO_ABBR.values():
        rows.append({
            "dt": "2026-08-21T10:00:00Z",
            "team": team_abbr,
            "player_name": f"Old {team_abbr} QB",
            "pos_abb": "QB",
            "pos_slot": 1,
            "pos_rank": 1,
        })
        rows.append({
            "dt": "2026-08-22T10:00:00Z",
            "team": team_abbr,
            "player_name": f"Current {team_abbr} QB",
            "pos_abb": "QB",
            "pos_slot": 1,
            "pos_rank": 1,
        })
        rows.append({
            "dt": "2026-08-22T10:00:00Z",
            "team": team_abbr,
            "player_name": f"Backup {team_abbr} QB",
            "pos_abb": "QB",
            "pos_slot": 1,
            "pos_rank": 2,
        })
    auto_path = tmp_path / "depth_charts.csv"
    pd.DataFrame(rows).to_csv(auto_path, index=False)

    normalized = load_depth_charts(auto_path)
    bills = normalized[(normalized["team_abbr"] == "BUF") & (normalized["Position"] == "QB")].iloc[0]
    assert bills["Starter"] == "Current BUF QB"
    assert bills["2nd String"] == "Backup BUF QB"
    assert "Old BUF QB" not in bills.tolist()
    assert normalized.attrs["source_name"] == "nflverse automatic depth chart"
    assert normalized["team_abbr"].nunique() == 32
    assert _resolve_depth_chart_path(tmp_path) == auto_path
