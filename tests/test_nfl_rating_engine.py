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


def test_builds_team_units_and_reports_prediction_influence(tmp_path):
    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir(); _madden_fixture(madden)
    pd.DataFrame([{"team_abbr": "BUF", "quarterback": 90, "offensive_line": 82,
                   "defensive_line": 84, "secondary": 83, "special_teams": 70,
                   "offense": 86, "defense": 84}]).to_csv(nfl / "team_snapshot.csv", index=False)
    players = build_player_ratings(madden, nfl)
    teams = build_team_ratings(players, nfl / "team_snapshot.csv")
    bills = teams["Buffalo Bills"]
    assert bills["overall_rating"] > 0
    assert bills["prediction_influence_enabled"] is True
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


def test_player_performance_uses_gsis_identity_not_name_collision(tmp_path):
    """Same-name players must never inherit each other's nflverse performance."""
    madden = tmp_path / "madden.csv"
    nfl = tmp_path / "nfl"
    nfl.mkdir()
    pd.DataFrame([
        {"player_name": "Justin Jefferson", "team": "Minnesota Vikings", "position": "WR", "overall": 94,
         "speed": 92, "acceleration": 91, "catching": 96, "awareness": 95},
        {"player_name": "Justin Jefferson", "team": "Cleveland Browns", "position": "LB", "overall": 67,
         "speed": 84, "acceleration": 85, "tackle": 70, "awareness": 65},
    ]).to_csv(madden, index=False)
    pd.DataFrame([
        {"full_name": "Justin Jefferson", "team": "MIN", "position": "WR", "gsis_id": "00-0036322", "status": "Active"},
        {"full_name": "Justin Jefferson", "team": "CLE", "position": "LB", "gsis_id": "00-0041075", "status": "Active"},
    ]).to_csv(nfl / "rosters.csv", index=False)
    pd.DataFrame([
        {"player_id": "00-0036322", "player_display_name": "Justin Jefferson", "team": "MIN", "position": "WR",
         "targets": 170, "receptions": 120, "receiving_yards": 1800, "receiving_tds": 12,
         "rushing_yards": 0, "macabets_performance_cap": 0.20},
    ]).to_csv(nfl / "player_weekly_stats.csv", index=False)

    players = build_player_ratings(madden, nfl)
    vikings = players[(players.player_name.eq("Justin Jefferson")) & (players.team_abbr.eq("MIN"))].iloc[0]
    browns = players[(players.player_name.eq("Justin Jefferson")) & (players.team_abbr.eq("CLE"))].iloc[0]
    assert vikings.gsis_id == "00-0036322"
    assert vikings.performance_weight > 0
    assert browns.gsis_id == "00-0041075"
    assert browns.performance_weight == 0
    assert pd.isna(browns.performance_grade)


def test_nflverse_arizona_roster_abbreviation_resolves_gsis(tmp_path):
    """nflverse uses AZ while Macabets/Sleeper use ARI; identity must still resolve."""
    madden = tmp_path / "madden.csv"
    nfl = tmp_path / "nfl"
    nfl.mkdir()
    pd.DataFrame([
        {"player_name": "Test Cardinal", "team": "Arizona Cardinals", "position": "WR", "overall": 80,
         "speed": 88, "acceleration": 87, "catching": 80, "awareness": 78},
    ]).to_csv(madden, index=False)
    pd.DataFrame([
        {"full_name": "Test Cardinal", "team": "AZ", "position": "WR", "gsis_id": "00-0099999", "status": "Active"},
    ]).to_csv(nfl / "rosters.csv", index=False)
    pd.DataFrame([
        {"player_id": "00-0099999", "player_display_name": "Test Cardinal", "team": "ARI", "position": "WR",
         "targets": 100, "receptions": 70, "receiving_yards": 1000, "receiving_tds": 8,
         "rushing_yards": 0, "macabets_performance_cap": 0.20},
    ]).to_csv(nfl / "player_weekly_stats.csv", index=False)

    players = build_player_ratings(madden, nfl)
    row = players.iloc[0]
    assert row.team_abbr == "ARI"
    assert row.gsis_id == "00-0099999"
    assert row.performance_weight > 0


def test_nickname_resolves_gsis_but_different_first_name_does_not(tmp_path):
    from engine.nfl_rating_engine import _fill_missing_ids_by_last_name

    players = pd.DataFrame([
        {"player_name": "Joshua Palmer", "team_abbr": "BUF", "position_family": "WR", "gsis_id": ""},
        {"player_name": "Chigoziem Okonkwo", "team_abbr": "WAS", "position_family": "TE", "gsis_id": ""},
        {"player_name": "Cody White", "team_abbr": "SEA", "position_family": "WR", "gsis_id": ""},
    ])
    roster = tmp_path / "weekly_rosters.csv"
    pd.DataFrame([
        {"full_name": "Josh Palmer", "first_name": "Josh", "team": "BUF", "position": "WR", "gsis_id": "00-PALMER", "week": 1},
        {"full_name": "Chig Okonkwo", "first_name": "Chigoziem", "team": "WAS", "position": "TE", "gsis_id": "00-OKONKWO", "week": 1},
        {"full_name": "Ricky White III", "first_name": "Ricky", "team": "SEA", "position": "WR", "gsis_id": "00-RWHITE", "week": 1},
    ]).to_csv(roster, index=False)

    out = _fill_missing_ids_by_last_name(players, roster).set_index("player_name")
    assert out.loc["Joshua Palmer", "gsis_id"] == "00-PALMER"
    assert out.loc["Chigoziem Okonkwo", "gsis_id"] == "00-OKONKWO"
    assert out.loc["Cody White", "gsis_id"] == ""


def _wr_madden(path: Path, names_ovr):
    rows = [{"player_name": n, "team": "BUF", "position": "WR", "overall": o,
             "speed": o, "catching": o, "awareness": o} for n, o in names_ovr]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_performance_uses_efficiency_not_volume(tmp_path, monkeypatch):
    from engine import nfl_rating_engine as engine
    monkeypatch.setattr(engine, "RATING_MODEL", "v1.5")
    """Same per-target production must earn the same grade; volume only adds trust."""
    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir()
    _wr_madden(madden, [("Busy Receiver", 85), ("Part Timer", 85), ("Dud Receiver", 85)])
    pd.DataFrame([
        {"player_display_name": "Busy Receiver", "team": "BUF", "position": "WR",
         "targets": 100, "receptions": 70, "receiving_yards": 900, "receiving_tds": 5},
        {"player_display_name": "Part Timer", "team": "BUF", "position": "WR",
         "targets": 20, "receptions": 14, "receiving_yards": 180, "receiving_tds": 1},
        {"player_display_name": "Dud Receiver", "team": "BUF", "position": "WR",
         "targets": 60, "receptions": 20, "receiving_yards": 150, "receiving_tds": 0},
    ]).to_csv(nfl / "player_weekly_stats.csv", index=False)
    players = build_player_ratings(madden, nfl).set_index("player_name")
    busy, part = players.loc["Busy Receiver"], players.loc["Part Timer"]
    assert abs(busy.performance_grade - part.performance_grade) < 0.01
    assert busy.performance_weight > part.performance_weight
    assert players.loc["Dud Receiver", "performance_grade"] < busy.performance_grade


def test_one_bad_game_only_nudges_an_elite_player(tmp_path, monkeypatch):
    from engine import nfl_rating_engine as engine
    monkeypatch.setattr(engine, "RATING_MODEL", "v1.5")
    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir()
    _wr_madden(madden, [("Elite Receiver", 99), ("Average Receiver", 80), ("Other Receiver", 78)])
    pd.DataFrame([
        {"player_display_name": "Elite Receiver", "team": "BUF", "position": "WR",
         "targets": 9, "receptions": 2, "receiving_yards": 11, "receiving_tds": 0},
        {"player_display_name": "Average Receiver", "team": "BUF", "position": "WR",
         "targets": 8, "receptions": 6, "receiving_yards": 90, "receiving_tds": 1},
        {"player_display_name": "Other Receiver", "team": "BUF", "position": "WR",
         "targets": 7, "receptions": 5, "receiving_yards": 70, "receiving_tds": 0},
    ]).to_csv(nfl / "player_weekly_stats.csv", index=False)
    players = build_player_ratings(madden, nfl).set_index("player_name")
    elite = players.loc["Elite Receiver"]
    assert elite.macabets_rating < elite.trait_grade
    assert elite.trait_grade - elite.macabets_rating < 3.0


def test_team_performance_weight_grows_with_games_and_stays_capped():
    from engine.nfl_rating_engine import TEAM_PERFORMANCE_CAP, team_performance_weight

    assert team_performance_weight(2026, 0, current_year=2026) == 0.0
    week1 = team_performance_weight(2026, 1, current_year=2026)
    week8 = team_performance_weight(2026, 8, current_year=2026)
    week17 = team_performance_weight(2026, 17, current_year=2026)
    assert 0 < week1 < 0.12
    assert week1 < week8 < week17 < TEAM_PERFORMANCE_CAP
    assert team_performance_weight(2025, 18, current_year=2026) == 0.20
    assert team_performance_weight(2027, 1, current_year=2026) == 0.0


def test_team_performance_is_rescaled_to_roster_scale():
    from engine.nfl_rating_engine import _rescale_to_roster

    perf = {"A": 50.0, "B": 68.0, "C": 86.0}
    roster = {"A": 76.0, "B": 79.0, "C": 82.0}
    out = _rescale_to_roster(perf, roster)
    assert round(sum(out.values()) / 3, 2) == 79.0
    assert out["A"] < out["B"] < out["C"]
    assert 74 < out["A"] < 79


def test_default_model_is_restored_v14_weighting():
    from engine import nfl_rating_engine as engine
    from engine.nfl_rating_engine import legacy_team_performance_weight

    assert engine.RATING_MODEL == "v1.4"
    assert legacy_team_performance_weight(2026, 1, current_year=2026) == 0.275
    assert legacy_team_performance_weight(2026, 10, current_year=2026) == 0.80
    assert legacy_team_performance_weight(2025, 18, current_year=2026) == 0.20


def test_v14_mode_matches_frozen_v14_engine(tmp_path):
    """Restored weighting must reproduce the frozen v1.4 engine on identical inputs."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("legacy_v14_for_test", root / "audit" / "legacy_nfl_rating_engine_v14.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)

    madden = tmp_path / "madden.csv"; nfl = tmp_path / "nfl"; nfl.mkdir()
    _wr_madden(madden, [("Receiver A", 90), ("Receiver B", 80), ("Receiver C", 72)])
    pd.DataFrame([
        {"player_display_name": "Receiver A", "team": "BUF", "position": "WR",
         "targets": 40, "receptions": 25, "receiving_yards": 300, "receiving_tds": 2},
        {"player_display_name": "Receiver B", "team": "BUF", "position": "WR",
         "targets": 60, "receptions": 45, "receiving_yards": 700, "receiving_tds": 6},
        {"player_display_name": "Receiver C", "team": "BUF", "position": "WR",
         "targets": 20, "receptions": 10, "receiving_yards": 90, "receiving_tds": 0},
    ]).to_csv(nfl / "player_weekly_stats.csv", index=False)
    snapshot = nfl / "team_snapshot.csv"
    pd.DataFrame([{"team_abbr": "BUF", "season": 2026, "through_week": 3, "offensive_line": 60,
                   "defensive_line": 90, "secondary": 55, "special_teams": 70, "defense": 75}]).to_csv(snapshot, index=False)
    no_chart = tmp_path / "none.csv"

    ours = build_player_ratings(madden, nfl, depth_chart_path=no_chart)
    theirs = legacy.build_player_ratings(madden, nfl, depth_chart_path=no_chart)
    assert ours["macabets_rating"].round(4).tolist() == theirs["macabets_rating"].round(4).tolist()

    our_team = build_team_ratings(ours, snapshot, no_chart, current_season=2026)["Buffalo Bills"]
    import engine.nfl_rating_engine as engine
    original = legacy.datetime

    class _Frozen(original):
        @classmethod
        def now(cls, tz=None):
            return original(2026, 9, 15, tzinfo=tz)

    legacy.datetime = _Frozen
    try:
        their_team = legacy.build_team_ratings(theirs, snapshot, no_chart)["Buffalo Bills"]
    finally:
        legacy.datetime = original
    for unit in ("offensive_line", "defensive_front", "secondary", "special_teams", "linebackers"):
        assert abs(our_team["units"][unit]["grade"] - their_team["units"][unit]["grade"]) < 1e-6, unit
    assert abs(our_team["overall_rating"] - their_team["overall_rating"]) < 1e-6
