import pandas as pd

from engine.nfl_availability import classify_availability, sleeper_payload_to_frame


def test_definitive_and_uncertain_statuses():
    assert classify_availability(roster_status="Active", injury_status="Out", active=True) == ("Out", True)
    assert classify_availability(roster_status="Injured Reserve", injury_status="", active=False) == ("Out", True)
    assert classify_availability(roster_status="Active", injury_status="Questionable", active=True) == ("Questionable", False)
    assert classify_availability(roster_status="Active", injury_status="Doubtful", active=True) == ("Doubtful", False)
    assert classify_availability(roster_status="Active", injury_status="", active=True) == ("Active", False)


def test_sleeper_payload_normalizes_player_rows():
    frame = sleeper_payload_to_frame({
        "1": {"first_name": "Josh", "last_name": "Allen", "team": "BUF", "position": "QB", "status": "Active", "active": True},
        "2": {"full_name": "Example Tackle", "team": "BUF", "position": "T", "status": "Injured Reserve", "active": False},
    }, updated_at_utc="2026-08-10T20:00:00+00:00")
    assert len(frame) == 2
    assert frame.loc[frame.player_name.eq("Josh Allen"), "availability_state"].iloc[0] == "Active"
    assert bool(frame.loc[frame.player_name.eq("Example Tackle"), "definitively_unavailable"].iloc[0]) is True

from engine.nfl_rating_engine import _apply_availability


def test_out_starter_promotes_same_role_backup():
    starters = pd.DataFrame([
        {"player_name": "Starter LT", "depth_chart_role": "LT", "macabets_rating": 88.0,
         "definitively_unavailable": True, "availability_state": "Out", "injury_status": "Out"},
        {"player_name": "Starter LG", "depth_chart_role": "LG", "macabets_rating": 84.0,
         "definitively_unavailable": False, "availability_state": "Active", "injury_status": ""},
    ])
    depth = pd.DataFrame([
        {"player_name": "Backup LT", "depth_chart_role": "LT", "macabets_rating": 74.0,
         "definitively_unavailable": False, "availability_state": "Active", "injury_status": ""},
        {"player_name": "Backup LG", "depth_chart_role": "LG", "macabets_rating": 72.0,
         "definitively_unavailable": False, "availability_state": "Active", "injury_status": ""},
    ])
    active, remaining, unavailable, promotions = _apply_availability(starters, depth)
    assert set(active.player_name) == {"Backup LT", "Starter LG"}
    assert "Backup LT" not in set(remaining.player_name)
    assert unavailable[0]["name"] == "Starter LT"
    assert promotions[0]["in"] == "Backup LT"


def test_questionable_starter_is_not_benched():
    starters = pd.DataFrame([
        {"player_name": "Questionable QB", "depth_chart_role": "QB", "macabets_rating": 90.0,
         "definitively_unavailable": False, "availability_state": "Questionable", "injury_status": "Questionable"},
    ])
    depth = pd.DataFrame([
        {"player_name": "Backup QB", "depth_chart_role": "QB", "macabets_rating": 70.0,
         "definitively_unavailable": False, "availability_state": "Active", "injury_status": ""},
    ])
    active, _, unavailable, promotions = _apply_availability(starters, depth)
    assert active.iloc[0].player_name == "Questionable QB"
    assert unavailable == []
    assert promotions == []

from engine.nfl_rating_engine import build_player_ratings, build_team_ratings


def test_sleeper_out_qb_activates_qb2_in_team_unit(tmp_path):
    madden = tmp_path / "madden.csv"
    nfl = tmp_path / "nfl"
    nfl.mkdir()
    pd.DataFrame([
        {"player_name": "Starter Quarterback", "team": "BUF", "position": "QB", "overall": 92,
         "speed": 80, "strength": 70, "agility": 78, "awareness": 90, "injury": 90, "change_of_direction": 75},
        {"player_name": "Backup Quarterback", "team": "BUF", "position": "QB", "overall": 72,
         "speed": 75, "strength": 70, "agility": 75, "awareness": 70, "injury": 90, "change_of_direction": 72},
    ]).to_csv(madden, index=False)
    depth = nfl / "footballguys_depth_charts.csv"
    pd.DataFrame([{
        "Team": "Buffalo Bills", "Unit": "Offense", "Position": "QB",
        "Starter": "Starter Quarterback", "2nd String": "Backup Quarterback",
        "3rd String": "", "4th String": "", "5th String": "", "Source URL": "fixture",
    }]).to_csv(depth, index=False)
    pd.DataFrame([
        {"player_name": "Starter Quarterback", "name_key": "starterquarterback", "team_abbr": "BUF",
         "roster_status": "Active", "injury_status": "Out", "practice_participation": "DNP",
         "availability_state": "Out", "definitively_unavailable": True,
         "updated_at_utc": "2026-08-10T20:00:00+00:00"},
        {"player_name": "Backup Quarterback", "name_key": "backupquarterback", "team_abbr": "BUF",
         "roster_status": "Active", "injury_status": "", "practice_participation": "Full",
         "availability_state": "Active", "definitively_unavailable": False,
         "updated_at_utc": "2026-08-10T20:00:00+00:00"},
    ]).to_csv(nfl / "sleeper_availability.csv", index=False)

    players = build_player_ratings(madden, nfl, depth_chart_path=depth)
    teams = build_team_ratings(players, snapshot_path=nfl / "missing.csv", depth_chart_path=depth)
    qb = teams["Buffalo Bills"]["units"]["quarterback"]
    assert qb["top_players"][0]["name"] == "Backup Quarterback"
    assert qb["availability_promotions"][0]["out"] == "Starter Quarterback"
    assert qb["availability_promotions"][0]["in"] == "Backup Quarterback"
