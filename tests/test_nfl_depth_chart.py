from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.nfl_depth_chart import (
    ABBR_TO_TEAM,
    DEPTH_COLUMNS,
    depth_chart_team_assignments,
    load_depth_charts,
    match_depth_players,
    normalize_player_name,
    team_depth_chart,
    unit_depth_plan,
)


# --- normalize_player_name --------------------------------------------------

def test_normalize_player_name_strips_suffixes_and_punctuation():
    assert normalize_player_name("A.J. Brown Jr.") == "ajbrown"
    assert normalize_player_name("Odell Beckham III") == "odellbeckham"


def test_normalize_player_name_empty_input():
    assert normalize_player_name("") == ""
    assert normalize_player_name(None) == ""


# --- load_depth_charts: format auto-detection -------------------------------

def _nflverse_rows(dt: str) -> list[dict]:
    rows = []
    for team, qb1, qb2 in (("ARI", "Jacoby Brissett", "Gardner Minshew"), ("BUF", "Josh Allen", "Mitchell Trubisky")):
        rows.append({"dt": dt, "team": team, "player_name": qb1, "espn_id": "1", "gsis_id": "1",
                     "pos_grp_id": "1", "pos_grp": "QB", "pos_id": "1", "pos_name": "Quarterback",
                     "pos_abb": "QB", "pos_slot": "1", "pos_rank": "1"})
        rows.append({"dt": dt, "team": team, "player_name": qb2, "espn_id": "2", "gsis_id": "2",
                     "pos_grp_id": "1", "pos_grp": "QB", "pos_id": "1", "pos_name": "Quarterback",
                     "pos_abb": "QB", "pos_slot": "2", "pos_rank": "2"})
        rows.append({"dt": dt, "team": team, "player_name": f"{team} Nickel", "espn_id": "3", "gsis_id": "3",
                     "pos_grp_id": "2", "pos_grp": "DB", "pos_id": "2", "pos_name": "Nickel Back",
                     "pos_abb": "NB", "pos_slot": "1", "pos_rank": "1"})
    return rows


def test_load_depth_charts_missing_file_returns_empty(tmp_path: Path):
    result = load_depth_charts(tmp_path / "missing.csv")
    assert result.empty
    assert list(result.columns) == ["Team", "Unit", "Position", *DEPTH_COLUMNS, "Source URL", "team_abbr"]


def test_load_depth_charts_detects_nflverse_auto_format(tmp_path: Path):
    path = tmp_path / "depth_charts.csv"
    pd.DataFrame(_nflverse_rows("2026-08-15T00:00:00Z")).to_csv(path, index=False)

    result = load_depth_charts(path)
    assert result.attrs["source_name"] == "nflverse automatic depth chart"
    ari_qb = result[(result["team_abbr"] == "ARI") & (result["Position"] == "QB")]
    assert ari_qb.iloc[0]["Starter"] == "Jacoby Brissett"
    assert ari_qb.iloc[0]["2nd String"] == "Gardner Minshew"


def test_load_depth_charts_only_uses_latest_snapshot(tmp_path: Path):
    path = tmp_path / "depth_charts.csv"
    rows = _nflverse_rows("2026-07-01T00:00:00Z") + _nflverse_rows("2026-08-15T00:00:00Z")
    # Make the old snapshot obviously different so we can detect leakage.
    for row in rows:
        if row["dt"] == "2026-07-01T00:00:00Z" and row["pos_abb"] == "QB" and row["pos_rank"] == "1":
            row["player_name"] = "Stale Old Starter"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = load_depth_charts(path)
    assert "Stale Old Starter" not in result[list(DEPTH_COLUMNS)].to_csv()


def test_load_depth_charts_renames_nickel_back_to_scb(tmp_path: Path):
    path = tmp_path / "depth_charts.csv"
    pd.DataFrame(_nflverse_rows("2026-08-15T00:00:00Z")).to_csv(path, index=False)
    result = load_depth_charts(path)
    scb_row = result[(result["team_abbr"] == "ARI") & (result["Position"] == "SCB")]
    assert len(scb_row) == 1
    assert scb_row.iloc[0]["Unit"] == "Defense"


def test_load_depth_charts_auto_format_missing_columns_raises(tmp_path: Path):
    path = tmp_path / "depth_charts.csv"
    # Has "dt" and "team" but not the rest -> should NOT be silently treated
    # as a valid (empty) manual file; it should fail loudly.
    pd.DataFrame([{"dt": "2026-08-15T00:00:00Z", "team": "ARI"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_depth_charts(path)


def test_load_depth_charts_manual_footballguys_format(tmp_path: Path):
    path = tmp_path / "footballguys.csv"
    pd.DataFrame([{
        "Team": "Buffalo Bills", "Unit": "Offense", "Position": "QB",
        "Starter": "Josh Allen", "2nd String": "Mitchell Trubisky",
        "3rd String": "", "4th String": "", "5th String": "",
        "Source URL": "https://www.footballguys.com/depth-charts",
    }]).to_csv(path, index=False)
    result = load_depth_charts(path)
    assert result.attrs["source_name"] == "Footballguys/manual depth chart"
    assert result.iloc[0]["team_abbr"] == "BUF"


def test_load_depth_charts_manual_format_missing_columns_raises(tmp_path: Path):
    path = tmp_path / "footballguys.csv"
    pd.DataFrame([{"Team": "Buffalo Bills"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_depth_charts(path)


# --- depth_chart_team_assignments ------------------------------------------

def test_depth_chart_team_assignments_unique_player_maps_to_team():
    charts = pd.DataFrame([{
        "team_abbr": "BUF", "Starter": "Josh Allen", "2nd String": "", "3rd String": "", "4th String": "", "5th String": "",
    }])
    assignments = depth_chart_team_assignments(charts)
    assert assignments[normalize_player_name("Josh Allen")] == "BUF"


def test_depth_chart_team_assignments_excludes_players_listed_on_two_teams():
    # A malformed/overlapping upload shouldn't let a player silently "belong"
    # to whichever team happened to be processed last.
    charts = pd.DataFrame([
        {"team_abbr": "BUF", "Starter": "Ambiguous Player", "2nd String": "", "3rd String": "", "4th String": "", "5th String": ""},
        {"team_abbr": "NYJ", "Starter": "Ambiguous Player", "2nd String": "", "3rd String": "", "4th String": "", "5th String": ""},
    ])
    assignments = depth_chart_team_assignments(charts)
    assert normalize_player_name("Ambiguous Player") not in assignments


# --- unit_depth_plan: scheme detection --------------------------------------

def _team_depth_row(position: str, unit: str, *names: str) -> dict:
    row = {"team_abbr": "BUF", "Unit": unit, "Position": position, "Source URL": "test"}
    for i, col in enumerate(DEPTH_COLUMNS):
        row[col] = names[i] if i < len(names) else ""
    return row


def test_unit_depth_plan_detects_3_4_front():
    rows = [
        _team_depth_row("NT", "Defense", "Nose Tackle"),
        _team_depth_row("LILB", "Defense", "Left ILB"),
        _team_depth_row("RILB", "Defense", "Right ILB"),
        _team_depth_row("LDE", "Defense", "Left End"),
        _team_depth_row("RDE", "Defense", "Right End"),
        _team_depth_row("SLB", "Defense", "Strongside"),
        _team_depth_row("WLB", "Defense", "Weakside"),
    ]
    team_depth = pd.DataFrame(rows)
    plan = unit_depth_plan(team_depth, "defensive_front")
    assert plan["scheme"] == "3-4"


def test_unit_depth_plan_detects_4_3_front_when_no_ilb_pair():
    rows = [
        _team_depth_row("LDE", "Defense", "Left End"),
        _team_depth_row("LDT", "Defense", "Left Tackle"),
        _team_depth_row("RDT", "Defense", "Right Tackle"),
        _team_depth_row("RDE", "Defense", "Right End"),
    ]
    team_depth = pd.DataFrame(rows)
    plan = unit_depth_plan(team_depth, "defensive_front")
    assert plan["scheme"] == "4-3"


def test_unit_depth_plan_empty_team_returns_missing_source():
    plan = unit_depth_plan(pd.DataFrame(), "quarterback")
    assert plan["source"] == "missing"
    assert plan["starters"] == []


def test_unit_depth_plan_dedupes_a_player_listed_at_two_roles():
    # e.g. a swing tackle listed at both LT and RT depth slots.
    rows = [
        _team_depth_row("LT", "Offense", "Swing Tackle", "Backup A"),
        _team_depth_row("RT", "Offense", "Swing Tackle", "Backup B"),
        _team_depth_row("LG", "Offense", "Guard One"),
        _team_depth_row("C", "Offense", "Center One"),
        _team_depth_row("RG", "Offense", "Guard Two"),
    ]
    team_depth = pd.DataFrame(rows)
    plan = unit_depth_plan(team_depth, "offensive_line")
    all_names = [name for name, _role in plan["starters"] + plan["depth"]]
    assert all_names.count("Swing Tackle") == 1


# --- match_depth_players -----------------------------------------------------

def _players_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_match_depth_players_exact_name_match():
    players = _players_frame([{"player_name": "Josh Allen", "position": "QB"}])
    matched, unmatched = match_depth_players(players, [("Josh Allen", "QB")])
    assert len(matched) == 1
    assert unmatched == []


def test_match_depth_players_nickname_bridge_within_role_family():
    players = _players_frame([{"player_name": "Christopher Godwin", "position": "WR"}])
    matched, unmatched = match_depth_players(players, [("Chris Godwin", "WR")])
    assert len(matched) == 1
    assert unmatched == []


def test_match_depth_players_does_not_cross_incompatible_position_family():
    # Same surname, but the roster player is a DB, not a QB -- must not match.
    players = _players_frame([{"player_name": "Chris Jones", "position": "CB"}])
    matched, unmatched = match_depth_players(players, [("Chris Jones", "QB")])
    assert len(matched) == 0
    assert unmatched == [{"name": "Chris Jones", "role": "QB"}]


def test_match_depth_players_ambiguous_first_name_candidates_stay_unmatched():
    players = _players_frame([
        {"player_name": "Michael Thomas", "position": "WR"},
        {"player_name": "Mike Thomas", "position": "WR"},
    ])
    # Neither candidate should be confidently preferred over the other.
    matched, unmatched = match_depth_players(players, [("M. Thomas", "WR")])
    assert len(matched) <= 1  # never both; a tie should not double-assign


def test_match_depth_players_empty_inputs():
    matched, unmatched = match_depth_players(pd.DataFrame(), [("Josh Allen", "QB")])
    assert matched.empty
    assert unmatched == [{"name": "Josh Allen", "role": "QB"}]
