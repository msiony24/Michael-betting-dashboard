"""Offline end-to-end check of audit/nfl_rating_backtest.py on a tiny synthetic league."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nfl_rating_backtest", ROOT / "audit" / "nfl_rating_backtest.py")
bt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bt
_spec.loader.exec_module(bt)  # type: ignore[union-attr]

TEAMS = ["BUF", "MIA", "NE", "NYJ"]
LAYOUT = {"QB": 2, "RB": 2, "WR": 4, "TE": 2, "OL": 6, "DL": 5, "LB": 4, "DB": 6, "K": 1, "P": 1, "LS": 1}
ROSTER_POS = {"OL": "T", "DL": "DE", "DB": "CB"}
SCHEDULE = {1: [("BUF", "MIA"), ("NE", "NYJ")], 2: [("MIA", "NE"), ("NYJ", "BUF")], 3: [("BUF", "NE"), ("MIA", "NYJ")]}


def _synthetic_inputs(season: int = 2025):
    rng = np.random.default_rng(3)
    madden, rosters, stats, pbp, games = [], [], [], [], []
    strength = {"BUF": 8, "MIA": 2, "NE": -3, "NYJ": -6}
    for team in TEAMS:
        for pos, count in LAYOUT.items():
            for i in range(count):
                name = f"{team} {pos} Player{i}"
                ovr = 72 + strength[team] + (6 if i == 0 else 0) + int(rng.integers(-2, 3))
                madden.append({"ea_player_id": len(madden) + 1, "player_name": name, "team": team,
                               "position": pos, "overall": ovr, "speed": ovr, "awareness": ovr})
                for week in SCHEDULE:
                    status = "INA" if (team == "BUF" and pos == "QB" and i == 0 and week == 3) else "ACT"
                    rosters.append({"season": season, "week": week, "team": team, "full_name": name,
                                    "position": ROSTER_POS.get(pos, pos), "status": status,
                                    "gsis_id": f"00-{len(madden):04d}"})
    for week, pairs in SCHEDULE.items():
        for home, away in pairs:
            home_pts = 24 + strength[home] - strength[away] // 2
            away_pts = 20 + strength[away]
            games.append({"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": "REG",
                          "week": week, "home_team": home, "away_team": away, "home_score": home_pts,
                          "away_score": away_pts, "location": "Home",
                          "home_moneyline": -150 if strength[home] > strength[away] else 130,
                          "away_moneyline": 130 if strength[home] > strength[away] else -150})
            for play in range(80):
                offense, defense = (home, away) if play % 2 else (away, home)
                pbp.append({"game_id": games[-1]["game_id"], "season": season, "season_type": "REG", "week": week,
                            "posteam": offense, "defteam": defense, "home_team": home, "away_team": away,
                            "epa": rng.normal(strength[offense] / 20 - strength[defense] / 20, 1),
                            "play_type": "pass" if play % 3 else "run", "play_id": play,
                            "total_home_score": home_pts, "total_away_score": away_pts,
                            "down": 1 + play % 4, "ydstogo": 10, "yardline_100": 50, "qtr": 1 + play // 20})
        for row in [r for r in rosters if r["week"] == week and r["status"] == "ACT"]:
            pos = row["position"]
            if pos not in {"QB", "RB", "WR", "TE"}:
                continue
            edge = strength[row["team"]] / 10
            stats.append({"player_id": row["gsis_id"], "player_display_name": row["full_name"], "position": pos,
                          "team": row["team"], "season": season, "week": week, "season_type": "REG",
                          "attempts": 30 if pos == "QB" else 0, "passing_yards": 230 + 20 * edge if pos == "QB" else 0,
                          "passing_tds": 2 if pos == "QB" else 0, "passing_interceptions": 1 if pos == "QB" else 0,
                          "sacks_suffered": 2 if pos == "QB" else 0, "carries": 12 if pos == "RB" else 0,
                          "rushing_yards": 50 + 5 * edge if pos == "RB" else 0,
                          "targets": 6 if pos in {"WR", "TE"} else 2,
                          "receptions": 4, "receiving_yards": 50 + 5 * edge,
                          "passing_epa": 3 * edge if pos == "QB" else 0.0, "rushing_epa": edge,
                          "receiving_epa": edge})
    inputs = bt.SeasonInputs(season, pd.DataFrame(games), pd.DataFrame(stats), pd.DataFrame(rosters), pd.DataFrame(pbp))
    return inputs, pd.DataFrame(madden)


def test_week_inputs_never_include_the_predicted_week(tmp_path):
    inputs, madden = _synthetic_inputs()
    info = bt.prepare_week(inputs, madden, 3, tmp_path / "w3")
    stats = pd.read_csv(tmp_path / "w3" / "player_weekly_stats_base.csv")
    snapshot = pd.read_csv(tmp_path / "w3" / "team_snapshot.csv")
    week_madden = pd.read_csv(tmp_path / "w3" / "madden.csv")
    assert stats["week"].max() == 2
    assert snapshot["through_week"].max() == 2
    assert "BUF QB Player0" not in set(week_madden["player_name"])  # inactive that week
    assert info["active_matched"] == len(week_madden)


def test_backtest_runs_end_to_end_and_scores_every_variant(tmp_path):
    inputs, madden = _synthetic_inputs()
    variants = bt.default_variants(include_grid=False) + [
        bt.Variant("grid_test", team_cap=0.3, team_stability=3.0, group="grid"),
    ]
    predictions, week_log = bt.run_backtest(inputs, madden, [2, 3], variants, work_dir=tmp_path, log=lambda _: None)
    assert len(week_log) == 2
    for v in variants:
        assert predictions[f"margin__{v.name}"].notna().all()
    # Performance blending must actually change something relative to Madden only.
    assert not np.allclose(predictions["margin__madden_only"], predictions["margin__v1_5_current"])

    summary = bt.summarize(predictions, variants)
    assert set(summary["table"]["variant"]) == {v.name for v in variants}
    assert "v1_5_current_vs_madden_only" in summary["comparisons"]
    assert "market_closing_moneyline" in summary["market"]
    report = bt.write_report(summary, {"season": 2025, "madden_iteration": "synthetic",
                                       "generated_at_utc": "now"}, tmp_path)
    text = report.read_text()
    assert "Main comparison" in text and "v1_4_legacy" in text


def test_current_engine_settings_are_restored_after_a_variant(tmp_path):
    from engine import nfl_rating_engine as engine

    before = (dict(engine.PERFORMANCE_STABILITY), engine.TEAM_PERFORMANCE_CAP, engine.TEAM_PERFORMANCE_STABILITY)
    with bt._current_settings(bt.Variant("x", player_stability_mult=2.0, team_cap=0.1, team_stability=1.0)):
        assert engine.TEAM_PERFORMANCE_CAP == 0.1
    assert (dict(engine.PERFORMANCE_STABILITY), engine.TEAM_PERFORMANCE_CAP, engine.TEAM_PERFORMANCE_STABILITY) == before
