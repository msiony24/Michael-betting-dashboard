"""Automated Macabets NFL player, unit, and team ratings.

The engine keeps the public app simple while maintaining richer internal grades.
Madden supplies the roster/trait baseline. nflverse supplies current rosters,
weekly production, and team performance. Missing optional datasets never invent
performance; they simply lower confidence and preserve the Madden baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd

from engine.madden_team_builder import TEAM_ALIASES, load_madden_players

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NFL_DIR = PROJECT_ROOT / "data" / "nfl"
DEFAULT_MADDEN_PATH = PROJECT_ROOT / "data" / "madden_27_players.csv"
DEFAULT_PLAYER_OUTPUT = DEFAULT_NFL_DIR / "player_ratings.csv"
DEFAULT_TEAM_OUTPUT = DEFAULT_NFL_DIR / "team_ratings_auto.json"
DEFAULT_STATUS_OUTPUT = DEFAULT_NFL_DIR / "rating_status.json"
DEFAULT_HISTORY_OUTPUT = DEFAULT_NFL_DIR / "rating_history.jsonl"

FULL_TO_ABBR = {full: abbr for abbr, full in TEAM_ALIASES.items() if len(abbr) <= 3}
FULL_TO_ABBR.update({"Arizona Cardinals": "ARI", "Washington Commanders": "WAS", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Green Bay Packers": "GB", "New England Patriots": "NE", "New Orleans Saints": "NO", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Las Vegas Raiders": "LV", "Los Angeles Rams": "LA"})

POSITION_GROUPS = {
    "quarterback": {"QB"},
    "running_backs": {"RB", "HB", "FB"},
    "receiving_weapons": {"WR", "TE"},
    "offensive_line": {"LT", "LG", "C", "RG", "RT", "OL", "G", "T"},
    "defensive_front": {"DE", "DT", "DL", "LE", "RE", "EDGE"},
    "linebackers": {"LB", "MLB", "ILB", "OLB", "LOLB", "ROLB"},
    "secondary": {"CB", "DB", "FS", "SS", "S"},
    "special_teams": {"K", "P", "LS"},
}

DEPTH_LIMITS = {
    "quarterback": 2, "running_backs": 4, "receiving_weapons": 7,
    "offensive_line": 8, "defensive_front": 8, "linebackers": 6,
    "secondary": 8, "special_teams": 3,
}
STARTER_COUNTS = {
    "quarterback": 1, "running_backs": 2, "receiving_weapons": 4,
    "offensive_line": 5, "defensive_front": 4, "linebackers": 3,
    "secondary": 5, "special_teams": 2,
}
TEAM_WEIGHTS = {
    "quarterback": 0.22, "running_backs": 0.07, "receiving_weapons": 0.14,
    "offensive_line": 0.16, "defensive_front": 0.14,
    "linebackers": 0.09, "secondary": 0.14, "special_teams": 0.04,
}

TRAIT_WEIGHTS = {
    "QB": {"overall": .38, "stats_awareness_value": .10, "stats_throwPower_value": .08,
           "stats_throwAccuracyShort_value": .10, "stats_throwAccuracyMid_value": .10,
           "stats_throwAccuracyDeep_value": .08, "stats_throwUnderPressure_value": .10,
           "stats_playAction_value": .03, "stats_throwOnTheRun_value": .03},
    "RB": {"overall": .44, "stats_speed_value": .10, "stats_acceleration_value": .08,
           "stats_agility_value": .06, "stats_bCVision_value": .10,
           "stats_carrying_value": .08, "stats_breakTackle_value": .08,
           "stats_changeOfDirection_value": .06},
    "WR": {"overall": .42, "stats_speed_value": .08, "stats_catching_value": .10,
           "stats_catchInTraffic_value": .07, "stats_release_value": .08,
           "stats_shortRouteRunning_value": .08, "stats_mediumRouteRunning_value": .09,
           "stats_deepRouteRunning_value": .08},
    "TE": {"overall": .45, "stats_catching_value": .10, "stats_catchInTraffic_value": .08,
           "stats_shortRouteRunning_value": .07, "stats_mediumRouteRunning_value": .07,
           "stats_runBlock_value": .08, "stats_impactBlocking_value": .08,
           "stats_strength_value": .07},
    "OL": {"overall": .38, "stats_awareness_value": .10, "stats_strength_value": .10,
           "stats_passBlock_value": .14, "stats_passBlockPower_value": .08,
           "stats_passBlockFinesse_value": .08, "stats_runBlock_value": .12},
    "DL": {"overall": .42, "stats_strength_value": .08, "stats_blockShedding_value": .13,
           "stats_powerMoves_value": .10, "stats_finesseMoves_value": .10,
           "stats_pursuit_value": .09, "stats_tackle_value": .08},
    "LB": {"overall": .43, "stats_speed_value": .08, "stats_awareness_value": .08,
           "stats_playRecognition_value": .11, "stats_pursuit_value": .10,
           "stats_tackle_value": .12, "stats_blockShedding_value": .08},
    "DB": {"overall": .42, "stats_speed_value": .08, "stats_agility_value": .05,
           "stats_awareness_value": .08, "stats_playRecognition_value": .08,
           "stats_manCoverage_value": .11, "stats_zoneCoverage_value": .11,
           "stats_press_value": .07},
    "ST": {"overall": .55, "stats_kickPower_value": .23, "stats_kickAccuracy_value": .22},
}

STAT_ALIASES = {
    "player_name": ("player_display_name", "player_name", "full_name", "name"),
    "team": ("recent_team", "team", "team_abbr"),
    "position": ("position", "position_group"),
    "attempts": ("attempts", "passing_attempts"),
    "passing_yards": ("passing_yards",), "passing_tds": ("passing_tds",),
    "interceptions": ("interceptions", "passing_interceptions"),
    "sacks": ("sacks", "sack_fumbles"), "carries": ("carries", "rushing_attempts"),
    "rushing_yards": ("rushing_yards",), "rushing_tds": ("rushing_tds",),
    "targets": ("targets",), "receptions": ("receptions",),
    "receiving_yards": ("receiving_yards",), "receiving_tds": ("receiving_tds",),
}


def _name_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text.lower())
    return re.sub(r"[^a-z0-9]", "", text)


def _position_family(position: Any) -> str:
    pos = str(position or "").upper().strip()
    if pos in {"QB"}: return "QB"
    if pos in {"RB", "HB", "FB"}: return "RB"
    if pos == "WR": return "WR"
    if pos == "TE": return "TE"
    if pos in POSITION_GROUPS["offensive_line"]: return "OL"
    if pos in POSITION_GROUPS["defensive_front"]: return "DL"
    if pos in POSITION_GROUPS["linebackers"]: return "LB"
    if pos in POSITION_GROUPS["secondary"]: return "DB"
    if pos in POSITION_GROUPS["special_teams"]: return "ST"
    return "OTHER"


def _weighted_trait_grade(row: pd.Series) -> float:
    family = _position_family(row.get("position"))
    weights = TRAIT_WEIGHTS.get(family, {"overall": 1.0})
    values, used = 0.0, 0.0
    baseline = pd.to_numeric(row.get("overall"), errors="coerce")
    baseline = 60.0 if pd.isna(baseline) else float(baseline)
    for col, weight in weights.items():
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(value):
            value = baseline
        values += float(value) * weight
        used += weight
    return round(max(0.0, min(99.0, values / used if used else baseline)), 2)


def _find_col(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup = {str(c).casefold(): c for c in frame.columns}
    return next((lookup[a.casefold()] for a in aliases if a.casefold() in lookup), None)


def _aggregate_weekly_stats(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    mapping = {key: _find_col(frame, aliases) for key, aliases in STAT_ALIASES.items()}
    if not mapping["player_name"]: return pd.DataFrame()
    clean = pd.DataFrame({key: frame[col] if col else 0 for key, col in mapping.items()})
    clean["name_key"] = clean["player_name"].map(_name_key)
    clean["team"] = clean["team"].astype(str).str.upper()
    clean["position"] = clean["position"].astype(str).str.upper()
    numeric = [c for c in clean.columns if c not in {"player_name", "name_key", "team", "position"}]
    clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
    agg = {c: "sum" for c in numeric}
    agg.update({"player_name": "last", "team": "last", "position": "last"})
    return clean.groupby("name_key", as_index=False).agg(agg)


def _percentile_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(numeric) <= 1 or numeric.nunique() <= 1:
        return pd.Series(67.5, index=series.index)
    return 45.0 + numeric.rank(pct=True) * 50.0


def _performance_grades(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty: return stats
    out = stats.copy()
    out["family"] = out["position"].map(_position_family)
    out["performance_grade"] = np.nan
    out["sample_size"] = 0.0

    formulas = {
        "QB": ({"passing_yards": .26, "passing_tds": .22, "attempts": .12,
                "interceptions": -.18, "rushing_yards": .10, "rushing_tds": .08,
                "sacks": -.04}, "attempts", 500),
        "RB": ({"rushing_yards": .34, "rushing_tds": .20, "carries": .12,
                "receiving_yards": .16, "receptions": .08, "receiving_tds": .10}, "carries", 250),
        "WR": ({"receiving_yards": .38, "receiving_tds": .22, "targets": .18,
                "receptions": .14, "rushing_yards": .08}, "targets", 150),
        "TE": ({"receiving_yards": .34, "receiving_tds": .24, "targets": .18,
                "receptions": .16, "rushing_yards": .08}, "targets", 120),
    }
    for family, (weights, sample_col, _) in formulas.items():
        mask = out["family"].eq(family)
        if not mask.any(): continue
        score = pd.Series(0.0, index=out.index)
        total_weight = 0.0
        for col, weight in weights.items():
            pct = _percentile_score(out.loc[mask, col])
            if weight < 0:
                pct = 140.0 - pct
            score.loc[mask] += pct * abs(weight)
            total_weight += abs(weight)
        out.loc[mask, "performance_grade"] = score.loc[mask] / total_weight
        out.loc[mask, "sample_size"] = pd.to_numeric(out.loc[mask, sample_col], errors="coerce").fillna(0)
    return out


def _load_roster_status(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    name_col = _find_col(frame, ("full_name", "player_name", "name"))
    if not name_col: return pd.DataFrame()
    result = pd.DataFrame({
        "name_key": frame[name_col].map(_name_key),
        "roster_status": frame[_find_col(frame, ("status",))] if _find_col(frame, ("status",)) else "",
        "gsis_id": frame[_find_col(frame, ("gsis_id", "player_id"))] if _find_col(frame, ("gsis_id", "player_id")) else "",
    })
    return result.drop_duplicates("name_key", keep="last")


def _load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    name_col = _find_col(frame, ("full_name", "player_name", "name"))
    if not name_col: return pd.DataFrame()
    status_col = _find_col(frame, ("report_status", "game_status", "practice_status", "status"))
    result = pd.DataFrame({"name_key": frame[name_col].map(_name_key), "injury_status": frame[status_col] if status_col else ""})
    return result.drop_duplicates("name_key", keep="last")


def build_player_ratings(
    madden_path: Path | str = DEFAULT_MADDEN_PATH,
    nfl_dir: Path | str = DEFAULT_NFL_DIR,
) -> pd.DataFrame:
    players = load_madden_players(madden_path).copy()
    players["name_key"] = players["player_name"].map(_name_key)
    players["team_abbr"] = players["team"].map(lambda x: FULL_TO_ABBR.get(str(x), str(x).upper()))
    valid_teams = set(FULL_TO_ABBR.values())
    players = players[players["team_abbr"].isin(valid_teams)].copy()
    players["position_family"] = players["position"].map(_position_family)
    players["trait_grade"] = players.apply(_weighted_trait_grade, axis=1)

    root = Path(nfl_dir)
    stats = _performance_grades(_aggregate_weekly_stats(root / "player_weekly_stats.csv"))
    if not stats.empty:
        players = players.merge(stats[["name_key", "performance_grade", "sample_size"]], on="name_key", how="left")
    else:
        players["performance_grade"] = np.nan
        players["sample_size"] = 0.0

    roster = _load_roster_status(root / "weekly_rosters.csv")
    if roster.empty:
        roster = _load_roster_status(root / "rosters.csv")

    # Madden 27's final player file is now pre-enriched from nflverse rosters and may
    # already contain roster_status / gsis_id. A second roster merge would create
    # roster_status_x / roster_status_y and gsis_id_x / gsis_id_y, which breaks the
    # output column selection below. Merge defensively and coalesce the freshest
    # roster values instead.
    for column in ("roster_status", "gsis_id"):
        if column not in players.columns:
            players[column] = ""

    if not roster.empty:
        roster_merge = roster.copy()
        rename_map = {}
        if "roster_status" in roster_merge.columns:
            rename_map["roster_status"] = "_fresh_roster_status"
        if "gsis_id" in roster_merge.columns:
            rename_map["gsis_id"] = "_fresh_gsis_id"
        roster_merge = roster_merge.rename(columns=rename_map)

        keep_cols = ["name_key"] + [
            column for column in ("_fresh_roster_status", "_fresh_gsis_id")
            if column in roster_merge.columns
        ]
        roster_merge = roster_merge[keep_cols].drop_duplicates("name_key", keep="first")
        players = players.merge(roster_merge, on="name_key", how="left")

        if "_fresh_roster_status" in players.columns:
            fresh = players["_fresh_roster_status"].fillna("").astype(str).str.strip()
            current = players["roster_status"].fillna("").astype(str).str.strip()
            players["roster_status"] = fresh.where(fresh.ne(""), current)
            players = players.drop(columns=["_fresh_roster_status"])

        if "_fresh_gsis_id" in players.columns:
            fresh = players["_fresh_gsis_id"].fillna("").astype(str).str.strip()
            current = players["gsis_id"].fillna("").astype(str).str.strip()
            players["gsis_id"] = fresh.where(fresh.ne(""), current)
            players = players.drop(columns=["_fresh_gsis_id"])

    injuries = _load_injuries(root / "injuries.csv")
    if not injuries.empty: players = players.merge(injuries, on="name_key", how="left")
    else: players["injury_status"] = ""

    thresholds = {"QB": 500, "RB": 250, "WR": 150, "TE": 120}
    players["performance_confidence"] = players.apply(
        lambda r: min(1.0, float(r.get("sample_size", 0) or 0) / thresholds.get(r["position_family"], 999999)), axis=1
    )
    players["performance_weight"] = players["performance_confidence"] * 0.55
    has_perf = players["performance_grade"].notna()
    players.loc[~has_perf, "performance_weight"] = 0.0
    players["base_rating"] = (
        players["trait_grade"] * (1 - players["performance_weight"])
        + players["performance_grade"].fillna(players["trait_grade"]) * players["performance_weight"]
    )

    injury_text = players["injury_status"].fillna("").astype(str).str.lower()
    adjustment = np.select(
        [injury_text.str.contains("out|reserve|ir"), injury_text.str.contains("doubt"), injury_text.str.contains("question")],
        [-8.0, -4.0, -1.5], default=0.0,
    )
    players["availability_adjustment"] = adjustment
    players["macabets_rating"] = (players["base_rating"] + players["availability_adjustment"]).clip(0, 99).round(2)
    players["rating_confidence"] = np.where(players["performance_weight"] > .35, "high", np.where(players["performance_weight"] > .10, "medium", "baseline"))
    players["rating_source"] = np.where(players["performance_weight"] > 0, "Madden 27 + nflverse performance", "Madden 27 baseline")

    keep = ["player_name", "team_abbr", "position", "position_family", "overall", "trait_grade",
            "performance_grade", "performance_weight", "availability_adjustment", "macabets_rating",
            "rating_confidence", "rating_source", "roster_status", "injury_status", "gsis_id"]
    return players[keep].sort_values(["team_abbr", "macabets_rating"], ascending=[True, False]).reset_index(drop=True)


def _unit_grade(team_players: pd.DataFrame, unit: str) -> dict[str, Any]:
    allowed = POSITION_GROUPS[unit]
    group = team_players[team_players["position"].isin(allowed)].sort_values("macabets_rating", ascending=False).head(DEPTH_LIMITS[unit])
    if group.empty:
        return {"grade": 50.0, "starter_grade": 50.0, "depth_grade": 50.0, "player_count": 0, "confidence": "missing", "top_players": []}
    starter_n = min(STARTER_COUNTS[unit], len(group))
    starters, depth = group.head(starter_n), group.iloc[starter_n:]
    starter_grade = float(starters["macabets_rating"].mean())
    depth_grade = float(depth["macabets_rating"].mean()) if not depth.empty else starter_grade
    grade = starter_grade * (.90 if unit == "quarterback" else .84) + depth_grade * (.10 if unit == "quarterback" else .16)
    return {
        "grade": round(grade, 2), "starter_grade": round(starter_grade, 2), "depth_grade": round(depth_grade, 2),
        "player_count": int(len(group)), "confidence": "high" if len(group) >= starter_n else "limited",
        "top_players": [{"name": r.player_name, "position": r.position, "rating": float(r.macabets_rating)} for r in group.head(5).itertuples()],
    }


def build_team_ratings(player_ratings: pd.DataFrame, snapshot_path: Path | str = DEFAULT_NFL_DIR / "team_snapshot.csv") -> dict[str, dict[str, Any]]:
    snapshot = pd.read_csv(snapshot_path) if Path(snapshot_path).exists() else pd.DataFrame()
    snap_by_abbr = {str(r["team_abbr"]): r for _, r in snapshot.iterrows()} if "team_abbr" in snapshot else {}
    result = {}
    for abbr, team_players in player_ratings.groupby("team_abbr"):
        units = {name: _unit_grade(team_players, name) for name in POSITION_GROUPS}
        row = snap_by_abbr.get(str(abbr))
        live_map = {
            "quarterback": "quarterback", "offensive_line": "offensive_line",
            "defensive_front": "defensive_line", "secondary": "secondary",
            "special_teams": "special_teams",
        }
        for unit, col in live_map.items():
            if row is not None and col in row and pd.notna(row[col]):
                roster_grade = units[unit]["grade"]
                units[unit]["roster_grade"] = roster_grade
                units[unit]["performance_grade"] = round(float(row[col]), 2)
                units[unit]["grade"] = round(roster_grade * .45 + float(row[col]) * .55, 2)
                units[unit]["source"] = "45% roster + 55% live performance"
            else:
                units[unit]["source"] = "roster rating"
        if row is not None and "offense" in row and pd.notna(row["offense"]):
            # Skill groups receive a modest team-efficiency adjustment, while preserving player quality.
            for unit in ("running_backs", "receiving_weapons"):
                units[unit]["roster_grade"] = units[unit]["grade"]
                units[unit]["performance_grade"] = round(float(row["offense"]), 2)
                units[unit]["grade"] = round(units[unit]["grade"] * .70 + float(row["offense"]) * .30, 2)
                units[unit]["source"] = "70% roster + 30% team performance"
        if row is not None and "defense" in row and pd.notna(row["defense"]):
            units["linebackers"]["roster_grade"] = units["linebackers"]["grade"]
            units["linebackers"]["performance_grade"] = round(float(row["defense"]), 2)
            units["linebackers"]["grade"] = round(units["linebackers"]["grade"] * .70 + float(row["defense"]) * .30, 2)
            units["linebackers"]["source"] = "70% roster + 30% team performance"

        overall = sum(units[u]["grade"] * w for u, w in TEAM_WEIGHTS.items())
        offense = units["quarterback"]["grade"] * .35 + units["running_backs"]["grade"] * .12 + units["receiving_weapons"]["grade"] * .25 + units["offensive_line"]["grade"] * .28
        defense = units["defensive_front"]["grade"] * .36 + units["linebackers"]["grade"] * .24 + units["secondary"]["grade"] * .40
        full_name = TEAM_ALIASES.get(str(abbr), str(abbr))
        result[full_name] = {
            "team_abbr": str(abbr), "overall_rating": round(overall, 2), "offense_rating": round(offense, 2),
            "defense_rating": round(defense, 2), "player_count": int(len(team_players)), "units": units,
            "source": "Macabets automated rating engine v1", "prediction_influence_enabled": False,
        }
    return dict(sorted(result.items()))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def save_rating_outputs(player_ratings: pd.DataFrame, team_ratings: dict[str, Any], *, nfl_dir: Path | str = DEFAULT_NFL_DIR) -> dict[str, Any]:
    root = Path(nfl_dir); root.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    player_path, team_path = root / "player_ratings.csv", root / "team_ratings_auto.json"
    temp = player_path.with_suffix(".csv.tmp"); player_ratings.to_csv(temp, index=False); temp.replace(player_path)
    _write_json(team_path, team_ratings)
    status = {
        "schema_version": "1.0", "engine_version": "1.0", "updated_at_utc": updated,
        "players_rated": int(len(player_ratings)), "teams_rated": int(len(team_ratings)),
        "players_with_performance_data": int((player_ratings["performance_weight"] > 0).sum()),
        "prediction_influence_enabled": False,
        "files": {"players": str(player_path), "teams": str(team_path)},
    }
    _write_json(root / "rating_status.json", status)
    history_path = root / "rating_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"updated_at_utc": updated, "teams": {k: v["overall_rating"] for k, v in team_ratings.items()}}, sort_keys=True) + "\n")
    return status


def build_and_save_ratings(*, madden_path: Path | str = DEFAULT_MADDEN_PATH, nfl_dir: Path | str = DEFAULT_NFL_DIR) -> dict[str, Any]:
    players = build_player_ratings(madden_path=madden_path, nfl_dir=nfl_dir)
    teams = build_team_ratings(players, snapshot_path=Path(nfl_dir) / "team_snapshot.csv")
    return save_rating_outputs(players, teams, nfl_dir=nfl_dir)
