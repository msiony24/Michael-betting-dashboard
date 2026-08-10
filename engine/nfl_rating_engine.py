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
from engine.nfl_depth_chart import (
    DEFAULT_DEPTH_CHART_PATH,
    depth_chart_team_assignments,
    load_depth_charts,
    match_depth_players,
    team_depth_chart,
    unit_depth_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NFL_DIR = PROJECT_ROOT / "data" / "nfl"
DEFAULT_MADDEN_PATH = PROJECT_ROOT / "data" / "madden_27_players.csv"
DEFAULT_PLAYER_OUTPUT = DEFAULT_NFL_DIR / "player_ratings.csv"
DEFAULT_TEAM_OUTPUT = DEFAULT_NFL_DIR / "team_ratings_auto.json"
DEFAULT_STATUS_OUTPUT = DEFAULT_NFL_DIR / "rating_status.json"
DEFAULT_HISTORY_OUTPUT = DEFAULT_NFL_DIR / "rating_history.jsonl"
DEFAULT_DEPTH_CHART_OUTPUT = DEFAULT_NFL_DIR / "footballguys_depth_charts.csv"

FULL_TO_ABBR = {full: abbr for abbr, full in TEAM_ALIASES.items() if len(abbr) <= 3}
FULL_TO_ABBR.update({"Arizona Cardinals": "ARI", "Washington Commanders": "WAS", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Green Bay Packers": "GB", "New England Patriots": "NE", "New Orleans Saints": "NO", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Las Vegas Raiders": "LV", "Los Angeles Rams": "LA"})

POSITION_GROUPS = {
    "quarterback": {"QB"},
    "running_backs": {"RB", "HB", "FB"},
    "receiving_weapons": {"WR", "TE"},
    "offensive_line": {"LT", "LG", "C", "RG", "RT", "OL", "G", "T"},
    "defensive_front": {"DE", "DT", "DL", "LE", "RE", "EDGE", "LEDG", "REDG"},
    "linebackers": {"LB", "MLB", "ILB", "OLB", "LOLB", "ROLB", "MIKE", "WILL", "SAM"},
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
    # Detailed Madden traits refine the official OVR rather than replacing it.
    # The official OVR remains the preseason talent anchor.
    "QB": {"awareness": .12, "throw_power": .11, "throw_accuracy_short": .14,
           "throw_accuracy_mid": .14, "throw_accuracy_deep": .10,
           "throw_under_pressure": .14, "throw_on_the_run": .10,
           "play_action": .05, "speed": .05, "acceleration": .05},
    "RB": {"speed": .13, "acceleration": .10, "agility": .08, "bc_vision": .15,
           "carrying": .13, "break_tackle": .13, "change_of_direction": .10,
           "catching": .08, "awareness": .10},
    "WR": {"speed": .10, "acceleration": .06, "catching": .14,
           "catch_in_traffic": .10, "spectacular_catch": .08, "release": .10,
           "short_route_running": .12, "medium_route_running": .15,
           "deep_route_running": .15},
    "TE": {"catching": .13, "catch_in_traffic": .11, "short_route_running": .10,
           "medium_route_running": .10, "release": .08, "run_block": .12,
           "impact_blocking": .10, "strength": .10, "awareness": .06,
           "speed": .10},
    "OL": {"awareness": .14, "strength": .12, "pass_block": .20,
           "pass_block_power": .11, "pass_block_finesse": .11,
           "run_block": .16, "run_block_power": .08, "run_block_finesse": .08},
    "DL": {"strength": .10, "block_shedding": .18, "power_moves": .16,
           "finesse_moves": .16, "pursuit": .13, "tackle": .12,
           "play_recognition": .10, "acceleration": .05},
    "LB": {"speed": .08, "acceleration": .06, "awareness": .10,
           "play_recognition": .15, "pursuit": .14, "tackle": .15,
           "block_shedding": .10, "man_coverage": .10, "zone_coverage": .12},
    "DB": {"speed": .10, "acceleration": .07, "agility": .06,
           "awareness": .10, "play_recognition": .11, "man_coverage": .17,
           "zone_coverage": .17, "press": .10, "catching": .06,
           "change_of_direction": .06},
    "ST": {"kick_power": .48, "kick_accuracy": .52},
}

# At preseason, traits may refine OVR but cannot manufacture a giant new talent gap.
TRAIT_BLEND_WEIGHT = 0.30
MAX_TRAIT_DEVIATION = 3.0


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
    "performance_cap": ("macabets_performance_cap", "performance_cap"),
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
    if pos in POSITION_GROUPS["defensive_front"] or pos in {"LEDG", "REDG"}: return "DL"
    if pos in POSITION_GROUPS["linebackers"] or pos in {"MIKE", "WILL", "SAM"}: return "LB"
    if pos in POSITION_GROUPS["secondary"]: return "DB"
    if pos in POSITION_GROUPS["special_teams"]: return "ST"
    return "OTHER"


def _weighted_trait_grade(row: pd.Series) -> float:
    family = _position_family(row.get("position"))
    weights = TRAIT_WEIGHTS.get(family, {})
    overall = pd.to_numeric(row.get("overall"), errors="coerce")
    overall = 60.0 if pd.isna(overall) else float(overall)
    if not weights:
        return round(overall, 2)

    weighted_sum = 0.0
    used_weight = 0.0
    for column, weight in weights.items():
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value):
            continue
        weighted_sum += float(value) * float(weight)
        used_weight += float(weight)

    # Missing traits are not silently replaced with OVR. We simply use the traits
    # that really exist and renormalize them.
    if used_weight <= 0:
        return round(overall, 2)
    trait_composite = weighted_sum / used_weight
    refined = overall * (1.0 - TRAIT_BLEND_WEIGHT) + trait_composite * TRAIT_BLEND_WEIGHT
    refined = max(overall - MAX_TRAIT_DEVIATION, min(overall + MAX_TRAIT_DEVIATION, refined))
    return round(max(0.0, min(99.0, refined)), 2)


def _find_col(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup = {str(c).casefold(): c for c in frame.columns}
    return next((lookup[a.casefold()] for a in aliases if a.casefold() in lookup), None)


def _aggregate_weekly_stats(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    mapping = {key: _find_col(frame, aliases) for key, aliases in STAT_ALIASES.items()}
    if not mapping["player_name"]: return pd.DataFrame()
    clean = pd.DataFrame({key: frame[col] if col else 0 for key, col in mapping.items()})
    # Current-season files without an explicit cap may use the normal 80% ceiling.
    # Preseason fallback files explicitly carry macabets_performance_cap=0.20.
    if mapping.get("performance_cap") is None:
        clean["performance_cap"] = 0.80
    clean["name_key"] = clean["player_name"].map(_name_key)
    clean["team"] = clean["team"].astype(str).str.upper()
    clean["position"] = clean["position"].astype(str).str.upper()
    numeric = [c for c in clean.columns if c not in {"player_name", "name_key", "team", "position"}]
    clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
    agg = {c: "sum" for c in numeric}
    if "performance_cap" in agg:
        agg["performance_cap"] = "max"
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
    depth_chart_path: Path | str | None = None,
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
        merge_cols = ["name_key", "performance_grade", "sample_size"]
        if "performance_cap" in stats.columns:
            merge_cols.append("performance_cap")
        players = players.merge(stats[merge_cols], on="name_key", how="left")
    else:
        players["performance_grade"] = np.nan
        players["sample_size"] = 0.0
        players["performance_cap"] = 0.0

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
    if "performance_cap" in players.columns:
        performance_cap = pd.to_numeric(players["performance_cap"], errors="coerce").fillna(0.80).clip(0.0, 0.80)
    else:
        performance_cap = pd.Series(0.80, index=players.index)
    players["performance_weight"] = players["performance_confidence"] * performance_cap
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

    # The depth chart is authoritative for current team assignment. This matters for
    # offseason trades/free-agent moves that Madden launch rosters or nflverse files
    # may not yet reflect. Madden still supplies the talent grade; it does not decide
    # which team the player currently belongs to.
    chart_path = Path(depth_chart_path) if depth_chart_path is not None else root / "footballguys_depth_charts.csv"
    depth_charts = load_depth_charts(chart_path)
    assignments = depth_chart_team_assignments(depth_charts)
    assigned = players["name_key"].map(assignments)
    players["depth_chart_team_override"] = assigned.notna() & assigned.ne(players["team_abbr"])
    players["team_abbr"] = assigned.where(assigned.notna(), players["team_abbr"])

    keep = ["player_name", "team_abbr", "position", "position_family", "overall", "trait_grade",
            "performance_grade", "performance_weight", "availability_adjustment", "macabets_rating",
            "rating_confidence", "rating_source", "roster_status", "injury_status", "gsis_id",
            "depth_chart_team_override"]
    return players[keep].sort_values(["team_abbr", "macabets_rating"], ascending=[True, False]).reset_index(drop=True)


def _legacy_unit_selection(team_players: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rating-order fallback used only when authoritative depth data is unavailable."""
    allowed = POSITION_GROUPS[unit]
    group = team_players[team_players["position"].isin(allowed)].sort_values("macabets_rating", ascending=False).head(DEPTH_LIMITS[unit])
    starter_n = min(STARTER_COUNTS[unit], len(group))
    return group.head(starter_n).copy(), group.iloc[starter_n:].copy()


def _unit_grade(team_players: pd.DataFrame, unit: str, team_depth: pd.DataFrame | None = None) -> dict[str, Any]:
    plan = unit_depth_plan(team_depth, unit) if team_depth is not None else {"starters": [], "depth": [], "scheme": "unknown", "source": "missing"}
    starters, unmatched_starters = match_depth_players(team_players, plan.get("starters", []))
    depth, unmatched_depth = match_depth_players(team_players, plan.get("depth", []))

    selection_source = "Footballguys depth chart"
    expected_starters = len(plan.get("starters", []))
    # Fall back only when a unit has no usable depth-chart starters. Partial matches
    # remain visible as limited confidence instead of silently promoting a higher-OVR backup.
    if starters.empty:
        starters, depth = _legacy_unit_selection(team_players, unit)
        selection_source = "rating-order fallback"
        expected_starters = len(starters)
        unmatched_starters = []
        unmatched_depth = []

    if starters.empty:
        return {
            "grade": 50.0, "starter_grade": 50.0, "depth_grade": 50.0, "player_count": 0,
            "confidence": "missing", "top_players": [], "selection_source": selection_source,
            "scheme": plan.get("scheme", "unknown"), "unmatched_depth_chart": [],
        }

    starter_grade = float(starters["macabets_rating"].mean())
    depth_grade = float(depth["macabets_rating"].mean()) if not depth.empty else starter_grade

    # Healthy starters define the current unit. Backup quality is retained as depth
    # information without materially diluting a healthy starting lineup. RB/QB/ST are
    # completely starter-driven until availability logic activates a backup.
    depth_blend = {
        "quarterback": 0.00, "running_backs": 0.00, "receiving_weapons": 0.05,
        "offensive_line": 0.05, "defensive_front": 0.05, "linebackers": 0.05,
        "secondary": 0.05, "special_teams": 0.00,
    }.get(unit, 0.05)
    grade = starter_grade * (1.0 - depth_blend) + depth_grade * depth_blend

    matched_starter_count = len(starters)
    confidence = "high" if selection_source == "Footballguys depth chart" and matched_starter_count >= max(1, expected_starters) else "limited"
    top = pd.concat([starters.assign(depth_order=0), depth.assign(depth_order=1)], ignore_index=True)
    top_players = []
    for _, r in top.head(8).iterrows():
        top_players.append({
            "name": r["player_name"], "position": r["position"], "rating": float(r["macabets_rating"]),
            "role": str(r.get("depth_chart_role", "") or ""), "starter": bool(int(r.get("depth_order", 1)) == 0),
        })

    return {
        "grade": round(grade, 2), "starter_grade": round(starter_grade, 2), "depth_grade": round(depth_grade, 2),
        "player_count": int(len(starters) + len(depth)), "starter_count": int(len(starters)),
        "depth_count": int(len(depth)), "confidence": confidence, "top_players": top_players,
        "selection_source": selection_source, "depth_chart_source": plan.get("source", ""),
        "scheme": plan.get("scheme", "unknown"),
        "unmatched_depth_chart": unmatched_starters + unmatched_depth,
    }


def build_team_ratings(
    player_ratings: pd.DataFrame,
    snapshot_path: Path | str = DEFAULT_NFL_DIR / "team_snapshot.csv",
    depth_chart_path: Path | str = DEFAULT_DEPTH_CHART_PATH,
) -> dict[str, dict[str, Any]]:
    snapshot = pd.read_csv(snapshot_path) if Path(snapshot_path).exists() else pd.DataFrame()
    depth_charts = load_depth_charts(depth_chart_path)
    snap_by_abbr = {str(r["team_abbr"]): r for _, r in snapshot.iterrows()} if "team_abbr" in snapshot else {}
    result = {}
    for abbr, team_players in player_ratings.groupby("team_abbr"):
        current_depth = team_depth_chart(depth_charts, str(abbr))
        units = {name: _unit_grade(team_players, name, current_depth) for name in POSITION_GROUPS}
        row = snap_by_abbr.get(str(abbr))
        # Team-unit performance gradually replaces the Madden roster prior as the
        # current season accumulates. Previous-season snapshots are capped at 20%.
        perf_weight = 0.0
        if row is not None:
            row_season = int(pd.to_numeric(row.get("season"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("season"), errors="coerce")) else 0
            through_week = int(pd.to_numeric(row.get("through_week"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("through_week"), errors="coerce")) else 0
            current_year = datetime.now(timezone.utc).year
            if row_season == current_year:
                perf_weight = min(0.80, 0.20 + max(0, through_week) * 0.075)
            elif row_season > 0:
                perf_weight = 0.20

        live_map = {
            # QB/RB/WR/TE already receive player-level weekly performance in build_player_ratings.
            # Do not blend the same prior-season evidence into those units a second time.
            "offensive_line": "offensive_line",
            "defensive_front": "defensive_line",
            "secondary": "secondary",
            "special_teams": "special_teams",
        }
        for unit, col in live_map.items():
            if row is not None and col in row and pd.notna(row[col]) and perf_weight > 0:
                roster_grade = units[unit]["grade"]
                units[unit]["roster_grade"] = roster_grade
                units[unit]["performance_grade"] = round(float(row[col]), 2)
                units[unit]["grade"] = round(roster_grade * (1 - perf_weight) + float(row[col]) * perf_weight, 2)
                units[unit]["performance_weight"] = round(perf_weight, 3)
                units[unit]["source"] = f"{(1-perf_weight):.0%} roster + {perf_weight:.0%} NFL performance"
            else:
                units[unit]["source"] = "Madden 27 roster rating"
        # Linebacker play currently lacks a clean player-level weekly metric in this pipeline.
        # Use only a modest team-defense proxy; never apply generic offense to RB/WR/TE.
        indirect_weight = min(perf_weight * 0.50, 0.30)
        if row is not None and "defense" in row and pd.notna(row["defense"]) and indirect_weight > 0:
            units["linebackers"]["roster_grade"] = units["linebackers"]["grade"]
            units["linebackers"]["performance_grade"] = round(float(row["defense"]), 2)
            units["linebackers"]["grade"] = round(units["linebackers"]["grade"] * (1-indirect_weight) + float(row["defense"]) * indirect_weight, 2)
            units["linebackers"]["performance_weight"] = round(indirect_weight, 3)
            units["linebackers"]["source"] = f"{(1-indirect_weight):.0%} roster + {indirect_weight:.0%} NFL team-defense proxy"

        overall = sum(units[u]["grade"] * w for u, w in TEAM_WEIGHTS.items())
        offense = units["quarterback"]["grade"] * .35 + units["running_backs"]["grade"] * .12 + units["receiving_weapons"]["grade"] * .25 + units["offensive_line"]["grade"] * .28
        defense = units["defensive_front"]["grade"] * .36 + units["linebackers"]["grade"] * .24 + units["secondary"]["grade"] * .40
        full_name = TEAM_ALIASES.get(str(abbr), str(abbr))
        result[full_name] = {
            "team_abbr": str(abbr), "overall_rating": round(overall, 2), "offense_rating": round(offense, 2),
            "defense_rating": round(defense, 2), "player_count": int(len(team_players)), "units": units,
            "source": "Macabets automated rating engine v1.2 - Footballguys depth chart + audited Madden 27 baseline", "prediction_influence_enabled": False,
            "personnel_matchup_influence_enabled": True,
            "depth_chart_source": "Footballguys" if not current_depth.empty else "rating-order fallback",
            "depth_chart_rows": int(len(current_depth)),
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
        "schema_version": "1.2", "engine_version": "1.2-depth-chart-first", "updated_at_utc": updated,
        "players_rated": int(len(player_ratings)), "teams_rated": int(len(team_ratings)),
        "players_with_performance_data": int((player_ratings["performance_weight"] > 0).sum()),
        "prediction_influence_enabled": False,
        "depth_chart_source": "Footballguys",
        "depth_chart_file": str(Path(nfl_dir) / "footballguys_depth_charts.csv"),
        "files": {"players": str(player_path), "teams": str(team_path)},
    }
    _write_json(root / "rating_status.json", status)
    history_path = root / "rating_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"updated_at_utc": updated, "teams": {k: v["overall_rating"] for k, v in team_ratings.items()}}, sort_keys=True) + "\n")
    return status


def build_and_save_ratings(*, madden_path: Path | str = DEFAULT_MADDEN_PATH, nfl_dir: Path | str = DEFAULT_NFL_DIR) -> dict[str, Any]:
    players = build_player_ratings(
        madden_path=madden_path,
        nfl_dir=nfl_dir,
        depth_chart_path=Path(nfl_dir) / "footballguys_depth_charts.csv",
    )
    teams = build_team_ratings(
        players,
        snapshot_path=Path(nfl_dir) / "team_snapshot.csv",
        depth_chart_path=Path(nfl_dir) / "footballguys_depth_charts.csv",
    )
    return save_rating_outputs(players, teams, nfl_dir=nfl_dir)
