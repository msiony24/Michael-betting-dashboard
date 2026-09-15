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
from engine.nfl_availability import (
    DEFAULT_AVAILABILITY_PATH,
    load_availability,
    load_availability_status,
)
from engine.nfl_qb_intelligence import apply_qb_replacement_adjustment

from engine.nfl_depth_chart import (
    AUTO_DEPTH_CHART_PATH,
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
    "quarterback": 1, "running_backs": 1, "receiving_weapons": 4,
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
    "player_id": ("player_id", "gsis_id"),
    "player_name": ("player_display_name", "player_name", "full_name", "name"),
    "team": ("recent_team", "team", "team_abbr"),
    "position": ("position", "position_group"),
    "attempts": ("attempts", "passing_attempts"),
    "passing_yards": ("passing_yards",), "passing_tds": ("passing_tds",),
    "interceptions": ("interceptions", "passing_interceptions"),
    # "sack_fumbles" is fumbles on sacks, not sacks, so it is not a valid alias.
    "sacks": ("sacks_suffered", "sacks"), "carries": ("carries", "rushing_attempts"),
    "rushing_yards": ("rushing_yards",), "rushing_tds": ("rushing_tds",),
    "targets": ("targets",), "receptions": ("receptions",),
    "receiving_yards": ("receiving_yards",), "receiving_tds": ("receiving_tds",),
    "passing_epa": ("passing_epa",), "rushing_epa": ("rushing_epa",),
    "receiving_epa": ("receiving_epa",),
    "performance_cap": ("macabets_performance_cap", "performance_cap"),
}

# Player performance model (v1.5)
# -------------------------------
# Performance is graded on per-opportunity EFFICIENCY (EPA per play, plus a
# yards/TD/turnover efficiency measure), never on season totals, so missing a
# game or playing on a low-volume offense is not punished as "bad play".
# Efficiency is converted to a z-score within the position and mapped onto the
# Madden starter scale for that position, so an average season lands at an
# average-starter rating instead of being dragged into the 60s.
# Its weight grows with opportunities as n / (n + K) and is capped by the
# season cap (0.80 current season, 0.20 prior-season fallback). K is the number
# of opportunities at which performance and Madden are trusted equally (before
# the cap), so one game can only nudge a rating.
PERFORMANCE_STABILITY = {"QB": 500.0, "RB": 200.0, "WR": 100.0, "TE": 80.0}
MADDEN_SCALE_POOL = {"QB": 32, "RB": 32, "WR": 96, "TE": 32}
PERFORMANCE_Z_CLIP = 3.0
MISSING_EPA_COLUMNS = ("passing_epa", "rushing_epa", "receiving_epa")


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

    data = {}
    for key, col in mapping.items():
        if col:
            data[key] = frame[col]
        elif key == "player_id":
            data[key] = ""
        elif key in MISSING_EPA_COLUMNS:
            data[key] = np.nan
        else:
            data[key] = 0
    clean = pd.DataFrame(data)

    # nflverse player_id is the stable GSIS identity and must remain a string.
    # Name-only aggregation can merge different NFL players who share a name
    # (for example the two Byron Youngs or the two Justin Jeffersons).
    clean["player_id"] = clean["player_id"].fillna("").astype(str).str.strip()
    clean.loc[clean["player_id"].str.lower().isin({"", "nan", "none", "0", "0.0"}), "player_id"] = ""

    # Current-season files without an explicit cap may use the normal 80% ceiling.
    # Preseason fallback files explicitly carry macabets_performance_cap=0.20.
    if mapping.get("performance_cap") is None:
        clean["performance_cap"] = 0.80
    clean["name_key"] = clean["player_name"].map(_name_key)
    clean["team"] = clean["team"].astype(str).str.upper()
    clean["position"] = clean["position"].astype(str).str.upper()
    numeric = [c for c in clean.columns if c not in {"player_id", "player_name", "name_key", "team", "position"}]
    clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce")
    # Remember whether real EPA exists; a filled 0 must not look like "average EPA".
    clean["epa_rows"] = clean[list(MISSING_EPA_COLUMNS)].notna().any(axis=1).astype(float)
    clean[numeric] = clean[numeric].fillna(0)
    numeric.append("epa_rows")
    agg = {c: "sum" for c in numeric}
    if "performance_cap" in agg:
        agg["performance_cap"] = "max"
    agg.update({"player_name": "last", "name_key": "last", "team": "last", "position": "last"})

    with_id = clean[clean["player_id"].ne("")].groupby("player_id", as_index=False).agg(agg)
    without_id = clean[clean["player_id"].eq("")].copy()
    if not without_id.empty:
        fallback_agg = dict(agg)
        fallback_agg.pop("name_key", None)
        without_id = without_id.groupby("name_key", as_index=False).agg(fallback_agg)
        without_id.insert(0, "player_id", "")

    if with_id.empty:
        return without_id
    if without_id.empty:
        return with_id
    return pd.concat([with_id, without_id], ignore_index=True, sort=False)


def _opportunities(frame: pd.DataFrame, family: str) -> pd.Series:
    col = lambda name: pd.to_numeric(frame.get(name, 0), errors="coerce").fillna(0)
    if family == "QB":
        return col("attempts") + col("sacks") + col("carries")
    if family == "RB":
        return col("carries") + col("targets")
    return col("targets") + col("carries")


def _yards_efficiency(frame: pd.DataFrame, family: str, opportunities: pd.Series) -> pd.Series:
    col = lambda name: pd.to_numeric(frame.get(name, 0), errors="coerce").fillna(0)
    if family == "QB":
        value = (col("passing_yards") + 20 * col("passing_tds") - 45 * col("interceptions")
                 + col("rushing_yards") + 20 * col("rushing_tds"))
    elif family == "RB":
        value = (col("rushing_yards") + col("receiving_yards")
                 + 20 * (col("rushing_tds") + col("receiving_tds")))
    else:
        value = (col("receiving_yards") + col("rushing_yards")
                 + 20 * (col("receiving_tds") + col("rushing_tds")))
    return value / opportunities.where(opportunities > 0)


def _epa_efficiency(frame: pd.DataFrame, family: str, opportunities: pd.Series) -> pd.Series:
    col = lambda name: pd.to_numeric(frame.get(name, 0), errors="coerce").fillna(0)
    if family == "QB":
        value = col("passing_epa") + col("rushing_epa")
    elif family == "RB":
        value = col("rushing_epa") + col("receiving_epa")
    else:
        value = col("receiving_epa") + col("rushing_epa")
    has_epa = pd.to_numeric(frame.get("epa_rows", 0), errors="coerce").fillna(0) > 0
    return (value / opportunities.where(opportunities > 0)).where(has_epa)


def _weighted_z(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Opportunity-weighted z-score; tiny samples barely move the mean/SD."""
    mask = values.notna() & (weights > 0)
    if mask.sum() < 2:
        return pd.Series(np.where(mask, 0.0, np.nan), index=values.index)
    v, w = values[mask].astype(float), weights[mask].astype(float)
    mean = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    if not np.isfinite(sd) or sd <= 1e-9:
        return pd.Series(np.where(mask, 0.0, np.nan), index=values.index)
    return ((values - mean) / sd).clip(-PERFORMANCE_Z_CLIP, PERFORMANCE_Z_CLIP)


def _performance_grades(stats: pd.DataFrame) -> pd.DataFrame:
    """Attach efficiency z-scores (performance_z) and opportunity counts (sample_size).

    performance_grade is filled later in build_player_ratings, once the Madden
    starter scale for each position is known.
    """
    if stats.empty: return stats
    out = stats.copy()
    out["family"] = out["position"].map(_position_family)
    out["performance_z"] = np.nan
    out["sample_size"] = 0.0
    out["performance_grade"] = np.nan
    for family in PERFORMANCE_STABILITY:
        mask = out["family"].eq(family)
        if not mask.any():
            continue
        group = out.loc[mask]
        opps = _opportunities(group, family)
        z_parts = [
            _weighted_z(_epa_efficiency(group, family, opps), opps),
            _weighted_z(_yards_efficiency(group, family, opps), opps),
        ]
        z = pd.concat(z_parts, axis=1).mean(axis=1, skipna=True)
        out.loc[mask, "performance_z"] = z.where(opps > 0)
        out.loc[mask, "sample_size"] = opps
    return out


def _madden_position_scale(players: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mean/SD of each position's likely starters on the Madden trait-grade scale."""
    scale: dict[str, tuple[float, float]] = {}
    for family, pool in MADDEN_SCALE_POOL.items():
        grades = pd.to_numeric(
            players.loc[players["position_family"].eq(family), "trait_grade"], errors="coerce"
        ).dropna().sort_values(ascending=False).head(pool)
        if grades.empty:
            continue
        sd = float(grades.std(ddof=0)) if len(grades) > 1 else 0.0
        scale[family] = (float(grades.mean()), sd if sd > 0 else 5.0)
    return scale


def _load_roster_status(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    name_col = _find_col(frame, ("full_name", "player_name", "name"))
    if not name_col: return pd.DataFrame()
    team_col = _find_col(frame, ("team", "recent_team", "team_abbr"))
    result = pd.DataFrame({
        "name_key": frame[name_col].map(_name_key),
        "team_abbr": (
            frame[team_col].astype(str).str.upper().str.strip().replace({"AZ": "ARI", "LAR": "LA", "OAK": "LV"})
            if team_col else ""
        ),
        "roster_status": frame[_find_col(frame, ("status",))] if _find_col(frame, ("status",)) else "",
        "gsis_id": frame[_find_col(frame, ("gsis_id", "player_id"))] if _find_col(frame, ("gsis_id", "player_id")) else "",
    })
    keys = ["name_key", "team_abbr"] if team_col else ["name_key"]
    return result.drop_duplicates(keys, keep="last")


def _load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    name_col = _find_col(frame, ("full_name", "player_name", "name"))
    if not name_col: return pd.DataFrame()
    team_col = _find_col(frame, ("team", "recent_team", "team_abbr"))
    status_col = _find_col(frame, ("report_status", "game_status", "practice_status", "status"))
    result = pd.DataFrame({
        "name_key": frame[name_col].map(_name_key),
        "team_abbr": frame[team_col].astype(str).str.upper().str.strip() if team_col else "",
        "injury_status": frame[status_col] if status_col else "",
    })
    keys = ["name_key", "team_abbr"] if team_col else ["name_key"]
    return result.drop_duplicates(keys, keep="last")


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()


def _last_name_key(value: Any) -> str:
    """Normalized last name with suffixes removed ("Ricky White III" -> "white")."""
    text = _clean_text(value)
    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in text.replace("-", " ").split()]
    tokens = [t for t in tokens if t and t not in _NAME_SUFFIXES]
    return tokens[-1] if tokens else ""


def _first_name_key(value: Any) -> str:
    text = _clean_text(value)
    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in text.split()]
    tokens = [t for t in tokens if t]
    return tokens[0] if tokens else ""


def _first_names_compatible(madden_first: str, roster_firsts: Iterable[str]) -> bool:
    """Josh/Joshua, Chig/Chigoziem, Drew/Andrew are compatible; Cody/Ricky is not."""
    if len(madden_first) < 2:
        return False
    for other in roster_firsts:
        if len(other) >= 2 and (other in madden_first or madden_first in other):
            return True
    return False


def _fill_missing_ids_by_last_name(players: pd.DataFrame, roster_path: Path) -> pd.DataFrame:
    """Recover GSIS IDs for nickname mismatches (Joshua/Josh, Chigoziem/Chig).

    Only players still missing a GSIS ID are considered, and a match is accepted
    only when last name + team + position family is unique on BOTH sides and the
    ID is not already used by another player. Anything ambiguous stays unmatched.
    """
    if not roster_path.exists():
        return players
    frame = pd.read_csv(roster_path)
    name_col = _find_col(frame, ("full_name", "player_name", "name"))
    team_col = _find_col(frame, ("team", "recent_team", "team_abbr"))
    pos_col = _find_col(frame, ("position",))
    id_col = _find_col(frame, ("gsis_id", "player_id"))
    if not all((name_col, team_col, pos_col, id_col)):
        return players

    if "week" in frame.columns:
        # Latest week last, so a mid-season trade resolves to the current team.
        frame = frame.assign(_week=pd.to_numeric(frame["week"], errors="coerce")).sort_values("_week", kind="stable")
    first_cols = [c for c in (name_col, _find_col(frame, ("first_name",)), _find_col(frame, ("football_name",))) if c]
    roster = pd.DataFrame({
        "firsts": [
            tuple(sorted({_first_name_key(v) for v in vals if _first_name_key(v)}))
            for vals in zip(*(frame[c] for c in first_cols))
        ],
        "last_key": frame[name_col].map(_last_name_key),
        "team_abbr": frame[team_col].astype(str).str.upper().str.strip().replace({"AZ": "ARI", "LAR": "LA", "OAK": "LV"}),
        "family": frame[pos_col].map(_position_family),
        "gsis_id": frame[id_col].fillna("").astype(str).str.strip(),
    })
    roster = roster[roster["gsis_id"].ne("") & roster["last_key"].ne("") & roster["family"].ne("OTHER")]
    roster = roster.drop_duplicates(["gsis_id"], keep="last")
    keys = ["last_key", "team_abbr", "family"]
    roster = roster[~roster.duplicated(keys, keep=False)]

    out = players.copy()
    out["_last_key"] = out["player_name"].map(_last_name_key)
    current_ids = out["gsis_id"].fillna("").astype(str).str.strip()
    unique_on_team = ~out.duplicated(["_last_key", "team_abbr", "position_family"], keep=False)
    used_ids = set(current_ids[current_ids.ne("")])

    lookup = {
        (r.last_key, r.team_abbr, r.family): (r.gsis_id, r.firsts)
        for r in roster.itertuples(index=False)
        if r.gsis_id not in used_ids
    }
    for idx in out.index[current_ids.eq("") & unique_on_team]:
        key = (out.at[idx, "_last_key"], out.at[idx, "team_abbr"], out.at[idx, "position_family"])
        found, firsts = lookup.get(key, ("", ()))
        if not _first_names_compatible(_first_name_key(out.at[idx, "player_name"]), firsts):
            continue
        if found and found not in used_ids:
            out.at[idx, "gsis_id"] = found
            used_ids.add(found)
    return out.drop(columns=["_last_key"])


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

    # Resolve the current team assignment before joining live roster/injury feeds.
    # Names are not globally unique in the NFL (for example, two Byron Youngs), so
    # availability must be matched by player name + team rather than name alone.
    chart_path = Path(depth_chart_path) if depth_chart_path is not None else root / "footballguys_depth_charts.csv"
    depth_charts = load_depth_charts(chart_path)
    assignments = depth_chart_team_assignments(depth_charts)
    assigned = players["name_key"].map(assignments)
    players["depth_chart_team_override"] = assigned.notna() & assigned.ne(players["team_abbr"])
    players["team_abbr"] = assigned.where(assigned.notna(), players["team_abbr"])

    roster = _load_roster_status(root / "weekly_rosters.csv")
    if roster.empty:
        roster = _load_roster_status(root / "rosters.csv")

    # Resolve the nflverse/GSIS identity before applying historical performance.
    # Current team + normalized name is only used to discover the stable ID; the
    # performance join itself uses GSIS so trades retain prior production without
    # leaking stats across different same-name players.
    if "gsis_id" not in players.columns:
        players["gsis_id"] = ""
    if not roster.empty and "gsis_id" in roster.columns:
        identity = roster[["name_key", "team_abbr", "gsis_id"]].copy()
        identity = identity.rename(columns={"gsis_id": "_identity_gsis_id"})
        identity = identity.drop_duplicates(["name_key", "team_abbr"], keep="last")
        players = players.merge(identity, on=["name_key", "team_abbr"], how="left")
        fresh_id = players["_identity_gsis_id"].fillna("").astype(str).str.strip()
        current_id = players["gsis_id"].fillna("").astype(str).str.strip()
        players["gsis_id"] = fresh_id.where(fresh_id.ne(""), current_id)
        players = players.drop(columns=["_identity_gsis_id"])

    # Second pass for nickname / legal-name differences between Madden and nflverse.
    roster_file = root / "weekly_rosters.csv"
    if not roster_file.exists():
        roster_file = root / "rosters.csv"
    players = _fill_missing_ids_by_last_name(players, roster_file)

    stats = _performance_grades(_aggregate_weekly_stats(root / "player_weekly_stats.csv"))
    if not stats.empty:
        perf_cols = ["performance_z", "sample_size"]
        if "performance_cap" in stats.columns:
            perf_cols.append("performance_cap")

        # Primary identity path: exact GSIS/player_id match.
        stats_by_id = stats[stats.get("player_id", pd.Series(index=stats.index, dtype=str)).fillna("").astype(str).str.strip().ne("")].copy()
        if not stats_by_id.empty:
            stats_by_id["_stats_identity"] = "id:" + stats_by_id["player_id"].astype(str).str.strip()
        # Safe fallback for fixtures/legacy rows that lack GSIS: only allow a name
        # alias when that normalized name belongs to exactly one stats identity.
        identity_counts = stats.groupby("name_key")["player_id"].apply(
            lambda values: max(1, len({str(v).strip() for v in values if str(v).strip()}))
        )
        unique_names = set(identity_counts[identity_counts.eq(1)].index)
        stats_by_name = stats[stats["name_key"].isin(unique_names)].drop_duplicates("name_key", keep="last").copy()
        stats_by_name["_stats_identity"] = "name:" + stats_by_name["name_key"].astype(str)

        stats_lookup = pd.concat([stats_by_id, stats_by_name], ignore_index=True, sort=False)
        stats_lookup = stats_lookup[["_stats_identity"] + perf_cols].drop_duplicates("_stats_identity", keep="first")

        player_gsis = players["gsis_id"].fillna("").astype(str).str.strip()
        players["_stats_identity"] = np.where(
            player_gsis.ne(""), "id:" + player_gsis, "name:" + players["name_key"].astype(str)
        )
        players = players.merge(stats_lookup, on="_stats_identity", how="left").drop(columns=["_stats_identity"])
    else:
        players["performance_z"] = np.nan
        players["sample_size"] = 0.0
        players["performance_cap"] = 0.0

    # Translate efficiency z-scores onto the Madden starter scale for the position.
    scale = _madden_position_scale(players)
    z = pd.to_numeric(players["performance_z"], errors="coerce")
    mu = players["position_family"].map(lambda f: scale.get(f, (np.nan, np.nan))[0])
    sd = players["position_family"].map(lambda f: scale.get(f, (np.nan, np.nan))[1])
    players["performance_grade"] = (mu + z * sd).clip(0, 99)

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

        join_keys = ["name_key", "team_abbr"] if "team_abbr" in roster_merge.columns else ["name_key"]
        keep_cols = join_keys + [
            column for column in ("_fresh_roster_status", "_fresh_gsis_id")
            if column in roster_merge.columns
        ]
        roster_merge = roster_merge[keep_cols].drop_duplicates(join_keys, keep="last")
        players = players.merge(roster_merge, on=join_keys, how="left")

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

    # Sleeper is the primary live availability feed. nflverse injuries remain a
    # fallback when a Sleeper snapshot has not been refreshed yet. Definitive
    # unavailable states drive depth-chart substitution later; Questionable and
    # Doubtful are preserved as uncertainty rather than being silently benched.
    sleeper = load_availability(root / "sleeper_availability.csv")
    if not sleeper.empty:
        sleeper_cols = [
            "name_key", "team_abbr", "roster_status", "injury_status",
            "practice_participation", "availability_state",
            "definitively_unavailable", "updated_at_utc",
        ]
        sleeper = sleeper[[c for c in sleeper_cols if c in sleeper.columns]].copy()
        sleeper = sleeper.rename(columns={
            "roster_status": "_sleeper_roster_status",
            "injury_status": "_sleeper_injury_status",
            "updated_at_utc": "availability_updated_at_utc",
        })
        sleeper = sleeper.drop_duplicates(["name_key", "team_abbr"], keep="last")
        players = players.merge(sleeper, on=["name_key", "team_abbr"], how="left")
        if "_sleeper_roster_status" in players.columns:
            live = players["_sleeper_roster_status"].fillna("").astype(str).str.strip()
            current = players["roster_status"].fillna("").astype(str).str.strip()
            players["roster_status"] = live.where(live.ne(""), current)
        players["injury_status"] = players.get("_sleeper_injury_status", "").fillna("").astype(str)
        players["availability_state"] = players.get("availability_state", "").fillna("").astype(str)
        players["definitively_unavailable"] = players.get("definitively_unavailable", False).fillna(False).astype(bool)
        players["practice_participation"] = players.get("practice_participation", "").fillna("").astype(str)
        players["availability_source"] = np.where(players["availability_state"].ne(""), "Sleeper", "")
    else:
        injuries = _load_injuries(root / "injuries.csv")
        if not injuries.empty:
            join_keys = ["name_key", "team_abbr"] if "team_abbr" in injuries.columns else ["name_key"]
            players = players.merge(injuries, on=join_keys, how="left")
        else:
            players["injury_status"] = ""
        injury_text = players["injury_status"].fillna("").astype(str).str.lower()
        players["availability_state"] = np.select(
            [injury_text.str.contains(r"\bout\b|reserve|\bir\b", regex=True),
             injury_text.str.contains("doubt"),
             injury_text.str.contains("question")],
            ["Out", "Doubtful", "Questionable"], default="Active",
        )
        players["definitively_unavailable"] = players["availability_state"].eq("Out")
        players["practice_participation"] = ""
        players["availability_updated_at_utc"] = ""
        players["availability_source"] = np.where(players["injury_status"].fillna("").astype(str).str.strip().ne(""), "nflverse fallback", "")

    opportunities = pd.to_numeric(players["sample_size"], errors="coerce").fillna(0).clip(lower=0)
    stability = players["position_family"].map(PERFORMANCE_STABILITY)
    players["performance_confidence"] = (opportunities / (opportunities + stability)).fillna(0.0)
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

    # Availability does not apply arbitrary point deductions to healthy/questionable
    # players. Definitively unavailable players are replaced by the next available
    # depth-chart option inside _unit_grade().
    players["availability_adjustment"] = 0.0
    players["macabets_rating"] = players["base_rating"].clip(0, 99).round(2)
    players["rating_confidence"] = np.where(players["performance_weight"] > .35, "high", np.where(players["performance_weight"] > .10, "medium", "baseline"))
    players["rating_source"] = np.where(players["performance_weight"] > 0, "Madden 27 + nflverse performance", "Madden 27 baseline")

    keep = ["player_name", "team_abbr", "position", "position_family", "overall", "trait_grade",
            "performance_grade", "performance_weight", "availability_adjustment", "macabets_rating",
            "rating_confidence", "rating_source", "roster_status", "injury_status", "gsis_id",
            "depth_chart_team_override", "availability_state", "definitively_unavailable",
            "practice_participation", "availability_updated_at_utc", "availability_source"]
    return players[keep].sort_values(["team_abbr", "macabets_rating"], ascending=[True, False]).reset_index(drop=True)


def _legacy_unit_selection(team_players: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rating-order fallback used only when authoritative depth data is unavailable."""
    allowed = POSITION_GROUPS[unit]
    group = team_players[team_players["position"].isin(allowed)].sort_values("macabets_rating", ascending=False).head(DEPTH_LIMITS[unit])
    starter_n = min(STARTER_COUNTS[unit], len(group))
    return group.head(starter_n).copy(), group.iloc[starter_n:].copy()


def _is_unavailable(row: pd.Series) -> bool:
    value = row.get("definitively_unavailable", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _apply_availability(starters: pd.DataFrame, depth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Promote same-role backups when a starter is definitively unavailable."""
    if starters.empty:
        return starters, depth, [], []

    starters = starters.copy()
    depth = depth.copy()
    unavailable: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    active_rows = []
    consumed_depth: set[int] = set()

    for _, starter in starters.iterrows():
        if not _is_unavailable(starter):
            active_rows.append(starter)
            continue

        role = str(starter.get("depth_chart_role", "") or "")
        unavailable.append({
            "name": str(starter.get("player_name", "")),
            "role": role,
            "status": str(starter.get("availability_state", "Out") or "Out"),
            "injury_status": str(starter.get("injury_status", "") or ""),
        })
        candidates = depth[depth.get("depth_chart_role", pd.Series(index=depth.index, dtype=str)).astype(str).eq(role)] if not depth.empty else depth
        replacement = None
        replacement_idx = None
        for idx, candidate in candidates.iterrows():
            if idx in consumed_depth or _is_unavailable(candidate):
                continue
            replacement = candidate.copy()
            replacement_idx = idx
            break
        if replacement is not None:
            active_rows.append(replacement)
            consumed_depth.add(replacement_idx)
            promotions.append({
                "role": role,
                "out": str(starter.get("player_name", "")),
                "in": str(replacement.get("player_name", "")),
                "replacement_rating": round(float(replacement.get("macabets_rating", 0.0)), 2),
            })

    active_starters = pd.DataFrame(active_rows) if active_rows else starters.iloc[0:0].copy()
    remaining_depth = depth.drop(index=list(consumed_depth), errors="ignore").copy()
    if not remaining_depth.empty:
        remaining_depth = remaining_depth[~remaining_depth.apply(_is_unavailable, axis=1)].copy()
    return active_starters.reset_index(drop=True), remaining_depth.reset_index(drop=True), unavailable, promotions


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

    # Preserve the healthy depth-chart baseline so the UI can show the exact
    # personnel-grade impact of an unavailable starter after a backup is activated.
    healthy_starters = starters.copy()
    healthy_depth = depth.copy()

    unavailable_starters: list[dict[str, Any]] = []
    availability_promotions: list[dict[str, Any]] = []
    if selection_source == "Footballguys depth chart" and not starters.empty:
        starters, depth, unavailable_starters, availability_promotions = _apply_availability(starters, depth)

    if starters.empty:
        return {
            "grade": 50.0, "starter_grade": 50.0, "depth_grade": 50.0, "player_count": 0,
            "confidence": "missing", "top_players": [], "selection_source": selection_source,
            "scheme": plan.get("scheme", "unknown"), "unmatched_depth_chart": unmatched_starters + unmatched_depth,
            "unavailable_starters": unavailable_starters, "availability_promotions": availability_promotions,
            "availability_source": "Sleeper" if unavailable_starters else "",
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

    qb_replacement_context = {}
    if unit == "quarterback" and unavailable_starters:
        grade, qb_replacement_context = apply_qb_replacement_adjustment(
            grade=grade,
            healthy_starters=healthy_starters,
            active_starters=starters,
            depth=depth,
        )

    if not healthy_starters.empty:
        healthy_starter_grade = float(healthy_starters["macabets_rating"].mean())
        healthy_depth_grade = float(healthy_depth["macabets_rating"].mean()) if not healthy_depth.empty else healthy_starter_grade
        healthy_grade = healthy_starter_grade * (1.0 - depth_blend) + healthy_depth_grade * depth_blend
    else:
        healthy_starter_grade = starter_grade
        healthy_depth_grade = depth_grade
        healthy_grade = grade

    matched_starter_count = len(starters)
    confidence = "high" if selection_source == "Footballguys depth chart" and matched_starter_count >= max(1, expected_starters) else "limited"
    top = pd.concat([starters.assign(depth_order=0), depth.assign(depth_order=1)], ignore_index=True)
    top_players = []
    for _, r in top.head(8).iterrows():
        top_players.append({
            "name": r["player_name"], "position": r["position"], "rating": float(r["macabets_rating"]),
            "role": str(r.get("depth_chart_role", "") or ""), "starter": bool(int(r.get("depth_order", 1)) == 0),
            "availability": str(r.get("availability_state", "Active") or "Active"),
            "injury_status": str(r.get("injury_status", "") or ""),
        })

    return {
        "grade": round(grade, 2), "starter_grade": round(starter_grade, 2), "depth_grade": round(depth_grade, 2),
        "healthy_grade": round(healthy_grade, 2),
        "healthy_starter_grade": round(healthy_starter_grade, 2),
        "availability_grade_delta": round(grade - healthy_grade, 2),
        "player_count": int(len(starters) + len(depth)), "starter_count": int(len(starters)),
        "depth_count": int(len(depth)), "confidence": confidence, "top_players": top_players,
        "selection_source": selection_source, "depth_chart_source": plan.get("source", ""),
        "scheme": plan.get("scheme", "unknown"),
        "unmatched_depth_chart": unmatched_starters + unmatched_depth,
        "unavailable_starters": unavailable_starters,
        "availability_promotions": availability_promotions,
        "availability_source": "Sleeper" if unavailable_starters or availability_promotions else "",
        "qb_replacement_context": qb_replacement_context,
        "availability_uncertainty": [
            {"name": str(r.get("player_name", "")), "role": str(r.get("depth_chart_role", "") or ""),
             "status": str(r.get("availability_state", ""))}
            for _, r in pd.concat([starters, depth], ignore_index=True).iterrows()
            if str(r.get("availability_state", "")) in {"Questionable", "Doubtful"}
        ],
    }


# Team-unit performance (v1.5)
# ----------------------------
# Team snapshot unit grades are single-number summaries of a small number of
# games, and they live on a different scale (league mean ~68) than roster unit
# grades (league mean ~79). They are first rescaled onto the roster scale by
# z-score, then blended with a weight that grows with games played:
#     weight = TEAM_PERFORMANCE_CAP * games / (games + TEAM_PERFORMANCE_STABILITY)
# Week 1 ~9%, Week 4 ~24%, Week 8 ~34%, Week 17 ~44%.
TEAM_PERFORMANCE_CAP = 0.60
TEAM_PERFORMANCE_STABILITY = 6.0
PRIOR_SEASON_TEAM_WEIGHT = 0.20
LINEBACKER_PROXY_SHARE = 0.50
LINEBACKER_PROXY_MAX = 0.30
TEAM_LIVE_MAP = {
    # QB/RB/WR/TE already receive player-level weekly performance in build_player_ratings.
    # Do not blend the same evidence into those units a second time.
    "offensive_line": "offensive_line",
    "defensive_front": "defensive_line",
    "secondary": "secondary",
    "special_teams": "special_teams",
}


def team_performance_weight(season: Any, through_week: Any, *, current_year: int | None = None) -> float:
    season_num = pd.to_numeric(season, errors="coerce")
    week_num = pd.to_numeric(through_week, errors="coerce")
    row_season = int(season_num) if pd.notna(season_num) else 0
    games = max(0, int(week_num)) if pd.notna(week_num) else 0
    year = current_year if current_year is not None else datetime.now(timezone.utc).year
    if row_season == year:
        if games <= 0:
            return 0.0
        return TEAM_PERFORMANCE_CAP * games / (games + TEAM_PERFORMANCE_STABILITY)
    if 0 < row_season < year:
        return PRIOR_SEASON_TEAM_WEIGHT
    return 0.0


def _rescale_to_roster(
    values: dict[str, float], roster: dict[str, float]
) -> dict[str, float]:
    """Map performance grades onto the roster-grade scale via league z-scores."""
    teams = [t for t in values if t in roster]
    if len(teams) < 2:
        return {t: roster[t] for t in teams}
    perf = np.array([values[t] for t in teams], dtype=float)
    base = np.array([roster[t] for t in teams], dtype=float)
    perf_sd = float(perf.std())
    base_mean, base_sd = float(base.mean()), float(base.std())
    if perf_sd <= 1e-9:
        return {t: base_mean for t in teams}
    z = np.clip((perf - perf.mean()) / perf_sd, -PERFORMANCE_Z_CLIP, PERFORMANCE_Z_CLIP)
    return {t: round(float(base_mean + zi * base_sd), 2) for t, zi in zip(teams, z)}


def build_team_ratings(
    player_ratings: pd.DataFrame,
    snapshot_path: Path | str = DEFAULT_NFL_DIR / "team_snapshot.csv",
    depth_chart_path: Path | str = DEFAULT_DEPTH_CHART_PATH,
    current_season: int | None = None,
) -> dict[str, dict[str, Any]]:
    """current_season defaults to this calendar year; backtests pass the replayed season."""
    snapshot = pd.read_csv(snapshot_path) if Path(snapshot_path).exists() else pd.DataFrame()
    depth_charts = load_depth_charts(depth_chart_path)
    snap_by_abbr = {str(r["team_abbr"]): r for _, r in snapshot.iterrows()} if "team_abbr" in snapshot else {}

    # Pass 1: roster-based unit grades for every team.
    team_units: dict[str, dict[str, dict[str, Any]]] = {}
    team_depth: dict[str, pd.DataFrame] = {}
    team_frames: dict[str, pd.DataFrame] = {}
    for abbr, team_players in player_ratings.groupby("team_abbr"):
        abbr = str(abbr)
        current_depth = team_depth_chart(depth_charts, abbr)
        team_depth[abbr] = current_depth
        team_frames[abbr] = team_players
        team_units[abbr] = {name: _unit_grade(team_players, name, current_depth) for name in POSITION_GROUPS}

    # League-wide rescaling of snapshot unit grades onto each unit's roster scale.
    rescaled: dict[str, dict[str, float]] = {}
    unit_sources = dict(TEAM_LIVE_MAP)
    unit_sources["linebackers"] = "defense"
    for unit, col in unit_sources.items():
        perf_values: dict[str, float] = {}
        for abbr, row in snap_by_abbr.items():
            if abbr in team_units and col in row and pd.notna(pd.to_numeric(row[col], errors="coerce")):
                perf_values[abbr] = float(row[col])
        roster_values = {abbr: float(team_units[abbr][unit]["grade"]) for abbr in perf_values}
        rescaled[unit] = _rescale_to_roster(perf_values, roster_values)

    # Pass 2: blend and assemble.
    result = {}
    for abbr, units in team_units.items():
        team_players = team_frames[abbr]
        current_depth = team_depth[abbr]
        row = snap_by_abbr.get(abbr)
        perf_weight = (
            team_performance_weight(row.get("season"), row.get("through_week"), current_year=current_season)
            if row is not None else 0.0
        )

        for unit, col in TEAM_LIVE_MAP.items():
            implied = rescaled.get(unit, {}).get(abbr)
            if implied is not None and perf_weight > 0:
                roster_grade = units[unit]["grade"]
                units[unit]["roster_grade"] = roster_grade
                units[unit]["performance_grade_raw"] = round(float(row[col]), 2)
                units[unit]["performance_grade"] = round(implied, 2)
                units[unit]["grade"] = round(roster_grade * (1 - perf_weight) + implied * perf_weight, 2)
                units[unit]["performance_weight"] = round(perf_weight, 3)
                units[unit]["source"] = f"{(1-perf_weight):.0%} roster + {perf_weight:.0%} NFL performance"
            else:
                units[unit]["source"] = "Madden 27 roster rating"
        # Linebacker play currently lacks a clean player-level weekly metric in this pipeline.
        # Use only a modest team-defense proxy; never apply generic offense to RB/WR/TE.
        indirect_weight = min(perf_weight * LINEBACKER_PROXY_SHARE, LINEBACKER_PROXY_MAX)
        lb_implied = rescaled.get("linebackers", {}).get(abbr)
        if lb_implied is not None and indirect_weight > 0:
            units["linebackers"]["roster_grade"] = units["linebackers"]["grade"]
            units["linebackers"]["performance_grade_raw"] = round(float(row["defense"]), 2)
            units["linebackers"]["performance_grade"] = round(lb_implied, 2)
            units["linebackers"]["grade"] = round(units["linebackers"]["grade"] * (1-indirect_weight) + lb_implied * indirect_weight, 2)
            units["linebackers"]["performance_weight"] = round(indirect_weight, 3)
            units["linebackers"]["source"] = f"{(1-indirect_weight):.0%} roster + {indirect_weight:.0%} NFL team-defense proxy"
        overall = sum(units[u]["grade"] * w for u, w in TEAM_WEIGHTS.items())
        offense = units["quarterback"]["grade"] * .35 + units["running_backs"]["grade"] * .12 + units["receiving_weapons"]["grade"] * .25 + units["offensive_line"]["grade"] * .28
        defense = units["defensive_front"]["grade"] * .36 + units["linebackers"]["grade"] * .24 + units["secondary"]["grade"] * .40
        full_name = TEAM_ALIASES.get(str(abbr), str(abbr))
        availability_sources = [str(v) for v in team_players.get("availability_source", pd.Series(dtype=str)).dropna().tolist() if str(v)]
        availability_updates = [str(v) for v in team_players.get("availability_updated_at_utc", pd.Series(dtype=str)).dropna().tolist() if str(v)]
        unavailable_count = sum(len(unit.get("unavailable_starters", []) or []) for unit in units.values())
        promotion_count = sum(len(unit.get("availability_promotions", []) or []) for unit in units.values())
        uncertain_count = sum(len(unit.get("availability_uncertainty", []) or []) for unit in units.values())
        depth_source = str(depth_charts.attrs.get("source_name") or "depth chart") if not current_depth.empty else "rating-order fallback"
        result[full_name] = {
            "team_abbr": str(abbr), "overall_rating": round(overall, 2), "offense_rating": round(offense, 2),
            "defense_rating": round(defense, 2), "player_count": int(len(team_players)), "units": units,
            "source": f"Macabets automated rating engine v1.5 - {depth_source} + Sleeper availability + audited Madden 27 baseline", "prediction_influence_enabled": True,
            "personnel_matchup_influence_enabled": True,
            "depth_chart_source": depth_source,
            "depth_chart_rows": int(len(current_depth)),
            "availability_source": "Sleeper" if "Sleeper" in availability_sources else (availability_sources[0] if availability_sources else ""),
            "availability_updated_at_utc": max(availability_updates) if availability_updates else "",
            "unavailable_starters": int(unavailable_count),
            "availability_promotions": int(promotion_count),
            "availability_uncertain": int(uncertain_count),
        }
    return dict(sorted(result.items()))




def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def save_rating_outputs(
    player_ratings: pd.DataFrame,
    team_ratings: dict[str, Any],
    *,
    nfl_dir: Path | str = DEFAULT_NFL_DIR,
    depth_chart_path: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(nfl_dir); root.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    player_path, team_path = root / "player_ratings.csv", root / "team_ratings_auto.json"
    temp = player_path.with_suffix(".csv.tmp"); player_ratings.to_csv(temp, index=False); temp.replace(player_path)
    _write_json(team_path, team_ratings)
    availability_status = load_availability_status(Path(nfl_dir) / "sleeper_availability_status.json")
    depth_sources = sorted({
        str(team.get("depth_chart_source") or "").strip()
        for team in team_ratings.values()
        if isinstance(team, dict) and str(team.get("depth_chart_source") or "").strip()
    })
    depth_source = ", ".join(depth_sources) if depth_sources else "Unavailable"
    resolved_depth_path = Path(depth_chart_path) if depth_chart_path is not None else None
    status = {
        "schema_version": "1.5", "engine_version": "1.5-efficiency-weighting", "updated_at_utc": updated,
        "players_rated": int(len(player_ratings)), "teams_rated": int(len(team_ratings)),
        "players_with_performance_data": int((player_ratings["performance_weight"] > 0).sum()),
        # These unit grades feed nfl_ratings_loader -> team power scores, so the
        # performance blend is live in NFL predictions.
        "prediction_influence_enabled": True,
        "depth_chart_source": depth_source,
        "depth_chart_file": str(resolved_depth_path) if resolved_depth_path is not None else "",
        "availability_source": availability_status.get("source", "Sleeper snapshot not available"),
        "availability_updated_at_utc": availability_status.get("updated_at_utc"),
        "availability_players": availability_status.get("players", 0),
        "availability_definitively_unavailable": availability_status.get("definitively_unavailable", 0),
        "availability_uncertain": availability_status.get("uncertain", 0),
        "files": {"players": str(player_path), "teams": str(team_path)},
    }
    _write_json(root / "rating_status.json", status)
    history_path = root / "rating_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"updated_at_utc": updated, "teams": {k: v["overall_rating"] for k, v in team_ratings.items()}}, sort_keys=True) + "\n")
    return status


def _resolve_depth_chart_path(nfl_dir: Path | str = DEFAULT_NFL_DIR) -> Path:
    """Resolve the best current depth chart for production ratings.

    The daily nflverse snapshot is preferred because it updates automatically with
    roster/depth changes. The existing Footballguys CSV remains a safe fallback if
    the automatic dataset is missing or malformed.
    """
    root = Path(nfl_dir)
    auto_candidates = [root / "depth_charts.csv", AUTO_DEPTH_CHART_PATH]
    for candidate in auto_candidates:
        if not candidate.exists():
            continue
        try:
            chart = load_depth_charts(candidate)
        except Exception:
            continue
        if not chart.empty and chart["team_abbr"].nunique() >= 32:
            return candidate

    fallback_candidates = [
        DEFAULT_DEPTH_CHART_PATH,
        root / "footballguys_depth_charts.csv",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    return DEFAULT_DEPTH_CHART_PATH


def build_and_save_ratings(*, madden_path: Path | str = DEFAULT_MADDEN_PATH, nfl_dir: Path | str = DEFAULT_NFL_DIR) -> dict[str, Any]:
    depth_chart_path = _resolve_depth_chart_path(nfl_dir)
    players = build_player_ratings(
        madden_path=madden_path,
        nfl_dir=nfl_dir,
        depth_chart_path=depth_chart_path,
    )
    teams = build_team_ratings(
        players,
        snapshot_path=Path(nfl_dir) / "team_snapshot.csv",
        depth_chart_path=depth_chart_path,
    )
    return save_rating_outputs(players, teams, nfl_dir=nfl_dir, depth_chart_path=depth_chart_path)
