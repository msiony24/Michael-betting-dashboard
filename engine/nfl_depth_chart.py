"""Depth-chart-first NFL personnel selection for Macabets.

Footballguys determines who is first/second/etc. on the depth chart. Madden and
current NFL performance determine how good those players are. This module keeps
those two responsibilities separate so a higher Madden OVR never silently
promotes a backup over the actual starter.
"""

from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH_CHART_PATH = PROJECT_ROOT / "data" / "footballguys_depth_charts.csv"

DEPTH_COLUMNS = ("Starter", "2nd String", "3rd String", "4th String", "5th String")

TEAM_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
ABBR_TO_TEAM = {abbr: team for team, abbr in TEAM_TO_ABBR.items()}


def normalize_player_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text.lower())
    return re.sub(r"[^a-z0-9]", "", text)


def load_depth_charts(path: Path | str = DEFAULT_DEPTH_CHART_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["Team", "Unit", "Position", *DEPTH_COLUMNS, "Source URL", "team_abbr"])
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"Team", "Unit", "Position", *DEPTH_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Depth chart file missing required columns: " + ", ".join(sorted(missing)))
    frame["Team"] = frame["Team"].astype(str).str.strip()
    frame["Unit"] = frame["Unit"].astype(str).str.strip()
    frame["Position"] = frame["Position"].astype(str).str.upper().str.strip()
    frame["team_abbr"] = frame["Team"].map(TEAM_TO_ABBR).fillna("")
    for col in DEPTH_COLUMNS:
        frame[col] = frame[col].astype(str).str.strip()
    return frame



def depth_chart_team_assignments(depth_charts: pd.DataFrame) -> dict[str, str]:
    """Map uniquely listed depth-chart players to their current team abbreviation."""
    if depth_charts.empty:
        return {}
    seen: dict[str, set[str]] = {}
    for _, row in depth_charts.iterrows():
        abbr = str(row.get("team_abbr", "") or "").strip()
        if not abbr:
            continue
        for col in DEPTH_COLUMNS:
            name = str(row.get(col, "") or "").strip()
            key = normalize_player_name(name)
            if key:
                seen.setdefault(key, set()).add(abbr)
    return {key: next(iter(teams)) for key, teams in seen.items() if len(teams) == 1}

def team_depth_chart(depth_charts: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    if depth_charts.empty:
        return depth_charts.copy()
    abbr = str(team_abbr or "").upper().strip()
    return depth_charts[depth_charts["team_abbr"].eq(abbr)].copy()


def _row(team_depth: pd.DataFrame, position: str) -> pd.Series | None:
    hit = team_depth[team_depth["Position"].eq(position)]
    return hit.iloc[0] if not hit.empty else None


def _names_from_role(team_depth: pd.DataFrame, role: str, *, starter_count: int = 1, depth_count: int = 2) -> tuple[list[str], list[str]]:
    row = _row(team_depth, role)
    if row is None:
        return [], []
    values = [str(row.get(col, "") or "").strip() for col in DEPTH_COLUMNS]
    values = [v for v in values if v]
    return values[:starter_count], values[starter_count:starter_count + depth_count]


def _dedupe(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, role in values:
        key = normalize_player_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((name, role))
    return out


def unit_depth_plan(team_depth: pd.DataFrame, unit: str) -> dict[str, Any]:
    """Return depth-chart roles/names for a Macabets unit.

    The defensive plan is scheme-aware: teams with NT + LILB/RILB are treated as
    3-4 fronts, so SLB/WLB join the line-of-scrimmage group rather than being
    double-counted again as off-ball linebackers.
    """
    if team_depth.empty:
        return {"starters": [], "depth": [], "scheme": "unknown", "source": "missing"}

    starters: list[tuple[str, str]] = []
    depth: list[tuple[str, str]] = []
    scheme = "base"

    def add(role: str, starter_count: int = 1, depth_count: int = 2) -> None:
        s, d = _names_from_role(team_depth, role, starter_count=starter_count, depth_count=depth_count)
        starters.extend((name, role) for name in s)
        depth.extend((name, role) for name in d)

    if unit == "quarterback":
        add("QB", 1, 3)
    elif unit == "running_backs":
        add("RB", 1, 3)
    elif unit == "receiving_weapons":
        # Footballguys' WR row is an ordered receiving depth chart rather than
        # separate X/Z/slot rows. Use the first three WRs plus TE1 as the healthy
        # starting receiving group.
        add("WR", 3, 2)
        add("TE", 1, 2)
    elif unit == "offensive_line":
        for role in ("LT", "LG", "C", "RG", "RT"):
            add(role, 1, 1)
    elif unit in {"defensive_front", "linebackers"}:
        has_34 = _row(team_depth, "NT") is not None and _row(team_depth, "LILB") is not None and _row(team_depth, "RILB") is not None
        scheme = "3-4" if has_34 else "4-3"
        if unit == "defensive_front":
            roles = ("LDE", "NT", "RDE", "SLB", "WLB") if has_34 else ("LDE", "LDT", "RDT", "RDE")
        else:
            roles = ("LILB", "RILB") if has_34 else ("SLB", "MLB", "WLB")
        for role in roles:
            add(role, 1, 1)
    elif unit == "secondary":
        for role in ("LCB", "RCB", "SCB", "SS", "FS"):
            add(role, 1, 1)
    elif unit == "special_teams":
        add("PK", 1, 0)
        add("P", 1, 0)

    source = "Footballguys depth chart"
    if "Source URL" in team_depth.columns:
        urls = [u for u in team_depth["Source URL"].astype(str).unique().tolist() if u]
        if urls:
            source = urls[0]
    return {
        "starters": _dedupe(starters),
        "depth": _dedupe(depth),
        "scheme": scheme,
        "source": source,
    }



def _name_parts(value: Any) -> tuple[str, str]:
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    raw = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", raw)
    parts = re.findall(r"[a-z0-9]+", raw)
    if not parts:
        return "", ""
    return parts[0], parts[-1]


def _role_compatible(position: Any, role: str) -> bool:
    pos = str(position or "").upper().strip()
    role = str(role or "").upper().strip()
    if role == "QB": return pos == "QB"
    if role == "RB": return pos in {"RB", "HB", "FB"}
    if role == "WR": return pos == "WR"
    if role == "TE": return pos == "TE"
    if role in {"LG", "RG"}: return pos in {"LG", "RG", "OL", "G"}
    if role in {"LT", "RT"}: return pos in {"LT", "RT", "OL", "T"}
    if role == "C": return pos in {"C", "OL"}
    if role in {"LDE", "RDE", "LDT", "RDT", "NT"}: return pos in {"DE", "DT", "DL", "LE", "RE", "EDGE", "LEDG", "REDG"}
    if role in {"SLB", "WLB", "MLB", "LILB", "RILB"}: return pos in {"LB", "MLB", "ILB", "OLB", "LOLB", "ROLB", "MIKE", "WILL", "SAM", "EDGE", "LEDG", "REDG"}
    if role in {"LCB", "RCB", "SCB", "SS", "FS"}: return pos in {"CB", "DB", "FS", "SS", "S"}
    if role == "PK": return pos == "K"
    if role == "P": return pos == "P"
    return True


def _fallback_name_match(working: pd.DataFrame, target_name: str, role: str, used: set[int]) -> int | None:
    """Conservative nickname/full-first-name bridge within the expected position family."""
    target_first, target_last = _name_parts(target_name)
    if not target_last:
        return None
    candidates = []
    for idx, row in working.iterrows():
        if idx in used or not _role_compatible(row.get("position"), role):
            continue
        cand_first, cand_last = _name_parts(row.get("player_name"))
        if cand_last != target_last:
            continue
        # Same surname + compatible football role is strong evidence. First-name
        # similarity prevents collisions such as unrelated players with common surnames.
        ratio = SequenceMatcher(None, target_first, cand_first).ratio() if target_first and cand_first else 0.0
        prefix = target_first[:2] == cand_first[:2] if len(target_first) >= 2 and len(cand_first) >= 2 else False
        if ratio >= 0.55 or prefix:
            candidates.append((ratio, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.12:
        return None
    return candidates[0][1]

def match_depth_players(team_players: pd.DataFrame, planned: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Match planned depth-chart players to a team's Macabets player ratings."""
    if team_players.empty or not planned:
        return team_players.iloc[0:0].copy(), [{"name": n, "role": r} for n, r in planned]

    working = team_players.copy()
    working["_depth_name_key"] = working["player_name"].map(normalize_player_name)
    lookup = {}
    for idx, row in working.iterrows():
        key = row["_depth_name_key"]
        if key and key not in lookup:
            lookup[key] = idx

    matched_rows = []
    unmatched = []
    used: set[int] = set()
    for name, role in planned:
        idx = lookup.get(normalize_player_name(name))
        if idx is None or idx in used:
            idx = _fallback_name_match(working, name, role, used)
        if idx is None or idx in used:
            unmatched.append({"name": name, "role": role})
            continue
        used.add(idx)
        row = working.loc[idx].copy()
        row["depth_chart_role"] = role
        matched_rows.append(row)

    if not matched_rows:
        return working.iloc[0:0].drop(columns=["_depth_name_key"], errors="ignore"), unmatched
    out = pd.DataFrame(matched_rows).drop(columns=["_depth_name_key"], errors="ignore")
    return out.reset_index(drop=True), unmatched
