"""Automated NFL roster-continuity priors.

Continuity is intentionally narrow: it measures how much of the current starting
lineup was already on the same club's prior-season roster. It is not a depth or
talent grade; those concepts are handled elsewhere by the personnel engine.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH_CHART = PROJECT_ROOT / "data" / "footballguys_depth_charts.csv"
DEFAULT_PRIOR_ROSTERS = PROJECT_ROOT / "data" / "nfl" / "prior_rosters.csv"

TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Current starters at the most stable offense/defense positions. K/P/LS are excluded
# because special teams already has its own rating.
EXCLUDED_POSITIONS = {"K", "P", "LS", "KR", "PR"}


def _norm_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _continuity_score(retained_rate: float) -> float:
    """Convert retention into a deliberately compressed 0-100 prior.

    A continuity rating is not a quality rating. Keeping the scale between 55 and
    85 prevents roster stability from becoming more important than talent.
    """
    rate = max(0.0, min(float(retained_rate), 1.0))
    return round(55.0 + 30.0 * rate, 1)


def build_continuity_priors(
    depth_chart_path: str | Path = DEFAULT_DEPTH_CHART,
    prior_rosters_path: str | Path = DEFAULT_PRIOR_ROSTERS,
) -> dict[str, dict[str, Any]]:
    depth_path = Path(depth_chart_path)
    roster_path = Path(prior_rosters_path)
    if not depth_path.exists() or not roster_path.exists():
        return {}

    depth = pd.read_csv(depth_path)
    rosters = pd.read_csv(roster_path)
    if not {"Team", "Position", "Starter"}.issubset(depth.columns):
        return {}
    if not {"team", "full_name"}.issubset(rosters.columns):
        return {}

    prior_names: dict[str, set[str]] = {}
    for abbr, group in rosters.groupby(rosters["team"].astype(str)):
        team_name = TEAM_ABBR_TO_NAME.get(str(abbr).upper())
        if not team_name:
            continue
        prior_names[team_name] = {
            name for name in (_norm_name(v) for v in group["full_name"].tolist()) if name
        }

    output: dict[str, dict[str, Any]] = {}
    for team_name, group in depth.groupby(depth["Team"].astype(str)):
        prior = prior_names.get(str(team_name), set())
        if not prior:
            continue
        starters = []
        for _, row in group.iterrows():
            position = str(row.get("Position") or "").strip().upper()
            starter = str(row.get("Starter") or "").strip()
            if not starter or starter.lower() == "nan" or position in EXCLUDED_POSITIONS:
                continue
            starters.append((starter, _norm_name(starter)))
        # De-duplicate players who appear on more than one depth-chart row.
        unique = {}
        for display, normalized in starters:
            if normalized:
                unique[normalized] = display
        if not unique:
            continue
        retained = [display for normalized, display in unique.items() if normalized in prior]
        rate = len(retained) / len(unique)
        output[str(team_name)] = {
            "rating": _continuity_score(rate),
            "retained_starters": len(retained),
            "starter_count": len(unique),
            "retained_rate": round(rate, 4),
            "status": "Automated prior-season starter retention",
            "source": "Footballguys current depth chart + nflverse prior-season roster",
        }
    return output


def load_continuity_priors(**kwargs) -> dict[str, dict[str, Any]]:
    return build_continuity_priors(**kwargs)
