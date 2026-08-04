"""Unified NFL ratings exposed to the Macabets app and prediction engine."""

from __future__ import annotations

from pathlib import Path

from engine.nfl_team_state import TEAM_STATE_WEIGHTS, build_all_team_states
from engine.nfl_foundation import load_foundation_status

NFL_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
]

VENUE_TYPES = ["Outdoor", "Dome", "Retractable roof"]
WEATHER_OPTIONS = ["Normal", "Rain", "Snow", "High wind", "Extreme heat", "Extreme cold"]

# Retain the public name already consumed by app.py.
TEAM_RATING_WEIGHTS = dict(TEAM_STATE_WEIGHTS)

try:
    _TEAM_STATES = build_all_team_states()
except Exception as exc:  # safe app fallback
    _TEAM_STATES = {}
    _LOAD_ERROR = str(exc)
else:
    _LOAD_ERROR = ""

NFL_TEAM_STATES = _TEAM_STATES
NFL_TEAM_RATINGS = {
    team: dict(_TEAM_STATES[team]["components"])
    for team in NFL_TEAMS
    if team in _TEAM_STATES
}

_FOUNDATION_STATUS = load_foundation_status()
_available_states = [state for state in _TEAM_STATES.values() if state.get("season") is not None]
_latest = max(_available_states, key=lambda state: (state.get("season") or 0, state.get("week") or 0), default=None)

NFL_DATA_STATUS = {
    "available": bool(_available_states),
    "teams": len(NFL_TEAM_RATINGS),
    "data_source": (_latest or {}).get("data_source", "manual priors"),
    "season": (_latest or {}).get("season"),
    "through_week": (_latest or {}).get("week"),
    "updated_at_utc": (_latest or {}).get("updated_at_utc"),
    "rating_mode": "unified weekly team state" if _available_states else "manual team-state priors",
    "reason": _LOAD_ERROR or ("Live snapshot not present; manual priors and neutral recent form are active." if not _available_states else ""),
    "foundation_updated_at_utc": _FOUNDATION_STATUS.get("updated_at_utc"),
    "foundation_available_datasets": _FOUNDATION_STATUS.get("available_datasets", 0),
    "foundation_total_datasets": _FOUNDATION_STATUS.get("total_datasets", 0),
    "foundation_requested_season": _FOUNDATION_STATUS.get("requested_season"),
    "foundation_performance_season": _FOUNDATION_STATUS.get("performance_season"),
}
