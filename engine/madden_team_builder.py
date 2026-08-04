from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAYERS_PATH = PROJECT_ROOT / "data" / "madden_26_players.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "madden_26_team_ratings.json"


POSITION_GROUPS = {
    "quarterback": {"QB"},
    "running_backs": {"HB", "RB", "FB"},
    "receiving_weapons": {"WR", "TE"},
    "offensive_line": {"LT", "LG", "C", "RG", "RT", "OL"},
    "defensive_front": {"LE", "RE", "DT", "DE", "DL", "EDGE"},
    "linebackers": {"LOLB", "MLB", "ROLB", "LB"},
    "secondary": {"CB", "FS", "SS", "DB"},
    "special_teams": {"K", "P"},
}


UNIT_DEPTH_LIMITS = {
    "quarterback": 2,
    "running_backs": 4,
    "receiving_weapons": 7,
    "offensive_line": 8,
    "defensive_front": 8,
    "linebackers": 6,
    "secondary": 8,
    "special_teams": 2,
}


UNIT_WEIGHTS = {
    "quarterback": 0.20,
    "running_backs": 0.08,
    "receiving_weapons": 0.15,
    "offensive_line": 0.17,
    "defensive_front": 0.14,
    "linebackers": 0.09,
    "secondary": 0.13,
    "special_teams": 0.04,
}


COLUMN_ALIASES = {
    "player_name": ["player_name", "player", "full_name", "name", "display_name"],
    "team": ["team", "team_name", "club", "team_label"],
    "position": ["position", "pos", "position_name", "position_group"],
    "overall": ["overall", "ovr", "overall_rating", "rating"],
    "speed": ["speed", "spd"],
    "strength": ["strength", "str"],
    "agility": ["agility", "agi"],
    "awareness": ["awareness", "awr"],
    "injury": ["injury", "inj"],
    "change_of_direction": [
        "change_of_direction",
        "cod",
        "change_of-direction",
        "change direction",
    ],
}


TEAM_ALIASES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "GNB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "KAN": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LVR": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NWE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NOR": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "SFO": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TAM": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
}


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {str(column).strip().casefold(): str(column) for column in columns}
    for alias in aliases:
        match = normalized.get(alias.casefold())
        if match:
            return match
    return None


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        match = _find_column(frame.columns, aliases)
        if match:
            rename_map[match] = canonical

    clean = frame.rename(columns=rename_map).copy()

    required = {"player_name", "team", "position", "overall"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(
            "Madden player file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return clean


def _normalize_team(value: object) -> str:
    team = str(value or "").strip()
    if not team:
        return ""
    return TEAM_ALIASES.get(team.upper(), team)


def _normalize_position(value: object) -> str:
    position = str(value or "").strip().upper()
    replacements = {
        "HB": "HB",
        "RB": "RB",
        "DT1": "DT",
        "DT2": "DT",
        "LE": "LE",
        "RE": "RE",
        "LOLB": "LOLB",
        "ROLB": "ROLB",
        "MLB": "MLB",
    }
    return replacements.get(position, position)


def load_madden_players(
    players_path: Path | str = DEFAULT_PLAYERS_PATH,
) -> pd.DataFrame:
    path = Path(players_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Madden player ratings file not found: {path}. "
            "Run the Madden updater first."
        )

    frame = pd.read_csv(path)
    frame = _standardize_columns(frame)

    frame["player_name"] = frame["player_name"].astype(str).str.strip()
    frame["team"] = frame["team"].map(_normalize_team)
    frame["position"] = frame["position"].map(_normalize_position)
    frame["overall"] = pd.to_numeric(frame["overall"], errors="coerce")

    for column in (
        "speed",
        "strength",
        "agility",
        "awareness",
        "injury",
        "change_of_direction",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["overall"])
    frame = frame[
        frame["player_name"].ne("")
        & frame["team"].ne("")
        & frame["position"].ne("")
    ].copy()

    frame["overall"] = frame["overall"].clip(0, 99)
    return frame.reset_index(drop=True)


def _unit_grade(players: pd.DataFrame, unit_name: str) -> dict:
    positions = POSITION_GROUPS[unit_name]
    unit_players = players[players["position"].isin(positions)].copy()
    unit_players = unit_players.sort_values("overall", ascending=False)

    depth_limit = UNIT_DEPTH_LIMITS[unit_name]
    selected = unit_players.head(depth_limit).copy()

    if selected.empty:
        return {
            "grade": 50.0,
            "starter_grade": 50.0,
            "depth_grade": 50.0,
            "player_count": 0,
            "top_players": [],
            "data_status": "missing",
        }

    if unit_name == "quarterback":
        starter_count = 1
    elif unit_name == "special_teams":
        starter_count = min(2, len(selected))
    elif unit_name == "offensive_line":
        starter_count = min(5, len(selected))
    elif unit_name in {"defensive_front", "secondary"}:
        starter_count = min(4, len(selected))
    elif unit_name == "linebackers":
        starter_count = min(3, len(selected))
    elif unit_name == "receiving_weapons":
        starter_count = min(4, len(selected))
    else:
        starter_count = min(2, len(selected))

    starters = selected.head(starter_count)
    depth = selected.iloc[starter_count:]

    starter_grade = float(starters["overall"].mean())
    depth_grade = (
        float(depth["overall"].mean())
        if not depth.empty
        else starter_grade
    )

    if unit_name == "quarterback":
        grade = starter_grade * 0.90 + depth_grade * 0.10
    else:
        grade = starter_grade * 0.82 + depth_grade * 0.18

    top_players = [
        {
            "name": row["player_name"],
            "position": row["position"],
            "overall": round(float(row["overall"]), 1),
        }
        for _, row in selected.head(5).iterrows()
    ]

    return {
        "grade": round(grade, 2),
        "starter_grade": round(starter_grade, 2),
        "depth_grade": round(depth_grade, 2),
        "player_count": int(len(unit_players)),
        "top_players": top_players,
        "data_status": "complete",
    }


def build_team_ratings(players: pd.DataFrame) -> Dict[str, dict]:
    team_ratings: Dict[str, dict] = {}

    for team_name, team_players in players.groupby("team"):
        units = {
            unit_name: _unit_grade(team_players, unit_name)
            for unit_name in POSITION_GROUPS
        }

        roster_grade = sum(
            units[unit_name]["grade"] * UNIT_WEIGHTS[unit_name]
            for unit_name in UNIT_WEIGHTS
        )

        team_ratings[str(team_name)] = {
            "source": "Madden NFL 26",
            "prediction_influence_enabled": False,
            "player_count": int(len(team_players)),
            "roster_grade": round(roster_grade, 2),
            "units": units,
        }

    return dict(sorted(team_ratings.items()))


def save_team_ratings(
    team_ratings: Dict[str, dict],
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(team_ratings, file, indent=2, sort_keys=True)

    return path


def build_and_save_team_ratings(
    players_path: Path | str = DEFAULT_PLAYERS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Dict[str, dict]:
    players = load_madden_players(players_path)
    team_ratings = build_team_ratings(players)
    save_team_ratings(team_ratings, output_path)
    return team_ratings


if __name__ == "__main__":
    ratings = build_and_save_team_ratings()
    print(f"Built Madden unit ratings for {len(ratings)} teams.")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")
