from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from engine.nfl_team_quality import (
    TeamQualityInputs,
    TeamQualityResult,
    calculate_team_quality,
)


DEFAULT_RATINGS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "nfl_team_ratings.json"
)


def load_all_team_ratings(
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
) -> Dict[str, dict]:
    path = Path(ratings_path)

    if not path.exists():
        raise FileNotFoundError(
            f"NFL team ratings file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        ratings = json.load(file)

    if not isinstance(ratings, dict):
        raise ValueError(
            "NFL team ratings file must contain a JSON object."
        )

    return ratings


def load_team_quality(
    team_name: str,
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
) -> TeamQualityResult:
    ratings = load_all_team_ratings(ratings_path)

    if team_name not in ratings:
        available_teams = ", ".join(sorted(ratings.keys()))
        raise KeyError(
            f"Team not found: {team_name}. "
            f"Available teams: {available_teams}"
        )

    team_data = ratings[team_name]

    required_fields = {
        "quarterback",
        "offense",
        "defense",
        "coaching",
        "offensive_line",
        "defensive_line",
        "skill_positions",
        "secondary",
        "special_teams",
        "continuity",
    }

    missing_fields = required_fields - team_data.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"{team_name} is missing required fields: {missing}"
        )

    inputs = TeamQualityInputs(
        quarterback=team_data["quarterback"],
        offense=team_data["offense"],
        defense=team_data["defense"],
        coaching=team_data["coaching"],
        offensive_line=team_data["offensive_line"],
        defensive_line=team_data["defensive_line"],
        skill_positions=team_data["skill_positions"],
        secondary=team_data["secondary"],
        special_teams=team_data["special_teams"],
        continuity=team_data["continuity"],
        injury_adjustment=team_data.get(
            "injury_adjustment",
            0,
        ),
        rookie_adjustment=team_data.get(
            "rookie_adjustment",
            0,
        ),
    )

    return calculate_team_quality(
        team=team_name,
        inputs=inputs,
    )
