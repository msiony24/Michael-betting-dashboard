from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from engine.nfl_coaching import load_coaching_priors
from engine.nfl_continuity import load_continuity_priors
from engine.nfl_team_quality import (
    TeamQualityInputs,
    TeamQualityResult,
    calculate_team_quality,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "nfl_team_ratings.json"
DEFAULT_MADDEN_RATINGS_PATH = PROJECT_ROOT / "data" / "nfl" / "team_ratings_auto.json"

DEFAULT_MADDEN_BLEND_WEIGHT = 1.00


REQUIRED_TEAM_FIELDS = {
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


def _load_json_object(path: Path, label: str) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object.")

    return payload


def load_manual_team_ratings(
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
) -> Dict[str, dict]:
    """Load the existing Macabets NFL team ratings."""
    return _load_json_object(Path(ratings_path), "NFL team ratings")


def load_madden_team_ratings(
    ratings_path: Path | str = DEFAULT_MADDEN_RATINGS_PATH,
) -> Dict[str, dict]:
    """Load Madden-derived team and unit ratings."""
    return _load_json_object(Path(ratings_path), "Madden team ratings")


def _number(value, default: float = 50.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _unit_grade(madden_team: dict, unit_name: str, default: float) -> float:
    units = madden_team.get("units", {})
    if not isinstance(units, dict):
        return float(default)

    unit = units.get(unit_name, {})
    if isinstance(unit, dict):
        return _number(unit.get("grade"), default)

    return _number(unit, default)


def _madden_category_ratings(
    manual_team: dict,
    madden_team: dict,
) -> Dict[str, float]:
    """Translate Madden roster units into the categories Macabets already uses."""
    quarterback = _unit_grade(
        madden_team,
        "quarterback",
        _number(manual_team.get("quarterback")),
    )
    running_backs = _unit_grade(
        madden_team,
        "running_backs",
        _number(manual_team.get("skill_positions")),
    )
    receiving = _unit_grade(
        madden_team,
        "receiving_weapons",
        _number(manual_team.get("skill_positions")),
    )
    offensive_line = _unit_grade(
        madden_team,
        "offensive_line",
        _number(manual_team.get("offensive_line")),
    )
    defensive_front = _unit_grade(
        madden_team,
        "defensive_front",
        _number(manual_team.get("defensive_line")),
    )
    linebackers = _unit_grade(
        madden_team,
        "linebackers",
        _number(manual_team.get("defense")),
    )
    secondary = _unit_grade(
        madden_team,
        "secondary",
        _number(manual_team.get("secondary")),
    )
    special_teams = _unit_grade(
        madden_team,
        "special_teams",
        _number(manual_team.get("special_teams")),
    )

    offense = (
        quarterback * 0.35
        + running_backs * 0.15
        + receiving * 0.25
        + offensive_line * 0.25
    )
    defense = (
        defensive_front * 0.35
        + linebackers * 0.25
        + secondary * 0.40
    )
    skill_positions = running_backs * 0.35 + receiving * 0.65

    return {
        "quarterback": quarterback,
        "offense": offense,
        "defense": defense,
        "offensive_line": offensive_line,
        "defensive_line": defensive_front,
        "skill_positions": skill_positions,
        "secondary": secondary,
        "special_teams": special_teams,
    }


def _blend(manual_value, madden_value, madden_weight: float) -> float:
    weight = min(max(float(madden_weight), 0.0), 1.0)
    manual = _number(manual_value)
    madden = _number(madden_value, manual)
    return round(manual * (1.0 - weight) + madden * weight, 2)


def merge_team_ratings(
    manual_ratings: Dict[str, dict],
    madden_ratings: Dict[str, dict],
    madden_weight: float = DEFAULT_MADDEN_BLEND_WEIGHT,
) -> Dict[str, dict]:
    """
    Merge the audited automated personnel profile into the app-facing team ratings.

    Player-driven categories use the audited Madden + nflverse automated grades
    directly by default. The legacy manual file is fallback/context only. Coaching,
    continuity, and manual injury/rookie adjustments remain controlled there.
    """
    merged: Dict[str, dict] = {}
    coaching_priors = load_coaching_priors()
    continuity_priors = load_continuity_priors()

    for team_name, manual_team in manual_ratings.items():
        if not isinstance(manual_team, dict):
            raise ValueError(f"{team_name} ratings must be a JSON object.")

        missing_fields = REQUIRED_TEAM_FIELDS - manual_team.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{team_name} is missing required fields: {missing}")

        madden_team = madden_ratings.get(team_name)
        if not isinstance(madden_team, dict):
            merged[team_name] = {
                **manual_team,
                "madden_status": "Unavailable",
                "madden_blend_weight": 0.0,
            }
            continue

        madden_categories = _madden_category_ratings(manual_team, madden_team)

        combined = dict(manual_team)
        for category, madden_value in madden_categories.items():
            # Default weight is 1.0: do not let stale manual player grades dilute
            # the audited personnel source. A caller can explicitly request a lower
            # weight for diagnostics, but production uses the automated grade.
            combined[category] = _blend(
                manual_team.get(category),
                madden_value,
                madden_weight,
            )

        coach = coaching_priors.get(team_name, {})
        if coach:
            combined["coaching"] = _number(coach.get("rating"), 70.0)
            combined["head_coach"] = str(coach.get("head_coach") or "")
            combined["coaching_experience_years"] = _number(coach.get("experience_years"), 0.0)
            combined["coaching_2025_record"] = str(coach.get("record_2025") or "--")
            combined["coaching_status"] = str(coach.get("status") or "2026 coaching prior")
            combined["coaching_source"] = str(coach.get("source_url") or "ESPN NFL Coaches")
        else:
            combined["coaching"] = 70.0
            combined["head_coach"] = "Unknown"
            combined["coaching_status"] = "Neutral fallback — current coach data unavailable"
            combined["coaching_source"] = "Neutral fallback"
        continuity = continuity_priors.get(team_name, {})
        if continuity:
            combined["continuity"] = _number(continuity.get("rating"), 67.5)
            combined["continuity_retained_starters"] = int(_number(continuity.get("retained_starters"), 0))
            combined["continuity_starter_count"] = int(_number(continuity.get("starter_count"), 0))
            combined["continuity_retained_rate"] = _number(continuity.get("retained_rate"), 0.0)
            combined["continuity_status"] = str(continuity.get("status") or "Automated roster continuity")
            combined["continuity_source"] = str(continuity.get("source") or "Automated roster continuity")
        else:
            # Neutral fallback: do not revive the legacy subjective continuity grade.
            combined["continuity"] = 67.5
            combined["continuity_retained_starters"] = 0
            combined["continuity_starter_count"] = 0
            combined["continuity_retained_rate"] = 0.0
            combined["continuity_status"] = "Neutral fallback — automated continuity data unavailable"
            combined["continuity_source"] = "Neutral fallback"
        # Automatic Sleeper availability is already reflected by starter replacement
        # inside the audited unit grades. Do not stack the legacy manual team-level
        # injury adjustment on top of the same personnel loss.
        if str(madden_team.get("availability_source", "")).strip().lower() == "sleeper":
            combined["injury_adjustment"] = 0.0
            combined["injury_adjustment_source"] = "Sleeper availability already embedded in personnel units"
        else:
            combined["injury_adjustment"] = _number(manual_team.get("injury_adjustment"), 0.0)
            combined["injury_adjustment_source"] = "Legacy manual fallback"
        combined["availability_source"] = str(madden_team.get("availability_source", "") or "")
        combined["availability_updated_at_utc"] = str(madden_team.get("availability_updated_at_utc", "") or "")
        combined["unavailable_starters"] = int(_number(madden_team.get("unavailable_starters"), 0))
        combined["availability_uncertain"] = int(_number(madden_team.get("availability_uncertain"), 0))
        combined["rookie_adjustment"] = _number(
            manual_team.get("rookie_adjustment"),
            0.0,
        )
        combined["madden_status"] = "Integrated"
        combined["madden_blend_weight"] = round(float(madden_weight), 3)
        combined["madden_roster_grade"] = _number(
            madden_team.get("roster_grade"),
            50.0,
        )

        merged[team_name] = combined

    return merged


def load_all_team_ratings(
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
    madden_ratings_path: Path | str = DEFAULT_MADDEN_RATINGS_PATH,
    use_madden: bool = True,
    madden_weight: float = DEFAULT_MADDEN_BLEND_WEIGHT,
) -> Dict[str, dict]:
    """
    Load Macabets team ratings with the audited automated personnel profile as
    the production source for player-driven categories.

    If the automated file has not been generated yet, Macabets safely falls back
    to the existing manual ratings instead of preventing the app from loading.
    """
    manual_ratings = load_manual_team_ratings(ratings_path)

    if not use_madden:
        return manual_ratings

    madden_path = Path(madden_ratings_path)
    if not madden_path.exists():
        return {
            team: {
                **ratings,
                "madden_status": "File not generated",
                "madden_blend_weight": 0.0,
            }
            for team, ratings in manual_ratings.items()
        }

    madden_ratings = load_madden_team_ratings(madden_path)
    return merge_team_ratings(
        manual_ratings,
        madden_ratings,
        madden_weight=madden_weight,
    )


def load_team_quality(
    team_name: str,
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
    madden_ratings_path: Path | str = DEFAULT_MADDEN_RATINGS_PATH,
    use_madden: bool = True,
    madden_weight: float = DEFAULT_MADDEN_BLEND_WEIGHT,
) -> TeamQualityResult:
    ratings = load_all_team_ratings(
        ratings_path=ratings_path,
        madden_ratings_path=madden_ratings_path,
        use_madden=use_madden,
        madden_weight=madden_weight,
    )

    if team_name not in ratings:
        available_teams = ", ".join(sorted(ratings.keys()))
        raise KeyError(
            f"Team not found: {team_name}. "
            f"Available teams: {available_teams}"
        )

    team_data = ratings[team_name]

    missing_fields = REQUIRED_TEAM_FIELDS - team_data.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{team_name} is missing required fields: {missing}")

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
        injury_adjustment=team_data.get("injury_adjustment", 0),
        rookie_adjustment=team_data.get("rookie_adjustment", 0),
    )

    return calculate_team_quality(
        team=team_name,
        inputs=inputs,
    )
