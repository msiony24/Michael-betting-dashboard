from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEAM_RATINGS = PROJECT_ROOT / "data" / "nfl" / "team_ratings_auto.json"
DEFAULT_MADDEN_RATINGS = PROJECT_ROOT / "data" / "madden_27_team_ratings.json"

MATCHUP_WEIGHTS = {
    "Passing attack vs secondary": 0.45,
    "Pass protection vs defensive front": 0.30,
    "Run game vs front seven": 0.25,
}


def _number(value: Any, default: float = 67.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _unit(team_data: dict[str, Any], name: str) -> dict[str, Any]:
    units = team_data.get("units") if isinstance(team_data, dict) else None
    unit = units.get(name) if isinstance(units, dict) else None
    if not isinstance(unit, dict):
        return {"grade": 67.5, "source": "neutral fallback", "top_players": []}
    return {
        "grade": _number(unit.get("grade")),
        "source": str(unit.get("source") or "Madden 27 roster rating"),
        "top_players": list(unit.get("top_players") or []),
    }


def _attack_grades(team_data: dict[str, Any]) -> dict[str, float]:
    qb = _unit(team_data, "quarterback")["grade"]
    rec = _unit(team_data, "receiving_weapons")["grade"]
    rb = _unit(team_data, "running_backs")["grade"]
    ol = _unit(team_data, "offensive_line")["grade"]
    return {
        "pass": qb * 0.45 + rec * 0.55,
        "protection": ol,
        "run": rb * 0.35 + ol * 0.65,
    }


def _defense_grades(team_data: dict[str, Any]) -> dict[str, float]:
    front = _unit(team_data, "defensive_front")["grade"]
    lb = _unit(team_data, "linebackers")["grade"]
    secondary = _unit(team_data, "secondary")["grade"]
    return {
        "secondary": secondary,
        "front": front,
        "front_seven": front * 0.70 + lb * 0.30,
    }


def _strength(edge: float) -> str:
    magnitude = abs(float(edge))
    if magnitude < 2.5:
        return "Even"
    if magnitude < 5.0:
        return "Slight"
    if magnitude < 9.0:
        return "Moderate"
    return "Strong"


def _matchup_row(label: str, offense_team: str, defense_team: str, attack: float, defense: float, source: str) -> dict[str, Any]:
    edge = float(attack) - float(defense)
    strength = _strength(edge)
    advantage = "Even" if strength == "Even" else (offense_team if edge > 0 else defense_team)
    return {
        "Matchup": f"{offense_team} {label}",
        "Advantage": advantage,
        "Strength": strength,
        "Edge": round(abs(edge), 1),
        "Attack Grade": round(float(attack), 1),
        "Defense Grade": round(float(defense), 1),
        "Source": source,
    }


def _source_summary(team_data: dict[str, Any]) -> str:
    units = team_data.get("units", {}) if isinstance(team_data, dict) else {}
    weights = [
        _number(v.get("performance_weight"), 0.0)
        for v in units.values() if isinstance(v, dict)
    ]
    average = sum(weights) / len(weights) if weights else 0.0
    if average >= 0.60:
        return "NFL-performance heavy"
    if average >= 0.30:
        return "Balanced Madden + NFL performance"
    if average > 0.0:
        return "Madden-heavy + prior NFL performance"
    return "Madden-heavy personnel baseline"


def build_personnel_matchup_context(
    *,
    away_team: str,
    home_team: str,
    week: int | None = None,
    team_ratings_path: Path | str = DEFAULT_TEAM_RATINGS,
    madden_ratings_path: Path | str = DEFAULT_MADDEN_RATINGS,
) -> dict[str, Any]:
    automated = _load_json(Path(team_ratings_path))
    baseline = _load_json(Path(madden_ratings_path))
    ratings = automated or baseline
    if not ratings:
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Personnel ratings are not available yet.",
            "matchups": [],
        }

    away_data = ratings.get(away_team)
    home_data = ratings.get(home_team)
    if not isinstance(away_data, dict) or not isinstance(home_data, dict):
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Personnel ratings are incomplete for this matchup.",
            "matchups": [],
        }

    away_attack = _attack_grades(away_data)
    home_attack = _attack_grades(home_data)
    away_def = _defense_grades(away_data)
    home_def = _defense_grades(home_data)

    away_source = _source_summary(away_data)
    home_source = _source_summary(home_data)
    common_source = away_source if away_source == home_source else "Blended team-specific Madden/NFL inputs"

    rows = [
        _matchup_row("passing attack vs secondary", away_team, home_team, away_attack["pass"], home_def["secondary"], common_source),
        _matchup_row("pass protection vs defensive front", away_team, home_team, away_attack["protection"], home_def["front"], common_source),
        _matchup_row("run game vs front seven", away_team, home_team, away_attack["run"], home_def["front_seven"], common_source),
        _matchup_row("passing attack vs secondary", home_team, away_team, home_attack["pass"], away_def["secondary"], common_source),
        _matchup_row("pass protection vs defensive front", home_team, away_team, home_attack["protection"], away_def["front"], common_source),
        _matchup_row("run game vs front seven", home_team, away_team, home_attack["run"], away_def["front_seven"], common_source),
    ]

    def signed_for(team: str, row: dict[str, Any]) -> float:
        if row["Advantage"] == "Even":
            return 0.0
        magnitude = float(row["Edge"])
        return magnitude if row["Advantage"] == team else -magnitude

    away_rows = rows[:3]
    home_rows = rows[3:]
    weights = list(MATCHUP_WEIGHTS.values())
    away_composite = sum(signed_for(away_team, row) * weight for row, weight in zip(away_rows, weights))
    home_composite = sum(signed_for(home_team, row) * weight for row, weight in zip(home_rows, weights))

    # Personnel is deliberately a secondary adjustment because the base NFL model
    # already contains team performance. Hard cap prevents double counting.
    net_home_edge = home_composite - away_composite
    home_margin_adjustment = max(-1.5, min(1.5, net_home_edge * 0.06))

    strongest = max(rows, key=lambda row: float(row["Edge"])) if rows else None
    away_wins = sum(row["Advantage"] == away_team for row in rows)
    home_wins = sum(row["Advantage"] == home_team for row in rows)
    leader = home_team if home_wins > away_wins else away_team if away_wins > home_wins else "Even"

    week_text = f"Week {int(week)}" if week is not None else "Current season"
    return {
        "available": True,
        "home_margin_adjustment": round(home_margin_adjustment, 2),
        "adjustment_cap": 1.5,
        "leader": leader,
        "away_advantages": away_wins,
        "home_advantages": home_wins,
        "even_matchups": sum(row["Advantage"] == "Even" for row in rows),
        "matchups": rows,
        "strongest_edge": strongest,
        "data_mode": common_source,
        "summary": (
            f"{week_text}: Madden 27 supplies the roster/talent baseline. As current NFL data accumulates, "
            "Macabets' automated player and unit ratings shift toward real performance, with current-season "
            "player production allowed to reach 80% of a skill player's rating."
        ),
    }
