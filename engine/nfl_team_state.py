"""Unified weekly NFL team-state profiles for Macabets.

This module merges the automated nflverse snapshot with the richer manual team
priors already stored in ``data/nfl_team_ratings.json``. It is intentionally
separate from the prediction engine so the data blend can be tested before it
changes any picks, fair spreads, or moneylines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from engine.nfl_ratings_loader import DEFAULT_RATINGS_PATH, load_all_team_ratings

DEFAULT_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "nfl" / "team_snapshot.csv"
)

# The order reflects the user's stated NFL philosophy: quarterback first,
# followed by offensive line, defense, recent form, coaching, and supporting
# units. Weights sum to 1.00.
TEAM_STATE_WEIGHTS: Mapping[str, float] = {
    "quarterback": 0.25,
    "offensive_line": 0.14,
    "defense": 0.16,
    "offense": 0.13,
    "recent_form": 0.12,
    "coaching": 0.08,
    "defensive_line": 0.04,
    "secondary": 0.025,
    "skill_positions": 0.025,
    "special_teams": 0.015,
    "continuity": 0.015,
}

LIVE_COMPONENTS = {
    "quarterback",
    "offense",
    "defense",
    "offensive_line",
    "defensive_line",
    "secondary",
    "special_teams",
    "recent_form",
}


@dataclass(frozen=True)
class NFLTeamState:
    team: str
    season: int | None
    week: int | None
    overall_rating: float
    base_rating: float
    injury_adjustment: float
    rookie_adjustment: float
    components: dict[str, float]
    component_sources: dict[str, str]
    data_source: str
    updated_at_utc: str | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(value: Any, fallback: float = 67.5) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        numeric = fallback
    return round(max(0.0, min(float(numeric), 100.0)), 2)


def _load_snapshot(path: Path | str) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(snapshot_path)
    if "team" not in frame.columns:
        return pd.DataFrame()
    return frame


def build_team_state(
    team: str,
    *,
    snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
    injury_adjustment: float | None = None,
) -> NFLTeamState:
    """Build one current team profile without changing prediction logic.

    Automated values replace manual priors only when the snapshot contains a
    valid rating for that component. Missing live fields safely retain their
    transparent prior instead of receiving invented values.
    """
    priors = load_all_team_ratings(ratings_path)
    if team not in priors:
        raise KeyError(f"No NFL prior profile exists for {team}.")

    prior = priors[team]
    snapshot = _load_snapshot(snapshot_path)
    row: pd.Series | None = None
    if not snapshot.empty:
        matches = snapshot[snapshot["team"].astype(str) == team]
        if not matches.empty:
            row = matches.iloc[-1]

    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []

    for component in TEAM_STATE_WEIGHTS:
        prior_value = prior.get(component, 67.5)
        live_value = row.get(component) if row is not None and component in row else None
        if component in LIVE_COMPONENTS and live_value is not None and pd.notna(live_value):
            components[component] = _score(live_value, _score(prior_value))
            sources[component] = "nflverse snapshot"
        else:
            # Recent form has no meaningful static prior. Neutral is safer until
            # the upgraded workflow has produced the field.
            fallback = 67.5 if component == "recent_form" else prior_value
            components[component] = _score(fallback)
            sources[component] = "neutral fallback" if component == "recent_form" else "manual prior"
            if component in LIVE_COMPONENTS:
                warnings.append(f"{component} is using {sources[component]}")

    base_rating = sum(
        components[name] * weight for name, weight in TEAM_STATE_WEIGHTS.items()
    )
    stored_injury = float(prior.get("injury_adjustment", 0.0))
    applied_injury = stored_injury if injury_adjustment is None else float(injury_adjustment)
    rookie_adjustment = float(prior.get("rookie_adjustment", 0.0))
    overall = max(0.0, min(100.0, base_rating + applied_injury + rookie_adjustment))

    season = None
    week = None
    updated_at = None
    data_source = "manual priors"
    if row is not None:
        season_value = pd.to_numeric(row.get("season"), errors="coerce")
        week_value = pd.to_numeric(row.get("through_week"), errors="coerce")
        season = int(season_value) if pd.notna(season_value) else None
        week = int(week_value) if pd.notna(week_value) else None
        updated_at = str(row.get("updated_at_utc")) if pd.notna(row.get("updated_at_utc")) else None
        data_source = str(row.get("data_source") or "nflverse snapshot")

    return NFLTeamState(
        team=team,
        season=season,
        week=week,
        overall_rating=round(overall, 2),
        base_rating=round(base_rating, 2),
        injury_adjustment=round(applied_injury, 2),
        rookie_adjustment=round(rookie_adjustment, 2),
        components=components,
        component_sources=sources,
        data_source=data_source,
        updated_at_utc=updated_at,
        warnings=sorted(set(warnings)),
    )


def build_all_team_states(
    *,
    snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
) -> dict[str, dict[str, Any]]:
    priors = load_all_team_ratings(ratings_path)
    return {
        team: build_team_state(
            team,
            snapshot_path=snapshot_path,
            ratings_path=ratings_path,
        ).to_dict()
        for team in sorted(priors)
    }
