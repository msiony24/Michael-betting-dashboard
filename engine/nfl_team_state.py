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

# Production team-power weights deliberately favor independent personnel units.
# Aggregate offense/defense remain available to the reasoning layer, but receive
# small weights here because QB/OL/skill and DL/secondary already carry much of
# the same information. This avoids counting one season of NFL performance
# several times in the base team-power number. Weights sum to 1.00.
TEAM_STATE_WEIGHTS: Mapping[str, float] = {
    "quarterback": 0.22,
    "offensive_line": 0.14,
    "skill_positions": 0.12,
    "defensive_line": 0.14,
    "secondary": 0.13,
    "offense": 0.03,
    "defense": 0.04,
    "recent_form": 0.04,
    "coaching": 0.07,
    "continuity": 0.04,
    "special_teams": 0.03,
}

PRIOR_SEASON_PERFORMANCE_WEIGHT = 0.0
CURRENT_SEASON_START_WEIGHT = 0.20
CURRENT_SEASON_WEEK_STEP = 0.06
CURRENT_SEASON_MAX_WEIGHT = 0.85

# The automated rating engine already blends current/prior NFL performance into
# QB, OL, DL, secondary, special teams and the derived offense/defense grades.
# Re-blending those same team_snapshot fields here would count the same evidence
# twice. Team-state keeps only recent form as a separate live snapshot signal.
LIVE_COMPONENTS = {"recent_form"}


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



def performance_evidence_weight(
    snapshot_season: int | None,
    through_week: int | None,
    *,
    target_season: int,
) -> float:
    """Return how much team-level NFL performance may influence a component.

    Prior-season team snapshots are not blended again here because the personnel layer already carries a capped prior-season performance adjustment. Current-season evidence
    starts conservatively and earns more weight as games accumulate. Future or
    invalid snapshots receive no weight.
    """
    if not snapshot_season or snapshot_season > target_season:
        return 0.0
    if snapshot_season < target_season:
        return PRIOR_SEASON_PERFORMANCE_WEIGHT
    week = max(0, int(through_week or 0))
    if week <= 0:
        return 0.0
    return min(
        CURRENT_SEASON_MAX_WEIGHT,
        CURRENT_SEASON_START_WEIGHT + max(0, week - 1) * CURRENT_SEASON_WEEK_STEP,
    )


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

    # The requested/current NFL season is the calendar year used by this build.
    # A prior-season snapshot is context only; it must not replace the audited
    # Madden/depth-chart personnel baseline.
    from datetime import datetime, timezone
    target_season = datetime.now(timezone.utc).year
    snapshot_season = None
    snapshot_week = None
    if row is not None:
        season_num = pd.to_numeric(row.get("season"), errors="coerce")
        week_num = pd.to_numeric(row.get("through_week"), errors="coerce")
        snapshot_season = int(season_num) if pd.notna(season_num) else None
        snapshot_week = int(week_num) if pd.notna(week_num) else None
    evidence_weight = performance_evidence_weight(
        snapshot_season, snapshot_week, target_season=target_season
    )

    for component in TEAM_STATE_WEIGHTS:
        prior_value = _score(prior.get(component, 67.5))
        live_value = row.get(component) if row is not None and component in row else None

        # Every non-recent-form component arrives here already carrying the
        # audited Madden/depth-chart baseline plus the appropriate NFL
        # performance blend from nfl_rating_engine.py. Keep that value intact so
        # current-season team performance is not counted a second time.
        if component != "recent_form":
            components[component] = prior_value
            sources[component] = "automated rating baseline"
            continue

        # Recent form is intentionally current-season only and is the one live
        # team_snapshot signal that is not already embedded in the rating engine.
        # Start from neutral so prior-season Week 18 momentum cannot leak forward.
        if (
            snapshot_season == target_season
            and live_value is not None
            and pd.notna(live_value)
            and evidence_weight > 0
        ):
            live_score = _score(live_value, 67.5)
            components[component] = round(
                67.5 * (1.0 - evidence_weight) + live_score * evidence_weight, 2
            )
            sources[component] = (
                f"{1.0-evidence_weight:.0%} neutral baseline + "
                f"{evidence_weight:.0%} current-season recent form"
            )
        else:
            components[component] = 67.5
            sources[component] = "neutral preseason baseline"
            if snapshot_season == target_season:
                warnings.append("recent_form is using neutral preseason baseline")

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
