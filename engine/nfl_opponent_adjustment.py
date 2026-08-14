from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = ROOT / "data" / "nfl" / "team_snapshot.csv"
OPPONENT_ADJUSTMENT_CAP = 0.40


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile(frame: pd.DataFrame, team: str) -> dict[str, Any] | None:
    rows = frame[frame["team"].astype(str).eq(team)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {column: row.get(column) for column in frame.columns}


def _evidence_weight(row: dict[str, Any], season: int, week: int | None) -> float:
    source_season = int(_num(row.get("season"), 0) or 0)
    if source_season != int(season):
        # Prior-season opponent quality is useful only as a small preseason prior.
        return 0.20
    resolved_week = int(week or _num(row.get("through_week"), 1) or 1)
    # Current-season data hard-switches in immediately, but one or two games are
    # still treated as a small sample rather than full-strength evidence.
    return min(1.0, 0.25 + 0.13 * max(0, resolved_week - 1))


def build_opponent_adjusted_context(
    *,
    away_team: str,
    home_team: str,
    season: int,
    week: int | None,
    snapshot_path: Path | str = DEFAULT_SNAPSHOT,
) -> dict[str, Any]:
    """Apply a small credit/debit for the quality of opponents already faced.

    The base Macabets power rating already rewards what a team produced. This
    layer does *not* score that production again. It only asks whether those
    results came against stronger or weaker offensive/defensive opposition than
    league average, then applies a heavily capped correction.
    """
    path = Path(snapshot_path)
    if not path.exists():
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Opponent-adjusted NFL performance data is not available yet.",
        }

    frame = pd.read_csv(path)
    required = {
        "team",
        "offense_epa_per_play",
        "defense_epa_allowed",
        "sos_opponent_offense_epa",
        "sos_opponent_defense_epa_allowed",
        "opponent_quality_epa",
        "opponent_adjusted_net_epa",
    }
    if frame.empty or not required.issubset(frame.columns):
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Opponent-adjusted performance data is incomplete. Run the NFL data refresh once after this update.",
        }

    away = _profile(frame, away_team)
    home = _profile(frame, home_team)
    if away is None or home is None:
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Opponent-adjusted performance data is not yet available for both teams in this matchup.",
        }

    away_quality = _num(away.get("opponent_quality_epa"), 0.0)
    home_quality = _num(home.get("opponent_quality_epa"), 0.0)
    raw_quality_gap = home_quality - away_quality

    evidence_weight = min(
        _evidence_weight(away, season, week),
        _evidence_weight(home, season, week),
    )

    # Roughly 0.10 EPA/play of opponent-quality separation is meaningful, but
    # this remains a refinement rather than a second team-strength model.
    raw_home_adjustment = raw_quality_gap * 3.0
    adjustment = max(
        -OPPONENT_ADJUSTMENT_CAP,
        min(OPPONENT_ADJUSTMENT_CAP, raw_home_adjustment * evidence_weight),
    )

    if abs(adjustment) < 0.04:
        leader = "Even"
        strength = "Even"
    else:
        leader = home_team if adjustment > 0 else away_team
        strength = "Slight" if abs(adjustment) < 0.20 else "Moderate"

    if leader == "Even":
        summary = (
            "After accounting for the quality of opponents already faced, neither team earns a meaningful extra performance credit."
        )
    else:
        summary = (
            f"{leader} gets the opponent-adjusted edge because its results came against the stronger overall slate of offenses and defenses faced. "
            "The adjustment is intentionally small so schedule quality cannot replace actual team performance."
        )

    def output(team: str, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "team": team,
            "opponent_offense_epa": _num(row.get("sos_opponent_offense_epa")),
            "opponent_defense_epa_allowed": _num(row.get("sos_opponent_defense_epa_allowed")),
            "opponent_quality_epa": _num(row.get("opponent_quality_epa")),
            "raw_net_epa": _num(row.get("offense_epa_per_play")) - _num(row.get("defense_epa_allowed")),
            "opponent_adjusted_net_epa": _num(row.get("opponent_adjusted_net_epa")),
            "season": int(_num(row.get("season"), 0) or 0),
            "through_week": int(_num(row.get("through_week"), 0) or 0),
        }

    return {
        "available": True,
        "away": output(away_team, away),
        "home": output(home_team, home),
        "overall_advantage": leader,
        "overall_strength": strength,
        "home_margin_adjustment": round(float(adjustment), 3),
        "evidence_weight": round(float(evidence_weight), 3),
        "summary": summary,
        "source": "nflverse regular-season play-by-play",
        "guardrail": (
            "Opponent adjustment is capped at ±0.40 points. Base power already scores team performance; this layer only gives a small correction for the quality of offenses and defenses already faced. Future schedule difficulty never receives a probability bonus."
        ),
    }
