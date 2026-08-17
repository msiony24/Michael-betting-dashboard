from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = ROOT / "data" / "nfl" / "team_snapshot.csv"
SITUATIONAL_ADJUSTMENT_CAP = 0.35


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rank(frame: pd.DataFrame, column: str, *, higher: bool = True) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(50.0, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    if not higher:
        values = -values
    return values.rank(pct=True).fillna(0.5) * 100.0


def _profile(frame: pd.DataFrame, team: str) -> dict[str, Any] | None:
    rows = frame[frame["team"].astype(str).eq(team)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {column: row.get(column) for column in frame.columns}


def _evidence_weight(row: dict[str, Any], season: int, week: int | None) -> float:
    source_season = int(_num(row.get("season"), 0) or 0)
    if source_season != int(season):
        # Prior-season situational splits are useful context, but are deliberately
        # weak because these rates are much noisier than core team quality.
        return 0.15
    resolved_week = int(week or _num(row.get("through_week"), 1) or 1)
    return min(1.0, 0.25 + 0.12 * max(0, resolved_week - 1))


def _edge_label(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude < 3.0:
        return "Even"
    if magnitude < 8.0:
        return "Slight"
    return "Clear"


def build_situational_matchup_context(
    *,
    away_team: str,
    home_team: str,
    season: int,
    week: int | None,
    snapshot_path: Path | str = DEFAULT_SNAPSHOT,
) -> dict[str, Any]:
    """Evaluate high-leverage execution without allowing noisy splits to dominate.

    The side adjustment is intentionally tiny and capped. Only third down, red
    zone and close-fourth-quarter EPA are scored here. Turnover and explosive-play
    rates remain visible for audit/context but are not re-awarded because they
    already contribute to the core offense/defense ratings.
    """
    path = Path(snapshot_path)
    if not path.exists():
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Situational NFL performance data is not available yet.",
        }

    frame = pd.read_csv(path)
    required = {
        "team",
        "third_down_conversion_rate",
        "third_down_conversion_allowed",
        "red_zone_td_rate",
        "red_zone_td_rate_allowed",
        "offense_turnover_rate",
        "defense_takeaway_rate",
        "offense_explosive_rate",
        "defense_explosive_allowed",
        "high_leverage_epa",
        "high_leverage_epa_allowed",
    }
    if frame.empty or not required.issubset(frame.columns):
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Situational NFL performance data is incomplete. Run the NFL data refresh once after this update.",
        }

    away = _profile(frame, away_team)
    home = _profile(frame, home_team)
    if away is None or home is None:
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "summary": "Situational performance data is not yet available for both teams in this matchup.",
        }

    grades = pd.DataFrame({"team": frame["team"]})
    grades["third_offense"] = _rank(frame, "third_down_conversion_rate", higher=True)
    grades["third_defense"] = _rank(frame, "third_down_conversion_allowed", higher=False)
    grades["rz_offense"] = _rank(frame, "red_zone_td_rate", higher=True)
    grades["rz_defense"] = _rank(frame, "red_zone_td_rate_allowed", higher=False)
    grades["turnover_offense"] = _rank(frame, "offense_turnover_rate", higher=False)
    grades["takeaway_defense"] = _rank(frame, "defense_takeaway_rate", higher=True)
    grades["explosive_offense"] = _rank(frame, "offense_explosive_rate", higher=True)
    grades["explosive_defense"] = _rank(frame, "defense_explosive_allowed", higher=False)
    grades["leverage_offense"] = _rank(frame, "high_leverage_epa", higher=True)
    grades["leverage_defense"] = _rank(frame, "high_leverage_epa_allowed", higher=False)
    table = grades.set_index("team")

    def side(offense: str, defense: str) -> dict[str, float]:
        third = float(table.loc[offense, "third_offense"] - table.loc[defense, "third_defense"])
        red_zone = float(table.loc[offense, "rz_offense"] - table.loc[defense, "rz_defense"])
        leverage = float(table.loc[offense, "leverage_offense"] - table.loc[defense, "leverage_defense"])
        turnovers = float(table.loc[offense, "turnover_offense"] - table.loc[defense, "takeaway_defense"])
        explosives = float(table.loc[offense, "explosive_offense"] - table.loc[defense, "explosive_defense"])
        combined = third * 0.35 + red_zone * 0.35 + leverage * 0.30
        return {
            "third_down_edge": third,
            "red_zone_edge": red_zone,
            "high_leverage_edge": leverage,
            "turnover_edge": turnovers,
            "explosive_edge": explosives,
            "combined": combined,
        }

    away_side = side(away_team, home_team)
    home_side = side(home_team, away_team)
    raw_home = (home_side["combined"] - away_side["combined"]) * 0.008
    evidence_weight = min(
        _evidence_weight(away, season, week),
        _evidence_weight(home, season, week),
    )
    adjustment = max(
        -SITUATIONAL_ADJUSTMENT_CAP,
        min(SITUATIONAL_ADJUSTMENT_CAP, raw_home * evidence_weight),
    )

    if abs(adjustment) < 0.04:
        leader = "Even"
        strength = "Even"
    else:
        leader = home_team if adjustment > 0 else away_team
        strength = "Slight" if abs(adjustment) < 0.18 else "Moderate"

    # Build a human-readable reason from the largest opponent-specific matchup
    # edges. These are conclusions, while the raw numbers remain audit-only.
    differential = {
        "third downs": home_side["third_down_edge"] - away_side["third_down_edge"],
        "red-zone execution": home_side["red_zone_edge"] - away_side["red_zone_edge"],
        "close-game execution": home_side["high_leverage_edge"] - away_side["high_leverage_edge"],
    }
    top = sorted(differential.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
    if leader == "Even":
        summary = "Neither team has a meaningful overall situational-execution edge; the high-leverage strengths largely offset each other."
    else:
        drivers = [name for name, value in top if (value > 0) == (leader == home_team)]
        if drivers:
            driver_text = " and ".join(drivers)
            summary = f"{leader} has the situational edge, driven mainly by {driver_text}. The adjustment stays small because these outcomes are volatile."
        else:
            summary = f"{leader} has a small overall situational-execution edge, but the individual high-leverage categories are mixed."

    def output(team: str, row: dict[str, Any], side_row: dict[str, float]) -> dict[str, Any]:
        return {
            "team": team,
            "third_down_conversion_rate": _num(row.get("third_down_conversion_rate")),
            "third_down_conversion_allowed": _num(row.get("third_down_conversion_allowed")),
            "red_zone_td_rate": _num(row.get("red_zone_td_rate")),
            "red_zone_td_rate_allowed": _num(row.get("red_zone_td_rate_allowed")),
            "offense_turnover_rate": _num(row.get("offense_turnover_rate")),
            "defense_takeaway_rate": _num(row.get("defense_takeaway_rate")),
            "offense_explosive_rate": _num(row.get("offense_explosive_rate")),
            "defense_explosive_allowed": _num(row.get("defense_explosive_allowed")),
            "high_leverage_epa": _num(row.get("high_leverage_epa")),
            "high_leverage_epa_allowed": _num(row.get("high_leverage_epa_allowed")),
            "matchup_edges": {key: round(float(value), 1) for key, value in side_row.items() if key != "combined"},
            "season": int(_num(row.get("season"), 0) or 0),
            "through_week": int(_num(row.get("through_week"), 0) or 0),
        }

    return {
        "available": True,
        "away": output(away_team, away, away_side),
        "home": output(home_team, home, home_side),
        "overall_advantage": leader,
        "overall_strength": strength,
        "home_margin_adjustment": round(float(adjustment), 3),
        "evidence_weight": round(float(evidence_weight), 3),
        "summary": summary,
        "source": "nflverse regular-season play-by-play",
        "guardrail": (
            "Situational execution is capped at ±0.35 points and receives reduced weight in small samples. "
            "Only third down, red zone and close-fourth-quarter execution affect the side projection here; turnover and explosive-play rates are context-only because they already live in core team quality."
        ),
    }
