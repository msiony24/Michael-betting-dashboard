"""NFL scheme and team-tendency intelligence for Macabets.

The snapshot is derived from nflverse play-by-play plus optional FTN charting
and participation data. The prediction adjustment is intentionally small and
uses *behavioral compatibility* with the existing personnel matchup grades,
not a second copy of team-performance strength.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

DEFAULT_SCHEME_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl" / "scheme_tendencies.csv"


def _num(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def _pct(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(numeric) else f"{float(numeric):.1%}"


def _load(path: Path | str = DEFAULT_SCHEME_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _row(frame: pd.DataFrame, team: str) -> pd.Series | None:
    if frame.empty or "team" not in frame.columns:
        return None
    found = frame[frame["team"].astype(str).eq(str(team))]
    return None if found.empty else found.iloc[-1]


def _matchup_edge(personnel: Mapping[str, Any], team: str, label: str) -> float:
    team_key = str(team).strip().lower()
    label_key = str(label).strip().lower()
    for item in personnel.get("matchups", []) or []:
        matchup = str(item.get("Matchup", "")).strip().lower()
        if not matchup.startswith(team_key + " ") or not matchup.endswith(label_key):
            continue
        edge = _num(item.get("Edge"), 0.0)
        leader = str(item.get("Advantage", "Even"))
        if leader == team:
            return abs(edge)
        if leader == "Even":
            return 0.0
        return -abs(edge)
    return 0.0


def _evidence_weight(row: pd.Series | None, target_season: int, week: int | None) -> float:
    if row is None:
        return 0.0
    season = int(_num(row.get("season"), 0))
    if season <= 0 or season > target_season:
        return 0.0
    if season < target_season:
        # Prior-year scheme is useful as a preseason prior, but coaching/personnel
        # changes mean it must never dominate the current matchup.
        return 0.35
    resolved_week = max(1, int(week or _num(row.get("through_week"), 1)))
    return min(1.0, 0.40 + 0.10 * max(0, resolved_week - 1))


def _tendency_label(value: Any, *, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "Unavailable"
    value = float(numeric)
    if value <= low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def _team_summary(team: str, row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {"team": team, "available": False}
    return {
        "team": team,
        "available": True,
        "season": int(_num(row.get("season"), 0)) or None,
        "through_week": int(_num(row.get("through_week"), 0)) or None,
        "pass_rate": _num(row.get("pass_rate"), float("nan")),
        "early_down_pass_rate": _num(row.get("early_down_pass_rate"), float("nan")),
        "neutral_early_down_pass_rate": _num(row.get("neutral_early_down_pass_rate"), float("nan")),
        "plays_per_game": _num(row.get("plays_per_game"), float("nan")),
        "seconds_per_play": _num(row.get("seconds_per_play"), float("nan")),
        "no_huddle_rate": _num(row.get("no_huddle_rate"), float("nan")),
        "motion_rate": _num(row.get("motion_rate"), float("nan")),
        "play_action_rate": _num(row.get("play_action_rate"), float("nan")),
        "rpo_rate": _num(row.get("rpo_rate"), float("nan")),
        "blitz_rate": _num(row.get("blitz_rate"), float("nan")),
        "man_rate": _num(row.get("man_rate"), float("nan")),
        "zone_rate": _num(row.get("zone_rate"), float("nan")),
        "offense_explosive_rate": _num(row.get("offense_explosive_rate"), float("nan")),
        "defense_explosive_allowed": _num(row.get("defense_explosive_allowed"), float("nan")),
        "red_zone_td_rate": _num(row.get("red_zone_td_rate"), float("nan")),
        "red_zone_td_rate_allowed": _num(row.get("red_zone_td_rate_allowed"), float("nan")),
        "pressure_rate": _num(row.get("pressure_rate"), float("nan")),
        "pressure_rate_allowed": _num(row.get("pressure_rate_allowed"), float("nan")),
        "data_source": str(row.get("data_source", "nflverse")),
        "updated_at_utc": str(row.get("updated_at_utc", "")),
    }


def build_scheme_matchup_context(
    *,
    away_team: str,
    home_team: str,
    season: int,
    week: int | None,
    personnel_context: Mapping[str, Any] | None = None,
    snapshot_path: Path | str = DEFAULT_SCHEME_PATH,
) -> dict[str, Any]:
    """Return a small opponent-specific scheme compatibility adjustment.

    Only behavioral tendencies alter the side projection. Explosive, red-zone,
    pressure, and coverage rates are surfaced for football context but are not
    re-awarded as generic team strength here because those signals overlap other
    Macabets layers and will receive dedicated treatment later.
    """
    frame = _load(snapshot_path)
    away_row = _row(frame, away_team)
    home_row = _row(frame, home_team)
    away = _team_summary(away_team, away_row)
    home = _team_summary(home_team, home_row)
    personnel = dict(personnel_context or {})

    if away_row is None or home_row is None:
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "confidence_penalty": 0.0,
            "away": away,
            "home": home,
            "summary": "Scheme tendency data is not available for both teams yet.",
            "guardrail": "No scheme adjustment was applied.",
        }

    away_weight = _evidence_weight(away_row, int(season), week)
    home_weight = _evidence_weight(home_row, int(season), week)
    evidence_weight = min(away_weight, home_weight)

    # Existing personnel matchup gaps are signed from the perspective of the
    # offense named here. Tendencies merely determine how often a team is likely
    # to lean into (or away from) those already-measured matchup paths.
    away_pass_edge = _matchup_edge(personnel, away_team, "passing attack vs secondary")
    home_pass_edge = _matchup_edge(personnel, home_team, "passing attack vs secondary")
    away_run_edge = _matchup_edge(personnel, away_team, "run game vs front seven")
    home_run_edge = _matchup_edge(personnel, home_team, "run game vs front seven")
    away_prot_edge = _matchup_edge(personnel, away_team, "pass protection vs defensive front")
    home_prot_edge = _matchup_edge(personnel, home_team, "pass protection vs defensive front")

    def centered(value: Any, baseline: float, scale: float, cap: float = 1.0) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return 0.0
        return max(-cap, min(cap, (float(numeric) - baseline) / scale))

    # League-center references are intentionally broad, not hard claims about a
    # specific season. They normalize the compatibility signal and the final
    # scoreboard effect is hard-capped below.
    away_pass_lean = centered(away_row.get("early_down_pass_rate"), 0.56, 0.10)
    home_pass_lean = centered(home_row.get("early_down_pass_rate"), 0.56, 0.10)
    away_run_lean = -away_pass_lean
    home_run_lean = -home_pass_lean
    away_blitz = centered(away_row.get("blitz_rate"), 0.25, 0.15)
    home_blitz = centered(home_row.get("blitz_rate"), 0.25, 0.15)

    away_scheme_score = (
        away_pass_lean * away_pass_edge * 0.045
        + away_run_lean * away_run_edge * 0.035
        - home_blitz * away_prot_edge * -0.025
    )
    home_scheme_score = (
        home_pass_lean * home_pass_edge * 0.045
        + home_run_lean * home_run_edge * 0.035
        - away_blitz * home_prot_edge * -0.025
    )
    raw_home = (home_scheme_score - away_scheme_score) * evidence_weight
    adjustment = max(-0.65, min(0.65, raw_home))

    rows = []
    for profile in (away, home):
        rows.append({
            "Team": profile["team"],
            "Pass/Run Identity": _tendency_label(
                profile.get("early_down_pass_rate"), low=0.50, high=0.62,
                low_label="Run-leaning", mid_label="Balanced", high_label="Pass-leaning",
            ),
            "Early-Down Pass": _pct(profile.get("early_down_pass_rate")),
            "Pace": (
                "Fast" if pd.notna(profile.get("seconds_per_play")) and profile.get("seconds_per_play") < 27
                else "Slow" if pd.notna(profile.get("seconds_per_play")) and profile.get("seconds_per_play") > 31
                else "Average" if pd.notna(profile.get("seconds_per_play")) else "Unavailable"
            ),
            "Blitz Rate": _pct(profile.get("blitz_rate")),
            "Man / Zone": (
                f"{_pct(profile.get('man_rate'))} / {_pct(profile.get('zone_rate'))}"
                if pd.notna(profile.get("man_rate")) or pd.notna(profile.get("zone_rate")) else "Unavailable"
            ),
            "Motion": _pct(profile.get("motion_rate")),
            "Play Action": _pct(profile.get("play_action_rate")),
            "RPO": _pct(profile.get("rpo_rate")),
            "Explosive O": _pct(profile.get("offense_explosive_rate")),
            "Explosive Allowed": _pct(profile.get("defense_explosive_allowed")),
            "RZ TD Rate": _pct(profile.get("red_zone_td_rate")),
        })

    leader = "Even"
    if abs(adjustment) >= 0.08:
        leader = home_team if adjustment > 0 else away_team
    strength = "Even" if leader == "Even" else "Slight" if abs(adjustment) < 0.30 else "Moderate"

    if leader == "Even":
        summary = "The teams' current scheme tendencies do not create a meaningful extra matchup edge after personnel compatibility is considered."
    else:
        summary = (
            f"{leader} gets a {strength.lower()} scheme-compatibility edge because its play-calling tendencies are better aligned with the personnel matchups already identified by Macabets."
        )

    source_season = int(_num(away_row.get("season"), 0))
    mode = "current-season" if source_season == int(season) else f"{source_season} preseason prior"
    return {
        "available": True,
        "away": away,
        "home": home,
        "rows": rows,
        "overall_advantage": leader,
        "overall_strength": strength,
        "home_margin_adjustment": round(adjustment, 3),
        "raw_home_margin_adjustment": round(raw_home, 3),
        "evidence_weight": round(evidence_weight, 3),
        "data_mode": mode,
        "summary": summary,
        "guardrail": (
            "Scheme compatibility is capped at ±0.65 scoreboard points and only amplifies or dampens existing opponent-specific personnel matchups. "
            "Explosive, red-zone, pressure, and coverage rates are shown as context but are not double-counted as a second team-strength rating."
        ),
        "source": "nflverse play-by-play + FTN charting/participation when available",
    }
