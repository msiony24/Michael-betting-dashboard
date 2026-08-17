from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import math
import re
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_PATH = ROOT / "data" / "ufc" / "fighter_profiles.csv"
CONTEXT_VERSION = "Macabets UFC Physical & Context v0.1"


@dataclass(frozen=True)
class UFCContextConfig:
    max_probability_adjustment: float = 0.02
    max_physical_adjustment: float = 0.009
    max_age_adjustment: float = 0.010
    max_activity_adjustment: float = 0.005
    full_profile_reliability: float = 0.85


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _parse_inches(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"--", "nan", "None"}:
        return None
    feet_match = re.search(r"(\d+)\s*'\s*(\d+)", text)
    if feet_match:
        return float(int(feet_match.group(1)) * 12 + int(feet_match.group(2)))
    number = re.search(r"\d+(?:\.\d+)?", text)
    return float(number.group()) if number else None


def _parse_weight(value: Any) -> float | None:
    text = str(value or "").strip()
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _parse_dob(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() in {"", "--", "nan", "None"}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def load_fighter_profiles(path: Path | str | None = None) -> pd.DataFrame:
    profile_path = Path(path) if path is not None else DEFAULT_PROFILES_PATH
    if not profile_path.exists() or profile_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(profile_path, low_memory=False)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if frame.empty:
        return frame
    rename = {
        "name": "fighter",
        "url": "fighter_url",
        "dob": "dob",
        "weight": "weight",
        "reach": "reach",
        "height": "height",
        "stance": "stance",
        "nickname": "nickname",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    if "fighter" not in frame.columns:
        return pd.DataFrame()
    frame["fighter"] = frame["fighter"].astype(str).str.strip()
    frame["fighter_key"] = frame["fighter"].map(_normalize_name)
    return frame.drop_duplicates(subset=["fighter_key"], keep="last").reset_index(drop=True)


def _profile_row(profiles: pd.DataFrame, fighter: str) -> dict[str, Any]:
    if profiles is None or profiles.empty or "fighter_key" not in profiles.columns:
        return {}
    match = profiles.loc[profiles["fighter_key"] == _normalize_name(fighter)]
    if match.empty:
        return {}
    row = match.iloc[-1]
    return {
        "fighter": fighter,
        "dob": None if pd.isna(row.get("dob")) else str(row.get("dob")),
        "height_inches": _parse_inches(row.get("height")),
        "reach_inches": _parse_inches(row.get("reach")),
        "weight_lbs": _parse_weight(row.get("weight")),
        "stance": "" if pd.isna(row.get("stance")) else str(row.get("stance") or "").strip(),
        "fighter_url": "" if pd.isna(row.get("fighter_url")) else str(row.get("fighter_url") or "").strip(),
    }


def _age_on(dob: pd.Timestamp | None, as_of: date) -> float | None:
    if dob is None:
        return None
    born = dob.date()
    return (as_of - born).days / 365.2425


def _decline_index(age: float | None) -> float | None:
    if age is None:
        return None
    # No automatic youth bonus. The context layer only applies a modest decline
    # signal once a fighter is beyond the typical prime window.
    return max(0.0, age - 32.0) + 0.75 * max(0.0, age - 36.0)


def _recent_activity(fights: pd.DataFrame, fighter: str, as_of: date) -> dict[str, Any]:
    if fights is None or fights.empty:
        return {"days_since_last_fight": None, "fights_180d": 0, "fights_365d": 0, "recent_divisions": []}
    frame = fights.copy()
    if "event_date" not in frame.columns or "fighter" not in frame.columns:
        return {"days_since_last_fight": None, "fights_180d": 0, "fights_365d": 0, "recent_divisions": []}
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    rows = frame.loc[
        frame["fighter"].astype(str).map(_normalize_name).eq(_normalize_name(fighter))
        & frame["event_date"].notna()
        & (frame["event_date"].dt.date <= as_of)
    ].sort_values("event_date", ascending=False)
    if rows.empty:
        return {"days_since_last_fight": None, "fights_180d": 0, "fights_365d": 0, "recent_divisions": []}
    last_date = rows.iloc[0]["event_date"].date()
    days = max(0, (as_of - last_date).days)
    deltas = (pd.Timestamp(as_of) - rows["event_date"]).dt.days
    divisions = []
    if "division" in rows.columns:
        divisions = [str(v) for v in rows["division"].dropna().astype(str).head(4).tolist() if str(v).strip()]
    return {
        "days_since_last_fight": int(days),
        "fights_180d": int((deltas <= 180).sum()),
        "fights_365d": int((deltas <= 365).sum()),
        "recent_divisions": divisions,
    }


def _current_division(ratings: pd.DataFrame, fighter: str) -> str:
    if ratings is None or ratings.empty or "fighter" not in ratings.columns:
        return ""
    match = ratings.loc[ratings["fighter"].astype(str).map(_normalize_name).eq(_normalize_name(fighter))]
    if match.empty:
        return ""
    return str(match.iloc[-1].get("division", "") or "")


def _division_transition(activity: dict[str, Any], current_division: str) -> bool:
    recent = [d for d in activity.get("recent_divisions", []) if d and "Catch" not in d]
    if not recent or not current_division:
        return False
    return any(d != current_division for d in recent[:3])


def build_fight_context(
    fighter_a: str,
    fighter_b: str,
    ratings: pd.DataFrame,
    fights: pd.DataFrame,
    *,
    profiles: pd.DataFrame | None = None,
    rounds: int = 3,
    fight_date: date | datetime | str | None = None,
    config: UFCContextConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCContextConfig()
    if fight_date is None:
        as_of = date.today()
    elif isinstance(fight_date, datetime):
        as_of = fight_date.date()
    elif isinstance(fight_date, date):
        as_of = fight_date
    else:
        parsed = pd.to_datetime(fight_date, errors="coerce")
        as_of = date.today() if pd.isna(parsed) else parsed.date()

    profiles = load_fighter_profiles() if profiles is None else profiles.copy()
    if not profiles.empty and "fighter_key" not in profiles.columns and "fighter" in profiles.columns:
        profiles["fighter_key"] = profiles["fighter"].map(_normalize_name)

    pa = _profile_row(profiles, fighter_a)
    pb = _profile_row(profiles, fighter_b)
    dob_a = _parse_dob(pa.get("dob"))
    dob_b = _parse_dob(pb.get("dob"))
    age_a = _age_on(dob_a, as_of)
    age_b = _age_on(dob_b, as_of)
    pa["age"] = age_a
    pb["age"] = age_b

    act_a = _recent_activity(fights, fighter_a, as_of)
    act_b = _recent_activity(fights, fighter_b, as_of)
    div_a = _current_division(ratings, fighter_a)
    div_b = _current_division(ratings, fighter_b)
    move_a = _division_transition(act_a, div_a)
    move_b = _division_transition(act_b, div_b)

    rows: list[dict[str, Any]] = []
    physical_adjustment = 0.0
    age_adjustment = 0.0
    activity_adjustment = 0.0

    reach_a = _safe_float(pa.get("reach_inches"))
    reach_b = _safe_float(pb.get("reach_inches"))
    height_a = _safe_float(pa.get("height_inches"))
    height_b = _safe_float(pb.get("height_inches"))
    if reach_a is not None and reach_b is not None:
        reach_gap = reach_a - reach_b
        reach_move = float(np.clip(reach_gap * 0.0012, -0.0075, 0.0075))
        physical_adjustment += reach_move
        advantage = "Even" if abs(reach_gap) < 1.0 else (fighter_a if reach_gap > 0 else fighter_b)
        rows.append({
            "category": "Reach / range",
            "advantage": advantage,
            "strength": "Even" if abs(reach_gap) < 1 else ("Slight" if abs(reach_gap) < 3 else "Clear"),
            "line_impact_a": reach_move,
            "why": f"Reach is {reach_a:.0f}\" vs {reach_b:.0f}\". Macabets treats reach as a small range modifier, not a standalone skill edge.",
        })
    if height_a is not None and height_b is not None:
        height_gap = height_a - height_b
        height_move = float(np.clip(height_gap * 0.0005, -0.0025, 0.0025))
        physical_adjustment += height_move
        advantage = "Even" if abs(height_gap) < 1.0 else (fighter_a if height_gap > 0 else fighter_b)
        rows.append({
            "category": "Height",
            "advantage": advantage,
            "strength": "Even" if abs(height_gap) < 1 else ("Slight" if abs(height_gap) < 3 else "Clear"),
            "line_impact_a": height_move,
            "why": f"Height is {height_a:.0f}\" vs {height_b:.0f}\". Height receives less weight than reach because taller is not automatically better in MMA.",
        })
    physical_adjustment = float(np.clip(physical_adjustment, -config.max_physical_adjustment, config.max_physical_adjustment))

    decline_a = _decline_index(age_a)
    decline_b = _decline_index(age_b)
    if decline_a is not None and decline_b is not None:
        decline_gap = decline_b - decline_a  # positive favors A (B has more decline exposure)
        age_adjustment = float(np.clip(decline_gap * 0.0015, -config.max_age_adjustment, config.max_age_adjustment))
        age_gap = age_a - age_b
        advantage = "Even" if abs(age_adjustment) < 0.001 else (fighter_a if age_adjustment > 0 else fighter_b)
        rows.append({
            "category": "Age curve",
            "advantage": advantage,
            "strength": "Even" if abs(age_adjustment) < 0.001 else ("Slight" if abs(age_adjustment) < 0.005 else "Moderate"),
            "line_impact_a": age_adjustment,
            "why": f"Fight-date ages are {age_a:.1f} vs {age_b:.1f}. Macabets applies no generic youth bonus and only prices modest post-prime decline risk.",
        })

    days_a = act_a.get("days_since_last_fight")
    days_b = act_b.get("days_since_last_fight")
    # Inactivity is already included in Strength v0.2. Do not award it again. Only a
    # very fast turnaround gets a small fatigue/load adjustment because that is a
    # different mechanism than inactivity/recency.
    def quick_turnaround(days: Any) -> float:
        if days is None:
            return 0.0
        d = int(days)
        if d <= 21:
            return -0.005
        if d <= 35:
            return -0.003
        if d <= 49:
            return -0.0015
        return 0.0

    fatigue_a = quick_turnaround(days_a)
    fatigue_b = quick_turnaround(days_b)
    activity_adjustment = float(np.clip(fatigue_a - fatigue_b, -config.max_activity_adjustment, config.max_activity_adjustment))
    rows.append({
        "category": "Activity / turnaround",
        "advantage": "Even" if abs(activity_adjustment) < 0.001 else (fighter_a if activity_adjustment > 0 else fighter_b),
        "strength": "Even" if abs(activity_adjustment) < 0.001 else "Slight",
        "line_impact_a": activity_adjustment,
        "why": f"Days since last UFCStats fight: {days_a if days_a is not None else 'N/A'} vs {days_b if days_b is not None else 'N/A'}. Long layoffs are not re-priced here because Strength v0.2 already handles inactivity; only unusually fast turnarounds can add a small fatigue signal.",
    })

    stance_a = pa.get("stance") or "Unknown"
    stance_b = pb.get("stance") or "Unknown"
    rows.append({
        "category": "Stance",
        "advantage": "Even",
        "strength": "Context only",
        "line_impact_a": 0.0,
        "why": f"Stance: {stance_a} vs {stance_b}. v0.1 displays stance but does not invent a directional edge until Macabets calibrates stance interactions against historical outcomes.",
    })

    if move_a or move_b:
        if move_a and not move_b:
            move_text = f"{fighter_a} has recent fights outside the current {div_a or 'listed'} division."
        elif move_b and not move_a:
            move_text = f"{fighter_b} has recent fights outside the current {div_b or 'listed'} division."
        else:
            move_text = "Both fighters have recent cross-division history."
        rows.append({
            "category": "Weight-class transition",
            "advantage": "Uncertain",
            "strength": "Context only",
            "line_impact_a": 0.0,
            "why": move_text + " Macabets treats the move as uncertainty rather than assigning an automatic up/down advantage.",
        })

    raw_adjustment = physical_adjustment + age_adjustment + activity_adjustment
    profile_fields = [reach_a, reach_b, height_a, height_b, age_a, age_b]
    profile_completeness = sum(v is not None for v in profile_fields) / len(profile_fields)
    stance_complete = int(stance_a != "Unknown" and stance_b != "Unknown")
    reliability = min(1.0, 0.25 + 0.65 * profile_completeness + 0.10 * stance_complete)
    adjustment = float(np.clip(raw_adjustment * reliability, -config.max_probability_adjustment, config.max_probability_adjustment))

    uncertainty_flags = int(move_a) + int(move_b)
    if profile_completeness < 0.5:
        uncertainty_flags += 1
    confidence_modifier = -min(3, uncertainty_flags)

    return {
        "available": bool(rows),
        "version": CONTEXT_VERSION,
        "fight_date": as_of.isoformat(),
        "adjustment_a": adjustment,
        "raw_adjustment_a": raw_adjustment,
        "physical_adjustment_a": physical_adjustment,
        "age_adjustment_a": age_adjustment,
        "activity_adjustment_a": activity_adjustment,
        "reliability": float(reliability),
        "confidence_modifier": int(confidence_modifier),
        "fighter_a_profile": pa,
        "fighter_b_profile": pb,
        "fighter_a_activity": act_a,
        "fighter_b_activity": act_b,
        "fighter_a_weight_class_transition": bool(move_a),
        "fighter_b_weight_class_transition": bool(move_b),
        "rows": rows,
        "guardrail": (
            "Physical & Context is capped at ±2 percentage points. Reach/height are small modifiers; stance is context-only; "
            "long-layoff inactivity is not counted twice because Strength v0.2 already prices inactivity, and weight-class changes reduce certainty rather than receiving an automatic directional bonus."
        ),
    }
