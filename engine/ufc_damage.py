from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import math

import numpy as np
import pandas as pd


DAMAGE_VERSION = "Macabets UFC Damage & Durability Risk v0.1"


@dataclass(frozen=True)
class UFCDamageConfig:
    recent_fights: int = 8
    acute_fights: int = 3
    min_sample_full_weight: int = 6
    max_probability_adjustment: float = 0.0075


def _safe(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _as_date(value: date | datetime | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    return date.today() if pd.isna(parsed) else parsed.date()


def _fight_minutes(round_number: Any, finish_time: Any) -> float:
    try:
        rnd = max(1, int(float(round_number)))
    except (TypeError, ValueError):
        rnd = 1
    text = str(finish_time or "").strip()
    seconds = 0
    if ":" in text:
        try:
            minute, second = text.split(":", 1)
            seconds = int(minute) * 60 + int(second)
        except (TypeError, ValueError):
            seconds = 0
    return max(1.0 / 60.0, (rnd - 1) * 5.0 + seconds / 60.0)


def _attach_opponent_damage(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if not {"fight_url", "fighter", "opponent"}.issubset(frame.columns):
        return frame
    stats = [c for c in ("kd", "head_landed", "sig_str_landed", "sig_str") if c in frame.columns]
    if not stats:
        return frame
    opponent = frame[["fight_url", "fighter"] + stats].copy()
    opponent = opponent.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stats}})
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _method_is_ko(value: Any) -> bool:
    text = str(value or "").casefold()
    return "ko" in text or "tko" in text


def fighter_damage_profile(
    fights: pd.DataFrame,
    fighter: str,
    *,
    fight_date: date | datetime | str | None = None,
    config: UFCDamageConfig | None = None,
) -> dict[str, Any]:
    """Estimate *damage trajectory risk*, not general fighting durability.

    The existing Performance layer already uses broad durability indicators. This layer
    deliberately focuses on recent damage concentration, recent KO recovery windows and
    career cage-time mileage so the same static defense metrics are not simply awarded twice.
    """
    config = config or UFCDamageConfig()
    as_of = _as_date(fight_date)
    frame = _attach_opponent_damage(fights)
    if "event_date" not in frame.columns or "fighter" not in frame.columns:
        return {"available": False, "version": DAMAGE_VERSION, "fighter": fighter, "reason": "Fight history is missing event dates or fighter names."}

    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    all_rows = frame.loc[
        frame["fighter"].astype(str).str.casefold().eq(str(fighter).strip().casefold())
        & frame["event_date"].notna()
        & (frame["event_date"].dt.date <= as_of)
    ].sort_values("event_date", ascending=False).copy()
    if all_rows.empty:
        return {"available": False, "version": DAMAGE_VERSION, "fighter": fighter, "sample": 0, "reason": "No historical UFC fight rows were found."}

    recent = all_rows.head(config.recent_fights).copy()
    acute = recent.head(config.acute_fights).copy()
    if "opponent_kd" not in recent.columns and "opponent_head_landed" not in recent.columns:
        return {
            "available": False,
            "version": DAMAGE_VERSION,
            "fighter": fighter,
            "sample": int(len(recent)),
            "reason": "Opponent knockdown/head-strike detail is unavailable. Re-run Update Macabets UFC Data with the enriched UFCStats history.",
        }

    def numeric(rows: pd.DataFrame, col: str) -> pd.Series:
        if col not in rows.columns:
            return pd.Series(np.nan, index=rows.index, dtype=float)
        return pd.to_numeric(rows[col], errors="coerce")

    opp_kd = numeric(acute, "opponent_kd")
    opp_head = numeric(acute, "opponent_head_landed")
    opp_sig = numeric(acute, "opponent_sig_str_landed")
    if not opp_sig.notna().any():
        opp_sig = numeric(acute, "opponent_sig_str")

    acute_kd = float(opp_kd.fillna(0).sum()) if opp_kd.notna().any() else None
    acute_head = float(opp_head.fillna(0).sum()) if opp_head.notna().any() else None
    acute_sig = float(opp_sig.fillna(0).sum()) if opp_sig.notna().any() else None

    results = recent.get("result", pd.Series("", index=recent.index)).astype(str).str.upper()
    ko_losses = recent.loc[results.eq("L") & recent.get("method", pd.Series("", index=recent.index)).map(_method_is_ko)]
    ko_losses_365 = 0
    days_since_last_ko = None
    if not ko_losses.empty:
        ko_dates = ko_losses["event_date"].dropna().sort_values(ascending=False)
        if not ko_dates.empty:
            days_since_last_ko = max(0, (as_of - ko_dates.iloc[0].date()).days)
            ko_losses_365 = int(sum((as_of - d.date()).days <= 365 for d in ko_dates))

    career_minutes = float(sum(_fight_minutes(r, t) for r, t in zip(all_rows.get("round", 1), all_rows.get("time", ""))))

    # Risk components are intentionally trajectory-focused. They do not attempt to
    # recreate the broad durability percentile already used in UFC Performance.
    kd_component = 0.0 if acute_kd is None else min(24.0, acute_kd * 8.0)
    head_component = 0.0 if acute_head is None else min(20.0, max(0.0, acute_head - 45.0) / 7.5)
    sig_component = 0.0 if acute_sig is None else min(8.0, max(0.0, acute_sig - 120.0) / 20.0)
    recent_ko_component = min(24.0, ko_losses_365 * 12.0)
    recovery_component = 0.0
    if days_since_last_ko is not None:
        if days_since_last_ko <= 120:
            recovery_component = 16.0
        elif days_since_last_ko <= 240:
            recovery_component = 10.0
        elif days_since_last_ko <= 365:
            recovery_component = 5.0
    mileage_component = min(8.0, max(0.0, career_minutes - 180.0) / 30.0)

    raw_risk = kd_component + head_component + sig_component + recent_ko_component + recovery_component + mileage_component
    sample_reliability = min(1.0, len(recent) / float(config.min_sample_full_weight))
    completeness_fields = [acute_kd, acute_head, acute_sig]
    completeness = sum(v is not None for v in completeness_fields) / len(completeness_fields)
    reliability = float(sample_reliability * (0.45 + 0.55 * completeness))
    risk_score = float(np.clip(raw_risk * (0.65 + 0.35 * reliability), 0.0, 100.0))

    if risk_score >= 55:
        label = "Elevated"
    elif risk_score >= 32:
        label = "Moderate"
    else:
        label = "Lower"

    return {
        "available": True,
        "version": DAMAGE_VERSION,
        "fighter": fighter,
        "sample": int(len(recent)),
        "acute_sample": int(len(acute)),
        "risk_score": risk_score,
        "risk_label": label,
        "reliability": reliability,
        "knockdowns_absorbed_last3": acute_kd,
        "head_strikes_absorbed_last3": acute_head,
        "significant_strikes_absorbed_last3": acute_sig,
        "ko_tko_losses_last365": int(ko_losses_365),
        "days_since_last_ko_tko_loss": days_since_last_ko,
        "career_ufc_minutes": career_minutes,
        "guardrail": (
            "Damage Risk focuses on recent damage concentration, KO recovery and accumulated UFC cage time. "
            "It is separate from the broad durability score, sample-shrunk, and cannot by itself create a large side adjustment."
        ),
    }


def build_damage_matchup(
    fights: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    *,
    fight_date: date | datetime | str | None = None,
    config: UFCDamageConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCDamageConfig()
    a = fighter_damage_profile(fights, fighter_a, fight_date=fight_date, config=config)
    b = fighter_damage_profile(fights, fighter_b, fight_date=fight_date, config=config)
    if not a.get("available") or not b.get("available"):
        return {
            "available": False,
            "version": DAMAGE_VERSION,
            "fighter_a_profile": a,
            "fighter_b_profile": b,
            "adjustment_a": 0.0,
            "reliability": 0.0,
        }

    # Higher risk is worse. Positive gap therefore favors fighter A when B has more risk.
    risk_gap = float(b["risk_score"] - a["risk_score"])
    reliability = float(min(a["reliability"], b["reliability"]))
    raw = (risk_gap / 45.0) * config.max_probability_adjustment
    adjustment = float(np.clip(raw * reliability, -config.max_probability_adjustment, config.max_probability_adjustment))
    advantage = "Even" if abs(risk_gap) < 4.0 else (fighter_a if risk_gap > 0 else fighter_b)
    return {
        "available": True,
        "version": DAMAGE_VERSION,
        "fighter_a_profile": a,
        "fighter_b_profile": b,
        "risk_gap": risk_gap,
        "advantage": advantage,
        "reliability": reliability,
        "adjustment_a": adjustment,
        "guardrail": (
            f"Damage & Durability Risk can move the side probability by at most ±{config.max_probability_adjustment:.2%}. "
            "The simulator may also use damage risk to redistribute a winner's finish method, but it never changes the winner probability a second time."
        ),
    }
