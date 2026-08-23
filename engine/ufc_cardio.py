from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd


CARDIO_VERSION = "Macabets UFC Round Cardio v0.1"


@dataclass(frozen=True)
class UFCCardioConfig:
    recent_fights: int = 8
    min_late_rounds_full_weight: int = 8
    max_adjustment_3r: float = 0.0075
    max_adjustment_5r: float = 0.015


def _safe(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _round_minutes(fight_round: Any, finish_time: Any, round_no: int) -> float:
    try:
        final_round = max(1, int(float(fight_round)))
    except (TypeError, ValueError):
        final_round = 1
    if round_no > final_round:
        return 0.0
    if round_no < final_round:
        return 5.0
    text = str(finish_time or "").strip()
    if ":" not in text:
        return 5.0
    try:
        minute, second = text.split(":", 1)
        return max(1.0 / 60.0, min(5.0, int(minute) + int(second) / 60.0))
    except (TypeError, ValueError):
        return 5.0


def _attach_opponent_round_stats(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if not {"fight_url", "fighter", "opponent"}.issubset(frame.columns):
        return frame
    round_cols = [c for c in frame.columns if c.startswith("r") and "_" in c]
    if not round_cols:
        return frame
    opponent = frame[["fight_url", "fighter"] + round_cols].copy()
    opponent = opponent.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in round_cols}})
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _round_summary(rows: pd.DataFrame, round_no: int) -> dict[str, Any]:
    if rows.empty:
        return {"round": round_no, "exposures": 0, "minutes": 0.0}
    prefix = f"r{round_no}_"
    reached = rows.loc[pd.to_numeric(rows.get("round", 0), errors="coerce").fillna(0).ge(round_no)].copy()
    if reached.empty:
        return {"round": round_no, "exposures": 0, "minutes": 0.0}

    minutes = np.array([
        _round_minutes(r, t, round_no)
        for r, t in zip(reached.get("round", 0), reached.get("time", ""))
    ], dtype=float)
    total_minutes = float(minutes.sum())
    if total_minutes <= 0:
        return {"round": round_no, "exposures": 0, "minutes": 0.0}

    def total(col: str) -> float | None:
        if col not in reached.columns:
            return None
        values = pd.to_numeric(reached[col], errors="coerce")
        return None if not values.notna().any() else float(values.fillna(0).sum())

    sig_l = total(prefix + "sig_str_landed")
    sig_a = total(prefix + "sig_str_attempted")
    opp_sig_l = total("opponent_" + prefix + "sig_str_landed")
    opp_sig_a = total("opponent_" + prefix + "sig_str_attempted")
    td_l = total(prefix + "td_landed")
    td_a = total(prefix + "td_attempted")
    control = total(prefix + "control_seconds")
    opp_control = total("opponent_" + prefix + "control_seconds")

    accuracy = None if sig_a in (None, 0) or sig_l is None else sig_l / sig_a
    defense = None if opp_sig_a in (None, 0) or opp_sig_l is None else 1.0 - opp_sig_l / opp_sig_a
    td_accuracy = None if td_a in (None, 0) or td_l is None else td_l / td_a
    control_share = None
    if control is not None or opp_control is not None:
        own = float(control or 0.0)
        opp = float(opp_control or 0.0)
        if own + opp > 0:
            control_share = own / (own + opp)

    return {
        "round": round_no,
        "exposures": int(len(reached)),
        "minutes": total_minutes,
        "sig_landed_per_min": None if sig_l is None else sig_l / total_minutes,
        "sig_attempted_per_min": None if sig_a is None else sig_a / total_minutes,
        "sig_absorbed_per_min": None if opp_sig_l is None else opp_sig_l / total_minutes,
        "sig_accuracy": accuracy,
        "sig_defense": defense,
        "td_attempted_per15": None if td_a is None else td_a / total_minutes * 15.0,
        "td_accuracy": td_accuracy,
        "control_share": control_share,
    }


def _retention(later: float | None, first: float | None, *, inverse: bool = False) -> float | None:
    if later is None or first is None:
        return None
    if inverse:
        # For absorbed strikes, lower later is better. Convert to a stability ratio.
        if later <= 0 and first <= 0:
            return 1.0
        if later <= 0:
            return 1.25
        if first <= 0:
            return 0.75
        ratio = first / later
    else:
        if first <= 0:
            return None
        ratio = later / first
    return float(np.clip(ratio, 0.50, 1.35))


def fighter_cardio_profile(
    fights: pd.DataFrame,
    fighter: str,
    *,
    config: UFCCardioConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCCardioConfig()
    if fights is None or fights.empty or "fighter" not in fights.columns:
        return {
            "available": False,
            "version": CARDIO_VERSION,
            "fighter": fighter,
            "sample": 0,
            "reason": "No fight history is available yet for this fighter.",
        }
    frame = _attach_opponent_round_stats(fights)
    if "event_date" in frame.columns:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    rows = frame.loc[frame["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()].copy()
    if "event_date" in rows.columns:
        rows = rows.sort_values("event_date", ascending=False)
    rows = rows.head(config.recent_fights)

    required = {"r1_sig_str_landed", "r2_sig_str_landed"}
    if rows.empty or not required.issubset(rows.columns):
        return {
            "available": False,
            "version": CARDIO_VERSION,
            "fighter": fighter,
            "sample": int(len(rows)),
            "reason": "Round-level UFCStats history is not available. Re-run Update Macabets UFC Data with the current pipeline.",
        }

    rounds = [_round_summary(rows, r) for r in range(1, 6)]
    r1 = rounds[0]
    late = [r for r in rounds[1:3] if r.get("exposures", 0) > 0]
    championship = [r for r in rounds[3:5] if r.get("exposures", 0) > 0]
    if not late:
        return {
            "available": False,
            "version": CARDIO_VERSION,
            "fighter": fighter,
            "sample": int(len(rows)),
            "rounds": rounds,
            "reason": "Not enough Round 2+ exposure to estimate degradation.",
        }

    def weighted_metric(name: str, selected: list[dict[str, Any]]) -> float | None:
        vals, weights = [], []
        for r in selected:
            value = _safe(r.get(name))
            if value is None:
                continue
            vals.append(value)
            weights.append(max(float(r.get("minutes", 0.0) or 0.0), 0.1))
        if not vals:
            return None
        return float(np.average(vals, weights=weights))

    later_sig_attempt = weighted_metric("sig_attempted_per_min", late)
    later_accuracy = weighted_metric("sig_accuracy", late)
    later_defense = weighted_metric("sig_defense", late)
    later_absorbed = weighted_metric("sig_absorbed_per_min", late)
    later_td_attempt = weighted_metric("td_attempted_per15", late)
    later_control = weighted_metric("control_share", late)

    retentions = {
        "output_retention": _retention(later_sig_attempt, _safe(r1.get("sig_attempted_per_min"))),
        "accuracy_retention": _retention(later_accuracy, _safe(r1.get("sig_accuracy"))),
        "defense_retention": _retention(later_defense, _safe(r1.get("sig_defense"))),
        "absorption_stability": _retention(later_absorbed, _safe(r1.get("sig_absorbed_per_min")), inverse=True),
        "wrestling_retention": _retention(later_td_attempt, _safe(r1.get("td_attempted_per15"))),
        "control_retention": _retention(later_control, _safe(r1.get("control_share"))),
    }
    clean = [v for v in retentions.values() if v is not None]
    if not clean:
        return {
            "available": False,
            "version": CARDIO_VERSION,
            "fighter": fighter,
            "sample": int(len(rows)),
            "rounds": rounds,
            "reason": "Round-level fields exist but are too incomplete for a cardio estimate.",
        }

    offense = np.mean([v for k, v in retentions.items() if k in {"output_retention", "accuracy_retention"} and v is not None]) if any(retentions[k] is not None for k in {"output_retention", "accuracy_retention"}) else 1.0
    defense = np.mean([v for k, v in retentions.items() if k in {"defense_retention", "absorption_stability"} and v is not None]) if any(retentions[k] is not None for k in {"defense_retention", "absorption_stability"}) else 1.0
    grappling = np.mean([v for k, v in retentions.items() if k in {"wrestling_retention", "control_retention"} and v is not None]) if any(retentions[k] is not None for k in {"wrestling_retention", "control_retention"}) else 1.0
    retention = float(0.45 * offense + 0.30 * defense + 0.25 * grappling)

    late_exposures = int(sum(int(r.get("exposures", 0) or 0) for r in late))
    championship_exposures = int(sum(int(r.get("exposures", 0) or 0) for r in championship))
    sample_reliability = min(1.0, late_exposures / float(config.min_late_rounds_full_weight))
    completeness = len(clean) / float(len(retentions))
    reliability = float(sample_reliability * (0.45 + 0.55 * completeness))
    cardio_score = float(np.clip(50.0 + (retention - 1.0) * 85.0, 20.0, 80.0))

    trend = "Stable"
    if retention >= 1.08:
        trend = "Improves / sustains late"
    elif retention >= 0.94:
        trend = "Stable"
    elif retention >= 0.82:
        trend = "Moderate fade"
    else:
        trend = "Sharp fade"

    return {
        "available": True,
        "version": CARDIO_VERSION,
        "fighter": fighter,
        "sample": int(len(rows)),
        "rounds": rounds,
        "retention": retention,
        "cardio_score": cardio_score,
        "trend": trend,
        "reliability": reliability,
        "late_round_exposures": late_exposures,
        "championship_round_exposures": championship_exposures,
        **retentions,
        "guardrail": (
            "Cardio is estimated from within-fighter Round 2+ retention versus that fighter's Round 1 rates. "
            "It is sample-shrunk and does not reward raw volume twice."
        ),
    }


def build_cardio_matchup(
    fights: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    *,
    rounds: int = 3,
    config: UFCCardioConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCCardioConfig()
    a = fighter_cardio_profile(fights, fighter_a, config=config)
    b = fighter_cardio_profile(fights, fighter_b, config=config)
    if not a.get("available") or not b.get("available"):
        return {
            "available": False,
            "version": CARDIO_VERSION,
            "fighter_a_profile": a,
            "fighter_b_profile": b,
            "adjustment_a": 0.0,
            "reliability": 0.0,
        }

    gap = float(a["cardio_score"] - b["cardio_score"])
    reliability = float(min(a["reliability"], b["reliability"]))
    max_adj = config.max_adjustment_5r if int(rounds) == 5 else config.max_adjustment_3r
    raw = (gap / 30.0) * max_adj
    adjustment = float(np.clip(raw * reliability, -max_adj, max_adj))
    if abs(gap) < 3:
        advantage = "Even"
    else:
        advantage = fighter_a if gap > 0 else fighter_b

    return {
        "available": True,
        "version": CARDIO_VERSION,
        "fighter_a_profile": a,
        "fighter_b_profile": b,
        "cardio_gap": gap,
        "advantage": advantage,
        "reliability": reliability,
        "adjustment_a": adjustment,
        "rounds": int(rounds),
        "guardrail": (
            f"Round Cardio can move the side probability by at most ±{max_adj:.1%} for a {int(rounds)}-round fight. "
            "The adjustment is based on degradation/retention, not standalone pace, and is heavily sample-shrunk."
        ),
    }
