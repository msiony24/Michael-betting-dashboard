from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd


STRIKING_VERSION = "Macabets UFC Advanced Striking v0.1"


@dataclass(frozen=True)
class UFCStrikingConfig:
    recent_fights: int = 8
    min_sample_for_full_weight: int = 6


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
    return max(1.0 / 60.0, ((rnd - 1) * 300 + seconds) / 60.0)


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _attach_opponent_striking_fields(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if not {"fight_url", "fighter", "opponent"}.issubset(frame.columns):
        return frame
    wanted = {
        "sig_str_landed", "sig_str_attempted", "kd",
        "head_landed", "head_attempted", "body_landed", "body_attempted",
        "leg_landed", "leg_attempted", "distance_landed", "distance_attempted",
        "clinch_landed", "clinch_attempted", "ground_landed", "ground_attempted",
    }
    stat_cols = [c for c in frame.columns if c in wanted]
    if not stat_cols:
        return frame
    opponent = frame[["fight_url", "fighter"] + stat_cols].copy()
    opponent = opponent.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stat_cols}})
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _fighter_raw_profile(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"sample": 0, "data_completeness": 0.0}
    rows = rows.sort_values("event_date", ascending=False).copy()
    minutes = max(float(sum(_fight_minutes(r, t) for r, t in zip(rows.get("round", 1), rows.get("time", "")))), 1.0)

    def totals(prefix: str) -> tuple[float, float, float, float]:
        own_l = float(_numeric(rows, f"{prefix}_landed").fillna(0).sum())
        own_a = float(_numeric(rows, f"{prefix}_attempted").fillna(0).sum())
        opp_l = float(_numeric(rows, f"opponent_{prefix}_landed").fillna(0).sum())
        opp_a = float(_numeric(rows, f"opponent_{prefix}_attempted").fillna(0).sum())
        return own_l, own_a, opp_l, opp_a

    profile: dict[str, Any] = {"sample": int(len(rows))}
    available = 0
    possible = 0
    for prefix in ("head", "body", "leg", "distance", "clinch", "ground"):
        own_l, own_a, opp_l, opp_a = totals(prefix)
        possible += 2
        if f"{prefix}_attempted" in rows.columns:
            available += 1
        if f"opponent_{prefix}_attempted" in rows.columns:
            available += 1
        profile[f"{prefix}_landed_per_min"] = own_l / minutes
        profile[f"{prefix}_accuracy"] = _safe_ratio(own_l, own_a)
        profile[f"{prefix}_absorbed_per_min"] = opp_l / minutes
        profile[f"{prefix}_accuracy_allowed"] = _safe_ratio(opp_l, opp_a)

    sig_landed = float(_numeric(rows, "sig_str_landed").fillna(0).sum())
    opp_sig_landed = float(_numeric(rows, "opponent_sig_str_landed").fillna(0).sum())
    kd = float(_numeric(rows, "kd").fillna(0).sum())
    opp_kd = float(_numeric(rows, "opponent_kd").fillna(0).sum())
    profile["power_kd_per100_sig"] = _safe_ratio(kd * 100.0, sig_landed)
    profile["kd_absorbed_per100_sig"] = _safe_ratio(opp_kd * 100.0, opp_sig_landed)
    if "kd" in rows.columns:
        available += 1
    if "opponent_kd" in rows.columns:
        available += 1
    possible += 2

    # Close-range striking is only striking production: clinch + ground. It does not use
    # takedowns/control, so the grappling engine remains responsible for positional control.
    close_l = float(_numeric(rows, "clinch_landed").fillna(0).sum() + _numeric(rows, "ground_landed").fillna(0).sum())
    close_a = float(_numeric(rows, "clinch_attempted").fillna(0).sum() + _numeric(rows, "ground_attempted").fillna(0).sum())
    opp_close_l = float(_numeric(rows, "opponent_clinch_landed").fillna(0).sum() + _numeric(rows, "opponent_ground_landed").fillna(0).sum())
    opp_close_a = float(_numeric(rows, "opponent_clinch_attempted").fillna(0).sum() + _numeric(rows, "opponent_ground_attempted").fillna(0).sum())
    profile["close_landed_per_min"] = close_l / minutes
    profile["close_accuracy"] = _safe_ratio(close_l, close_a)
    profile["close_absorbed_per_min"] = opp_close_l / minutes
    profile["close_accuracy_allowed"] = _safe_ratio(opp_close_l, opp_close_a)
    profile["data_completeness"] = float(available / max(possible, 1))
    return profile


def _avg(row: pd.Series | dict[str, Any], names: list[str]) -> float | None:
    vals = []
    for name in names:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        vals.append(float(value))
    return float(np.mean(vals)) if vals else None


def build_striking_table(fights: pd.DataFrame, ratings: pd.DataFrame, *, config: UFCStrikingConfig | None = None) -> pd.DataFrame:
    config = config or UFCStrikingConfig()
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date"])
    frame = _attach_opponent_striking_fields(frame)

    pool = ratings.copy()
    if "active_pool" in pool.columns:
        mask = pool["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        pool = pool.loc[mask]

    frame = frame.sort_values("event_date", ascending=False)
    frame["_fighter_key"] = frame["fighter"].astype(str).str.casefold()
    recent_by_fighter = {
        key: group.head(config.recent_fights)
        for key, group in frame.groupby("_fighter_key", sort=False)
    }

    records = []
    for _, rating in pool.iterrows():
        fighter = str(rating.get("fighter", "")).strip()
        if not fighter:
            continue
        recent = recent_by_fighter.get(fighter.casefold(), frame.iloc[0:0])
        records.append({"fighter": fighter, "division": str(rating.get("division", "Unknown") or "Unknown"), **_fighter_raw_profile(recent)})
    table = pd.DataFrame(records)
    if table.empty:
        return table

    metrics: dict[str, bool] = {}
    for prefix in ("head", "body", "leg", "distance"):
        metrics[f"{prefix}_landed_per_min"] = True
        metrics[f"{prefix}_accuracy"] = True
        metrics[f"{prefix}_absorbed_per_min"] = False
        metrics[f"{prefix}_accuracy_allowed"] = False
    metrics.update({
        "close_landed_per_min": True,
        "close_accuracy": True,
        "close_absorbed_per_min": False,
        "close_accuracy_allowed": False,
        "power_kd_per100_sig": True,
        "kd_absorbed_per100_sig": False,
    })

    for metric, higher_better in metrics.items():
        table[f"{metric}_pct"] = np.nan
        for _, idx in table.groupby("division").groups.items():
            values = pd.to_numeric(table.loc[idx, metric], errors="coerce")
            count = int(values.notna().sum())
            if count < 3:
                continue
            ranks = values.rank(pct=True, method="average") * 100.0
            if not higher_better:
                ranks = 100.0 - ranks + (100.0 / count)
            table.loc[idx, f"{metric}_pct"] = ranks

    for prefix in ("head", "body", "leg", "distance", "close"):
        table[f"{prefix}_attack_score"] = table.apply(lambda r, p=prefix: _avg(r, [f"{p}_landed_per_min_pct", f"{p}_accuracy_pct"]), axis=1)
        table[f"{prefix}_defense_score"] = table.apply(lambda r, p=prefix: _avg(r, [f"{p}_absorbed_per_min_pct", f"{p}_accuracy_allowed_pct"]), axis=1)
    table["power_score"] = table["power_kd_per100_sig_pct"]
    table["knockdown_resistance_score"] = table["kd_absorbed_per100_sig_pct"]
    return table


def fighter_striking_profile(table: pd.DataFrame, fighter: str) -> dict[str, Any]:
    if table is None or table.empty:
        return {"fighter": fighter, "sample": 0, "data_completeness": 0.0}
    match = table.loc[table["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()]
    if match.empty:
        return {"fighter": fighter, "sample": 0, "data_completeness": 0.0}
    result = match.iloc[-1].to_dict()
    for key, value in list(result.items()):
        if isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
    return result


def _score(profile: dict[str, Any], key: str) -> float | None:
    try:
        value = float(profile.get(key))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _gap(a_attack: float | None, b_defense: float | None, b_attack: float | None, a_defense: float | None) -> float | None:
    if None in (a_attack, b_defense, b_attack, a_defense):
        return None
    return float((a_attack - b_defense) - (b_attack - a_defense))


def _strength(gap: float) -> str:
    mag = abs(float(gap))
    if mag < 5:
        return "Even"
    if mag < 12:
        return "Slight"
    if mag < 22:
        return "Moderate"
    return "Clear"


def build_advanced_striking_matchup(table: pd.DataFrame, fighter_a: str, fighter_b: str, *, config: UFCStrikingConfig | None = None) -> dict[str, Any]:
    config = config or UFCStrikingConfig()
    a = fighter_striking_profile(table, fighter_a)
    b = fighter_striking_profile(table, fighter_b)
    specs = [
        ("Head attack vs head defense", "head_attack_score", "head_defense_score", 0.20, "Head-strike volume and efficiency are compared with the opponent's recent head-strike prevention profile."),
        ("Body attack vs body defense", "body_attack_score", "body_defense_score", 0.12, "Body-strike production is compared with how effectively the opponent has limited body offense."),
        ("Leg attack vs leg defense", "leg_attack_score", "leg_defense_score", 0.13, "Leg-strike production is compared with the opponent's demonstrated ability to limit leg attacks."),
        ("Distance efficiency vs range defense", "distance_attack_score", "distance_defense_score", 0.22, "Distance striking rate and accuracy are compared with the opponent's range-striking defense."),
        ("Power vs knockdown resistance", "power_score", "knockdown_resistance_score", 0.23, "Knockdown efficiency per significant strike is compared with the opponent's knockdown resistance, reducing simple volume bias."),
        ("Close-range striking vs close defense", "close_attack_score", "close_defense_score", 0.10, "Clinch and ground striking efficiency are compared with close-range strike defense without reusing takedown or control statistics."),
    ]
    rows = []
    weighted_gap = 0.0
    used_weight = 0.0
    for category, attack_key, defense_key, weight, why in specs:
        gap = _gap(_score(a, attack_key), _score(b, defense_key), _score(b, attack_key), _score(a, defense_key))
        if gap is None:
            continue
        advantage = "Even" if abs(gap) < 5 else (fighter_a if gap > 0 else fighter_b)
        rows.append({"category": category, "advantage": advantage, "strength": _strength(gap), "interaction_gap": float(gap), "weight": float(weight), "why": why})
        weighted_gap += float(gap) * float(weight)
        used_weight += float(weight)
    if not rows or used_weight <= 0:
        return {"available": False, "version": STRIKING_VERSION, "fighter_a_profile": a, "fighter_b_profile": b, "rows": [], "weighted_gap": 0.0, "reliability": 0.0, "reason": "Detailed target and position striking history is not complete enough for advanced striking interactions."}
    weighted_gap /= used_weight
    sample_rel = min(1.0, min(int(a.get("sample", 0) or 0), int(b.get("sample", 0) or 0)) / float(config.min_sample_for_full_weight))
    completeness = min(float(a.get("data_completeness", 0.0) or 0.0), float(b.get("data_completeness", 0.0) or 0.0))
    reliability = sample_rel * (0.45 + 0.55 * completeness) * (len(rows) / len(specs))
    return {
        "available": True,
        "version": STRIKING_VERSION,
        "fighter_a_profile": a,
        "fighter_b_profile": b,
        "rows": rows,
        "weighted_gap": float(weighted_gap),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "guardrail": (
            "Advanced striking replaces the generic striking row inside the existing Style Matchups cap; it does not add a separate probability adjustment. "
            "Power is normalized by significant strikes landed, and close-range striking uses strikes only so takedown/control skill remains in the grappling engine. Stance is not directionally priced until Macabets has evidence that the effect calibrates out of sample."
        ),
    }
