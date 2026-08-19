from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd


PERFORMANCE_VERSION = "Macabets UFC Performance v0.1"


@dataclass(frozen=True)
class UFCPerformanceConfig:
    recent_fights: int = 8
    min_sample_for_full_weight: int = 6
    max_probability_adjustment: float = 0.05


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


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
    elapsed = (rnd - 1) * 300 + seconds
    return max(1.0 / 60.0, elapsed / 60.0)


def _attach_opponent_fields(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if not {"fight_url", "fighter", "opponent"}.issubset(frame.columns):
        return frame
    stat_cols = [
        c for c in (
            "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
            "kd", "sub_att", "control_seconds", "total_str_landed", "total_str_attempted",
            "head_landed", "body_landed", "leg_landed", "distance_landed", "clinch_landed",
            "ground_landed",
        ) if c in frame.columns
    ]
    if not stat_cols:
        return frame
    opponent = frame[["fight_url", "fighter"] + stat_cols].copy()
    opponent = opponent.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stat_cols}})
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _profile_for_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"sample": 0, "data_completeness": 0.0}

    rows = rows.sort_values("event_date", ascending=False).copy()
    minutes = pd.Series(
        [_fight_minutes(r, t) for r, t in zip(rows.get("round", 1), rows.get("time", ""))],
        index=rows.index,
        dtype=float,
    )
    total_minutes = float(minutes.sum()) if minutes.notna().any() else 0.0
    if total_minutes <= 0:
        total_minutes = float(len(rows) * 15.0)

    sig_landed = _numeric(rows, "sig_str_landed")
    sig_attempted = _numeric(rows, "sig_str_attempted")
    if not sig_landed.notna().any() and "sig_str" in rows.columns:
        sig_landed = _numeric(rows, "sig_str")
    opp_sig_landed = _numeric(rows, "opponent_sig_str_landed")
    opp_sig_attempted = _numeric(rows, "opponent_sig_str_attempted")
    if not opp_sig_landed.notna().any() and "opponent_sig_str" in rows.columns:
        opp_sig_landed = _numeric(rows, "opponent_sig_str")

    td_landed = _numeric(rows, "td_landed")
    td_attempted = _numeric(rows, "td_attempted")
    if not td_landed.notna().any() and "td" in rows.columns:
        td_landed = _numeric(rows, "td")
    opp_td_landed = _numeric(rows, "opponent_td_landed")
    opp_td_attempted = _numeric(rows, "opponent_td_attempted")

    kd = _numeric(rows, "kd")
    opp_kd = _numeric(rows, "opponent_kd")
    sub = _numeric(rows, "sub_att")
    control = _numeric(rows, "control_seconds")
    opp_control = _numeric(rows, "opponent_control_seconds")

    def per_minute(series: pd.Series) -> float | None:
        valid = series.notna()
        if not valid.any() or total_minutes <= 0:
            return None
        return float(series[valid].sum() / total_minutes)

    sig_lpm = per_minute(sig_landed)
    sig_apm = per_minute(sig_attempted)
    sig_abs_lpm = per_minute(opp_sig_landed)
    kd_per15 = None if not kd.notna().any() else float(kd.sum() / total_minutes * 15.0)
    kd_abs_per15 = None if not opp_kd.notna().any() else float(opp_kd.sum() / total_minutes * 15.0)
    td_per15 = None if not td_landed.notna().any() else float(td_landed.sum() / total_minutes * 15.0)
    sub_per15 = None if not sub.notna().any() else float(sub.sum() / total_minutes * 15.0)

    sig_accuracy = None
    if sig_attempted.notna().any() and sig_attempted.sum() > 0:
        sig_accuracy = float(sig_landed.fillna(0).sum() / sig_attempted.fillna(0).sum())
    sig_defense = None
    if opp_sig_attempted.notna().any() and opp_sig_attempted.sum() > 0:
        sig_defense = float(1.0 - opp_sig_landed.fillna(0).sum() / opp_sig_attempted.fillna(0).sum())
    td_accuracy = None
    if td_attempted.notna().any() and td_attempted.sum() > 0:
        td_accuracy = float(td_landed.fillna(0).sum() / td_attempted.fillna(0).sum())
    td_defense = None
    if opp_td_attempted.notna().any() and opp_td_attempted.sum() > 0:
        td_defense = float(1.0 - opp_td_landed.fillna(0).sum() / opp_td_attempted.fillna(0).sum())

    control_share = None
    if control.notna().any() or opp_control.notna().any():
        own = float(control.fillna(0).sum())
        opp = float(opp_control.fillna(0).sum())
        if own + opp > 0:
            control_share = own / (own + opp)

    results = rows.get("result", pd.Series("", index=rows.index)).astype(str).str.upper()
    losses = rows.loc[results.eq("L")]
    finish_losses = 0
    if not losses.empty and "method" in losses.columns:
        finish_losses = int(losses["method"].astype(str).str.contains("KO|TKO|SUB", case=False, regex=True).sum())
    finish_loss_rate = float(finish_losses / len(losses)) if len(losses) else 0.0

    ground = _numeric(rows, "ground_landed")
    clinch = _numeric(rows, "clinch_landed")
    distance = _numeric(rows, "distance_landed")
    position_total = ground.fillna(0).sum() + clinch.fillna(0).sum() + distance.fillna(0).sum()
    ground_share = float(ground.fillna(0).sum() / position_total) if position_total > 0 else None
    clinch_share = float(clinch.fillna(0).sum() / position_total) if position_total > 0 else None
    distance_share = float(distance.fillna(0).sum() / position_total) if position_total > 0 else None

    detailed_inputs = [sig_attempted, opp_sig_attempted, td_attempted, opp_td_attempted, control]
    completeness = sum(1 for series in detailed_inputs if series.notna().any()) / len(detailed_inputs)

    return {
        "sample": int(len(rows)),
        "minutes": total_minutes,
        "data_completeness": float(completeness),
        "sig_landed_per_min": sig_lpm,
        "sig_attempted_per_min": sig_apm,
        "sig_absorbed_per_min": sig_abs_lpm,
        "sig_diff_per_min": None if sig_lpm is None or sig_abs_lpm is None else sig_lpm - sig_abs_lpm,
        "sig_accuracy": sig_accuracy,
        "sig_defense": sig_defense,
        "kd_per15": kd_per15,
        "kd_absorbed_per15": kd_abs_per15,
        "td_per15": td_per15,
        "td_accuracy": td_accuracy,
        "td_defense": td_defense,
        "sub_attempts_per15": sub_per15,
        "control_share": control_share,
        "finish_loss_rate": finish_loss_rate,
        "ground_strike_share": ground_share,
        "clinch_strike_share": clinch_share,
        "distance_strike_share": distance_share,
    }


def build_performance_table(
    fights: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    config: UFCPerformanceConfig | None = None,
    active_only: bool = True,
) -> pd.DataFrame:
    config = config or UFCPerformanceConfig()
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date"])
    frame = _attach_opponent_fields(frame)

    active = ratings.copy()
    if active_only and "active_pool" in active.columns:
        mask = active["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        active = active.loc[mask]

    # Pre-index recent fighter rows once. Re-filtering the full fight history for
    # every fighter made UFC analysis scale roughly as fighters x history rows.
    # A single grouped lookup preserves the same rows/order while cutting the
    # derived-table build to near-linear work.
    frame = frame.sort_values("event_date", ascending=False)
    frame["_fighter_key"] = frame["fighter"].astype(str).str.casefold()
    recent_by_fighter = {
        key: group.head(config.recent_fights)
        for key, group in frame.groupby("_fighter_key", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for _, rating in active.iterrows():
        fighter = str(rating.get("fighter", "")).strip()
        if not fighter:
            continue
        recent = recent_by_fighter.get(fighter.casefold(), frame.iloc[0:0])
        profile = _profile_for_rows(recent)
        rows.append({
            "fighter": fighter,
            "division": str(rating.get("division", "Unknown")),
            **profile,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    metrics = {
        "sig_diff_per_min": True,
        "sig_accuracy": True,
        "sig_defense": True,
        "kd_per15": True,
        "td_per15": True,
        "td_accuracy": True,
        "td_defense": True,
        "sub_attempts_per15": True,
        "control_share": True,
        "kd_absorbed_per15": False,
        "finish_loss_rate": False,
        "sig_attempted_per_min": True,
    }

    for metric, higher_better in metrics.items():
        if metric not in table.columns:
            continue
        pct_name = f"{metric}_pct"
        table[pct_name] = np.nan
        for _, idx in table.groupby("division").groups.items():
            values = pd.to_numeric(table.loc[idx, metric], errors="coerce")
            if values.notna().sum() < 3:
                continue
            ranks = values.rank(pct=True, method="average") * 100.0
            if not higher_better:
                ranks = 100.0 - ranks + (100.0 / max(values.notna().sum(), 1))
            table.loc[idx, pct_name] = ranks

    def average_pct(row: pd.Series, names: list[str]) -> float | None:
        values = [row.get(name) for name in names if pd.notna(row.get(name))]
        return float(np.mean(values)) if values else None

    composites = []
    for _, row in table.iterrows():
        striking = average_pct(row, [
            "sig_diff_per_min_pct", "sig_accuracy_pct", "sig_defense_pct", "kd_per15_pct"
        ])
        wrestling = average_pct(row, [
            "td_per15_pct", "td_accuracy_pct", "td_defense_pct", "control_share_pct"
        ])
        grappling = average_pct(row, [
            "sub_attempts_per15_pct", "control_share_pct", "td_per15_pct"
        ])
        durability = average_pct(row, [
            "sig_defense_pct", "kd_absorbed_per15_pct", "finish_loss_rate_pct"
        ])
        pace = row.get("sig_attempted_per_min_pct")
        composites.append((striking, wrestling, grappling, durability, pace))

    table[["striking_score", "wrestling_score", "grappling_score", "durability_score", "pace_score"]] = pd.DataFrame(
        composites, index=table.index
    )
    return table


def fighter_performance(table: pd.DataFrame, fighter: str) -> dict[str, Any]:
    match = table.loc[table["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()]
    if match.empty:
        return {"fighter": fighter, "sample": 0, "data_completeness": 0.0}
    row = match.iloc[-1]
    result = row.to_dict()
    for key, value in list(result.items()):
        if isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
    return result


def matchup_performance_adjustment(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    *,
    rounds: int = 3,
    config: UFCPerformanceConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCPerformanceConfig()
    if int(rounds) == 5:
        weights = {
            "striking_score": 0.28,
            "wrestling_score": 0.22,
            "grappling_score": 0.15,
            "durability_score": 0.20,
            "pace_score": 0.15,
        }
    else:
        weights = {
            "striking_score": 0.32,
            "wrestling_score": 0.25,
            "grappling_score": 0.15,
            "durability_score": 0.15,
            "pace_score": 0.13,
        }

    parts: list[dict[str, Any]] = []
    weighted_gap = 0.0
    used_weight = 0.0
    for metric, weight in weights.items():
        a = profile_a.get(metric)
        b = profile_b.get(metric)
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            continue
        gap = float(a) - float(b)
        weighted_gap += gap * weight
        used_weight += weight
        parts.append({"metric": metric.replace("_score", "").replace("_", " ").title(), "gap": gap, "weight": weight})

    if used_weight <= 0:
        return {
            "available": False,
            "adjustment_a": 0.0,
            "weighted_gap": 0.0,
            "reliability": 0.0,
            "components": [],
        }

    weighted_gap /= used_weight
    sample_reliability = min(
        1.0,
        min(int(profile_a.get("sample", 0) or 0), int(profile_b.get("sample", 0) or 0))
        / float(config.min_sample_for_full_weight),
    )
    completeness = min(
        float(profile_a.get("data_completeness", 0.0) or 0.0),
        float(profile_b.get("data_completeness", 0.0) or 0.0),
    )
    reliability = sample_reliability * (0.45 + 0.55 * completeness)
    raw_adjustment = (weighted_gap / 50.0) * config.max_probability_adjustment
    adjustment = float(np.clip(raw_adjustment * reliability, -config.max_probability_adjustment, config.max_probability_adjustment))

    return {
        "available": True,
        "adjustment_a": adjustment,
        "weighted_gap": weighted_gap,
        "reliability": reliability,
        "components": parts,
        "five_round_weighting": int(rounds) == 5,
    }
