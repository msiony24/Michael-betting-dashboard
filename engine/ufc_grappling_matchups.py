from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd


GRAPPLING_VERSION = "Macabets UFC Advanced Wrestling & Grappling v0.1"


@dataclass(frozen=True)
class UFCGrapplingConfig:
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


def _attach_opponent_grappling_fields(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    required = {"fight_url", "fighter", "opponent"}
    if not required.issubset(frame.columns):
        return frame
    stat_cols = [
        c for c in frame.columns
        if c in {"td_landed", "td_attempted", "control_seconds", "sub_att"}
        or any(c == f"r{rnd}_{metric}" for rnd in range(1, 6) for metric in ("td_landed", "td_attempted", "control_seconds", "sub_att"))
    ]
    if not stat_cols:
        return frame
    opponent = frame[["fight_url", "fighter"] + stat_cols].copy()
    opponent = opponent.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stat_cols}})
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _fighter_raw_profile(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"sample": 0, "data_completeness": 0.0}

    rows = rows.sort_values("event_date", ascending=False).copy()
    total_minutes = sum(_fight_minutes(r, t) for r, t in zip(rows.get("round", 1), rows.get("time", "")))
    total_minutes = max(float(total_minutes), 1.0)

    td_landed = float(_numeric(rows, "td_landed").fillna(0).sum())
    td_attempted = float(_numeric(rows, "td_attempted").fillna(0).sum())
    opp_td_landed = float(_numeric(rows, "opponent_td_landed").fillna(0).sum())
    opp_td_attempted = float(_numeric(rows, "opponent_td_attempted").fillna(0).sum())
    control_seconds = float(_numeric(rows, "control_seconds").fillna(0).sum())
    opp_control_seconds = float(_numeric(rows, "opponent_control_seconds").fillna(0).sum())
    sub_attempts = float(_numeric(rows, "sub_att").fillna(0).sum())
    opp_sub_attempts = float(_numeric(rows, "opponent_sub_att").fillna(0).sum())

    active_wrestling_rounds = 0
    repeat_attempt_rounds = 0
    round_level_available = False
    for rnd in range(1, 6):
        col = f"r{rnd}_td_attempted"
        if col not in rows.columns:
            continue
        attempts = _numeric(rows, col)
        if attempts.notna().any():
            round_level_available = True
            active_wrestling_rounds += int((attempts.fillna(0) >= 1).sum())
            repeat_attempt_rounds += int((attempts.fillna(0) >= 2).sum())

    repeat_attempt_rate = _safe_ratio(repeat_attempt_rounds, active_wrestling_rounds)
    td_accuracy = _safe_ratio(td_landed, td_attempted)
    td_defense = None if opp_td_attempted <= 0 else 1.0 - (opp_td_landed / opp_td_attempted)
    control_per_td_landed = _safe_ratio(control_seconds, td_landed)
    opp_control_per_td_allowed = _safe_ratio(opp_control_seconds, opp_td_landed)

    results = rows.get("result", pd.Series("", index=rows.index)).astype(str).str.upper()
    losses = rows.loc[results.eq("L")]
    sub_losses = 0
    if not losses.empty and "method" in losses.columns:
        sub_losses = int(losses["method"].astype(str).str.contains("SUB", case=False, regex=True).sum())
    submission_loss_rate = float(sub_losses / len(losses)) if len(losses) else 0.0

    available_inputs = [
        td_attempted > 0,
        opp_td_attempted > 0,
        control_seconds > 0 or "control_seconds" in rows.columns,
        opp_control_seconds > 0 or "opponent_control_seconds" in rows.columns,
        "sub_att" in rows.columns,
        "opponent_sub_att" in rows.columns,
        round_level_available,
    ]
    completeness = sum(bool(v) for v in available_inputs) / len(available_inputs)

    return {
        "sample": int(len(rows)),
        "data_completeness": float(completeness),
        "td_attempts_per15": float(td_attempted / total_minutes * 15.0),
        "td_landed_per15": float(td_landed / total_minutes * 15.0),
        "td_accuracy": td_accuracy,
        "repeat_attempt_rate": repeat_attempt_rate,
        "control_per15": float(control_seconds / 60.0 / total_minutes * 15.0),
        "control_seconds_per_td_landed": control_per_td_landed,
        "td_defense": td_defense,
        "opponent_td_attempts_per15": float(opp_td_attempted / total_minutes * 15.0),
        "opponent_control_per15": float(opp_control_seconds / 60.0 / total_minutes * 15.0),
        "opponent_control_seconds_per_td_allowed": opp_control_per_td_allowed,
        "sub_attempts_per15": float(sub_attempts / total_minutes * 15.0),
        "opponent_sub_attempts_per15": float(opp_sub_attempts / total_minutes * 15.0),
        "submission_loss_rate": submission_loss_rate,
        "active_wrestling_rounds": int(active_wrestling_rounds),
        "repeat_attempt_rounds": int(repeat_attempt_rounds),
    }


def _avg(row: pd.Series, names: list[str]) -> float | None:
    values = [float(row.get(name)) for name in names if pd.notna(row.get(name))]
    return float(np.mean(values)) if values else None


def build_grappling_table(
    fights: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    config: UFCGrapplingConfig | None = None,
) -> pd.DataFrame:
    config = config or UFCGrapplingConfig()
    frame = fights.copy()
    if "event_date" in frame.columns:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        frame = frame.dropna(subset=["event_date"])
    frame = _attach_opponent_grappling_fields(frame)

    pool = ratings.copy()
    if "active_pool" in pool.columns:
        mask = pool["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        pool = pool.loc[mask]

    records: list[dict[str, Any]] = []
    for _, rating in pool.iterrows():
        fighter = str(rating.get("fighter", "")).strip()
        if not fighter:
            continue
        recent = frame.loc[frame["fighter"].astype(str).str.casefold() == fighter.casefold()]
        recent = recent.sort_values("event_date", ascending=False).head(config.recent_fights)
        records.append({
            "fighter": fighter,
            "division": str(rating.get("division", "Unknown") or "Unknown"),
            **_fighter_raw_profile(recent),
        })

    table = pd.DataFrame(records)
    if table.empty:
        return table

    rank_specs = {
        "td_attempts_per15": True,
        "td_landed_per15": True,
        "td_accuracy": True,
        "repeat_attempt_rate": True,
        "control_per15": True,
        "control_seconds_per_td_landed": True,
        "td_defense": True,
        "opponent_control_per15": False,
        "opponent_control_seconds_per_td_allowed": False,
        "sub_attempts_per15": True,
        "opponent_sub_attempts_per15": False,
        "submission_loss_rate": False,
    }
    for metric, higher_better in rank_specs.items():
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

    composites = []
    for _, row in table.iterrows():
        chain = _avg(row, ["td_attempts_per15_pct", "repeat_attempt_rate_pct", "td_accuracy_pct"])
        control = _avg(row, ["control_per15_pct", "control_seconds_per_td_landed_pct"])
        td_resistance = _avg(row, ["td_defense_pct", "opponent_control_per15_pct"])
        bottom_escape = _avg(row, ["opponent_control_per15_pct", "opponent_control_seconds_per_td_allowed_pct"])
        sub_pressure = _avg(row, ["sub_attempts_per15_pct", "control_per15_pct"])
        sub_resistance = _avg(row, ["submission_loss_rate_pct", "opponent_sub_attempts_per15_pct", "bottom_escape_score"])
        composites.append((chain, control, td_resistance, bottom_escape, sub_pressure, sub_resistance))

    # bottom_escape_score is needed when constructing submission resistance, so set in two passes.
    table[["chain_wrestling_score", "control_retention_score", "takedown_resistance_score", "bottom_escape_score", "submission_pressure_score", "_sub_resistance_seed"]] = pd.DataFrame(composites, index=table.index)
    table["submission_resistance_score"] = table.apply(
        lambda row: _avg(row, ["submission_loss_rate_pct", "opponent_sub_attempts_per15_pct", "bottom_escape_score"]), axis=1
    )
    table = table.drop(columns=["_sub_resistance_seed"], errors="ignore")
    return table


def fighter_grappling_profile(table: pd.DataFrame, fighter: str) -> dict[str, Any]:
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


def _interaction_gap(a_attack: float | None, b_defense: float | None, b_attack: float | None, a_defense: float | None) -> float | None:
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


def build_advanced_grappling_matchup(
    table: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    *,
    config: UFCGrapplingConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCGrapplingConfig()
    a = fighter_grappling_profile(table, fighter_a)
    b = fighter_grappling_profile(table, fighter_b)

    specs = [
        (
            "Chain wrestling pressure vs resistance",
            "chain_wrestling_score", "takedown_resistance_score", 0.50,
            "Repeated takedown pressure is compared with opponent takedown resistance and ability to avoid extended control after wrestling exchanges.",
        ),
        (
            "Control retention vs bottom escape",
            "control_retention_score", "bottom_escape_score", 0.28,
            "Top-control retention is compared with a bottom-escape proxy based on how much control opponents sustain after successful takedowns.",
        ),
        (
            "Submission pressure vs submission resistance",
            "submission_pressure_score", "submission_resistance_score", 0.22,
            "Submission activity and control pressure are compared with recent submission-loss history, opponent submission activity and escape resistance.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    used_weight = 0.0
    for category, attack_key, defense_key, weight, explanation in specs:
        gap = _interaction_gap(_score(a, attack_key), _score(b, defense_key), _score(b, attack_key), _score(a, defense_key))
        if gap is None:
            continue
        advantage = "Even" if abs(gap) < 5 else (fighter_a if gap > 0 else fighter_b)
        rows.append({
            "category": category,
            "advantage": advantage,
            "strength": _strength(gap),
            "interaction_gap": float(gap),
            "weight": float(weight),
            "why": explanation,
        })
        weighted_gap += float(gap) * float(weight)
        used_weight += float(weight)

    if not rows or used_weight <= 0:
        return {
            "available": False,
            "version": GRAPPLING_VERSION,
            "fighter_a_profile": a,
            "fighter_b_profile": b,
            "rows": [],
            "weighted_gap": 0.0,
            "reliability": 0.0,
            "reason": "Detailed takedown/control history is not complete enough for advanced grappling interactions.",
        }

    weighted_gap /= used_weight
    sample_rel = min(1.0, min(int(a.get("sample", 0) or 0), int(b.get("sample", 0) or 0)) / float(config.min_sample_for_full_weight))
    completeness = min(float(a.get("data_completeness", 0.0) or 0.0), float(b.get("data_completeness", 0.0) or 0.0))
    reliability = sample_rel * (0.45 + 0.55 * completeness) * (len(rows) / len(specs))
    return {
        "available": True,
        "version": GRAPPLING_VERSION,
        "fighter_a_profile": a,
        "fighter_b_profile": b,
        "rows": rows,
        "weighted_gap": float(weighted_gap),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "guardrail": (
            "Advanced grappling upgrades the existing Style Matchups layer; it does not add a separate probability adjustment. "
            "Bottom escape and scramble behavior are statistical proxies because the tracked UFCStats schema does not contain explicit scramble/reversal events."
        ),
    }
