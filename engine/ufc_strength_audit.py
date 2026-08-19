from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd

from engine.ufc_ratings import UFCRatingConfig, _dominance_multiplier, _finish_multiplier, _is_ranking_division, _score


STRENGTH_AUDIT_VERSION = "Macabets UFC Strength Audit v0.1 — Leakage-Safe Backbone Calibration"


@dataclass(frozen=True)
class UFCStrengthAuditConfig:
    start_date: str = "2018-01-01"
    max_bouts: int = 1800
    min_prior_fights: int = 2
    holdout_fraction: float = 0.30
    min_group_sample: int = 20


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _expected(a: float, b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((float(b) - float(a)) / float(scale)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    if not len(p):
        return float("nan")
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _winner_accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y >= 0.5))) if len(p) else float("nan")


def _shrink(value: float, center: float, observations: int, prior_strength: float) -> float:
    weight = observations / (observations + prior_strength) if observations > 0 else 0.0
    return center + (value - center) * weight


def _recent_form_score(history: list[dict[str, Any]], today: pd.Timestamp, cfg: UFCRatingConfig) -> float:
    if not history:
        return 50.0
    recent = history[-cfg.recent_fights:][::-1]
    weighted_points = 0.0
    total_weight = 0.0
    for i, row in enumerate(recent):
        score = _score(row.get("result"))
        if score is None:
            continue
        event_date = pd.Timestamp(row.get("event_date"))
        days = max(0, int((today - event_date).days))
        weight = math.exp(-days / 720.0) * (0.88 ** i)
        weighted_points += float(score) * weight
        total_weight += weight
    return 50.0 if total_weight <= 0 else 100.0 * weighted_points / total_weight


def _schedule_score(history: list[dict[str, Any]], base: float) -> float:
    recent = history[-8:][::-1]
    if not recent:
        return base
    vals: list[float] = []
    weights: list[float] = []
    for i, row in enumerate(recent):
        vals.append(float(row.get("opponent_rating_before", base) or base))
        weights.append(0.86 ** i)
    return float(np.average(vals, weights=weights))


def _fighter_pre_fight_rating(
    fighter: str,
    target_division: str,
    bout_date: pd.Timestamp,
    global_ratings: dict[str, float],
    division_ratings: dict[tuple[str, str], float],
    histories: dict[str, list[dict[str, Any]]],
    division_counts: dict[tuple[str, str], int],
    cfg: UFCRatingConfig,
) -> dict[str, float]:
    hist = histories.get(fighter, [])
    total_fights = len(hist)
    raw_global = float(global_ratings.get(fighter, cfg.base_elo))
    raw_division = float(division_ratings.get((fighter, target_division), cfg.base_elo))
    division_fights = int(division_counts.get((fighter, target_division), 0)) if _is_ranking_division(target_division) else 0
    division_evidence = division_fights / (division_fights + 3.0) if division_fights else 0.0
    division_component = raw_global + (raw_division - raw_global) * division_evidence

    recent_form = _recent_form_score(hist, bout_date, cfg)
    recent_obs = min(cfg.recent_fights, total_fights)
    adjusted_form = _shrink(recent_form, 50.0, recent_obs, 3.0)
    schedule_rating = _schedule_score(hist, cfg.base_elo)
    adjusted_schedule = _shrink(schedule_rating, cfg.base_elo, recent_obs, 4.0)
    form_component = cfg.base_elo + (adjusted_form - 50.0) * 3.0

    composite = (
        raw_global
        + 0.18 * (division_component - raw_global)
        + 0.10 * (form_component - cfg.base_elo)
        + 0.08 * (adjusted_schedule - cfg.base_elo)
    )
    experience_reliability = min(1.0, 0.45 + 0.08 * total_fights)
    composite = cfg.base_elo + (composite - cfg.base_elo) * experience_reliability

    inactivity_penalty = 0.0
    if hist:
        last_date = pd.Timestamp(hist[-1]["event_date"])
        days_inactive = max(0, int((bout_date - last_date).days))
        penalty_days = max(0, days_inactive - cfg.inactivity_grace_days)
        inactivity_penalty = min(
            cfg.max_inactivity_penalty,
            (penalty_days / 30.0) * cfg.inactivity_penalty_per_30d,
        )
    else:
        days_inactive = 0
    adjusted_rating = composite - inactivity_penalty
    ranking_confidence = 100.0 * min(1.0, 0.68 * experience_reliability + 0.32 * division_evidence)
    return {
        "rating": float(adjusted_rating),
        "global_elo": raw_global,
        "division_elo": raw_division,
        "division_component": float(division_component),
        "division_evidence": float(division_evidence),
        "division_fights": float(division_fights),
        "total_fights": float(total_fights),
        "ranking_confidence": float(ranking_confidence),
        "inactivity_penalty": float(inactivity_penalty),
        "days_inactive": float(days_inactive),
    }


def _prepare(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date", "fight_url", "fighter", "opponent"])
    stat_map = {"sig_str": "opponent_sig_str", "kd": "opponent_kd", "td": "opponent_td", "sub_att": "opponent_sub_att"}
    cols = [c for c in stat_map if c in frame.columns]
    opp = frame[["fight_url", "fighter"] + cols].copy().rename(columns={"fighter": "opponent", **{c: stat_map[c] for c in cols}})
    frame = frame.merge(opp, on=["fight_url", "opponent"], how="left")
    return frame.sort_values(["event_date", "fight_url", "fighter"]).reset_index(drop=True)


def _fit_temperature(train_p: np.ndarray, train_y: np.ndarray) -> float:
    # Temperature on centered Elo-logit: >1 softens confidence, <1 sharpens it.
    if not len(train_p):
        return 1.0
    logits = np.log(np.clip(train_p, 1e-6, 1 - 1e-6) / np.clip(1 - train_p, 1e-6, 1 - 1e-6))
    best_t = 1.0
    best_loss = float("inf")
    for t in np.linspace(0.65, 2.00, 136):
        calibrated = 1.0 / (1.0 + np.exp(-logits / t))
        loss = _log_loss(calibrated, train_y)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return 1.0 / (1.0 + np.exp(-logits / float(temperature)))


def _calibration_table(frame: pd.DataFrame, p_col: str) -> list[dict[str, Any]]:
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90%+"]
    fav_p = np.maximum(frame[p_col].to_numpy(float), 1.0 - frame[p_col].to_numpy(float))
    fav_win = ((frame[p_col].to_numpy(float) >= 0.5) == (frame["a_win"].to_numpy(int) == 1)).astype(int)
    bucketed = pd.DataFrame({"p": fav_p, "win": fav_win})
    bucketed["bucket"] = pd.cut(bucketed["p"], bins=bins, labels=labels, right=False, include_lowest=True)
    out: list[dict[str, Any]] = []
    for label in labels:
        part = bucketed.loc[bucketed["bucket"].astype(str) == label]
        if part.empty:
            continue
        out.append({
            "bucket": label,
            "sample": int(len(part)),
            "mean_predicted": float(part["p"].mean()),
            "actual_win_rate": float(part["win"].mean()),
            "gap": float(part["win"].mean() - part["p"].mean()),
        })
    return out




def _scale_metrics(frame: pd.DataFrame, scale: float) -> dict[str, float]:
    if frame.empty:
        return {"sample": 0, "scale": float(scale), "brier": float("nan"), "log_loss": float("nan"), "winner_accuracy": float("nan"), "mean_favorite_probability": float("nan")}
    gap = frame["rating_a"].to_numpy(float) - frame["rating_b"].to_numpy(float)
    p = 1.0 / (1.0 + np.power(10.0, -gap / float(scale)))
    y = frame["a_win"].to_numpy(float)
    return {
        "sample": int(len(frame)),
        "scale": float(scale),
        "brier": _brier(p, y),
        "log_loss": _log_loss(p, y),
        "winner_accuracy": _winner_accuracy(p, y),
        "mean_favorite_probability": float(np.maximum(p, 1.0 - p).mean()),
    }


def _scale_stability(backtest: pd.DataFrame) -> dict[str, Any]:
    """Stress-test Elo probability scale across several recent chronological windows.

    This intentionally does not select a live scale. A scale is considered stable only
    when neighboring recent windows prefer broadly similar values. Large disagreement
    is a signal that one global sharpening/softening constant would be fragile.
    """
    if backtest.empty:
        return {"available": False}
    candidates = [280, 300, 320, 340, 360, 380, 400, 420, 450, 500]
    requested_windows = [240, 360, 540, 780]
    windows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for requested in requested_windows:
        size = min(int(requested), len(backtest))
        if size < 120 or size in seen:
            continue
        seen.add(size)
        part = backtest.tail(size)
        rows = [_scale_metrics(part, scale) for scale in candidates]
        best = min(rows, key=lambda row: (row["log_loss"], row["brier"]))
        current = next(row for row in rows if int(row["scale"]) == 400)
        windows.append({
            "window": int(size),
            "date_start": str(part["event_date"].min()),
            "date_end": str(part["event_date"].max()),
            "best_scale": int(best["scale"]),
            "best_brier": float(best["brier"]),
            "best_log_loss": float(best["log_loss"]),
            "scale_400_brier": float(current["brier"]),
            "scale_400_log_loss": float(current["log_loss"]),
            "brier_gain_vs_400": float(current["brier"] - best["brier"]),
            "log_loss_gain_vs_400": float(current["log_loss"] - best["log_loss"]),
        })
    best_scales = [row["best_scale"] for row in windows]
    spread = (max(best_scales) - min(best_scales)) if best_scales else 0
    if not best_scales:
        recommendation = "insufficient sample"
    elif spread <= 40:
        recommendation = "stable enough to consider a live scale trial"
    elif spread <= 100:
        recommendation = "mixed; keep 400 live and gather more evidence"
    else:
        recommendation = "unstable across time; do not apply one global scale"
    return {
        "available": bool(windows),
        "candidate_scales": candidates,
        "windows": windows,
        "best_scale_spread": int(spread),
        "recommendation": recommendation,
    }


def run_strength_backbone_audit(
    fights: pd.DataFrame,
    *,
    config: UFCStrengthAuditConfig | None = None,
    rating_config: UFCRatingConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCStrengthAuditConfig()
    cfg = rating_config or UFCRatingConfig()
    frame = _prepare(fights)
    if frame.empty:
        return {"available": False, "version": STRENGTH_AUDIT_VERSION, "reason": "No fight history."}

    global_ratings: dict[str, float] = {}
    division_ratings: dict[tuple[str, str], float] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    division_counts: dict[tuple[str, str], int] = {}
    records: list[dict[str, Any]] = []

    for fight_url, bout in frame.groupby("fight_url", sort=False):
        if len(bout) < 2:
            continue
        bout = bout.iloc[:2]
        a = bout.iloc[0]
        b = bout.iloc[1]
        score_a = _score(a.get("result"))
        score_b = _score(b.get("result"))
        if score_a is None or score_b is None or {score_a, score_b} != {0.0, 1.0}:
            continue
        fighter_a = str(a.get("fighter"))
        fighter_b = str(b.get("fighter"))
        division = str(a.get("division") or "Unknown")
        bout_date = pd.Timestamp(a.get("event_date"))

        ra = _fighter_pre_fight_rating(fighter_a, division, bout_date, global_ratings, division_ratings, histories, division_counts, cfg)
        rb = _fighter_pre_fight_rating(fighter_b, division, bout_date, global_ratings, division_ratings, histories, division_counts, cfg)
        p_strength = _expected(ra["rating"], rb["rating"])
        p_global = _expected(ra["global_elo"], rb["global_elo"])

        eligible = (
            bout_date >= pd.Timestamp(config.start_date)
            and int(ra["total_fights"]) >= config.min_prior_fights
            and int(rb["total_fights"]) >= config.min_prior_fights
        )
        if eligible:
            records.append({
                "fight_url": str(fight_url),
                "event_date": bout_date.date().isoformat(),
                "division": division,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "a_win": int(score_a == 1.0),
                "p_strength": float(p_strength),
                "p_global": float(p_global),
                "rating_a": ra["rating"],
                "rating_b": rb["rating"],
                "rating_gap": abs(ra["rating"] - rb["rating"]),
                "min_confidence": min(ra["ranking_confidence"], rb["ranking_confidence"]),
                "division_transition": int(min(ra["division_fights"], rb["division_fights"]) <= 1),
                "a_division_fights": int(ra["division_fights"]),
                "b_division_fights": int(rb["division_fights"]),
            })

        # Update ratings only after recording the pre-fight prediction.
        ga = global_ratings.get(fighter_a, cfg.base_elo)
        gb = global_ratings.get(fighter_b, cfg.base_elo)
        exp_a = _expected(ga, gb)
        exp_b = 1.0 - exp_a
        finish_mult = max(_finish_multiplier(a.get("method")), _finish_multiplier(b.get("method")))
        dom_a = _dominance_multiplier(a)
        dom_b = _dominance_multiplier(b)
        global_ratings[fighter_a] = ga + cfg.k_factor * finish_mult * dom_a * (score_a - exp_a)
        global_ratings[fighter_b] = gb + cfg.k_factor * finish_mult * dom_b * (score_b - exp_b)

        if _is_ranking_division(division):
            key_a = (fighter_a, division)
            key_b = (fighter_b, division)
            da = division_ratings.get(key_a, cfg.base_elo)
            db = division_ratings.get(key_b, cfg.base_elo)
            dexp_a = _expected(da, db)
            dexp_b = 1.0 - dexp_a
            division_ratings[key_a] = da + cfg.division_k_factor * finish_mult * dom_a * (score_a - dexp_a)
            division_ratings[key_b] = db + cfg.division_k_factor * finish_mult * dom_b * (score_b - dexp_b)
            division_counts[key_a] = division_counts.get(key_a, 0) + 1
            division_counts[key_b] = division_counts.get(key_b, 0) + 1

        histories.setdefault(fighter_a, []).append({
            "event_date": bout_date,
            "result": a.get("result"),
            "opponent_rating_before": gb,
        })
        histories.setdefault(fighter_b, []).append({
            "event_date": bout_date,
            "result": b.get("result"),
            "opponent_rating_before": ga,
        })

    backtest = pd.DataFrame(records)
    if config.max_bouts and len(backtest) > int(config.max_bouts):
        backtest = backtest.tail(int(config.max_bouts)).reset_index(drop=True)
    if backtest.empty:
        return {"available": False, "version": STRENGTH_AUDIT_VERSION, "reason": "No eligible bouts."}

    split = max(1, min(len(backtest) - 1, int(round(len(backtest) * (1.0 - config.holdout_fraction))))) if len(backtest) > 1 else 1
    train = backtest.iloc[:split]
    holdout = backtest.iloc[split:] if split < len(backtest) else backtest.iloc[0:0]
    train_p = train["p_strength"].to_numpy(float)
    train_y = train["a_win"].to_numpy(float)
    temperature = _fit_temperature(train_p, train_y)
    backtest["p_calibrated"] = _apply_temperature(backtest["p_strength"].to_numpy(float), temperature)

    def metrics(part: pd.DataFrame, col: str) -> dict[str, float]:
        p = part[col].to_numpy(float)
        y = part["a_win"].to_numpy(float)
        return {
            "sample": int(len(part)),
            "winner_accuracy": _winner_accuracy(p, y),
            "brier": _brier(p, y),
            "log_loss": _log_loss(p, y),
            "mean_favorite_probability": float(np.maximum(p, 1 - p).mean()) if len(p) else float("nan"),
        }

    groups: list[dict[str, Any]] = []
    group_specs = [
        ("New-division / <=1 prior division fight", backtest["division_transition"].eq(1)),
        ("Established division", backtest["division_transition"].eq(0)),
        ("High-confidence pair (>=75)", backtest["min_confidence"].ge(75)),
        ("Lower-confidence pair (<75)", backtest["min_confidence"].lt(75)),
        ("Large rating gap (>=125)", backtest["rating_gap"].ge(125)),
        ("Close rating gap (<75)", backtest["rating_gap"].lt(75)),
    ]
    for label, mask in group_specs:
        part = backtest.loc[mask]
        if len(part) < config.min_group_sample:
            continue
        m = metrics(part, "p_strength")
        groups.append({"group": label, **m})

    holdout = backtest.iloc[split:] if split < len(backtest) else backtest.iloc[0:0]
    holdout_strength = metrics(holdout, "p_strength") if len(holdout) else {}
    holdout_global = metrics(holdout, "p_global") if len(holdout) else {}
    holdout_calibrated = metrics(holdout, "p_calibrated") if len(holdout) else {}

    scale_stability = _scale_stability(backtest)

    return {
        "available": True,
        "version": STRENGTH_AUDIT_VERSION,
        "sample": int(len(backtest)),
        "date_start": str(backtest["event_date"].min()),
        "date_end": str(backtest["event_date"].max()),
        "train_sample": int(len(train)),
        "holdout_sample": int(len(holdout)),
        "all_strength": metrics(backtest, "p_strength"),
        "all_global_elo": metrics(backtest, "p_global"),
        "holdout_strength": holdout_strength,
        "holdout_global_elo": holdout_global,
        "temperature_fit": float(temperature),
        "holdout_temperature_calibrated": holdout_calibrated,
        "favorite_calibration": _calibration_table(holdout if len(holdout) else backtest, "p_strength"),
        "favorite_calibration_calibrated": _calibration_table(holdout if len(holdout) else backtest, "p_calibrated"),
        "groups": groups,
        "scale_stability": scale_stability,
        "interpretation": {
            "temperature_direction": "soften" if temperature > 1.05 else ("sharpen" if temperature < 0.95 else "leave near current scale"),
            "calibration_note": "Temperature is fitted on the earlier training segment and evaluated only on the later holdout segment. It is diagnostic and is not automatically applied to live Macabets probabilities.",
        },
        "limitations": [
            "This audit reconstructs the Strength v0.3 rating backbone leakage-safely before each bout. It does not reconstruct later Performance, Style, Cardio, Damage, Context, or Simulation side adjustments.",
            "Upcoming fight division is used as the target division for the pre-fight division-residual calculation; no bout result or post-fight statistic is used before prediction.",
            "The temperature fit is a one-parameter calibration diagnostic trained on the earlier chronological segment and tested on the later segment. It should not be promoted into the live model unless the holdout improvement is persistent across larger and future samples.",
            "No historical sportsbook closing odds are available here, so this measures predictive calibration rather than betting ROI or CLV.",
            "Probability-scale stability is tested across multiple recent chronological windows. If those windows disagree materially, Macabets should not promote one global Elo scale into production.",
        ],
        "records": backtest.to_dict(orient="records"),
    }
