from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd

from engine.ufc_markets import build_derivative_markets
from engine.ufc_ratings import build_elo_history
from engine.ufc_simulation import simulate_fight


VALIDATION_VERSION = "Macabets UFC Historical Validation v0.1"


@dataclass(frozen=True)
class UFCValidationConfig:
    recent_fights: int = 8
    min_prior_fights: int = 2
    start_date: str = "2018-01-01"
    max_bouts: int = 1800
    min_division_sample: int = 25


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _method_bucket(method: Any) -> str:
    text = str(method or "").casefold()
    if "ko" in text or "tko" in text:
        return "KO/TKO"
    if "sub" in text:
        return "Submission"
    if "decision" in text:
        return "Decision"
    return "Other"


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
    return float((rnd - 1) * 5.0 + seconds / 60.0)


def _brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    if len(probabilities) == 0:
        return float("nan")
    return float(np.mean((probabilities - outcomes) ** 2))


def _log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    if len(probabilities) == 0:
        return float("nan")
    p = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def _calibration_rows(frame: pd.DataFrame, probability_col: str, outcome_col: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    bins = [0.0, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00001]
    labels = ["<45%", "45-55%", "55-65%", "65-75%", "75-85%", "85%+"]
    work = frame[[probability_col, outcome_col]].dropna().copy()
    work["bucket"] = pd.cut(work[probability_col], bins=bins, labels=labels, right=False, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for label in labels:
        part = work.loc[work["bucket"].astype(str) == label]
        if part.empty:
            continue
        predicted = float(part[probability_col].mean())
        actual = float(part[outcome_col].mean())
        rows.append({
            "bucket": label,
            "sample": int(len(part)),
            "predicted": predicted,
            "actual": actual,
            "calibration_gap": actual - predicted,
        })
    return rows


def _recent_profile(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    recent = rows[-limit:]
    if not recent:
        return {
            "sample": 0,
            "finish_rate": 0.0,
            "ko_win_rate": 0.0,
            "submission_win_rate": 0.0,
            "decision_win_rate": 0.0,
            "loss_finish_rate": 0.0,
        }
    wins = [r for r in recent if str(r.get("result", "")).upper() == "W"]
    losses = [r for r in recent if str(r.get("result", "")).upper() == "L"]
    win_methods = [_method_bucket(r.get("method")) for r in wins]
    loss_methods = [_method_bucket(r.get("method")) for r in losses]
    ko = sum(m == "KO/TKO" for m in win_methods)
    sub = sum(m == "Submission" for m in win_methods)
    dec = sum(m == "Decision" for m in win_methods)
    finish_losses = sum(m in {"KO/TKO", "Submission"} for m in loss_methods)
    return {
        "sample": int(len(recent)),
        "finish_rate": float((ko + sub) / len(wins)) if wins else 0.0,
        "ko_win_rate": float(ko / len(wins)) if wins else 0.0,
        "submission_win_rate": float(sub / len(wins)) if wins else 0.0,
        "decision_win_rate": float(dec / len(wins)) if wins else 0.0,
        "loss_finish_rate": float(finish_losses / len(losses)) if losses else 0.0,
    }


def _performance_proxy(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Build only the simulator traits available in the compact tracked fight-history schema.

    Historical snapshots of the full percentile performance table are not stored. This proxy
    therefore uses pre-fight UFC event-summary stats and deliberately avoids pretending it is an
    exact reconstruction of today's Performance/Style layers.
    """
    recent = rows[-limit:]
    if not recent:
        return {"sample": 0, "data_completeness": 0.0}

    total_minutes = sum(max(1.0 / 60.0, float(r.get("minutes", 0.0) or 0.0)) for r in recent)
    kd = sum(float(r.get("kd", 0.0) or 0.0) for r in recent)
    sub = sum(float(r.get("sub_att", 0.0) or 0.0) for r in recent)
    sig = sum(float(r.get("sig_str", 0.0) or 0.0) for r in recent)
    losses = [r for r in recent if str(r.get("result", "")).upper() == "L"]
    finish_losses = sum(_method_bucket(r.get("method")) in {"KO/TKO", "Submission"} for r in losses)

    kd_per15 = kd / max(total_minutes, 1e-9) * 15.0
    sub_per15 = sub / max(total_minutes, 1e-9) * 15.0
    sig_per15 = sig / max(total_minutes, 1e-9) * 15.0
    finish_loss_rate = float(finish_losses / len(losses)) if losses else 0.0

    return {
        "sample": int(len(recent)),
        "data_completeness": 0.35,
        "kd_per15_pct": _clip(kd_per15 * 28.0, 0.0, 100.0),
        "sub_attempts_per15_pct": _clip(sub_per15 * 38.0, 0.0, 100.0),
        "durability_score": _clip(100.0 * (1.0 - 0.75 * finish_loss_rate), 0.0, 100.0),
        "pace_score": _clip(sig_per15 / 115.0 * 100.0, 0.0, 100.0),
    }


def _canonical_bouts(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date", "fight_url", "fighter", "opponent"])
    rows: list[dict[str, Any]] = []
    for fight_url, bout in frame.sort_values(["event_date", "fight_url", "fighter"]).groupby("fight_url", sort=False):
        if len(bout) < 2:
            continue
        bout = bout.iloc[:2]
        a = bout.iloc[0]
        b = bout.iloc[1]
        ra = str(a.get("result", "")).upper()
        rb = str(b.get("result", "")).upper()
        if {ra, rb} != {"W", "L"}:
            continue
        rows.append({
            "fight_url": str(fight_url),
            "event_date": pd.Timestamp(a["event_date"]),
            "division": str(a.get("division", "Unknown") or "Unknown"),
            "fighter_a": str(a["fighter"]),
            "fighter_b": str(b["fighter"]),
            "result_a": ra,
            "result_b": rb,
            "method": str(a.get("method", "") or b.get("method", "")),
            "round": int(float(a.get("round", 0) or 0)) if str(a.get("round", "")).strip() else 0,
            "time": str(a.get("time", "") or ""),
        })
    return pd.DataFrame(rows).sort_values(["event_date", "fight_url"]).reset_index(drop=True)


def run_historical_validation(
    fights: pd.DataFrame,
    *,
    config: UFCValidationConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCValidationConfig()
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date"])

    elo_history, _ = build_elo_history(frame)
    if elo_history.empty:
        return {"available": False, "version": VALIDATION_VERSION, "reason": "No valid Elo history."}
    elo_lookup = {
        (str(row["fight_url"]), str(row["fighter"])): float(row["expected_win_prob"])
        for _, row in elo_history.iterrows()
    }

    bouts = _canonical_bouts(frame)
    start = pd.Timestamp(config.start_date)
    bouts = bouts.loc[bouts["event_date"] >= start].copy()
    if config.max_bouts and len(bouts) > int(config.max_bouts):
        bouts = bouts.tail(int(config.max_bouts)).copy()

    history: dict[str, list[dict[str, Any]]] = {}
    # Seed rolling histories with fights before the selected validation window.
    cutoff = bouts["event_date"].min() if not bouts.empty else start
    prior = frame.loc[frame["event_date"] < cutoff].sort_values(["event_date", "fight_url", "fighter"])
    for _, row in prior.iterrows():
        fighter = str(row.get("fighter", ""))
        history.setdefault(fighter, []).append({
            "event_date": row["event_date"],
            "result": row.get("result"),
            "method": row.get("method"),
            "round": row.get("round"),
            "time": row.get("time"),
            "minutes": _fight_minutes(row.get("round"), row.get("time")),
            "kd": row.get("kd", 0.0),
            "sig_str": row.get("sig_str", 0.0),
            "td": row.get("td", 0.0),
            "sub_att": row.get("sub_att", 0.0),
        })

    raw_by_fight = {str(k): v.copy() for k, v in frame.groupby(frame["fight_url"].astype(str), sort=False)}
    records: list[dict[str, Any]] = []

    for _, bout in bouts.iterrows():
        fight_url = str(bout["fight_url"])
        a = str(bout["fighter_a"])
        b = str(bout["fighter_b"])
        hist_a = history.get(a, [])
        hist_b = history.get(b, [])

        if len(hist_a) >= config.min_prior_fights and len(hist_b) >= config.min_prior_fights:
            p_a = elo_lookup.get((fight_url, a))
            if p_a is not None:
                recent_a = _recent_profile(hist_a, config.recent_fights)
                recent_b = _recent_profile(hist_b, config.recent_fights)
                perf_a = _performance_proxy(hist_a, config.recent_fights)
                perf_b = _performance_proxy(hist_b, config.recent_fights)

                # Compact history does not store scheduled rounds. A bout that reaches R4/R5 is
                # known to be five rounds; otherwise v0.1 uses three rounds and flags the limitation.
                observed_round = int(bout.get("round", 0) or 0)
                scheduled_rounds = 5 if observed_round >= 4 else 3
                sim = simulate_fight(a, b, float(p_a), recent_a, recent_b, perf_a, perf_b, rounds=scheduled_rounds)
                markets = build_derivative_markets(sim, a, b)

                result_a = 1 if str(bout["result_a"]).upper() == "W" else 0
                method = _method_bucket(bout.get("method"))
                actual_finish = 0 if method == "Decision" else 1
                elapsed = _fight_minutes(bout.get("round"), bout.get("time"))

                record = {
                    "fight_url": fight_url,
                    "event_date": bout["event_date"].date().isoformat(),
                    "division": bout.get("division", "Unknown"),
                    "fighter_a": a,
                    "fighter_b": b,
                    "p_a": float(p_a),
                    "a_win": result_a,
                    "predicted_winner_correct": int((p_a >= 0.5) == bool(result_a)),
                    "finish_probability": float(sim["finish_probability"]),
                    "actual_finish": actual_finish,
                    "goes_distance_probability": float(sim["goes_distance_probability"]),
                    "actual_goes_distance": 1 - actual_finish,
                    "a_ko_probability": float(sim["a_ko_tko_probability"]),
                    "a_sub_probability": float(sim["a_submission_probability"]),
                    "a_dec_probability": float(sim["a_decision_probability"]),
                    "b_ko_probability": float(sim["b_ko_tko_probability"]),
                    "b_sub_probability": float(sim["b_submission_probability"]),
                    "b_dec_probability": float(sim["b_decision_probability"]),
                    "actual_method": method,
                    "actual_elapsed_minutes": elapsed,
                    "scheduled_rounds_proxy": scheduled_rounds,
                }

                path_probs = {
                    "a_ko": record["a_ko_probability"],
                    "a_sub": record["a_sub_probability"],
                    "a_dec": record["a_dec_probability"],
                    "b_ko": record["b_ko_probability"],
                    "b_sub": record["b_sub_probability"],
                    "b_dec": record["b_dec_probability"],
                }
                actual_path = ("a_" if result_a else "b_") + ({"KO/TKO": "ko", "Submission": "sub", "Decision": "dec"}.get(method, "other"))
                predicted_path = max(path_probs, key=path_probs.get)
                record["actual_path"] = actual_path
                record["predicted_path"] = predicted_path
                record["top_path_correct"] = int(actual_path == predicted_path)

                for total in markets.get("round_totals", []):
                    line = float(total["line"])
                    key = str(line).replace(".", "_")
                    record[f"over_{key}_probability"] = float(total["over_probability"])
                    record[f"actual_over_{key}"] = int(elapsed > line * 5.0)
                records.append(record)

        raw = raw_by_fight.get(fight_url)
        if raw is not None:
            for _, row in raw.iterrows():
                fighter = str(row.get("fighter", ""))
                history.setdefault(fighter, []).append({
                    "event_date": row.get("event_date"),
                    "result": row.get("result"),
                    "method": row.get("method"),
                    "round": row.get("round"),
                    "time": row.get("time"),
                    "minutes": _fight_minutes(row.get("round"), row.get("time")),
                    "kd": row.get("kd", 0.0),
                    "sig_str": row.get("sig_str", 0.0),
                    "td": row.get("td", 0.0),
                    "sub_att": row.get("sub_att", 0.0),
                })

    backtest = pd.DataFrame(records)
    if backtest.empty:
        return {"available": False, "version": VALIDATION_VERSION, "reason": "No eligible historical bouts."}

    p = backtest["p_a"].to_numpy(float)
    y = backtest["a_win"].to_numpy(float)
    finish_p = backtest["finish_probability"].to_numpy(float)
    finish_y = backtest["actual_finish"].to_numpy(float)

    totals: dict[str, Any] = {}
    # Only 1.5 and 2.5 are reported in v0.1 because compact history does not store
    # scheduled rounds. Reporting 3.5/4.5 would select only bouts observed in R4/R5
    # and create severe survivorship bias.
    for line in (1.5, 2.5):
        key = str(line).replace(".", "_")
        p_col = f"over_{key}_probability"
        y_col = f"actual_over_{key}"
        if p_col not in backtest or y_col not in backtest:
            continue
        part = backtest[[p_col, y_col]].dropna()
        if part.empty:
            continue
        totals[str(line)] = {
            "sample": int(len(part)),
            "mean_predicted_over": float(part[p_col].mean()),
            "actual_over_rate": float(part[y_col].mean()),
            "brier": _brier(part[p_col].to_numpy(float), part[y_col].to_numpy(float)),
            "calibration_gap": float(part[y_col].mean() - part[p_col].mean()),
        }

    divisions = []
    for division, part in backtest.groupby("division"):
        if len(part) < config.min_division_sample:
            continue
        divisions.append({
            "division": str(division),
            "sample": int(len(part)),
            "winner_accuracy": float(part["predicted_winner_correct"].mean()),
            "moneyline_brier": _brier(part["p_a"].to_numpy(float), part["a_win"].to_numpy(float)),
            "predicted_finish_rate": float(part["finish_probability"].mean()),
            "actual_finish_rate": float(part["actual_finish"].mean()),
            "finish_gap": float(part["actual_finish"].mean() - part["finish_probability"].mean()),
        })
    divisions.sort(key=lambda row: row["sample"], reverse=True)

    method_summary = {
        "ko_tko": {
            "predicted": float((backtest["a_ko_probability"] + backtest["b_ko_probability"]).mean()),
            "actual": float(backtest["actual_method"].eq("KO/TKO").mean()),
        },
        "submission": {
            "predicted": float((backtest["a_sub_probability"] + backtest["b_sub_probability"]).mean()),
            "actual": float(backtest["actual_method"].eq("Submission").mean()),
        },
        "decision": {
            "predicted": float((backtest["a_dec_probability"] + backtest["b_dec_probability"]).mean()),
            "actual": float(backtest["actual_method"].eq("Decision").mean()),
        },
    }
    for item in method_summary.values():
        item["gap"] = item["actual"] - item["predicted"]

    finish_gap = float(backtest["actual_finish"].mean() - backtest["finish_probability"].mean())
    if finish_gap <= -0.035:
        finish_bias = "Too finish-heavy"
    elif finish_gap >= 0.035:
        finish_bias = "Too decision-heavy"
    else:
        finish_bias = "Finish rate reasonably centered"

    return {
        "available": True,
        "version": VALIDATION_VERSION,
        "sample": int(len(backtest)),
        "date_start": str(backtest["event_date"].min()),
        "date_end": str(backtest["event_date"].max()),
        "moneyline": {
            "winner_accuracy": float(backtest["predicted_winner_correct"].mean()),
            "brier": _brier(p, y),
            "log_loss": _log_loss(p, y),
            "mean_favorite_probability": float(np.maximum(p, 1.0 - p).mean()),
            "calibration": _calibration_rows(backtest, "p_a", "a_win"),
        },
        "finish_distance": {
            "predicted_finish_rate": float(backtest["finish_probability"].mean()),
            "actual_finish_rate": float(backtest["actual_finish"].mean()),
            "finish_gap": finish_gap,
            "finish_brier": _brier(finish_p, finish_y),
            "bias_label": finish_bias,
            "calibration": _calibration_rows(backtest, "finish_probability", "actual_finish"),
        },
        "methods": method_summary,
        "top_method_path_accuracy": float(backtest["top_path_correct"].mean()),
        "round_totals": totals,
        "divisions": divisions[:12],
        "limitations": [
            "This is a leakage-safe retrospective proxy, not an exact historical reconstruction of today's full Macabets UFC stack. Elo win probabilities and all recent tendencies are taken only from fights that occurred before each tested bout.",
            "The tracked compact fight-history CSV does not preserve historical full Performance/Style percentile snapshots, so v0.1 validates the side baseline plus simulation/method decomposition using only pre-fight summary stats available in the repository.",
            "Scheduled-round count is not stored in the compact history. Bouts observed in rounds 4-5 are known five-round fights; earlier finishes in five-round fights may be treated as three-round bouts in this validation proxy. For that reason v0.1 reports only 1.5- and 2.5-round total calibration, avoiding biased 3.5/4.5 samples.",
            "No sportsbook closing prices are stored for historical UFC fights, so this report validates probability calibration and outcomes, not historical betting ROI or CLV.",
            "Use this report to calibrate simulation finish/method behavior first. Do not change Performance/Style/Context probability caps solely from this proxy backtest.",
        ],
        "records": backtest.to_dict(orient="records"),
    }
