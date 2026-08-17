from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np
import pandas as pd

from engine.ufc_performance import (
    PERFORMANCE_VERSION,
    build_performance_table,
    fighter_performance,
    matchup_performance_adjustment,
)
from engine.ufc_style_matchups import STYLE_VERSION, build_style_matchup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS_PATH = ROOT / "data" / "ufc" / "fighter_ratings.csv"
DEFAULT_FIGHTS_PATH = ROOT / "data" / "ufc" / "ufc_fight_history.csv"
MODEL_VERSION = "Macabets UFC Analysis v0.3"
RATING_VERSION = "Macabets UFC Strength v0.2"


class UFCAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class UFCAnalysisConfig:
    elo_scale: float = 400.0
    recent_fights: int = 8
    min_confidence: int = 45
    max_confidence: int = 74
    bet_roi_threshold: float = 0.05
    watch_roi_threshold: float = 0.02


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def _implied_probability(odds: int | float) -> float:
    decimal = _american_to_decimal(odds)
    return 1.0 / decimal


def _probability_to_american(probability: float) -> int:
    p = _clip(float(probability), 0.001, 0.999)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def _elo_probability(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((float(rating_b) - float(rating_a)) / float(scale)))


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise UFCAnalysisError(
            f"{label} is not available at {path}. Run the Update Macabets UFC Data workflow first."
        )
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise UFCAnalysisError(f"Could not read {label}: {exc}") from exc
    if frame.empty:
        raise UFCAnalysisError(f"{label} is empty. Run the Update Macabets UFC Data workflow first.")
    return frame


def load_ufc_ratings(path: Path | str | None = None) -> pd.DataFrame:
    ratings_path = Path(path) if path is not None else DEFAULT_RATINGS_PATH
    frame = _load_csv(ratings_path, "UFC fighter ratings")
    required = {
        "fighter", "division", "macabets_rating", "strength_score", "ranking_confidence",
        "active_pool", "division_rank", "recent_form_adjusted", "schedule_rating",
        "ufc_wins", "ufc_losses", "ufc_draws", "ufc_finishes", "division_fights",
        "days_inactive",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise UFCAnalysisError(
            "UFC fighter ratings use an older schema. Re-run the UFC workflow after installing "
            f"Strength v0.2. Missing: {', '.join(sorted(missing))}"
        )
    frame["fighter"] = frame["fighter"].astype(str).str.strip()
    frame["active_pool"] = frame["active_pool"].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame


def load_ufc_fights(path: Path | str | None = None) -> pd.DataFrame:
    fights_path = Path(path) if path is not None else DEFAULT_FIGHTS_PATH
    frame = _load_csv(fights_path, "UFC fight history")
    required = {"event_date", "fight_url", "fighter", "opponent", "result", "division", "method"}
    missing = required.difference(frame.columns)
    if missing:
        raise UFCAnalysisError(
            "UFC fight history is missing required fields: " + ", ".join(sorted(missing))
        )
    frame["fighter"] = frame["fighter"].astype(str).str.strip()
    frame["opponent"] = frame["opponent"].astype(str).str.strip()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    return frame.dropna(subset=["event_date"]).copy()


def fighter_names(
    ratings: pd.DataFrame | None = None,
    *,
    active_only: bool = True,
) -> list[str]:
    ratings = load_ufc_ratings() if ratings is None else ratings.copy()
    if active_only and "active_pool" in ratings:
        mask = ratings["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        ratings = ratings.loc[mask]
    return sorted(ratings["fighter"].dropna().astype(str).str.strip().unique().tolist())


def _fighter_row(ratings: pd.DataFrame, fighter: str) -> pd.Series:
    match = ratings.loc[ratings["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()]
    if match.empty:
        raise UFCAnalysisError(f"{fighter} was not found in the UFC fighter ratings.")
    # One current row per fighter is expected; last is defensive if schema ever duplicates.
    return match.iloc[-1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _method_bucket(method: Any) -> str:
    text = str(method or "").casefold()
    if "ko" in text or "tko" in text:
        return "KO/TKO"
    if "sub" in text:
        return "Submission"
    if "decision" in text:
        return "Decision"
    return "Other"


def _attach_opponent_stats(fights: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's event-summary totals to each fighter row.

    The stored UFC history intentionally uses one row per fighter. Keeping the raw
    snapshot at that grain avoids duplication, while this lightweight self-join lets
    the analysis UI show descriptive differentials without changing the rating model.
    """
    frame = fights.copy()
    stats = [c for c in ("sig_str", "kd", "td", "sub_att") if c in frame.columns]
    if not stats or not {"fight_url", "fighter", "opponent"}.issubset(frame.columns):
        return frame
    opponent = frame[["fight_url", "fighter"] + stats].copy()
    opponent = opponent.rename(
        columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stats}}
    )
    return frame.merge(opponent, on=["fight_url", "opponent"], how="left")


def _recent_profile(fights: pd.DataFrame, fighter: str, limit: int) -> dict[str, Any]:
    rows = fights.loc[
        fights["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()
    ].sort_values("event_date", ascending=False).head(limit).copy()

    if rows.empty:
        return {
            "sample": 0,
            "record": "0-0",
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "finish_rate": 0.0,
            "ko_win_rate": 0.0,
            "submission_win_rate": 0.0,
            "decision_win_rate": 0.0,
            "loss_finish_rate": 0.0,
            "sig_str_per_fight": None,
            "sig_str_diff_per_fight": None,
            "kd_per_fight": None,
            "td_per_fight": None,
            "sub_att_per_fight": None,
            "last_fight_date": "",
        }

    results = rows["result"].astype(str).str.upper()
    wins = int(results.eq("W").sum())
    losses = int(results.eq("L").sum())
    draws = int(results.isin(["D", "DRAW"]).sum())
    sample = len(rows)

    method_buckets = rows["method"].map(_method_bucket)
    win_rows = rows.loc[results.eq("W")].copy()
    loss_rows = rows.loc[results.eq("L")].copy()
    win_methods = win_rows["method"].map(_method_bucket) if not win_rows.empty else pd.Series(dtype=str)
    loss_methods = loss_rows["method"].map(_method_bucket) if not loss_rows.empty else pd.Series(dtype=str)

    def mean_col(column: str) -> float | None:
        if column not in rows.columns:
            return None
        values = pd.to_numeric(rows[column], errors="coerce")
        return None if not values.notna().any() else float(values.mean())

    sig_for = mean_col("sig_str")
    sig_diff = None
    if "sig_str" in rows.columns and "opponent_sig_str" in rows.columns:
        a = pd.to_numeric(rows["sig_str"], errors="coerce")
        b = pd.to_numeric(rows["opponent_sig_str"], errors="coerce")
        valid = a.notna() & b.notna()
        if valid.any():
            sig_diff = float((a[valid] - b[valid]).mean())

    decisions = int(win_methods.eq("Decision").sum())
    ko_wins = int(win_methods.eq("KO/TKO").sum())
    sub_wins = int(win_methods.eq("Submission").sum())
    finish_wins = ko_wins + sub_wins
    loss_finishes = int(loss_methods.isin(["KO/TKO", "Submission"]).sum())

    return {
        "sample": int(sample),
        "record": f"{wins}-{losses}" + (f"-{draws}" if draws else ""),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "finish_rate": float(finish_wins / wins) if wins else 0.0,
        "ko_win_rate": float(ko_wins / wins) if wins else 0.0,
        "submission_win_rate": float(sub_wins / wins) if wins else 0.0,
        "decision_win_rate": float(decisions / wins) if wins else 0.0,
        "loss_finish_rate": float(loss_finishes / losses) if losses else 0.0,
        "sig_str_per_fight": sig_for,
        "sig_str_diff_per_fight": sig_diff,
        "kd_per_fight": mean_col("kd"),
        "td_per_fight": mean_col("td"),
        "sub_att_per_fight": mean_col("sub_att"),
        "last_fight_date": rows.iloc[0]["event_date"].date().isoformat(),
    }


def _confidence(
    row_a: pd.Series,
    row_b: pd.Series,
    probability_a: float,
    config: UFCAnalysisConfig,
) -> tuple[int, str]:
    reliability = min(
        _safe_float(row_a.get("ranking_confidence"), 50.0),
        _safe_float(row_b.get("ranking_confidence"), 50.0),
    )
    sample = min(
        _safe_int(row_a.get("division_fights"), 0),
        _safe_int(row_b.get("division_fights"), 0),
    )
    sample_component = min(10.0, sample * 1.1)
    separation_component = min(8.0, abs(probability_a - 0.5) * 40.0)
    score = 34.0 + reliability * 0.28 + sample_component + separation_component
    score = int(round(_clip(score, config.min_confidence, config.max_confidence)))
    if score >= 70:
        band = "High baseline confidence"
    elif score >= 60:
        band = "Moderate baseline confidence"
    else:
        band = "Limited baseline confidence"
    return score, band


def _market_evaluation(
    probability_a: float,
    odds_a: int | None,
    odds_b: int | None,
    confidence: int,
    config: UFCAnalysisConfig,
) -> dict[str, Any]:
    if odds_a in (None, 0) or odds_b in (None, 0):
        return {"available": False}

    p_a_market = _implied_probability(int(odds_a))
    p_b_market = _implied_probability(int(odds_b))
    hold = p_a_market + p_b_market - 1.0
    denom = p_a_market + p_b_market
    no_vig_a = p_a_market / denom
    no_vig_b = p_b_market / denom
    probability_b = 1.0 - probability_a

    roi_a = probability_a * (_american_to_decimal(int(odds_a)) - 1.0) - probability_b
    roi_b = probability_b * (_american_to_decimal(int(odds_b)) - 1.0) - probability_a

    def verdict(roi: float) -> str:
        # Ranking-only v0.1 is deliberately conservative. Even a price edge needs
        # decent baseline confidence before it can be called bettable.
        if roi >= config.bet_roi_threshold and confidence >= 60:
            return "BET"
        if roi >= config.watch_roi_threshold:
            return "WATCH"
        return "PASS"

    return {
        "available": True,
        "sportsbook_hold": hold,
        "no_vig_probability_a": no_vig_a,
        "no_vig_probability_b": no_vig_b,
        "edge_a": probability_a - no_vig_a,
        "edge_b": probability_b - no_vig_b,
        "roi_a": roi_a,
        "roi_b": roi_b,
        "verdict_a": verdict(roi_a),
        "verdict_b": verdict(roi_b),
        "market_odds_a": int(odds_a),
        "market_odds_b": int(odds_b),
    }


def _difference_label(diff: float, small: float, medium: float) -> str:
    magnitude = abs(diff)
    if magnitude < small:
        return "Even"
    if magnitude < medium:
        return "Slight"
    return "Clear"


def _matchup_rows(row_a: pd.Series, row_b: pd.Series, fighter_a: str, fighter_b: str) -> list[dict[str, Any]]:
    rating_diff = _safe_float(row_a.get("macabets_rating")) - _safe_float(row_b.get("macabets_rating"))
    form_diff = _safe_float(row_a.get("recent_form_adjusted"), 50.0) - _safe_float(row_b.get("recent_form_adjusted"), 50.0)
    schedule_diff = _safe_float(row_a.get("schedule_rating"), 1500.0) - _safe_float(row_b.get("schedule_rating"), 1500.0)
    confidence_diff = _safe_float(row_a.get("ranking_confidence"), 50.0) - _safe_float(row_b.get("ranking_confidence"), 50.0)

    def advantage(diff: float) -> str:
        if abs(diff) < 1e-9:
            return "Even"
        return fighter_a if diff > 0 else fighter_b

    return [
        {
            "category": "Overall fighter strength",
            "advantage": advantage(rating_diff),
            "strength": _difference_label(rating_diff, 15.0, 45.0),
            "difference": rating_diff,
            "why": "Opponent-adjusted Macabets Strength v0.2 rating, including division context and inactivity.",
            "model_role": "Primary baseline",
        },
        {
            "category": "Recent trajectory",
            "advantage": advantage(form_diff),
            "strength": _difference_label(form_diff, 6.0, 18.0),
            "difference": form_diff,
            "why": "Recency-weighted recent UFC results. This is already reflected conservatively in the strength rating and is shown here as evidence, not awarded again.",
            "model_role": "Supporting evidence only",
        },
        {
            "category": "Competition faced",
            "advantage": advantage(schedule_diff),
            "strength": _difference_label(schedule_diff, 12.0, 35.0),
            "difference": schedule_diff,
            "why": "Strength of recent UFC opposition. It is displayed for context and is not double-counted as a separate probability adjustment.",
            "model_role": "Supporting evidence only",
        },
        {
            "category": "Ranking reliability",
            "advantage": advantage(confidence_diff),
            "strength": _difference_label(confidence_diff, 6.0, 16.0),
            "difference": confidence_diff,
            "why": "How much UFC/division evidence supports each current strength estimate.",
            "model_role": "Confidence only",
        },
    ]


def _reason_lines(
    row_a: pd.Series,
    row_b: pd.Series,
    winner: str,
    loser: str,
    winner_row: pd.Series,
    loser_row: pd.Series,
    probability: float,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    risks: list[str] = []

    winner_rank = winner_row.get("division_rank")
    loser_rank = loser_row.get("division_rank")
    if pd.notna(winner_rank) and pd.notna(loser_rank):
        reasons.append(
            f"{winner} carries the stronger current divisional baseline (Macabets #{int(winner_rank)} vs #{int(loser_rank)})."
        )
    else:
        reasons.append(f"{winner} has the stronger opponent-adjusted Macabets rating.")

    form_gap = _safe_float(winner_row.get("recent_form_adjusted"), 50.0) - _safe_float(loser_row.get("recent_form_adjusted"), 50.0)
    if form_gap >= 7:
        reasons.append(f"{winner} also has the better recent UFC trajectory, supporting the rating edge.")
    elif form_gap <= -7:
        risks.append(f"Recent form favors {loser}, so the long-run strength edge is not perfectly aligned with current trajectory.")

    schedule_gap = _safe_float(winner_row.get("schedule_rating"), 1500.0) - _safe_float(loser_row.get("schedule_rating"), 1500.0)
    if schedule_gap >= 15:
        reasons.append(f"{winner}'s recent strength was built against the tougher schedule.")
    elif schedule_gap <= -15:
        risks.append(f"{loser} has faced the tougher recent schedule, which creates an upset path not fully captured by the headline rank comparison.")

    loser_conf = _safe_float(loser_row.get("ranking_confidence"), 50.0)
    if loser_conf < 70:
        risks.append(f"{loser}'s rating has a smaller UFC/division sample, so there is more uncertainty around the underdog's true ceiling.")

    if probability < 0.60:
        risks.append("The baseline probability is close enough that style, cardio, and tactical matchup details could easily change the final side once those layers are added.")
    else:
        risks.append("This is still a ranking-based baseline; style, wrestling/striking interaction, durability, and five-round cardio are not yet priced into v0.1.")

    return reasons[:4], risks[:4]


def analyze(
    fighter_a: str,
    fighter_b: str,
    *,
    rounds: int = 3,
    market_odds_a: int | None = None,
    market_odds_b: int | None = None,
    ratings: pd.DataFrame | None = None,
    fights: pd.DataFrame | None = None,
    config: UFCAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build the first Macabets UFC matchup report.

    v0.1 deliberately uses Strength v0.2 as the only probability-driving football-equivalent
    baseline. Recent form and strength of schedule are shown as supporting evidence because they
    already influence the rating architecture; awarding them again would double count them.
    Detailed striking/wrestling/grappling/style adjustments belong to the next engine layer.
    """
    config = config or UFCAnalysisConfig()
    if str(fighter_a).strip().casefold() == str(fighter_b).strip().casefold():
        raise UFCAnalysisError("Select two different UFC fighters.")
    if int(rounds) not in {3, 5}:
        raise UFCAnalysisError("UFC analysis currently supports 3-round or 5-round fights.")

    ratings = load_ufc_ratings() if ratings is None else ratings.copy()
    fights = load_ufc_fights() if fights is None else fights.copy()
    if "event_date" in fights:
        fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    performance_table = build_performance_table(fights, ratings)
    performance_a = fighter_performance(performance_table, fighter_a)
    performance_b = fighter_performance(performance_table, fighter_b)
    performance_matchup = matchup_performance_adjustment(
        performance_a, performance_b, rounds=int(rounds)
    )
    fights = _attach_opponent_stats(fights)

    row_a = _fighter_row(ratings, fighter_a)
    row_b = _fighter_row(ratings, fighter_b)

    rating_a = _safe_float(row_a.get("macabets_rating"), 1500.0)
    rating_b = _safe_float(row_b.get("macabets_rating"), 1500.0)
    raw_probability_a = _elo_probability(rating_a, rating_b, config.elo_scale)

    # Shrink slightly toward 50/50 when either fighter has limited ranking evidence.
    # This is a confidence/reliability correction, not a new performance signal.
    reliability = min(
        _safe_float(row_a.get("ranking_confidence"), 50.0),
        _safe_float(row_b.get("ranking_confidence"), 50.0),
    ) / 100.0
    reliability_multiplier = 0.80 + 0.20 * _clip(reliability, 0.0, 1.0)
    baseline_probability_a = 0.5 + (raw_probability_a - 0.5) * reliability_multiplier
    baseline_probability_a = _clip(baseline_probability_a, 0.08, 0.92)

    performance_adjustment_a = float(performance_matchup.get("adjustment_a", 0.0) or 0.0)
    style_matchup = build_style_matchup(
        performance_a, performance_b, fighter_a, fighter_b, rounds=int(rounds)
    )
    style_adjustment_a = float(style_matchup.get("adjustment_a", 0.0) or 0.0)

    # Performance and style are correlated because both originate in UFCStats. Keep a
    # hard combined cap so the two layers cannot collectively overwhelm the Strength baseline.
    combined_matchup_adjustment_a = float(
        np.clip(performance_adjustment_a + style_adjustment_a, -0.075, 0.075)
    )
    probability_a = _clip(baseline_probability_a + combined_matchup_adjustment_a, 0.08, 0.92)
    probability_b = 1.0 - probability_a

    confidence, confidence_band = _confidence(row_a, row_b, probability_a, config)
    if performance_matchup.get("available"):
        perf_reliability = float(performance_matchup.get("reliability", 0.0) or 0.0)
        if perf_reliability >= 0.75:
            confidence = min(config.max_confidence + 4, confidence + 3)
        elif perf_reliability >= 0.50:
            confidence = min(config.max_confidence + 2, confidence + 1)
        if confidence >= 72:
            confidence_band = "High developing-model confidence"
        elif confidence >= 60:
            confidence_band = "Moderate developing-model confidence"
        else:
            confidence_band = "Limited developing-model confidence"
    fair_a = _probability_to_american(probability_a)
    fair_b = _probability_to_american(probability_b)

    winner = fighter_a if probability_a >= probability_b else fighter_b
    loser = fighter_b if winner == fighter_a else fighter_a
    winner_probability = max(probability_a, probability_b)
    winner_row = row_a if winner == fighter_a else row_b
    loser_row = row_b if winner == fighter_a else row_a

    profile_a = _recent_profile(fights, fighter_a, config.recent_fights)
    profile_b = _recent_profile(fights, fighter_b, config.recent_fights)
    reasons, risks = _reason_lines(
        row_a, row_b, winner, loser, winner_row, loser_row, winner_probability
    )

    if performance_matchup.get("available"):
        perf_adj = float(performance_matchup.get("adjustment_a", 0.0) or 0.0)
        if abs(perf_adj) >= 0.005:
            perf_side = fighter_a if perf_adj > 0 else fighter_b
            reasons.insert(
                0,
                f"{perf_side} owns the stronger recent underlying performance profile; the performance layer moves the fair win probability by {abs(perf_adj):.1%}.",
            )
        if float(performance_matchup.get("reliability", 0.0) or 0.0) < 0.55:
            risks.insert(
                0,
                "The detailed performance layer has a limited recent sample or incomplete landed/attempted/control data, so its adjustment is heavily shrunk.",
            )

    if style_matchup.get("available"):
        style_adj = float(style_matchup.get("adjustment_a", 0.0) or 0.0)
        if abs(style_adj) >= 0.003:
            style_side = fighter_a if style_adj > 0 else fighter_b
            strongest = sorted(
                style_matchup.get("rows", []),
                key=lambda item: abs(float(item.get("interaction_gap", 0.0) or 0.0)),
                reverse=True,
            )
            top = strongest[0] if strongest else {}
            reasons.insert(
                0,
                f"{style_side} has the better opponent-specific style interaction, led by {str(top.get('category', 'the matchup profile')).lower()}; Style Matchups moves the fair probability by {abs(style_adj):.1%}.",
            )
        if float(style_matchup.get("reliability", 0.0) or 0.0) < 0.55:
            risks.insert(
                0,
                "The style interaction layer is based on a limited or incomplete detailed-stat sample, so Macabets heavily shrinks the matchup adjustment.",
            )

    reasons = reasons[:4]
    risks = risks[:4]
    market = _market_evaluation(
        probability_a,
        market_odds_a,
        market_odds_b,
        confidence,
        config,
    )

    same_division = str(row_a.get("division", "")) == str(row_b.get("division", ""))
    division_note = (
        str(row_a.get("division", ""))
        if same_division
        else f"Cross-division comparison: {row_a.get('division', 'Unknown')} vs {row_b.get('division', 'Unknown')}"
    )

    return {
        "model_version": MODEL_VERSION,
        "rating_version": RATING_VERSION,
        "model_stage": "Ranking + underlying performance + style matchup",
        "performance_version": PERFORMANCE_VERSION,
        "style_version": STYLE_VERSION,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "rounds": int(rounds),
        "division_context": division_note,
        "same_division": same_division,
        "projected_winner": winner,
        "projected_loser": loser,
        "win_probability_a": probability_a,
        "win_probability_b": probability_b,
        "projected_winner_probability": winner_probability,
        "raw_rating_probability_a": raw_probability_a,
        "ranking_baseline_probability_a": baseline_probability_a,
        "performance_adjustment_a": performance_adjustment_a,
        "style_adjustment_a": style_adjustment_a,
        "combined_matchup_adjustment_a": combined_matchup_adjustment_a,
        "fair_moneyline_a": fair_a,
        "fair_moneyline_b": fair_b,
        "confidence": confidence,
        "confidence_band": confidence_band,
        "rating_a": rating_a,
        "rating_b": rating_b,
        "fighter_a_summary": {
            "division": str(row_a.get("division", "Unknown")),
            "division_rank": None if pd.isna(row_a.get("division_rank")) else int(row_a.get("division_rank")),
            "strength_score": _safe_float(row_a.get("strength_score"), 50.0),
            "macabets_rating": rating_a,
            "recent_form": _safe_float(row_a.get("recent_form_adjusted"), 50.0),
            "schedule_rating": _safe_float(row_a.get("schedule_rating"), 1500.0),
            "ranking_confidence": _safe_float(row_a.get("ranking_confidence"), 50.0),
            "ufc_record": f"{_safe_int(row_a.get('ufc_wins'))}-{_safe_int(row_a.get('ufc_losses'))}" + (f"-{_safe_int(row_a.get('ufc_draws'))}" if _safe_int(row_a.get('ufc_draws')) else ""),
            "days_inactive": _safe_int(row_a.get("days_inactive")),
        },
        "fighter_b_summary": {
            "division": str(row_b.get("division", "Unknown")),
            "division_rank": None if pd.isna(row_b.get("division_rank")) else int(row_b.get("division_rank")),
            "strength_score": _safe_float(row_b.get("strength_score"), 50.0),
            "macabets_rating": rating_b,
            "recent_form": _safe_float(row_b.get("recent_form_adjusted"), 50.0),
            "schedule_rating": _safe_float(row_b.get("schedule_rating"), 1500.0),
            "ranking_confidence": _safe_float(row_b.get("ranking_confidence"), 50.0),
            "ufc_record": f"{_safe_int(row_b.get('ufc_wins'))}-{_safe_int(row_b.get('ufc_losses'))}" + (f"-{_safe_int(row_b.get('ufc_draws'))}" if _safe_int(row_b.get('ufc_draws')) else ""),
            "days_inactive": _safe_int(row_b.get("days_inactive")),
        },
        "recent_profile_a": profile_a,
        "recent_profile_b": profile_b,
        "performance_profile_a": performance_a,
        "performance_profile_b": performance_b,
        "performance_matchup": performance_matchup,
        "style_matchup": style_matchup,
        "matchup_breakdown": _matchup_rows(row_a, row_b, fighter_a, fighter_b),
        "reasons_for_lean": reasons,
        "risk_factors": risks,
        "market": market,
        "limitations": [
            "v0.3 combines the Strength v0.2 baseline with a capped underlying-performance layer and a separate opponent-specific style interaction layer.",
            "Style Matchups compares directional attack traits against the opponent's corresponding defensive traits instead of reusing standalone composite strength.",
            "Performance is capped at ±5 percentage points, Style Matchups at ±3, and their correlated combined impact at ±7.5 percentage points.",
            "Three-round versus five-round context changes performance/style weighting modestly; true round-by-round cardio decay, physical measurements, stance and short-notice context remain future layers.",
        ],
    }
