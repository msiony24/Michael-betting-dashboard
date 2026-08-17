from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re

import numpy as np
import pandas as pd


BASE_ELO = 1500.0
ELO_SCALE = 400.0
DEFAULT_K = 32.0


@dataclass(frozen=True)
class UFCRatingConfig:
    base_elo: float = BASE_ELO
    k_factor: float = DEFAULT_K
    recent_fights: int = 8
    recent_window_days: int = 1460
    active_window_days: int = 730
    inactivity_grace_days: int = 365
    inactivity_penalty_per_30d: float = 4.0
    max_inactivity_penalty: float = 48.0


def _expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / ELO_SCALE))


def _score(result: object) -> float | None:
    text = str(result or "").strip().upper()
    if text == "W":
        return 1.0
    if text == "L":
        return 0.0
    if text in {"D", "DRAW"}:
        return 0.5
    return None


def _finish_multiplier(method: object) -> float:
    text = str(method or "").casefold()
    if "ko" in text or "tko" in text or "submission" in text or "sub" == text.strip():
        return 1.12
    return 1.0


def _dominance_multiplier(row: pd.Series) -> float:
    # Event-summary statistics are intentionally only a modest modifier. They are
    # not a replacement for the future detailed-fight performance layer.
    try:
        sig = float(row.get("sig_str", np.nan))
        opp_sig = float(row.get("opponent_sig_str", np.nan))
        kd = float(row.get("kd", np.nan))
        opp_kd = float(row.get("opponent_kd", np.nan))
        td = float(row.get("td", np.nan))
        opp_td = float(row.get("opponent_td", np.nan))
    except (TypeError, ValueError):
        return 1.0

    raw = 0.0
    if not np.isnan(sig) and not np.isnan(opp_sig):
        raw += np.clip((sig - opp_sig) / 80.0, -0.35, 0.35)
    if not np.isnan(kd) and not np.isnan(opp_kd):
        raw += np.clip((kd - opp_kd) * 0.12, -0.24, 0.24)
    if not np.isnan(td) and not np.isnan(opp_td):
        raw += np.clip((td - opp_td) * 0.035, -0.14, 0.14)
    return float(np.clip(1.0 + raw * 0.20, 0.90, 1.10))


def _prepare_fights(fights: pd.DataFrame) -> pd.DataFrame:
    required = {"event_date", "fight_url", "fighter", "opponent", "result", "division", "method"}
    missing = required - set(fights.columns)
    if missing:
        raise ValueError(f"UFC fight history is missing required columns: {', '.join(sorted(missing))}")

    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.dropna(subset=["event_date", "fighter", "opponent", "fight_url"])

    stat_map = {
        "sig_str": "opponent_sig_str",
        "kd": "opponent_kd",
        "td": "opponent_td",
        "sub_att": "opponent_sub_att",
    }
    opponent_rows = frame[["fight_url", "fighter"] + [c for c in stat_map if c in frame]].copy()
    opponent_rows = opponent_rows.rename(columns={"fighter": "opponent", **stat_map})
    frame = frame.merge(opponent_rows, on=["fight_url", "opponent"], how="left")
    return frame.sort_values(["event_date", "fight_url", "fighter"]).reset_index(drop=True)


def build_elo_history(fights: pd.DataFrame, config: UFCRatingConfig | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    config = config or UFCRatingConfig()
    frame = _prepare_fights(fights)
    ratings: dict[str, float] = {}
    history: list[dict[str, object]] = []

    # Process each bout once. UFCStats emits one row per fighter.
    for fight_url, bout in frame.groupby("fight_url", sort=False):
        if len(bout) < 2:
            continue
        bout = bout.iloc[:2]
        a = bout.iloc[0]
        b = bout.iloc[1]
        score_a = _score(a["result"])
        score_b = _score(b["result"])
        if score_a is None or score_b is None:
            continue

        fighter_a = str(a["fighter"])
        fighter_b = str(b["fighter"])
        rating_a = ratings.get(fighter_a, config.base_elo)
        rating_b = ratings.get(fighter_b, config.base_elo)
        exp_a = _expected(rating_a, rating_b)
        exp_b = 1.0 - exp_a

        # A finish gets a small extra update, and a one-sided event-summary line
        # can modestly increase/decrease the update. Opponent quality remains the
        # dominant component through the expected-score term.
        finish_mult = max(_finish_multiplier(a.get("method")), _finish_multiplier(b.get("method")))
        dom_a = _dominance_multiplier(a)
        dom_b = _dominance_multiplier(b)
        k_a = config.k_factor * finish_mult * dom_a
        k_b = config.k_factor * finish_mult * dom_b

        new_a = rating_a + k_a * (score_a - exp_a)
        new_b = rating_b + k_b * (score_b - exp_b)
        ratings[fighter_a] = new_a
        ratings[fighter_b] = new_b

        for row, before, after, expected in (
            (a, rating_a, new_a, exp_a),
            (b, rating_b, new_b, exp_b),
        ):
            history.append(
                {
                    "event_date": row["event_date"].date().isoformat(),
                    "fight_url": fight_url,
                    "fighter": row["fighter"],
                    "opponent": row["opponent"],
                    "division": row["division"],
                    "result": row["result"],
                    "method": row["method"],
                    "rating_before": round(before, 3),
                    "opponent_rating_before": round(rating_b if row["fighter"] == fighter_a else rating_a, 3),
                    "expected_win_prob": round(expected, 5),
                    "rating_after": round(after, 3),
                }
            )

    return pd.DataFrame(history), ratings


def _recent_form_score(rows: pd.DataFrame, today: pd.Timestamp, config: UFCRatingConfig) -> float:
    if rows.empty:
        return 50.0
    rows = rows.sort_values("event_date", ascending=False).head(config.recent_fights)
    weighted_points = 0.0
    total_weight = 0.0
    for i, (_, row) in enumerate(rows.iterrows()):
        score = _score(row["result"])
        if score is None:
            continue
        days = max(0, int((today - row["event_date"]).days))
        recency = math.exp(-days / 900.0)
        ordering = 0.90 ** i
        weight = recency * ordering
        weighted_points += score * weight
        total_weight += weight
    return 50.0 if total_weight <= 0 else 100.0 * weighted_points / total_weight


def _schedule_score(rows: pd.DataFrame, rating_before_lookup: dict[tuple[str, str], float], base: float) -> float:
    values: list[float] = []
    weights: list[float] = []
    rows = rows.sort_values("event_date", ascending=False).head(8)
    for i, (_, row) in enumerate(rows.iterrows()):
        key = (str(row["fight_url"]), str(row["opponent"]))
        opp = rating_before_lookup.get(key, base)
        values.append(opp)
        weights.append(0.88 ** i)
    if not values:
        return base
    return float(np.average(values, weights=weights))


def _to_strength_score(rating: float) -> float:
    # Fixed logistic map keeps scores stable as the roster changes. 1500 = 50,
    # 1700 ~= 76, 1800 ~= 86, 1900 ~= 93.
    return 100.0 / (1.0 + math.exp(-(rating - 1500.0) / 170.0))


def build_fighter_ratings(
    fights: pd.DataFrame,
    *,
    as_of: date | str | None = None,
    config: UFCRatingConfig | None = None,
) -> pd.DataFrame:
    config = config or UFCRatingConfig()
    frame = _prepare_fights(fights)
    if frame.empty:
        return pd.DataFrame()

    as_of_ts = pd.Timestamp(as_of or date.today())
    frame = frame[frame["event_date"] <= as_of_ts].copy()
    history, current = build_elo_history(frame, config)
    if history.empty:
        return pd.DataFrame()

    history_lookup = {
        (str(row["fight_url"]), str(row["fighter"])): float(row["rating_before"])
        for _, row in history.iterrows()
    }

    output: list[dict[str, object]] = []
    for fighter, rows in frame.groupby("fighter"):
        rows = rows.sort_values("event_date")
        last = rows.iloc[-1]
        days_inactive = max(0, int((as_of_ts - last["event_date"]).days))
        penalty_days = max(0, days_inactive - config.inactivity_grace_days)
        inactivity_penalty = min(
            config.max_inactivity_penalty,
            (penalty_days / 30.0) * config.inactivity_penalty_per_30d,
        )

        raw_rating = float(current.get(str(fighter), config.base_elo))
        adjusted_rating = raw_rating - inactivity_penalty
        recent_form = _recent_form_score(rows, as_of_ts, config)
        schedule_rating = _schedule_score(rows, history_lookup, config.base_elo)

        valid = rows["result"].map(_score).notna()
        wins = int((rows.loc[valid, "result"].astype(str).str.upper() == "W").sum())
        losses = int((rows.loc[valid, "result"].astype(str).str.upper() == "L").sum())
        draws = int((rows.loc[valid, "result"].astype(str).str.upper().isin(["D", "DRAW"])).sum())
        finishes = int(
            rows.loc[rows["result"].astype(str).str.upper().eq("W"), "method"]
            .astype(str)
            .str.contains(r"KO|TKO|SUB", case=False, regex=True)
            .sum()
        )

        recent_cutoff = as_of_ts - pd.Timedelta(days=config.recent_window_days)
        recent_count = int((rows["event_date"] >= recent_cutoff).sum())
        active = days_inactive <= config.active_window_days

        division = str(last.get("division") or "Unknown")
        strength_score = _to_strength_score(adjusted_rating)
        output.append(
            {
                "fighter": fighter,
                "division": division,
                "macabets_rating": round(adjusted_rating, 2),
                "strength_score": round(strength_score, 1),
                "raw_elo": round(raw_rating, 2),
                "recent_form_score": round(recent_form, 1),
                "schedule_rating": round(schedule_rating, 2),
                "ufc_wins": wins,
                "ufc_losses": losses,
                "ufc_draws": draws,
                "ufc_finishes": finishes,
                "recent_fights_4y": recent_count,
                "last_fight_date": last["event_date"].date().isoformat(),
                "days_inactive": days_inactive,
                "inactivity_penalty": round(inactivity_penalty, 2),
                "active_pool": bool(active),
            }
        )

    ratings = pd.DataFrame(output)
    if ratings.empty:
        return ratings

    # Division rank is based only on the active pool. Historical/inactive fighters
    # retain ratings for opponent-strength calculations but do not clutter the
    # current ranking table.
    ratings["division_rank"] = pd.NA
    active_idx = ratings.index[ratings["active_pool"]]
    active = ratings.loc[active_idx].copy()
    active["division_rank"] = (
        active.groupby("division")["macabets_rating"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    ratings.loc[active.index, "division_rank"] = active["division_rank"]
    ratings["division_rank"] = ratings["division_rank"].astype("Int64")
    return ratings.sort_values(["active_pool", "division", "macabets_rating"], ascending=[False, True, False]).reset_index(drop=True)
