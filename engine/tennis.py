from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from .player_identities import get_player_identity
from .player_profiles import (
    build_player_profile,
    compare_experience,
    experience_reliability,
)
from .tennis_serve_return import serve_return_profile, serve_return_matchup_adjustment


ROUND_MAP = {
    "Qualifying": "Q",
    "R128": "R128",
    "R64": "R64",
    "R32": "R32",
    "R16": "R16",
    "Quarterfinal": "QF",
    "Semifinal": "SF",
    "Final": "F",
}


class TennisDataValidationError(ValueError):
    """Raised when Macabets cannot build a trustworthy player profile."""


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


from .tennis_identity import (
    canonical_player_key,
    player_name_signature,
    resolve_player_name,
)


def safe_int(value, default: int = 0) -> int:
    """Convert user/data values to int without failing on blanks or NaN."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def american_from_probability(probability: float) -> int:
    p = float(np.clip(probability, 0.001, 0.999))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def implied_probability(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def player_names(matches: pd.DataFrame) -> list[str]:
    cutoff = matches["tourney_date"].max() - pd.Timedelta(days=1095)
    recent = matches[matches["tourney_date"] >= cutoff]
    names = pd.concat([recent["winner_name"], recent["loser_name"]]).dropna()
    return names.value_counts().index.tolist()


def tournament_names(matches: pd.DataFrame) -> list[str]:
    recent = matches.sort_values("tourney_date").drop_duplicates("tourney_name", keep="last")
    recent = recent[recent["tourney_level"].isin(["G", "M", "A", "F", "D"])]
    return sorted(recent["tourney_name"].dropna().unique().tolist())


def tournament_surface(matches: pd.DataFrame, tournament: str) -> str:
    rows = matches[matches["tourney_name"] == tournament].sort_values("tourney_date")
    if rows.empty:
        return "Hard"
    value = rows.iloc[-1]["surface"]
    return value if value in {"Hard", "Clay", "Grass", "Carpet"} else "Hard"


TOURNAMENT_LEVEL_LABELS = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP Tour",
    "F": "Tour Finals",
    "D": "Davis Cup",
    "C": "Challenger",
}


def tournament_category(matches: pd.DataFrame, tournament: str) -> str:
    """Infer the most recent event category for a tournament."""
    rows = matches[matches["tourney_name"] == tournament].sort_values("tourney_date")
    if rows.empty:
        return "ATP 250"

    level = str(rows.iloc[-1].get("tourney_level", "A"))
    if level == "G":
        return "Grand Slam"
    if level == "M":
        return "Masters 1000"
    if level == "F":
        return "Tour Finals"
    if level == "D":
        return "Davis Cup"
    if level == "C":
        return "Challenger"

    # ATP data does not consistently separate 250 and 500 in tourney_level.
    # Use a conservative name-based inference and leave the UI editable.
    name = norm(tournament)
    known_500 = {
        "rotterdam", "rio de janeiro", "dubai", "acapulco", "barcelona",
        "halle", "queens club", "hamburg", "washington", "beijing",
        "tokyo", "vienna", "basel",
    }
    if any(token in name for token in known_500):
        return "ATP 500"
    return "ATP 250"


def context_weights(
    surface: str,
    tournament_category_label: str,
    round_label: str,
    environment: str,
    match_format: str,
) -> dict:
    """Return transparent dynamic weights for the specific match context."""
    surface_key = str(surface).casefold()
    environment_key = str(environment).casefold()
    category_key = str(tournament_category_label).casefold()
    round_key = ROUND_MAP.get(round_label, round_label)

    overall_weight = 0.48
    surface_weight = 0.37
    rank_weight = 0.15

    if surface_key == "grass":
        overall_weight, surface_weight, rank_weight = 0.42, 0.43, 0.15
    elif surface_key == "clay":
        overall_weight, surface_weight, rank_weight = 0.43, 0.42, 0.15
    elif "indoor" in environment_key:
        overall_weight, surface_weight, rank_weight = 0.44, 0.40, 0.16

    serve_multiplier = 1.0
    return_multiplier = 1.0
    if surface_key == "grass":
        serve_multiplier = 1.25
        return_multiplier = 0.90
    elif surface_key == "clay":
        serve_multiplier = 0.85
        return_multiplier = 1.18
    elif "indoor" in environment_key:
        serve_multiplier = 1.18
        return_multiplier = 0.95

    category_pressure = {
        "grand slam": 1.35,
        "masters 1000": 1.15,
        "atp 500": 1.00,
        "atp 250": 0.80,
        "challenger": 0.65,
        "tour finals": 1.25,
        "davis cup": 1.20,
    }.get(category_key, 0.90)

    round_pressure = {
        "Q": 0.25,
        "R128": 0.30,
        "R64": 0.40,
        "R32": 0.55,
        "R16": 0.75,
        "QF": 1.00,
        "SF": 1.20,
        "F": 1.35,
    }.get(round_key, 0.55)

    format_pressure = 1.12 if str(match_format).casefold() == "best of 5" else 1.0
    pressure_multiplier = category_pressure * round_pressure * format_pressure

    form_multiplier = 1.12 if round_key in {"Q", "R128", "R64", "R32"} else 0.95
    fatigue_multiplier = 1.18 if round_key in {"QF", "SF", "F"} else 1.0
    deciding_multiplier = 1.20 if str(match_format).casefold() == "best of 5" else 1.0

    return {
        "overall_elo": overall_weight,
        "surface_elo": surface_weight,
        "ranking": rank_weight,
        "serve": serve_multiplier,
        "return": return_multiplier,
        "form": form_multiplier,
        "fatigue": fatigue_multiplier,
        "pressure": pressure_multiplier,
        "deciding": deciding_multiplier,
    }


def perspective(matches: pd.DataFrame, player: str, event_date: date) -> pd.DataFrame:
    key = canonical_player_key(player)
    subset = matches[
        (
            matches["winner_name"].map(canonical_player_key).eq(key)
            | matches["loser_name"].map(canonical_player_key).eq(key)
        )
        & (matches["tourney_date"] < pd.Timestamp(event_date))
    ].sort_values("tourney_date")

    rows = []
    for _, match in subset.iterrows():
        won = canonical_player_key(match["winner_name"]) == key
        side = "w" if won else "l"
        other = "l" if won else "w"
        rows.append({
            "date": match["tourney_date"],
            "won": won,
            "surface": match["surface"],
            "level": match["tourney_level"],
            "round": match["round"],
            "score": match["score"],
            "rank": match.get("winner_rank" if won else "loser_rank", np.nan),
            "age": match.get("winner_age" if won else "loser_age", np.nan),
            "svpt": match.get(f"{side}_svpt", np.nan),
            "first_won": match.get(f"{side}_1stWon", np.nan),
            "second_won": match.get(f"{side}_2ndWon", np.nan),
            "opp_svpt": match.get(f"{other}_svpt", np.nan),
            "opp_first_won": match.get(f"{other}_1stWon", np.nan),
            "opp_second_won": match.get(f"{other}_2ndWon", np.nan),
        })
    return pd.DataFrame(rows)


def safe_ratio(num: float, den: float, default: float) -> float:
    if pd.isna(den) or den <= 0:
        return default
    return float(num) / float(den)


def profile(rows: pd.DataFrame, surface: str, event_date: date) -> dict:
    """Build a player profile without disguising missing data as real statistics."""
    if rows.empty:
        return {
            "rank": np.nan, "recent_win": np.nan, "surface_win": np.nan,
            "serve_points_won": np.nan, "return_points_won": np.nan,
            "matches_7": 0, "matches_14": 0, "rest_days": 30,
            "advanced_win": .5, "big_event_win": .5,
            "deciding_win": .5, "sample": 0,
            "surface_sample": 0, "serve_sample": 0, "return_sample": 0,
            "data_flags": ["no_match_history"],
        }

    event_ts = pd.Timestamp(event_date)
    two_year = rows[rows["date"] >= event_ts - pd.Timedelta(days=730)]
    one_year = rows[rows["date"] >= event_ts - pd.Timedelta(days=365)]
    recent = rows.tail(10)
    surf = two_year[two_year["surface"].astype(str).str.casefold() == surface.casefold()]
    advanced = two_year[two_year["round"].isin(["QF", "SF", "F"])]
    big = two_year[two_year["level"].isin(["G", "M", "F"])]
    deciding = two_year[two_year["score"].astype(str).str.count(r"\d+-\d+") >= 3]

    valid_serve = one_year[["svpt", "first_won", "second_won"]].dropna()
    valid_return = one_year[["opp_svpt", "opp_first_won", "opp_second_won"]].dropna()
    serve_den = valid_serve["svpt"].sum() if not valid_serve.empty else 0
    serve_num = (valid_serve["first_won"] + valid_serve["second_won"]).sum() if not valid_serve.empty else 0
    return_den = valid_return["opp_svpt"].sum() if not valid_return.empty else 0
    return_num = (
        valid_return["opp_svpt"] - valid_return["opp_first_won"] - valid_return["opp_second_won"]
    ).sum() if not valid_return.empty else 0

    ranks = rows["rank"].dropna()
    last_date = rows["date"].max()
    recent_win = float(recent["won"].mean()) if len(recent) else np.nan
    flags = []
    if surf.empty:
        flags.append("no_surface_history")
    if len(valid_serve) < 3:
        flags.append("insufficient_serve_stats")
    if len(valid_return) < 3:
        flags.append("insufficient_return_stats")
    if ranks.empty:
        flags.append("missing_ranking")

    return {
        "rank": float(ranks.iloc[-1]) if len(ranks) else np.nan,
        "recent_win": recent_win,
        "surface_win": float(surf["won"].mean()) if len(surf) else recent_win,
        "serve_points_won": float(serve_num / serve_den) if serve_den > 0 else np.nan,
        "return_points_won": float(return_num / return_den) if return_den > 0 else np.nan,
        "matches_7": int((rows["date"] >= event_ts - pd.Timedelta(days=7)).sum()),
        "matches_14": int((rows["date"] >= event_ts - pd.Timedelta(days=14)).sum()),
        "rest_days": max(0, safe_int((event_ts - last_date).days, 30)),
        "advanced_win": float(advanced["won"].mean()) if len(advanced) >= 4 else .5,
        "big_event_win": float(big["won"].mean()) if len(big) >= 5 else .5,
        "deciding_win": float(deciding["won"].mean()) if len(deciding) >= 4 else .5,
        "sample": len(two_year),
        "surface_sample": len(surf),
        "serve_sample": len(valid_serve),
        "return_sample": len(valid_return),
        "data_flags": flags,
    }


def opponent_strength_profile(
    matches: pd.DataFrame,
    player: str,
    event_date: date,
    overall_elo: dict[str, float],
    lookback_matches: int = 10,
) -> dict:
    """
    Evaluate recent form *and* the strength of the opposition that produced it.

    Strength of schedule is intentionally kept separate from results. That prevents
    Macabets from double-counting a good win as both "good form" and a stronger
    schedule. ``quality_adjusted_form`` then blends the raw result with opponent
    quality so recent form gets appropriate context without becoming a dominant
    input.
    """
    key = canonical_player_key(player)
    history = matches[
        (
            matches["winner_name"].map(canonical_player_key).eq(key)
            | matches["loser_name"].map(canonical_player_key).eq(key)
        )
        & (matches["tourney_date"] < pd.Timestamp(event_date))
    ].sort_values("tourney_date", ascending=False).head(lookback_matches)

    if history.empty:
        return {
            "matches": 0,
            "raw_win_rate": 0.5,
            "avg_opponent_rank": None,
            "avg_opponent_elo": 1500.0,
            "quality_form": 0.5,
            "quality_adjusted_form": 0.5,
            "top_50_record": "0-0",
            "top_100_record": "0-0",
            "quality_wins": [],
            "bad_losses": [],
            "strength_score": 0.5,
        }

    opponent_ranks = []
    opponent_elos = []
    quality_results = []
    results = []
    quality_wins = []
    bad_losses = []
    top_50_wins = top_50_losses = 0
    top_100_wins = top_100_losses = 0

    for _, row in history.iterrows():
        won = canonical_player_key(row["winner_name"]) == key
        results.append(1.0 if won else 0.0)
        opponent_name = row["loser_name"] if won else row["winner_name"]
        opponent_rank = row.get("loser_rank" if won else "winner_rank", np.nan)
        opponent_elo = float(overall_elo.get(canonical_player_key(opponent_name), 1500.0))

        opponent_elos.append(opponent_elo)
        rank_value = None
        if pd.notna(opponent_rank):
            rank_value = float(opponent_rank)
            opponent_ranks.append(rank_value)

            if rank_value <= 50:
                if won:
                    top_50_wins += 1
                else:
                    top_50_losses += 1
            if rank_value <= 100:
                if won:
                    top_100_wins += 1
                else:
                    top_100_losses += 1

            if won and rank_value <= 100:
                quality_wins.append({"opponent": str(opponent_name), "rank": int(rank_value)})
            elif not won and rank_value > 100:
                bad_losses.append({"opponent": str(opponent_name), "rank": int(rank_value)})

        # Result quality: strong-opponent wins receive more credit, while losses
        # to strong opponents are treated more gently than losses to weak ones.
        opponent_quality = 1 / (1 + math.exp(-(opponent_elo - 1500.0) / 170.0))
        quality_results.append(
            0.55 + 0.45 * opponent_quality
            if won
            else 0.45 * opponent_quality
        )

    avg_rank = (
        float(sum(opponent_ranks) / len(opponent_ranks))
        if opponent_ranks else None
    )
    avg_elo = float(sum(opponent_elos) / len(opponent_elos))
    raw_win_rate = float(sum(results) / len(results))
    quality_form = float(sum(quality_results) / len(quality_results))

    # Schedule score uses opponent quality only -- not wins/losses. This makes
    # strength of schedule a clean, modest context signal rather than a second
    # version of recent form.
    elo_component = 1 / (1 + math.exp(-(avg_elo - 1500.0) / 160.0))
    rank_component = (
        1 / (1 + math.exp((avg_rank - 75.0) / 35.0))
        if avg_rank is not None else 0.5
    )
    strength_score = float(np.clip(
        0.60 * elo_component + 0.40 * rank_component,
        0.0,
        1.0,
    ))

    # Keep the player's actual recent record as the majority of the form signal.
    # Opponent-aware result quality supplies context, but cannot overwhelm it.
    quality_adjusted_form = float(np.clip(
        0.60 * raw_win_rate + 0.40 * quality_form,
        0.0,
        1.0,
    ))

    quality_wins.sort(key=lambda item: item["rank"])
    bad_losses.sort(key=lambda item: item["rank"], reverse=True)

    return {
        "matches": int(len(history)),
        "raw_win_rate": raw_win_rate,
        "avg_opponent_rank": avg_rank,
        "avg_opponent_elo": avg_elo,
        "quality_form": quality_form,
        "quality_adjusted_form": quality_adjusted_form,
        "top_50_record": f"{top_50_wins}-{top_50_losses}",
        "top_100_record": f"{top_100_wins}-{top_100_losses}",
        "quality_wins": quality_wins[:3],
        "bad_losses": bad_losses[:3],
        "strength_score": strength_score,
    }


def breakout_trajectory_profile(
    matches: pd.DataFrame,
    player: str,
    event_date: date,
) -> dict:
    """Estimate whether an emerging player's older baseline is becoming stale.

    This is deliberately a persistence detector, not a youth bonus and not a hot-streak
    bonus.  It rewards a strong current season only when the improvement is spread over
    time/events, includes credible opposition, and is accompanied by ranking progression.
    The final probability impact is capped and is intended to make Macabets adapt faster
    to genuine breakout seasons without chasing one-tournament noise.
    """
    key = canonical_player_key(player)
    event_ts = pd.Timestamp(event_date)
    history = matches[
        (
            matches["winner_name"].map(canonical_player_key).eq(key)
            | matches["loser_name"].map(canonical_player_key).eq(key)
        )
        & (matches["tourney_date"] < event_ts)
    ].sort_values("tourney_date").copy()

    neutral = {
        "score": 0.0,
        "probability_uplift": 0.0,
        "season_matches": 0,
        "season_win_rate": 0.5,
        "recent_90_matches": 0,
        "recent_90_win_rate": 0.5,
        "preseason_matches": 0,
        "start_rank": None,
        "current_rank": None,
        "rank_progress": 0.0,
        "top_50_wins": 0,
        "top_100_wins": 0,
        "quality_win_event_weeks": 0,
        "active_months": 0,
        "persistence": 0.0,
        "emerging_context": 0.0,
    }
    if history.empty:
        return neutral

    season_start = pd.Timestamp(year=event_ts.year, month=1, day=1)
    season = history[history["tourney_date"] >= season_start].copy()
    preseason = history[history["tourney_date"] < season_start]
    recent_90 = season[season["tourney_date"] >= event_ts - pd.Timedelta(days=90)].copy()

    if len(season) < 6:
        return {**neutral, "season_matches": int(len(season)), "preseason_matches": int(len(preseason))}

    def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["won"] = out["winner_name"].map(canonical_player_key).eq(key)
        out["own_rank"] = np.where(out["won"], out.get("winner_rank", np.nan), out.get("loser_rank", np.nan))
        out["opp_rank"] = np.where(out["won"], out.get("loser_rank", np.nan), out.get("winner_rank", np.nan))
        return out

    season = _decorate(season)
    recent_90 = _decorate(recent_90) if not recent_90.empty else recent_90

    season_win_rate = float(season["won"].mean())
    recent_90_win_rate = float(recent_90["won"].mean()) if len(recent_90) else 0.5

    ranks = pd.to_numeric(season["own_rank"], errors="coerce").dropna()
    if len(ranks):
        start_rank = float(ranks.head(min(5, len(ranks))).median())
        current_rank = float(ranks.tail(min(5, len(ranks))).median())
    else:
        start_rank = current_rank = None

    if start_rank and current_rank and start_rank > 0 and current_rank > 0:
        # A fourfold ranking improvement is already enough to saturate this signal.
        rank_progress = float(np.clip(math.log(start_rank / current_rank) / math.log(4.0), 0.0, 1.0))
    else:
        rank_progress = 0.0

    wins = season[season["won"]].copy()
    win_ranks = pd.to_numeric(wins.get("opp_rank", pd.Series(dtype=float)), errors="coerce")
    top_50_wins = int((win_ranks <= 50).sum())
    top_100_wins = int((win_ranks <= 100).sum())

    # Count event-weeks rather than raw tournament names because provider feeds can
    # spell the same event differently. This also prevents one hot tournament from
    # masquerading as sustained evidence.
    quality_wins = wins[pd.to_numeric(wins["opp_rank"], errors="coerce") <= 100].copy()
    if len(quality_wins):
        quality_event_weeks = int(
            pd.to_datetime(quality_wins["tourney_date"]).dt.to_period("W").nunique()
        )
    else:
        quality_event_weeks = 0
    active_months = int(pd.to_datetime(season["tourney_date"]).dt.to_period("M").nunique())

    season_signal = float(np.clip((season_win_rate - 0.56) / 0.22, 0.0, 1.0))
    recent_signal = float(np.clip((recent_90_win_rate - 0.58) / 0.22, 0.0, 1.0))
    quality_signal = float(np.clip((top_50_wins + 0.35 * max(0, top_100_wins - top_50_wins)) / 6.0, 0.0, 1.0))
    breadth_signal = float(np.clip(quality_event_weeks / 4.0, 0.0, 1.0))

    # Persistence is the anti-hot-streak gate: meaningful season sample, multiple
    # months, and quality results across multiple event weeks all have to be present.
    sample_gate = float(np.clip(len(season) / 24.0, 0.0, 1.0))
    month_gate = float(np.clip(active_months / 4.0, 0.0, 1.0))
    event_gate = float(np.clip(quality_event_weeks / 3.0, 0.0, 1.0))
    persistence = sample_gate * month_gate * event_gate

    # With no reliable birth-date field in every provider feed, limited pre-season
    # tour history is used as a conservative proxy for an emerging/young player.
    # Established players therefore do not receive a generic "good season" bonus.
    emerging_context = float(np.clip(1.0 - len(preseason) / 120.0, 0.0, 1.0))

    evidence_score = (
        0.24 * season_signal
        + 0.18 * recent_signal
        + 0.24 * rank_progress
        + 0.20 * quality_signal
        + 0.14 * breadth_signal
    )
    score = float(np.clip(evidence_score * persistence * emerging_context, 0.0, 1.0))

    # No material model movement until the evidence clears a meaningful threshold.
    # Maximum uplift is 2.8 percentage points for an exceptionally strong, sustained
    # breakout. This is large enough to correct a stale baseline but too small to
    # overwhelm Elo, ranking, surface, matchup, and fatigue inputs.
    probability_uplift = float(np.clip(max(0.0, score - 0.35) / 0.65 * 0.028, 0.0, 0.028))

    return {
        "score": score,
        "probability_uplift": probability_uplift,
        "season_matches": int(len(season)),
        "season_win_rate": season_win_rate,
        "recent_90_matches": int(len(recent_90)),
        "recent_90_win_rate": recent_90_win_rate,
        "preseason_matches": int(len(preseason)),
        "start_rank": start_rank,
        "current_rank": current_rank,
        "rank_progress": rank_progress,
        "top_50_wins": top_50_wins,
        "top_100_wins": top_100_wins,
        "quality_win_event_weeks": quality_event_weeks,
        "active_months": active_months,
        "persistence": persistence,
        "emerging_context": emerging_context,
    }


def elo_tables(matches: pd.DataFrame, surface: str, event_date: date) -> tuple[dict, dict]:
    history = matches[
        (matches["tourney_date"] < pd.Timestamp(event_date))
        & (matches["tourney_date"] >= pd.Timestamp(event_date) - pd.Timedelta(days=1460))
    ].sort_values("tourney_date")

    overall: dict[str, float] = {}
    surface_table: dict[str, float] = {}

    def update(table: dict[str, float], winner: str, loser: str) -> None:
        rw = table.get(winner, 1500.0)
        rl = table.get(loser, 1500.0)
        expected = 1 / (1 + 10 ** ((rl - rw) / 400))
        k = 24.0
        table[winner] = rw + k * (1 - expected)
        table[loser] = rl - k * (1 - expected)

    for _, row in history.iterrows():
        winner, loser = canonical_player_key(row["winner_name"]), canonical_player_key(row["loser_name"])
        update(overall, winner, loser)
        if str(row["surface"]).casefold() == surface.casefold():
            update(surface_table, winner, loser)

    return overall, surface_table


def rating_probability(a: float, b: float) -> float:
    return 1 / (1 + 10 ** ((b - a) / 400))


def simulate_matches(
    probability: float,
    simulations: int,
    best_of_five: bool,
    seed: int | None = None,
) -> dict:
    """Monte Carlo match simulation, vectorized with numpy.

    Mathematically equivalent to simulating each match one set at a time in
    a Python loop (each set is still an independent Bernoulli(set_p) draw,
    and a match still ends the instant either side reaches sets_needed) --
    but instead of a Python-level loop running `simulations` times (each
    with its own inner while-loop), every simulation's sets are drawn and
    resolved together as numpy array operations. This is the same model,
    just not run one match at a time in pure Python.

    Note: because the sets are drawn in one batched array rather than one
    `rng.random()` call per set with early stopping, the exact sequence of
    random draws differs from the old loop implementation even for the same
    seed. The output distribution is the same; a specific seed will not
    reproduce the old function's exact historical numbers.
    """
    rng = np.random.default_rng(seed)
    sets_needed = 3 if best_of_five else 2
    max_sets = 2 * sets_needed - 1  # a match can never go longer than this

    # Convert match-level strength to a set-level probability, then simulate sets.
    p = float(np.clip(probability, .05, .95))
    set_p = float(np.clip(.5 + (p - .5) * .72, .08, .92))

    # Draw every set for every simulation at once. Sets beyond the point a
    # match would actually have clinched are drawn too (cheap) and simply
    # ignored below -- this is what lets numpy do the work instead of Python.
    a_wins_set = rng.random((simulations, max_sets)) < set_p  # bool grid

    a_cum = np.cumsum(a_wins_set, axis=1)
    sets_played = np.arange(1, max_sets + 1)
    b_cum = sets_played[None, :] - a_cum

    # First column where either side has clinched the match.
    clinched = (a_cum >= sets_needed) | (b_cum >= sets_needed)
    end_idx = np.argmax(clinched, axis=1)  # guaranteed True by the last column

    rows = np.arange(simulations)
    a_final = a_cum[rows, end_idx]
    b_final = b_cum[rows, end_idx]

    a_won = a_final > b_final
    wins_a = int(np.count_nonzero(a_won))
    straight_a = int(np.count_nonzero(a_won & (b_final == 0)))
    straight_b = int(np.count_nonzero(~a_won & (a_final == 0)))
    deciding = int(np.count_nonzero(
        (np.maximum(a_final, b_final) == sets_needed)
        & (np.minimum(a_final, b_final) == sets_needed - 1)
    ))

    set_score_counts: dict[str, int] = {}
    for winner_sets, loser_sets in [(sets_needed, n) for n in range(sets_needed)]:
        a_count = int(np.count_nonzero((a_final == winner_sets) & (b_final == loser_sets)))
        if a_count:
            set_score_counts[f"{winner_sets}-{loser_sets}"] = a_count
        b_count = int(np.count_nonzero((a_final == loser_sets) & (b_final == winner_sets)))
        if b_count:
            set_score_counts[f"{loser_sets}-{winner_sets}"] = b_count

    return {
        "simulations": simulations,
        "win_probability": wins_a / simulations,
        "straight_sets_a": straight_a / simulations,
        "straight_sets_b": straight_b / simulations,
        "deciding_set": deciding / simulations,
        "set_scores": {
            key: value / simulations
            for key, value in sorted(set_score_counts.items())
        },
    }



def _sets_played_from_score(score: object) -> int:
    """Best-effort set count from ATP score text."""
    text = str(score or "").upper()
    if not text or text == "NAN":
        return 0
    return len(re.findall(r"\d+\s*-\s*\d+", text))


def fatigue_profile(rows: pd.DataFrame, event_date: date) -> dict:
    """
    Expanded workload profile using only information available in the match database.
    Manual travel, late-finish and injury inputs are added separately in analyze().
    """
    if rows.empty:
        return {
            "matches_3": 0,
            "matches_7": 0,
            "matches_14": 0,
            "sets_3": 0,
            "sets_7": 0,
            "deciders_7": 0,
            "consecutive_weeks": 0,
            "rest_days": 30,
            "score": 0.0,
        }

    event_ts = pd.Timestamp(event_date)
    recent_3 = rows[rows["date"] >= event_ts - pd.Timedelta(days=3)]
    recent_7 = rows[rows["date"] >= event_ts - pd.Timedelta(days=7)]
    recent_14 = rows[rows["date"] >= event_ts - pd.Timedelta(days=14)]

    sets_3 = safe_int(recent_3["score"].map(_sets_played_from_score).sum())
    sets_7 = safe_int(recent_7["score"].map(_sets_played_from_score).sum())
    deciders_7 = safe_int(
        (recent_7["score"].map(_sets_played_from_score) >= 3).sum()
    )

    active_weeks = set(
        pd.to_datetime(recent_14["date"]).dt.to_period("W").astype(str).tolist()
    )
    consecutive_weeks = min(len(active_weeks), 3)
    rest_days = max(0, safe_int((event_ts - rows["date"].max()).days, 30))

    score = (
        len(recent_3) * 1.20
        + len(recent_7) * 0.65
        + len(recent_14) * 0.18
        + sets_3 * 0.22
        + sets_7 * 0.10
        + deciders_7 * 0.75
        + max(0, consecutive_weeks - 1) * 0.70
        - min(rest_days, 7) * 0.25
    )

    return {
        "matches_3": int(len(recent_3)),
        "matches_7": int(len(recent_7)),
        "matches_14": int(len(recent_14)),
        "sets_3": sets_3,
        "sets_7": sets_7,
        "deciders_7": deciders_7,
        "consecutive_weeks": consecutive_weeks,
        "rest_days": rest_days,
        "score": float(score),
    }


def surface_transition_profile(
    rows: pd.DataFrame,
    current_surface: str,
    event_date: date,
) -> dict:
    """Measure how recently and how often the player has competed on this surface."""
    if rows.empty:
        return {
            "previous_surface": None,
            "surface_changed": False,
            "matches_current_surface_30": 0,
            "days_since_current_surface": None,
            "adaptation_score": 0.5,
        }

    event_ts = pd.Timestamp(event_date)
    previous = rows.iloc[-1]
    previous_surface = str(previous.get("surface", ""))
    surface_changed = previous_surface.casefold() != str(current_surface).casefold()

    same_surface = rows[
        rows["surface"].astype(str).str.casefold() == str(current_surface).casefold()
    ]
    recent_same = same_surface[
        same_surface["date"] >= event_ts - pd.Timedelta(days=30)
    ]

    if same_surface.empty:
        days_since = None
    else:
        days_since = max(0, safe_int((event_ts - same_surface["date"].max()).days, 999))

    recent_matches = int(len(recent_same))
    adaptation = 0.50
    if not surface_changed:
        adaptation += 0.20
    adaptation += min(recent_matches, 4) * 0.075
    if surface_changed and recent_matches == 0:
        adaptation -= 0.18
    if days_since is not None and days_since > 60:
        adaptation -= 0.10

    return {
        "previous_surface": previous_surface or None,
        "surface_changed": bool(surface_changed),
        "matches_current_surface_30": recent_matches,
        "days_since_current_surface": days_since,
        "adaptation_score": float(np.clip(adaptation, 0.10, 0.90)),
    }


def style_profile(profile_data: dict, manual_style: str = "Auto") -> dict:
    """Create a style label only when serve/return inputs are actually available."""
    serve = profile_data.get("serve_points_won", np.nan)
    ret = profile_data.get("return_points_won", np.nan)
    if manual_style and manual_style != "Auto":
        return {"label": manual_style, "serve_score": serve, "return_score": ret, "manual": True, "available": True}
    if pd.isna(serve) or pd.isna(ret):
        return {
            "label": "Style data unavailable",
            "serve_score": np.nan,
            "return_score": np.nan,
            "manual": False,
            "available": False,
        }

    serve = float(serve)
    ret = float(ret)
    if serve >= 0.665 and ret < 0.385:
        label = "Big Server"
    elif ret >= 0.405 and serve < 0.635:
        label = "Elite Returner"
    elif serve >= 0.645 and ret >= 0.395:
        label = "Aggressive All-Court"
    elif serve < 0.625 and ret >= 0.395:
        label = "Counterpuncher"
    else:
        label = "Balanced Baseliner"
    return {"label": label, "serve_score": serve, "return_score": ret, "manual": False, "available": True}


def style_matchup_adjustment(
    style_a: dict,
    style_b: dict,
    handedness_a: str,
    handedness_b: str,
    surface: str,
) -> tuple[float, str]:
    """Small matchup adjustment; capped because style tags are coarse."""
    if not style_a.get("available", True) or not style_b.get("available", True):
        return 0.0, "Automatic style adjustment skipped because verified serve/return inputs are unavailable"

    a = style_a["label"]
    b = style_b["label"]
    surface_key = str(surface).casefold()
    impact = 0.0
    notes = []

    if a == "Elite Returner" and b == "Big Server":
        impact += 0.018
        notes.append("Player A's return profile counters Player B's serve dependence")
    elif b == "Elite Returner" and a == "Big Server":
        impact -= 0.018
        notes.append("Player B's return profile counters Player A's serve dependence")

    if a == "Counterpuncher" and b in {"Aggressive All-Court", "Big Server"}:
        impact += 0.009 if surface_key == "clay" else 0.003
        notes.append("Player A's defensive profile gains value in longer exchanges")
    elif b == "Counterpuncher" and a in {"Aggressive All-Court", "Big Server"}:
        impact -= 0.009 if surface_key == "clay" else 0.003
        notes.append("Player B's defensive profile gains value in longer exchanges")

    if a == "Big Server" and surface_key in {"grass", "carpet"}:
        impact += 0.008
    if b == "Big Server" and surface_key in {"grass", "carpet"}:
        impact -= 0.008

    # Handedness-specific performance is handled separately by the historical
    # lefty/righty split engine. Do not apply a generic left-hander bonus here;
    # that would double-count handedness and assume an effect the player's own
    # results may not support.

    reason = "; ".join(notes) if notes else "No material style interaction detected"
    return float(np.clip(impact, -0.025, 0.025)), reason


def injury_risk_score(status: str) -> float:
    return {
        "Clear": 0.0,
        "Minor concern": 0.012,
        "Recent medical timeout": 0.022,
        "Returning from layoff": 0.028,
        "Recent retirement": 0.040,
        "Significant concern": 0.050,
    }.get(str(status), 0.0)


def motivation_score(
    home_event: bool,
    defending_status: str,
    priority: str,
    ranking_pressure: str,
) -> float:
    score = 0.0
    if home_event:
        score += 0.006
    score += {
        "None": 0.0,
        "Defending meaningful points": 0.006,
        "Defending title/final": 0.010,
    }.get(str(defending_status), 0.0)
    score += {
        "Low": -0.010,
        "Normal": 0.0,
        "High": 0.008,
    }.get(str(priority), 0.0)
    score += {
        "None": 0.0,
        "Moderate": 0.003,
        "High": 0.006,
    }.get(str(ranking_pressure), 0.0)
    return score



def detailed_player_archetype(profile_data: dict, style_data: dict, surface: str) -> dict:
    """Build a richer, evidence-limited archetype without inventing unavailable traits."""
    serve = profile_data.get("serve_points_won", np.nan)
    ret = profile_data.get("return_points_won", np.nan)
    surface_win = profile_data.get("surface_win", np.nan)
    base_style = str(style_data.get("label", "Style data unavailable"))

    if pd.isna(serve) or pd.isna(ret):
        return {
            "label": base_style,
            "base_style": base_style,
            "traits": ["Verified serve/return detail is unavailable"],
            "reliability": "Low",
        }

    serve = float(serve)
    ret = float(ret)
    traits = []
    if serve >= 0.665:
        traits.append("serve-led point construction")
    elif serve < 0.625:
        traits.append("limited free-point production")
    else:
        traits.append("balanced serving profile")

    if ret >= 0.405:
        traits.append("strong return pressure")
    elif ret < 0.380:
        traits.append("return-dependent vulnerability")
    else:
        traits.append("average return pressure")

    if base_style == "Big Server":
        label = "Serve-First Aggressor"
    elif base_style == "Elite Returner":
        label = "Return-Pressure Baseliner"
    elif base_style == "Aggressive All-Court":
        label = "Two-Way First-Strike Player"
    elif base_style == "Counterpuncher":
        label = "Defensive Counterpuncher"
    else:
        label = "Balanced Baseline Player"

    if str(surface).casefold() == "clay" and ret >= 0.400:
        label = "Clay-Court Return Grinder"
    elif str(surface).casefold() in {"grass", "carpet"} and serve >= 0.655:
        label = "Fast-Court Serve Aggressor"

    if pd.notna(surface_win):
        traits.append(f"{surface} win rate {float(surface_win):.0%}")

    sample = min(
        int(profile_data.get("serve_sample", 0) or 0),
        int(profile_data.get("return_sample", 0) or 0),
    )
    reliability = "High" if sample >= 15 else "Moderate" if sample >= 6 else "Low"
    return {
        "label": label,
        "base_style": base_style,
        "traits": traits,
        "reliability": reliability,
    }


def build_match_intelligence(
    player_a: str,
    player_b: str,
    probability_a: float,
    profile_a: dict,
    profile_b: dict,
    archetype_a: dict,
    archetype_b: dict,
    factors: list[tuple[str, float, str]],
    simulation: dict,
    fatigue_a: dict,
    fatigue_b: dict,
    transition_a: dict,
    transition_b: dict,
    injury_status_a: str,
    injury_status_b: str,
) -> dict:
    """Describe stability, volatility and realistic upset paths separately from price."""
    probability_a = float(np.clip(probability_a, 0.01, 0.99))
    favorite_is_a = probability_a >= 0.50
    favorite = player_a if favorite_is_a else player_b
    underdog = player_b if favorite_is_a else player_a
    favorite_probability = probability_a if favorite_is_a else 1 - probability_a

    signed_for_favorite = [impact if favorite_is_a else -impact for _, impact, _ in factors]
    material = [value for value in signed_for_favorite if abs(value) >= 0.004]
    supporting = sum(value > 0 for value in material)
    opposing = sum(value < 0 for value in material)
    consensus = supporting / max(supporting + opposing, 1)

    minimum_sample = min(int(profile_a.get("sample", 0)), int(profile_b.get("sample", 0)))
    sample_score = min(minimum_sample / 40.0, 1.0)
    surface_sample = min(int(profile_a.get("surface_sample", 0)), int(profile_b.get("surface_sample", 0)))
    surface_score = min(surface_sample / 20.0, 1.0)

    deciding = float(simulation.get("deciding_set", 0.0) or 0.0)
    closeness = 1.0 - min(abs(favorite_probability - 0.50) / 0.42, 1.0)

    serves = [profile_a.get("serve_points_won", np.nan), profile_b.get("serve_points_won", np.nan)]
    returns = [profile_a.get("return_points_won", np.nan), profile_b.get("return_points_won", np.nan)]
    if all(pd.notna(value) for value in serves + returns):
        serve_dominance = float(np.clip((np.mean(serves) - np.mean(returns) - 0.245) / 0.055, 0.0, 1.0))
    else:
        serve_dominance = 0.45

    health_uncertainty = min(
        int(injury_status_a != "Clear") + int(injury_status_b != "Clear"), 2
    ) / 2.0
    transition_uncertainty = abs(
        float(transition_a.get("adaptation_score", 0.5))
        - float(transition_b.get("adaptation_score", 0.5))
    )
    fatigue_gap = min(abs(float(fatigue_a.get("score", 0.0)) - float(fatigue_b.get("score", 0.0))) / 8.0, 1.0)

    volatility = round(100 * np.clip(
        0.34 * deciding
        + 0.25 * closeness
        + 0.18 * serve_dominance
        + 0.10 * health_uncertainty
        + 0.07 * transition_uncertainty
        + 0.06 * fatigue_gap,
        0.0,
        1.0,
    ))

    stability = round(100 * np.clip(
        0.31 * ((favorite_probability - 0.50) / 0.42)
        + 0.25 * consensus
        + 0.18 * sample_score
        + 0.10 * surface_score
        + 0.16 * (1.0 - volatility / 100.0),
        0.0,
        1.0,
    ))

    def band(score: int, high_good: bool = True) -> str:
        if high_good:
            return "Elite" if score >= 85 else "Strong" if score >= 70 else "Moderate" if score >= 55 else "Fragile"
        return "Extreme" if score >= 75 else "High" if score >= 60 else "Moderate" if score >= 40 else "Low"

    path_templates = {
        "Experience Engine": f"{underdog} can shorten the experience gap by starting quickly and preventing the match from becoming a composure test.",
        "Context-weighted matchup": f"{underdog} can win by attacking the favorite's weaker serve-or-return pattern and controlling first-strike points.",
        "Context-weighted recent form": f"{underdog} has a path if recent form carries over and {favorite} fails to reach its normal level.",
        "Opponent strength": f"{underdog}'s recent competition may have prepared them better for the pace and quality of this matchup.",
        "Surface": f"{underdog} can turn the match into surface-specific patterns where their recent results have been stronger.",
        "Fatigue 2.0": f"{underdog} can extend rallies and sets, forcing {favorite} to pay for the heavier recent workload.",
        "Surface transition": f"{underdog} can build an early lead while {favorite} is still adjusting timing and movement to the surface.",
        "Style matchup": f"{underdog}'s archetype can disrupt {favorite}'s preferred patterns and force an uncomfortable plan B.",
        "Injury / retirement risk": f"Any physical limitation to {favorite} creates a direct upset path through reduced movement, serve quality or endurance.",
        "Tournament motivation": f"{underdog} can close the talent gap by bringing greater urgency to the event and key points.",
        "Draw context": f"Tournament circumstances may give {underdog} the freer competitive position.",
        "Event pressure": f"{underdog} can win if {favorite} tightens in the important games and pressure execution swings the match.",
        "Deciding-match history": f"If the match reaches a deciding set, {underdog}'s late-match history becomes a credible upset route.",
    }

    opposing_factors = []
    for name, impact, reason in factors:
        favorite_impact = impact if favorite_is_a else -impact
        if favorite_impact < -0.002:
            opposing_factors.append((abs(favorite_impact), name, reason))
    opposing_factors.sort(reverse=True)

    upset_paths = []
    for _, name, _ in opposing_factors[:3]:
        text = path_templates.get(name)
        if text and text not in upset_paths:
            upset_paths.append(text)

    if deciding >= 0.38:
        upset_paths.append(f"A long match or deciding set increases variance and gives {underdog} more chances to flip a small number of key points.")
    if serve_dominance >= 0.60:
        upset_paths.append(f"Serve-heavy conditions can compress the match into tiebreaks, where a few points can erase {favorite}'s broader edge.")
    if not upset_paths:
        upset_paths.append(f"{underdog}'s clearest route is to neutralize {favorite}'s main advantage early and force a higher-variance match than projected.")

    drivers = []
    if deciding >= 0.38:
        drivers.append("high deciding-set probability")
    if serve_dominance >= 0.60:
        drivers.append("serve-driven point variance")
    if closeness >= 0.60:
        drivers.append("limited separation between the players")
    if health_uncertainty > 0:
        drivers.append("health uncertainty")
    if opposing >= 2:
        drivers.append("several factors supporting the underdog")
    if not drivers:
        drivers.append("few major structural upset signals")

    return {
        "favorite": favorite,
        "underdog": underdog,
        "favorite_probability": favorite_probability,
        "stability_score": int(stability),
        "stability_band": band(int(stability), True),
        "volatility_score": int(volatility),
        "volatility_band": band(int(volatility), False),
        "factor_consensus": float(consensus),
        "supporting_factors": int(supporting),
        "opposing_factors": int(opposing),
        "drivers": drivers[:4],
        "upset_paths": upset_paths[:4],
        "archetype_a": archetype_a,
        "archetype_b": archetype_b,
    }

def analyze(
    matches: pd.DataFrame,
    player_a: str,
    player_b: str,
    tournament: str,
    round_label: str,
    surface: str,
    event_date: date,
    simulations: int = 20000,
    tournament_category_label: str | None = None,
    environment: str = "Outdoor",
    match_format: str | None = None,
    style_a: str = "Auto",
    style_b: str = "Auto",
    handedness_a: str = "Right",
    handedness_b: str = "Right",
    injury_status_a: str = "Clear",
    injury_status_b: str = "Clear",
    travel_load_a: str = "None",
    travel_load_b: str = "None",
    late_finish_a: bool = False,
    late_finish_b: bool = False,
    home_event_a: bool = False,
    home_event_b: bool = False,
    defending_status_a: str = "None",
    defending_status_b: str = "None",
    priority_a: str = "Normal",
    priority_b: str = "Normal",
    ranking_pressure_a: str = "None",
    ranking_pressure_b: str = "None",
    draw_pressure_a: str = "Normal",
    draw_pressure_b: str = "Normal",
) -> dict:
    resolved_a, resolution_a = resolve_player_name(matches, player_a)
    resolved_b, resolution_b = resolve_player_name(matches, player_b)
    unresolved = [item["requested"] for item in (resolution_a, resolution_b) if item["resolved"] is None]
    if unresolved:
        raise TennisDataValidationError(
            "Insufficient verified data: could not match " + ", ".join(unresolved)
            + " to the historical tennis database. No projection was generated."
        )

    rows_a = perspective(matches, resolved_a, event_date)
    rows_b = perspective(matches, resolved_b, event_date)
    pa = profile(rows_a, surface, event_date)
    pb = profile(rows_b, surface, event_date)

    # Build reusable long-term player intelligence profiles. API enrichment is
    # optional and safely falls back to the historical database when unavailable.
    intelligence_a = build_player_profile(
        matches, resolved_a, event_date, include_api=True
    )
    intelligence_b = build_player_profile(
        matches, resolved_b, event_date, include_api=True
    )
    intelligence_a.requested_name = player_a
    intelligence_b.requested_name = player_b
    experience = compare_experience(
        intelligence_a, intelligence_b, surface,
        maximum_probability_adjustment=0.04,
    )

    validation_errors = []
    for requested, resolved, profile_data in (
        (player_a, resolved_a, pa), (player_b, resolved_b, pb)
    ):
        if profile_data["sample"] < 5:
            validation_errors.append(f"{requested}: only {profile_data['sample']} matches in the two-year sample")
        if pd.isna(profile_data["rank"]):
            validation_errors.append(f"{requested}: ranking unavailable")

    if validation_errors:
        raise TennisDataValidationError(
            "Insufficient verified data. No projection was generated. " + "; ".join(validation_errors)
        )

    fatigue_profile_a = fatigue_profile(rows_a, event_date)
    fatigue_profile_b = fatigue_profile(rows_b, event_date)
    transition_a = surface_transition_profile(rows_a, surface, event_date)
    transition_b = surface_transition_profile(rows_b, surface, event_date)
    playing_style_a = style_profile(pa, style_a)
    playing_style_b = style_profile(pb, style_b)

    overall, surface_table = elo_tables(matches, surface, event_date)
    opponent_strength_a = opponent_strength_profile(
        matches, player_a, event_date, overall, lookback_matches=10
    )
    opponent_strength_b = opponent_strength_profile(
        matches, player_b, event_date, overall, lookback_matches=10
    )
    trajectory_a = breakout_trajectory_profile(matches, resolved_a, event_date)
    trajectory_b = breakout_trajectory_profile(matches, resolved_b, event_date)
    ka, kb = canonical_player_key(resolved_a), canonical_player_key(resolved_b)
    if ka not in overall or kb not in overall:
        missing = [name for name, key in ((player_a, ka), (player_b, kb)) if key not in overall]
        raise TennisDataValidationError(
            "Insufficient verified data: Elo could not be calculated for "
            + ", ".join(missing) + ". No projection was generated."
        )
    oa, ob = overall[ka], overall[kb]
    # Surface Elo may be unavailable for a player new to the surface. In that case,
    # use the verified overall Elo and disclose the fallback in diagnostics.
    sa, sb = surface_table.get(ka, oa), surface_table.get(kb, ob)

    overall_p = rating_probability(oa, ob)
    surface_p = rating_probability(sa, sb)

    category = tournament_category_label or tournament_category(matches, tournament)
    if match_format is None:
        match_format = (
            "Best of 5"
            if category == "Grand Slam" and str(round_label) != "Qualifying"
            else "Best of 3"
        )
    weights = context_weights(
        surface=surface,
        tournament_category_label=category,
        round_label=round_label,
        environment=environment,
        match_format=match_format,
    )

    rank_a = pa["rank"]
    rank_b = pb["rank"]
    rank_p = 1 / (1 + math.exp(-((-math.log(max(rank_a, 1))) - (-math.log(max(rank_b, 1)))) * .9))

    rating_blend_probability = (
        weights["overall_elo"] * overall_p
        + weights["surface_elo"] * surface_p
        + weights["ranking"] * rank_p
    )
    base = float(rating_blend_probability)

    # Quietly correct for a stale historical baseline when an emerging player has
    # demonstrated a sustained breakout across the season. This is intentionally
    # not exposed as a separate UI factor.
    trajectory_adjustment = float(np.clip(
        trajectory_a["probability_uplift"] - trajectory_b["probability_uplift"],
        -0.028,
        0.028,
    ))
    base = float(np.clip(base + trajectory_adjustment, 0.05, 0.95))

    # Quietly account for each player's verified historical performance against
    # the opponent's handedness. Small samples are strongly shrunk toward the
    # player's normal baseline and the matchup effect is capped. This remains an
    # internal model input rather than a new UI factor.
    from .tennis_handedness import handedness_matchup_profile
    handedness_profile_a = handedness_matchup_profile(
        matches, resolved_a, handedness_b, event_date, surface
    )
    handedness_profile_b = handedness_matchup_profile(
        matches, resolved_b, handedness_a, event_date, surface
    )
    handedness_adjustment = float(np.clip(
        handedness_profile_a.get("adjustment", 0.0)
        - handedness_profile_b.get("adjustment", 0.0),
        -0.025,
        0.025,
    ))
    base = float(np.clip(base + handedness_adjustment, 0.05, 0.95))

    # Serve/Return Engine: build opponent-aware, sample-shrunk profiles from
    # verified match-level point statistics. It blends the last year, recent 90
    # days, and surface history, then compares A's serve against B's return and
    # vice versa. The normal UI does not need a new section; diagnostics are
    # retained for Challenge Macabets and later calibration.
    serve_return_profile_a = serve_return_profile(matches, resolved_a, event_date, surface)
    serve_return_profile_b = serve_return_profile(matches, resolved_b, event_date, surface)
    serve_return_matchup = serve_return_matchup_adjustment(
        serve_return_profile_a, serve_return_profile_b
    )
    serve_return_available = bool(serve_return_matchup.get("available"))
    matchup = float(serve_return_matchup.get("probability_adjustment_a", 0.0))
    if serve_return_available:
        matchup_reason = (
            f"{surface}, {environment}, {match_format}. "
            f"{serve_return_matchup['reason']} "
            f"Profiles: {player_a} serve {serve_return_profile_a['serve_points_won']:.1%}, "
            f"return {serve_return_profile_a['return_points_won']:.1%}; "
            f"{player_b} serve {serve_return_profile_b['serve_points_won']:.1%}, "
            f"return {serve_return_profile_b['return_points_won']:.1%}."
        )
    else:
        matchup_reason = str(serve_return_matchup.get("reason") or (
            "Verified serve/return point totals are unavailable, so the factor was excluded."
        ))
    # Recent form is quality-adjusted: raw last-10 results remain the majority
    # signal, while opponent quality gives those results context.
    form = float(np.clip(
        (
            opponent_strength_a["quality_adjusted_form"]
            - opponent_strength_b["quality_adjusted_form"]
        ) * .04 * weights["form"],
        -.045, .045
    ))
    opponent_strength = float(np.clip(
        (
            opponent_strength_a["strength_score"]
            - opponent_strength_b["strength_score"]
        ) * 0.045,
        -0.025,
        0.025,
    ))
    surface_adj = float(np.clip(
        (pa["surface_win"] - pb["surface_win"]) * .045,
        -.04, .04
    ))

    travel_penalty = {"None": 0.0, "Moderate": 0.9, "Heavy": 1.8}
    # Workload signals fatigue risk, not automatic performance deterioration.
    # Sustained high-level results during a busy stretch can soften (never erase) it.
    def _fatigue_resilience(profile_data: dict, strength_data: dict) -> float:
        if profile_data.get("matches_7", 0) < 3:
            return 0.0
        form = float(strength_data.get("quality_adjusted_form", 0.5))
        raw = float(strength_data.get("raw_win_rate", 0.5))
        evidence = max(0.0, (0.65 * form + 0.35 * raw) - 0.58)
        return min(evidence * 2.0, 0.30)

    fatigue_resilience_a = _fatigue_resilience(fatigue_profile_a, opponent_strength_a)
    fatigue_resilience_b = _fatigue_resilience(fatigue_profile_b, opponent_strength_b)
    fatigue_score_a = (
        fatigue_profile_a["score"] * (1.0 - fatigue_resilience_a)
        + travel_penalty.get(travel_load_a, 0.0)
        + (1.2 if late_finish_a else 0.0)
    )
    fatigue_score_b = (
        fatigue_profile_b["score"] * (1.0 - fatigue_resilience_b)
        + travel_penalty.get(travel_load_b, 0.0)
        + (1.2 if late_finish_b else 0.0)
    )
    fatigue = float(np.clip(
        (fatigue_score_b - fatigue_score_a) * .007 * weights["fatigue"],
        -.055, .055
    ))

    transition = float(np.clip(
        (
            transition_a["adaptation_score"]
            - transition_b["adaptation_score"]
        ) * 0.045,
        -0.035,
        0.035,
    ))

    style_matchup, style_reason = style_matchup_adjustment(
        playing_style_a,
        playing_style_b,
        handedness_a,
        handedness_b,
        surface,
    )
    # Auto style labels are derived directly from the same serve/return point data
    # used by the dedicated Serve/Return Engine. When that engine is available,
    # applying an additional automatic style adjustment would count the same signal
    # twice. Preserve manual style overrides because they can contain independent
    # scouting information that is not encoded in the point-stat profile.
    auto_style_suppressed = bool(
        serve_return_available and style_a == "Auto" and style_b == "Auto"
    )
    if auto_style_suppressed:
        original_auto_style_matchup = float(style_matchup)
        style_matchup = 0.0
        style_reason = (
            "Automatic style adjustment suppressed because its labels are derived from "
            "the same serve/return statistics already priced by the Serve/Return Engine. "
            f"Pre-guard style impact would have been {original_auto_style_matchup:+.2%}."
        )
    else:
        original_auto_style_matchup = float(style_matchup)

    injury_a = injury_risk_score(injury_status_a)
    injury_b = injury_risk_score(injury_status_b)
    injury = float(np.clip(injury_b - injury_a, -0.05, 0.05))

    motivation_a = motivation_score(
        home_event_a, defending_status_a, priority_a, ranking_pressure_a
    )
    motivation_b = motivation_score(
        home_event_b, defending_status_b, priority_b, ranking_pressure_b
    )
    motivation = float(np.clip(motivation_a - motivation_b, -0.025, 0.025))

    draw_values = {"Favorable": 0.003, "Normal": 0.0, "Difficult": -0.004}
    draw_context = float(np.clip(
        draw_values.get(draw_pressure_a, 0.0)
        - draw_values.get(draw_pressure_b, 0.0),
        -0.008,
        0.008,
    ))

    pressure = float(np.clip(
        (
            (pa["advanced_win"] - pb["advanced_win"]) * .035
            + (pa["big_event_win"] - pb["big_event_win"]) * .025
        ) * weights["pressure"],
        -.05, .05
    ))
    deciding = float(np.clip(
        (pa["deciding_win"] - pb["deciding_win"])
        * .02
        * weights["deciding"],
        -.03, .03
    ))

    # Form, opponent strength and surface win rate overlap with Elo and ranking.
    # Discount them as a group so the same underlying performance is not counted repeatedly.
    correlated_discount = 0.65
    form *= correlated_discount
    opponent_strength *= correlated_discount
    surface_adj *= correlated_discount

    experience_adjustment = float(experience["probability_adjustment_a"])
    # A sustained breakout makes a generic "lack of career experience" penalty less
    # trustworthy because the player's historical sample may no longer represent his
    # current level. Situational experience still matters; only the generic penalty is
    # attenuated, and never reversed.
    if experience_adjustment < 0:
        experience_adjustment *= 1.0 - 0.55 * trajectory_a["score"]
    elif experience_adjustment > 0:
        experience_adjustment *= 1.0 - 0.55 * trajectory_b["score"]
    experience_reason = (
        f"Career matches: {player_a} {intelligence_a.career_matches}, "
        f"{player_b} {intelligence_b.career_matches}. {surface} matches: "
        f"{player_a} {experience['surface_matches_a']}, "
        f"{player_b} {experience['surface_matches_b']}. Grand Slam/Masters matches: "
        f"{player_a} {intelligence_a.grand_slam_matches + intelligence_a.masters_matches}, "
        f"{player_b} {intelligence_b.grand_slam_matches + intelligence_b.masters_matches}. "
        f"The experience impact is capped at ±{experience['maximum_adjustment']:.0%}."
    )

    factors = [
        ("Experience Engine", experience_adjustment, experience_reason),
        ("Context-weighted matchup", matchup, matchup_reason),
        ("Context-weighted recent form", form,
         f"Last-10 win rate: {player_a} {opponent_strength_a['raw_win_rate']:.0%}; "
         f"{player_b} {opponent_strength_b['raw_win_rate']:.0%}. Quality-adjusted form: "
         f"{opponent_strength_a['quality_adjusted_form']:.0%} vs "
         f"{opponent_strength_b['quality_adjusted_form']:.0%}. The adjustment considers "
         f"who produced those results while keeping raw form as the majority signal. "
         f"Context multiplier: {weights['form']:.2f}x; correlation discount applied."),
        ("Opponent strength", opponent_strength,
         f"Last-{opponent_strength_a['matches']}/{opponent_strength_b['matches']} schedule score: "
         f"{player_a} {opponent_strength_a['strength_score']:.0%} vs {player_b} "
         f"{opponent_strength_b['strength_score']:.0%}. Average opponent Elo: "
         f"{opponent_strength_a['avg_opponent_elo']:.0f} vs "
         f"{opponent_strength_b['avg_opponent_elo']:.0f}; average opponent rank: "
         f"{opponent_strength_a['avg_opponent_rank'] or 'N/A'} vs "
         f"{opponent_strength_b['avg_opponent_rank'] or 'N/A'}. Top-50 records: "
         f"{opponent_strength_a['top_50_record']} vs {opponent_strength_b['top_50_record']}. "
         f"Quality wins (top 100): {player_a} "
         f"{', '.join(x['opponent'] + ' (#' + str(x['rank']) + ')' for x in opponent_strength_a['quality_wins']) or 'none'}; "
         f"{player_b} {', '.join(x['opponent'] + ' (#' + str(x['rank']) + ')' for x in opponent_strength_b['quality_wins']) or 'none'}. "
         f"Bad losses (outside top 100): {player_a} "
         f"{', '.join(x['opponent'] + ' (#' + str(x['rank']) + ')' for x in opponent_strength_a['bad_losses']) or 'none'}; "
         f"{player_b} {', '.join(x['opponent'] + ' (#' + str(x['rank']) + ')' for x in opponent_strength_b['bad_losses']) or 'none'}. "
         f"Strength of schedule is capped as a modest context adjustment; correlation discount applied."),
        ("Surface", surface_adj,
         f"Two-year {surface} win rate: {player_a} {pa['surface_win']:.0%}; "
         f"{player_b} {pb['surface_win']:.0%}. A correlation discount is applied because "
         f"surface results overlap with surface Elo."),
        ("Fatigue 2.0", fatigue,
         f"{player_a}: {fatigue_profile_a['matches_7']} matches, "
         f"{fatigue_profile_a['sets_7']} sets, {fatigue_profile_a['deciders_7']} deciders "
         f"in 7 days, {fatigue_profile_a['rest_days']} rest days, travel {travel_load_a}"
         f"{', late finish' if late_finish_a else ''}. {player_b}: "
         f"{fatigue_profile_b['matches_7']} matches, {fatigue_profile_b['sets_7']} sets, "
         f"{fatigue_profile_b['deciders_7']} deciders, "
         f"{fatigue_profile_b['rest_days']} rest days, travel {travel_load_b}"
         f"{', late finish' if late_finish_b else ''}. High-level recent performance offsets "
         f"{player_a}'s workload score by {fatigue_resilience_a:.0%} and {player_b}'s by "
         f"{fatigue_resilience_b:.0%}; workload remains a risk signal rather than assumed deterioration."),
        ("Surface transition", transition,
         f"{player_a}: previous surface {transition_a['previous_surface'] or 'unknown'}, "
         f"{transition_a['matches_current_surface_30']} current-surface matches in 30 days, "
         f"adaptation {transition_a['adaptation_score']:.0%}. {player_b}: previous surface "
         f"{transition_b['previous_surface'] or 'unknown'}, "
         f"{transition_b['matches_current_surface_30']} current-surface matches, "
         f"adaptation {transition_b['adaptation_score']:.0%}."),
        ("Style matchup", style_matchup,
         f"{player_a}: {playing_style_a['label']} ({handedness_a}-handed). "
         f"{player_b}: {playing_style_b['label']} ({handedness_b}-handed). {style_reason}."),
        ("Injury / retirement risk", injury,
         f"{player_a}: {injury_status_a}. {player_b}: {injury_status_b}."),
        ("Tournament motivation", motivation,
         f"{player_a}: priority {priority_a}, defending {defending_status_a}, "
         f"ranking pressure {ranking_pressure_a}, home event {home_event_a}. "
         f"{player_b}: priority {priority_b}, defending {defending_status_b}, "
         f"ranking pressure {ranking_pressure_b}, home event {home_event_b}."),
        ("Draw context", draw_context,
         f"Forward draw pressure: {player_a} {draw_pressure_a}; "
         f"{player_b} {draw_pressure_b}. This factor is deliberately capped."),
        ("Event pressure", pressure,
         f"{category}, {round_label}, {match_format}. Advanced-round win rate: "
         f"{player_a} {pa['advanced_win']:.0%}; {player_b} {pb['advanced_win']:.0%}. "
         f"Pressure multiplier: {weights['pressure']:.2f}x."),
        ("Deciding-match history", deciding,
         f"Deciding-match win rate: {player_a} {pa['deciding_win']:.0%}; "
         f"{player_b} {pb['deciding_win']:.0%}."),
    ]

    # Cap the combined secondary adjustment. Context should refine the core rating,
    # not overpower it when several related factors all point in the same direction.
    uncapped_secondary_adjustment = float(sum(v for _, v, _ in factors))
    total_adjustment = float(np.clip(uncapped_secondary_adjustment, -0.12, 0.12))
    raw_model = float(np.clip(base + total_adjustment, 0.05, 0.95))

    # Calibrate extreme outputs back toward 50%. Sparse data receives more shrinkage.
    minimum_sample = min(pa["sample"], pb["sample"])
    if minimum_sample >= 40 and serve_return_available:
        calibration_strength = 0.88
    elif minimum_sample >= 20:
        calibration_strength = 0.82
    else:
        calibration_strength = 0.76
    final_model = float(np.clip(
        0.50 + (raw_model - 0.50) * calibration_strength,
        0.08,
        0.92,
    ))
    best_of_five = str(match_format).casefold() == "best of 5"
    simulation = simulate_matches(final_model, simulations, best_of_five)

    # Narrative identities are explanation-only and do not feed the model.
    archetype_a = get_player_identity(player_a, pa, playing_style_a, surface)
    archetype_b = get_player_identity(player_b, pb, playing_style_b, surface)
    match_intelligence = build_match_intelligence(
        player_a=player_a,
        player_b=player_b,
        probability_a=simulation["win_probability"],
        profile_a=pa,
        profile_b=pb,
        archetype_a=archetype_a,
        archetype_b=archetype_b,
        factors=factors,
        simulation=simulation,
        fatigue_a=fatigue_profile_a,
        fatigue_b=fatigue_profile_b,
        transition_a=transition_a,
        transition_b=transition_b,
        injury_status_a=injury_status_a,
        injury_status_b=injury_status_b,
    )

    sample = min(pa["sample"], pb["sample"])
    quality = int(np.clip(round(3 + min(sample, 50) / 8), 3, 10))
    if not serve_return_available:
        quality = max(3, quality - 2)

    # Prediction confidence is a reliability score, not another win-probability
    # estimate. In particular, do not award confidence simply because the model
    # probability is farther from 50%. Win probability/fair line already carry
    # that information.
    data_score = float(np.clip(quality * 10.0, 0.0, 100.0))
    sample_score = float(np.clip(sample / 40.0 * 100.0, 0.0, 100.0))
    min_surface_sample = min(pa.get("surface_sample", 0), pb.get("surface_sample", 0))
    surface_score = float(np.clip(min_surface_sample / 20.0 * 100.0, 0.0, 100.0))
    min_serve_sample = min(pa.get("serve_sample", 0), pb.get("serve_sample", 0))
    min_return_sample = min(pa.get("return_sample", 0), pb.get("return_sample", 0))
    serve_return_sample = min(min_serve_sample, min_return_sample)
    serve_return_score = float(
        np.clip(serve_return_sample / 15.0 * 100.0, 0.0, 100.0)
        if serve_return_available else 0.0
    )

    core_probabilities = np.array([overall_p, surface_p, rank_p], dtype=float)
    core_dispersion = float(np.std(core_probabilities))
    core_agreement_score = float(np.clip(100.0 - (core_dispersion / 0.12) * 100.0, 0.0, 100.0))

    cap_excess = max(0.0, abs(uncapped_secondary_adjustment) - abs(total_adjustment))
    simulation_shift = abs(float(simulation["win_probability"] - final_model))
    stability_penalty = min(100.0, cap_excess / 0.06 * 60.0 + simulation_shift / 0.06 * 40.0)
    stability_score = float(np.clip(100.0 - stability_penalty, 0.0, 100.0))

    health_penalty = (
        int(injury_status_a != "Clear") + int(injury_status_b != "Clear")
    ) * 18.0
    style_penalty = (int(style_a == "Auto") + int(style_b == "Auto")) * 5.0
    context_score = float(np.clip(100.0 - health_penalty - style_penalty, 0.0, 100.0))

    confidence_overall = int(round(np.clip(
        data_score * 0.20
        + sample_score * 0.18
        + surface_score * 0.14
        + serve_return_score * 0.10
        + core_agreement_score * 0.18
        + stability_score * 0.12
        + context_score * 0.08,
        0.0,
        100.0,
    )))
    confidence = int(np.clip(round(confidence_overall / 10.0), 1, 10))
    confidence_reliability = {
        "version": "Macabets Tennis Confidence v1.0 — Reliability",
        "overall": confidence_overall,
        "data": round(data_score),
        "sample": round(sample_score),
        "surface": round(surface_score),
        "serve_return": round(serve_return_score),
        "core_agreement": round(core_agreement_score),
        "stability": round(stability_score),
        "context": round(context_score),
        "minimum_sample": int(sample),
        "minimum_surface_sample": int(min_surface_sample),
        "minimum_serve_return_sample": int(serve_return_sample),
        "core_probability_dispersion": round(core_dispersion, 4),
        "simulation_shift": round(simulation_shift, 4),
        "secondary_cap_excess": round(cap_excess, 4),
        "note": "Reliability only. Win-probability extremeness does not increase this score.",
    }

    return {
        "player_a": player_a,
        "player_b": player_b,
        "tournament": tournament,
        "round": round_label,
        "surface": surface,
        "tournament_category": category,
        "environment": environment,
        "match_format": match_format,
        "context_weights": weights,
        "opponent_strength_a": opponent_strength_a,
        "opponent_strength_b": opponent_strength_b,
        "recent_resume_comparison": {
            "player_a": opponent_strength_a,
            "player_b": opponent_strength_b,
            "quality_adjusted_form_edge_a": float(opponent_strength_a["quality_adjusted_form"] - opponent_strength_b["quality_adjusted_form"]),
            "schedule_strength_edge_a": float(opponent_strength_a["strength_score"] - opponent_strength_b["strength_score"]),
        },
        # Internal diagnostics only. The Streamlit UI does not render these fields.
        "trajectory_engine": {
            "player_a": trajectory_a,
            "player_b": trajectory_b,
            "probability_adjustment_a": trajectory_adjustment,
        },
        # Internal diagnostics only. Challenge Macabets can use the same verified
        # split records through tennis_evidence, while the normal UI stays clean.
        "handedness_engine": {
            "player_a_vs_opponent_hand": handedness_profile_a,
            "player_b_vs_opponent_hand": handedness_profile_b,
            "probability_adjustment_a": handedness_adjustment,
        },
        "serve_return_engine": {
            "player_a": serve_return_profile_a,
            "player_b": serve_return_profile_b,
            "matchup": serve_return_matchup,
            "probability_adjustment_a": matchup,
        },
        "fatigue_resilience_a": fatigue_resilience_a,
        "fatigue_resilience_b": fatigue_resilience_b,
        "fatigue_profile_a": fatigue_profile_a,
        "fatigue_profile_b": fatigue_profile_b,
        "surface_transition_a": transition_a,
        "surface_transition_b": transition_b,
        "playing_style_a": playing_style_a,
        "playing_style_b": playing_style_b,
        "handedness_a": handedness_a,
        "handedness_b": handedness_b,
        "injury_status_a": injury_status_a,
        "injury_status_b": injury_status_b,
        "motivation_context": {
            "player_a": {
                "home_event": home_event_a,
                "defending_status": defending_status_a,
                "priority": priority_a,
                "ranking_pressure": ranking_pressure_a,
                "draw_pressure": draw_pressure_a,
            },
            "player_b": {
                "home_event": home_event_b,
                "defending_status": defending_status_b,
                "priority": priority_b,
                "ranking_pressure": ranking_pressure_b,
                "draw_pressure": draw_pressure_b,
            },
        },
        "base_probability": base,
        "raw_model_probability": raw_model,
        "total_secondary_adjustment": total_adjustment,
        "calibration_strength": calibration_strength,
        "model_probability": final_model,
        "win_probability": simulation["win_probability"],
        "probability_decomposition": {
            "rating_core": {
                "overall_elo_probability_a": float(overall_p),
                "surface_elo_probability_a": float(surface_p),
                "ranking_probability_a": float(rank_p),
                "overall_elo_weight": float(weights["overall_elo"]),
                "surface_elo_weight": float(weights["surface_elo"]),
                "ranking_weight": float(weights["ranking"]),
                "blended_probability_a": float(rating_blend_probability),
            },
            "internal_baseline_adjustments": [
                {"name": "Breakout trajectory", "impact": float(trajectory_adjustment)},
                {"name": "Handedness history", "impact": float(handedness_adjustment)},
            ],
            "baseline_after_internal_adjustments": float(base),
            "secondary_factors": [
                {"name": name, "impact": float(impact)} for name, impact, _ in factors
            ],
            "feature_overlap_audit": {
                "version": "Macabets Tennis Feature Overlap Guard v0.1",
                "auto_style_suppressed": auto_style_suppressed,
                "pre_guard_auto_style_impact": float(original_auto_style_matchup),
                "serve_return_available": serve_return_available,
                "existing_correlated_discount": 0.65,
                "discounted_group": [
                    "Context-weighted recent form",
                    "Opponent strength",
                    "Surface",
                ],
                "notes": [
                    "Recent form, opponent strength, and surface win rate already receive a 0.65 correlation discount because they overlap with Elo/ranking.",
                    "Automatic style is now suppressed whenever the Serve/Return Engine is available because both are derived from the same serve/return point statistics.",
                    "Breakout trajectory remains capped and persistence-gated because it is intended to correct stale Elo/ranking rather than duplicate a hot streak.",
                    "Fatigue uses recent performance only to soften workload risk; it does not add recent form a second time as a standalone positive factor.",
                    "Event-pressure and deciding-match inputs remain small/capped and should be judged with forward results before further reweighting.",
                ],
            },
            "uncapped_secondary_adjustment": float(uncapped_secondary_adjustment),
            "secondary_cap": 0.12,
            "capped_secondary_adjustment": float(total_adjustment),
            "raw_probability_before_calibration": float(raw_model),
            "calibration_strength": float(calibration_strength),
            "probability_after_calibration": float(final_model),
            "calibration_shift": float(final_model - raw_model),
            "simulation_probability": float(simulation["win_probability"]),
            "simulation_shift": float(simulation["win_probability"] - final_model),
            "simulation_note": (
                "The engine converts the calibrated match probability to a set probability, "
                "then reconstructs best-of-3/best-of-5 match outcomes by Monte Carlo. The audit "
                "reports any probability shift introduced by that transformation."
            ),
        },
        "fair_line": american_from_probability(simulation["win_probability"]),
        "confidence": confidence,
        "confidence_reliability": confidence_reliability,
        "data_quality": quality,
        "overall_elo": (oa, ob),
        "surface_elo": (sa, sb),
        "profile_a": pa,
        "profile_b": pb,
        "player_intelligence_a": intelligence_a.to_dict(),
        "player_intelligence_b": intelligence_b.to_dict(),
        "experience_engine": {
            **experience,
            "reliability_a": experience_reliability(intelligence_a),
            "reliability_b": experience_reliability(intelligence_b),
        },
        "data_validation": {
            "status": "verified",
            "player_a": {
                **resolution_a,
                "historical_matches": int(len(rows_a)),
                "two_year_sample": int(pa["sample"]),
                "surface_sample": int(pa["surface_sample"]),
                "serve_sample": int(pa["serve_sample"]),
                "return_sample": int(pa["return_sample"]),
                "overall_elo_found": ka in overall,
                "surface_elo_found": ka in surface_table,
                "flags": pa["data_flags"],
            },
            "player_b": {
                **resolution_b,
                "historical_matches": int(len(rows_b)),
                "two_year_sample": int(pb["sample"]),
                "surface_sample": int(pb["surface_sample"]),
                "serve_sample": int(pb["serve_sample"]),
                "return_sample": int(pb["return_sample"]),
                "overall_elo_found": kb in overall,
                "surface_elo_found": kb in surface_table,
                "flags": pb["data_flags"],
            },
        },
        "factors": [
            {"name": name, "impact": impact, "reason": reason}
            for name, impact, reason in factors
        ],
        "player_archetype_a": archetype_a,
        "player_archetype_b": archetype_b,
        "match_intelligence": match_intelligence,
        "simulation": simulation,
    }
