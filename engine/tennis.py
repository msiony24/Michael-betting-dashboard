from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
import math
import re

import numpy as np
import pandas as pd


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


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


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
    key = norm(player)
    subset = matches[
        (
            matches["winner_name"].map(norm).eq(key)
            | matches["loser_name"].map(norm).eq(key)
        )
        & (matches["tourney_date"] < pd.Timestamp(event_date))
    ].sort_values("tourney_date")

    rows = []
    for _, match in subset.iterrows():
        won = norm(match["winner_name"]) == key
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
    if rows.empty:
        return {
            "rank": np.nan, "recent_win": .5, "surface_win": .5,
            "serve_points_won": .62, "return_points_won": .38,
            "matches_7": 0, "matches_14": 0, "rest_days": 30,
            "advanced_win": .5, "big_event_win": .5,
            "deciding_win": .5, "sample": 0,
        }

    event_ts = pd.Timestamp(event_date)
    two_year = rows[rows["date"] >= event_ts - pd.Timedelta(days=730)]
    one_year = rows[rows["date"] >= event_ts - pd.Timedelta(days=365)]
    recent = rows.tail(10)
    surf = two_year[two_year["surface"].str.casefold() == surface.casefold()]
    advanced = two_year[two_year["round"].isin(["QF", "SF", "F"])]
    big = two_year[two_year["level"].isin(["G", "M", "F"])]
    deciding = two_year[two_year["score"].str.count(r"\d+-\d+") >= 3]

    serve_den = one_year["svpt"].sum()
    serve_num = (one_year["first_won"].fillna(0) + one_year["second_won"].fillna(0)).sum()
    return_den = one_year["opp_svpt"].sum()
    return_num = (
        one_year["opp_svpt"].fillna(0)
        - one_year["opp_first_won"].fillna(0)
        - one_year["opp_second_won"].fillna(0)
    ).sum()

    ranks = rows["rank"].dropna()
    last_date = rows["date"].max()

    return {
        "rank": float(ranks.iloc[-1]) if len(ranks) else np.nan,
        "recent_win": float(recent["won"].mean()) if len(recent) else .5,
        "surface_win": float(surf["won"].mean()) if len(surf) else .5,
        "serve_points_won": safe_ratio(serve_num, serve_den, .62),
        "return_points_won": safe_ratio(return_num, return_den, .38),
        "matches_7": int((rows["date"] >= event_ts - pd.Timedelta(days=7)).sum()),
        "matches_14": int((rows["date"] >= event_ts - pd.Timedelta(days=14)).sum()),
        "rest_days": max(0, safe_int((event_ts - last_date).days, 30)),
        "advanced_win": float(advanced["won"].mean()) if len(advanced) >= 4 else .5,
        "big_event_win": float(big["won"].mean()) if len(big) >= 5 else .5,
        "deciding_win": float(deciding["won"].mean()) if len(deciding) >= 4 else .5,
        "sample": len(two_year),
    }


def opponent_strength_profile(
    matches: pd.DataFrame,
    player: str,
    event_date: date,
    overall_elo: dict[str, float],
    lookback_matches: int = 10,
) -> dict:
    """
    Evaluate the quality of a player's recent opposition.

    Uses the opponent's Elo at the analysis date, ranking recorded in the match,
    and whether the player won or lost. The result is centered around 50%.
    """
    key = norm(player)
    history = matches[
        (
            matches["winner_name"].map(norm).eq(key)
            | matches["loser_name"].map(norm).eq(key)
        )
        & (matches["tourney_date"] < pd.Timestamp(event_date))
    ].sort_values("tourney_date", ascending=False).head(lookback_matches)

    if history.empty:
        return {
            "matches": 0,
            "avg_opponent_rank": None,
            "avg_opponent_elo": 1500.0,
            "quality_form": 0.5,
            "top_50_record": "0-0",
            "top_100_record": "0-0",
            "strength_score": 0.5,
        }

    opponent_ranks = []
    opponent_elos = []
    quality_results = []
    top_50_wins = top_50_losses = 0
    top_100_wins = top_100_losses = 0

    for _, row in history.iterrows():
        won = norm(row["winner_name"]) == key
        opponent_name = row["loser_name"] if won else row["winner_name"]
        opponent_rank = row.get("loser_rank" if won else "winner_rank", np.nan)
        opponent_elo = overall_elo.get(norm(opponent_name), 1500.0)

        opponent_elos.append(float(opponent_elo))
        if pd.notna(opponent_rank):
            opponent_rank = float(opponent_rank)
            opponent_ranks.append(opponent_rank)

            if opponent_rank <= 50:
                if won:
                    top_50_wins += 1
                else:
                    top_50_losses += 1
            if opponent_rank <= 100:
                if won:
                    top_100_wins += 1
                else:
                    top_100_losses += 1

        # A win over a strong opponent earns more credit; a loss to a strong
        # opponent is penalized less than a loss to weak opposition.
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
    quality_form = float(sum(quality_results) / len(quality_results))

    elo_component = 1 / (1 + math.exp(-(avg_elo - 1500.0) / 160.0))
    rank_component = (
        1 / (1 + math.exp((avg_rank - 75.0) / 35.0))
        if avg_rank is not None else 0.5
    )

    strength_score = float(np.clip(
        0.45 * elo_component
        + 0.30 * rank_component
        + 0.25 * quality_form,
        0.0,
        1.0,
    ))

    return {
        "matches": int(len(history)),
        "avg_opponent_rank": avg_rank,
        "avg_opponent_elo": avg_elo,
        "quality_form": quality_form,
        "top_50_record": f"{top_50_wins}-{top_50_losses}",
        "top_100_record": f"{top_100_wins}-{top_100_losses}",
        "strength_score": strength_score,
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
        winner, loser = norm(row["winner_name"]), norm(row["loser_name"])
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
    rng = np.random.default_rng(seed)
    sets_needed = 3 if best_of_five else 2

    # Convert match-level strength to a set-level probability, then simulate sets.
    p = float(np.clip(probability, .05, .95))
    set_p = float(np.clip(.5 + (p - .5) * .72, .08, .92))

    wins_a = 0
    straight_a = 0
    straight_b = 0
    deciding = 0
    set_score_counts: dict[str, int] = {}

    for _ in range(simulations):
        a_sets = 0
        b_sets = 0
        while a_sets < sets_needed and b_sets < sets_needed:
            if rng.random() < set_p:
                a_sets += 1
            else:
                b_sets += 1

        key = f"{a_sets}-{b_sets}"
        set_score_counts[key] = set_score_counts.get(key, 0) + 1

        if a_sets > b_sets:
            wins_a += 1
            if b_sets == 0:
                straight_a += 1
        else:
            if a_sets == 0:
                straight_b += 1

        if max(a_sets, b_sets) == sets_needed and min(a_sets, b_sets) == sets_needed - 1:
            deciding += 1

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
    """Create a transparent high-level playing-style label."""
    if manual_style and manual_style != "Auto":
        return {
            "label": manual_style,
            "serve_score": float(profile_data["serve_points_won"]),
            "return_score": float(profile_data["return_points_won"]),
            "manual": True,
        }

    serve = float(profile_data["serve_points_won"])
    ret = float(profile_data["return_points_won"])

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

    return {
        "label": label,
        "serve_score": serve,
        "return_score": ret,
        "manual": False,
    }


def style_matchup_adjustment(
    style_a: dict,
    style_b: dict,
    handedness_a: str,
    handedness_b: str,
    surface: str,
) -> tuple[float, str]:
    """Small matchup adjustment; capped because style tags are coarse."""
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

    if handedness_a != handedness_b and "Left" in {handedness_a, handedness_b}:
        # Do not assume the left-hander always benefits; record the asymmetry and
        # apply only a very small adjustment toward the left-handed player.
        impact += 0.004 if handedness_a == "Left" else -0.004
        notes.append("Opposite-handed matchup creates a small left-handed asymmetry")

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
    rows_a = perspective(matches, player_a, event_date)
    rows_b = perspective(matches, player_b, event_date)
    pa = profile(rows_a, surface, event_date)
    pb = profile(rows_b, surface, event_date)

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
    ka, kb = norm(player_a), norm(player_b)
    oa, ob = overall.get(ka, 1500.0), overall.get(kb, 1500.0)
    sa, sb = surface_table.get(ka, 1500.0), surface_table.get(kb, 1500.0)

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

    rank_a = pa["rank"] if not pd.isna(pa["rank"]) else 250
    rank_b = pb["rank"] if not pd.isna(pb["rank"]) else 250
    rank_p = 1 / (1 + math.exp(-((-math.log(max(rank_a, 1))) - (-math.log(max(rank_b, 1)))) * .9))

    base = (
        weights["overall_elo"] * overall_p
        + weights["surface_elo"] * surface_p
        + weights["ranking"] * rank_p
    )

    serve_difference = pa["serve_points_won"] - pb["serve_points_won"]
    return_difference = pa["return_points_won"] - pb["return_points_won"]
    matchup = float(np.clip(
        (
            serve_difference * weights["serve"]
            + return_difference * weights["return"]
        ) * .35,
        -.055, .055
    ))
    form = float(np.clip(
        (pa["recent_win"] - pb["recent_win"]) * .04 * weights["form"],
        -.045, .045
    ))
    opponent_strength = float(np.clip(
        (
            opponent_strength_a["strength_score"]
            - opponent_strength_b["strength_score"]
        ) * 0.055,
        -0.035,
        0.035,
    ))
    surface_adj = float(np.clip(
        (pa["surface_win"] - pb["surface_win"]) * .045,
        -.04, .04
    ))

    travel_penalty = {"None": 0.0, "Moderate": 0.9, "Heavy": 1.8}
    fatigue_score_a = (
        fatigue_profile_a["score"]
        + travel_penalty.get(travel_load_a, 0.0)
        + (1.2 if late_finish_a else 0.0)
    )
    fatigue_score_b = (
        fatigue_profile_b["score"]
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

    factors = [
        ("Context-weighted matchup", matchup,
         f"{surface}, {environment}, {match_format}: serve weight {weights['serve']:.2f}x and "
         f"return weight {weights['return']:.2f}x. Profiles: {player_a} "
         f"{pa['serve_points_won']:.1%}/{pa['return_points_won']:.1%}; "
         f"{player_b} {pb['serve_points_won']:.1%}/{pb['return_points_won']:.1%}."),
        ("Context-weighted recent form", form,
         f"Last-10 win rate: {player_a} {pa['recent_win']:.0%}; {player_b} "
         f"{pb['recent_win']:.0%}. Context multiplier: {weights['form']:.2f}x."),
        ("Opponent strength", opponent_strength,
         f"Recent opposition score: {player_a} "
         f"{opponent_strength_a['strength_score']:.0%} vs {player_b} "
         f"{opponent_strength_b['strength_score']:.0%}. Average opponent Elo: "
         f"{opponent_strength_a['avg_opponent_elo']:.0f} vs "
         f"{opponent_strength_b['avg_opponent_elo']:.0f}."),
        ("Surface", surface_adj,
         f"Two-year {surface} win rate: {player_a} {pa['surface_win']:.0%}; "
         f"{player_b} {pb['surface_win']:.0%}."),
        ("Fatigue 2.0", fatigue,
         f"{player_a}: {fatigue_profile_a['matches_7']} matches, "
         f"{fatigue_profile_a['sets_7']} sets, {fatigue_profile_a['deciders_7']} deciders "
         f"in 7 days, {fatigue_profile_a['rest_days']} rest days, travel {travel_load_a}"
         f"{', late finish' if late_finish_a else ''}. {player_b}: "
         f"{fatigue_profile_b['matches_7']} matches, {fatigue_profile_b['sets_7']} sets, "
         f"{fatigue_profile_b['deciders_7']} deciders, "
         f"{fatigue_profile_b['rest_days']} rest days, travel {travel_load_b}"
         f"{', late finish' if late_finish_b else ''}."),
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

    final_model = float(np.clip(base + sum(v for _, v, _ in factors), .05, .95))
    best_of_five = str(match_format).casefold() == "best of 5"
    simulation = simulate_matches(final_model, simulations, best_of_five)

    sample = min(pa["sample"], pb["sample"])
    quality = int(np.clip(round(3 + min(sample, 50) / 8), 3, 10))
    uncertainty_penalty = (
        int(injury_status_a != "Clear")
        + int(injury_status_b != "Clear")
        + int(style_a == "Auto")
        + int(style_b == "Auto")
    ) * 0.25
    confidence = int(np.clip(
        round(
            5
            + abs(simulation["win_probability"] - .5) * 8
            + (quality - 6) * .3
            - uncertainty_penalty
        ),
        1,
        10,
    ))

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
        "model_probability": final_model,
        "win_probability": simulation["win_probability"],
        "fair_line": american_from_probability(simulation["win_probability"]),
        "confidence": confidence,
        "data_quality": quality,
        "overall_elo": (oa, ob),
        "surface_elo": (sa, sb),
        "profile_a": pa,
        "profile_b": pb,
        "factors": [
            {"name": name, "impact": impact, "reason": reason}
            for name, impact, reason in factors
        ],
        "simulation": simulation,
    }
