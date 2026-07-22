
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
        "rest_days": max(0, int((event_ts - last_date).days)),
        "advanced_win": float(advanced["won"].mean()) if len(advanced) >= 4 else .5,
        "big_event_win": float(big["won"].mean()) if len(big) >= 5 else .5,
        "deciding_win": float(deciding["won"].mean()) if len(deciding) >= 4 else .5,
        "sample": len(two_year),
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


def analyze(
    matches: pd.DataFrame,
    player_a: str,
    player_b: str,
    tournament: str,
    round_label: str,
    surface: str,
    event_date: date,
    simulations: int = 20000,
) -> dict:
    rows_a = perspective(matches, player_a, event_date)
    rows_b = perspective(matches, player_b, event_date)
    pa = profile(rows_a, surface, event_date)
    pb = profile(rows_b, surface, event_date)

    overall, surface_table = elo_tables(matches, surface, event_date)
    ka, kb = norm(player_a), norm(player_b)
    oa, ob = overall.get(ka, 1500.0), overall.get(kb, 1500.0)
    sa, sb = surface_table.get(ka, 1500.0), surface_table.get(kb, 1500.0)

    overall_p = rating_probability(oa, ob)
    surface_p = rating_probability(sa, sb)

    rank_a = pa["rank"] if not pd.isna(pa["rank"]) else 250
    rank_b = pb["rank"] if not pd.isna(pb["rank"]) else 250
    rank_p = 1 / (1 + math.exp(-((-math.log(max(rank_a, 1))) - (-math.log(max(rank_b, 1)))) * .9))

    base = .48 * overall_p + .37 * surface_p + .15 * rank_p

    matchup = float(np.clip(
        ((pa["serve_points_won"] + pa["return_points_won"])
        - (pb["serve_points_won"] + pb["return_points_won"])) * .35,
        -.045, .045
    ))
    form = float(np.clip((pa["recent_win"] - pb["recent_win"]) * .04, -.04, .04))
    surface_adj = float(np.clip((pa["surface_win"] - pb["surface_win"]) * .045, -.04, .04))

    fatigue_a = pa["matches_7"] * .7 + pa["matches_14"] * .22 - min(pa["rest_days"], 7) * .1
    fatigue_b = pb["matches_7"] * .7 + pb["matches_14"] * .22 - min(pb["rest_days"], 7) * .1
    fatigue = float(np.clip((fatigue_b - fatigue_a) * .006, -.035, .035))

    late_round = ROUND_MAP.get(round_label) in {"QF", "SF", "F"}
    pressure_weight = 1.0 if late_round else .4
    pressure = float(np.clip(
        (
            (pa["advanced_win"] - pb["advanced_win"]) * .035
            + (pa["big_event_win"] - pb["big_event_win"]) * .025
        ) * pressure_weight,
        -.04, .04
    ))
    deciding = float(np.clip((pa["deciding_win"] - pb["deciding_win"]) * .02, -.025, .025))

    factors = [
        ("Matchup", matchup,
         f"Serve/return profile: {player_a} {pa['serve_points_won']:.1%}/{pa['return_points_won']:.1%}; "
         f"{player_b} {pb['serve_points_won']:.1%}/{pb['return_points_won']:.1%}."),
        ("Recent form", form,
         f"Last-10 win rate: {player_a} {pa['recent_win']:.0%}; {player_b} {pb['recent_win']:.0%}."),
        ("Surface", surface_adj,
         f"Two-year {surface} win rate: {player_a} {pa['surface_win']:.0%}; "
         f"{player_b} {pb['surface_win']:.0%}."),
        ("Fitness and fatigue", fatigue,
         f"Last 7/14 days: {player_a} {pa['matches_7']}/{pa['matches_14']} matches with "
         f"{pa['rest_days']} rest days; {player_b} {pb['matches_7']}/{pb['matches_14']} "
         f"with {pb['rest_days']} rest days."),
        ("Event pressure", pressure,
         f"Advanced-round win rate: {player_a} {pa['advanced_win']:.0%}; "
         f"{player_b} {pb['advanced_win']:.0%}."),
        ("Deciding-match history", deciding,
         f"Deciding-match win rate: {player_a} {pa['deciding_win']:.0%}; "
         f"{player_b} {pb['deciding_win']:.0%}."),
    ]

    final_model = float(np.clip(base + sum(v for _, v, _ in factors), .05, .95))
    best_of_five = tournament in {"Australian Open", "Roland Garros", "Wimbledon", "US Open"}
    simulation = simulate_matches(final_model, simulations, best_of_five)

    sample = min(pa["sample"], pb["sample"])
    quality = int(np.clip(round(3 + min(sample, 50) / 8), 3, 10))
    confidence = int(np.clip(round(5 + abs(simulation["win_probability"] - .5) * 8 + (quality - 6) * .3), 1, 10))

    return {
        "player_a": player_a,
        "player_b": player_b,
        "tournament": tournament,
        "round": round_label,
        "surface": surface,
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
