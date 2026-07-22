
import math
import re
from datetime import date, timedelta
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st


DATA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
MATCH_YEARS = list(range(max(2019, date.today().year - 5), date.today().year + 1))
ROUND_ORDER = {
    "RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7
}
ROUND_LABELS = {
    "Qualifying": "Q", "Round Robin": "RR", "R128": "R128", "R64": "R64",
    "R32": "R32", "R16": "R16", "Quarterfinal": "QF",
    "Semifinal": "SF", "Final": "F"
}
LEVEL_LABELS = {
    "G": "Grand Slam", "M": "Masters 1000", "A": "ATP Tour",
    "F": "Tour Finals", "D": "Davis Cup", "C": "Challenger"
}


def _download_csv(url):
    request = Request(url, headers={"User-Agent": "Macabets/1.0"})
    with urlopen(request, timeout=20) as response:
        return pd.read_csv(StringIO(response.read().decode("utf-8", errors="replace")))


@st.cache_data(ttl=21600, show_spinner=False)
def load_tennis_data():
    frames = []
    errors = []
    for year in MATCH_YEARS:
        url = f"{DATA_BASE}/atp_matches_{year}.csv"
        try:
            frame = _download_csv(url)
            frame["source_year"] = year
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{year}: {exc}")

    if not frames:
        raise RuntimeError("Macabets could not download the ATP match database.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches = matches.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    matches["surface"] = matches.get("surface", "").fillna("Unknown")
    matches["tourney_name"] = matches.get("tourney_name", "").fillna("Unknown")
    matches["tourney_level"] = matches.get("tourney_level", "").fillna("")
    matches["round"] = matches.get("round", "").fillna("")
    matches["score"] = matches.get("score", "").fillna("")

    numeric_columns = [
        "winner_id", "loser_id", "winner_rank", "loser_rank", "winner_age", "loser_age",
        "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt", "w_1stIn", "l_1stIn",
        "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon", "w_SvGms", "l_SvGms",
        "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced"
    ]
    for col in numeric_columns:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    return matches, errors


def american_from_probability(probability):
    probability = min(max(float(probability), 0.001), 0.999)
    if probability >= 0.5:
        return int(round(-100 * probability / (1 - probability)))
    return int(round(100 * (1 - probability) / probability))


def implied_probability(odds):
    if odds is None or odds == 0:
        return None
    odds = int(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def normalize_name(name):
    return re.sub(r"\s+", " ", str(name).strip()).casefold()


def player_options(matches):
    cutoff = matches["tourney_date"].max() - pd.Timedelta(days=730)
    recent = matches[matches["tourney_date"] >= cutoff]
    names = pd.concat([recent["winner_name"], recent["loser_name"]]).dropna()
    counts = names.value_counts()
    return counts.index.tolist()


def tournament_catalog(matches):
    recent = matches.sort_values("tourney_date").drop_duplicates("tourney_name", keep="last")
    recent = recent[recent["tourney_level"].isin(["G", "M", "A", "F", "D"])]
    recent = recent.sort_values(["tourney_level", "tourney_name"])
    catalog = {}
    for _, row in recent.iterrows():
        catalog[row["tourney_name"]] = {
            "surface": row["surface"],
            "level_code": row["tourney_level"],
            "level": LEVEL_LABELS.get(row["tourney_level"], "ATP Tour"),
        }
    return catalog


def _player_matches(matches, player, before_date=None):
    key = normalize_name(player)
    mask = (
        matches["winner_name"].map(normalize_name).eq(key)
        | matches["loser_name"].map(normalize_name).eq(key)
    )
    result = matches[mask].copy()
    if before_date is not None:
        result = result[result["tourney_date"] < pd.Timestamp(before_date)]
    return result.sort_values("tourney_date")


def _perspective_rows(matches, player, before_date=None):
    pm = _player_matches(matches, player, before_date)
    key = normalize_name(player)
    rows = []
    for _, m in pm.iterrows():
        won = normalize_name(m["winner_name"]) == key
        prefix = "w" if won else "l"
        opp_prefix = "l" if won else "w"
        rows.append({
            "date": m["tourney_date"],
            "won": bool(won),
            "surface": m["surface"],
            "level": m["tourney_level"],
            "round": m["round"],
            "score": m["score"],
            "opponent": m[f"{'loser' if won else 'winner'}_name"],
            "player_rank": m.get(f"{'winner' if won else 'loser'}_rank", np.nan),
            "opponent_rank": m.get(f"{'loser' if won else 'winner'}_rank", np.nan),
            "player_age": m.get(f"{'winner' if won else 'loser'}_age", np.nan),
            "ace": m.get(f"{prefix}_ace", np.nan),
            "df": m.get(f"{prefix}_df", np.nan),
            "svpt": m.get(f"{prefix}_svpt", np.nan),
            "first_in": m.get(f"{prefix}_1stIn", np.nan),
            "first_won": m.get(f"{prefix}_1stWon", np.nan),
            "second_won": m.get(f"{prefix}_2ndWon", np.nan),
            "service_games": m.get(f"{prefix}_SvGms", np.nan),
            "bp_saved": m.get(f"{prefix}_bpSaved", np.nan),
            "bp_faced": m.get(f"{prefix}_bpFaced", np.nan),
            "opp_svpt": m.get(f"{opp_prefix}_svpt", np.nan),
            "opp_first_in": m.get(f"{opp_prefix}_1stIn", np.nan),
            "opp_first_won": m.get(f"{opp_prefix}_1stWon", np.nan),
            "opp_second_won": m.get(f"{opp_prefix}_2ndWon", np.nan),
        })
    return pd.DataFrame(rows)


def _safe_rate(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _stat_profile(rows, surface, event_date):
    if rows.empty:
        return {
            "matches": 0, "overall_win": 0.5, "surface_win": 0.5, "recent_win": 0.5,
            "serve_points_won": 0.62, "return_points_won": 0.38, "rank": np.nan,
            "age": np.nan, "matches_7": 0, "matches_14": 0, "rest_days": 30,
            "advanced_round_win": 0.5, "big_event_win": 0.5, "deciding_win": 0.5,
            "retirements_180": 0, "data_matches": 0
        }

    event_ts = pd.Timestamp(event_date)
    last_365 = rows[rows["date"] >= event_ts - pd.Timedelta(days=365)]
    last_730 = rows[rows["date"] >= event_ts - pd.Timedelta(days=730)]
    surface_rows = last_730[last_730["surface"].str.casefold() == str(surface).casefold()]
    recent = rows.tail(10)

    serve_num = (last_365["first_won"].fillna(0) + last_365["second_won"].fillna(0)).sum()
    serve_den = last_365["svpt"].sum()
    return_num = (
        last_365["opp_svpt"].fillna(0)
        - last_365["opp_first_won"].fillna(0)
        - last_365["opp_second_won"].fillna(0)
    ).sum()
    return_den = last_365["opp_svpt"].sum()

    advanced = last_730[last_730["round"].isin(["QF", "SF", "F"])]
    big_event = last_730[last_730["level"].isin(["G", "M", "F"])]
    deciding = last_730[last_730["score"].str.count(r"\d+-\d+") >= 3]

    last_date = rows["date"].max()
    recent_rank = rows["player_rank"].dropna()
    recent_age = rows["player_age"].dropna()

    return {
        "matches": len(last_365),
        "overall_win": float(last_365["won"].mean()) if len(last_365) else 0.5,
        "surface_win": float(surface_rows["won"].mean()) if len(surface_rows) else 0.5,
        "recent_win": float(recent["won"].mean()) if len(recent) else 0.5,
        "serve_points_won": _safe_rate(serve_num, serve_den) if serve_den else 0.62,
        "return_points_won": _safe_rate(return_num, return_den) if return_den else 0.38,
        "rank": float(recent_rank.iloc[-1]) if len(recent_rank) else np.nan,
        "age": float(recent_age.iloc[-1]) if len(recent_age) else np.nan,
        "matches_7": int((rows["date"] >= event_ts - pd.Timedelta(days=7)).sum()),
        "matches_14": int((rows["date"] >= event_ts - pd.Timedelta(days=14)).sum()),
        "rest_days": max(0, int((event_ts - last_date).days)) if pd.notna(last_date) else 30,
        "advanced_round_win": float(advanced["won"].mean()) if len(advanced) >= 4 else 0.5,
        "big_event_win": float(big_event["won"].mean()) if len(big_event) >= 5 else 0.5,
        "deciding_win": float(deciding["won"].mean()) if len(deciding) >= 4 else 0.5,
        "retirements_180": int(
            rows[
                (rows["date"] >= event_ts - pd.Timedelta(days=180))
                & rows["score"].str.contains("RET|W/O", case=False, regex=True)
            ].shape[0]
        ),
        "data_matches": len(last_730)
    }


def _elo_ratings(matches, surface, event_date):
    history = matches[matches["tourney_date"] < pd.Timestamp(event_date)].sort_values("tourney_date")
    history = history[history["tourney_date"] >= pd.Timestamp(event_date) - pd.Timedelta(days=1460)]
    overall = {}
    surface_ratings = {}
    k = 24.0

    def update(table, winner, loser):
        rw = table.get(winner, 1500.0)
        rl = table.get(loser, 1500.0)
        expected = 1 / (1 + 10 ** ((rl - rw) / 400))
        table[winner] = rw + k * (1 - expected)
        table[loser] = rl - k * (1 - expected)

    for _, row in history.iterrows():
        winner = normalize_name(row["winner_name"])
        loser = normalize_name(row["loser_name"])
        update(overall, winner, loser)
        if str(row["surface"]).casefold() == str(surface).casefold():
            update(surface_ratings, winner, loser)

    return overall, surface_ratings


def _prob_from_rating(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _clip_adjustment(value, limit=0.06):
    return float(np.clip(value, -limit, limit))


def analyze_matchup(matches, player_a, player_b, tournament, round_label, surface,
                    form_a, form_b, event_date):
    rows_a = _perspective_rows(matches, player_a, event_date)
    rows_b = _perspective_rows(matches, player_b, event_date)
    pa = _stat_profile(rows_a, surface, event_date)
    pb = _stat_profile(rows_b, surface, event_date)

    overall_elo, surface_elo = _elo_ratings(matches, surface, event_date)
    ka, kb = normalize_name(player_a), normalize_name(player_b)
    oa, ob = overall_elo.get(ka, 1500.0), overall_elo.get(kb, 1500.0)
    sa, sb = surface_elo.get(ka, 1500.0), surface_elo.get(kb, 1500.0)

    overall_prob = _prob_from_rating(oa, ob)
    surface_prob = _prob_from_rating(sa, sb)

    rank_a = pa["rank"] if not pd.isna(pa["rank"]) else 200
    rank_b = pb["rank"] if not pd.isna(pb["rank"]) else 200
    rank_strength_a = -math.log(max(rank_a, 1))
    rank_strength_b = -math.log(max(rank_b, 1))
    rank_prob = 1 / (1 + math.exp(-(rank_strength_a - rank_strength_b) * 0.9))

    base_probability = (
        0.42 * overall_prob
        + 0.33 * surface_prob
        + 0.15 * rank_prob
        + 0.10 * 0.5
    )

    matchup_adj = _clip_adjustment(
        ((pa["serve_points_won"] + pa["return_points_won"])
         - (pb["serve_points_won"] + pb["return_points_won"])) * 0.35,
        0.045
    )
    recent_data_adj = _clip_adjustment((pa["recent_win"] - pb["recent_win"]) * 0.035, 0.035)
    user_form_adj = _clip_adjustment(((form_a - form_b) / 10) * 0.035, 0.035)
    form_adj = recent_data_adj + user_form_adj

    surface_adj = _clip_adjustment((pa["surface_win"] - pb["surface_win"]) * 0.045, 0.04)

    fatigue_score_a = pa["matches_7"] * 0.7 + pa["matches_14"] * 0.25 - min(pa["rest_days"], 7) * 0.12
    fatigue_score_b = pb["matches_7"] * 0.7 + pb["matches_14"] * 0.25 - min(pb["rest_days"], 7) * 0.12
    fitness_adj = _clip_adjustment((fatigue_score_b - fatigue_score_a) * 0.006, 0.035)
    fitness_adj += _clip_adjustment((pb["retirements_180"] - pa["retirements_180"]) * 0.008, 0.02)

    round_code = ROUND_LABELS.get(round_label, round_label)
    pressure_weight = 1.0 if round_code in ["QF", "SF", "F"] else 0.45
    pressure_adj = _clip_adjustment(
        ((pa["advanced_round_win"] - pb["advanced_round_win"]) * 0.035
         + (pa["big_event_win"] - pb["big_event_win"]) * 0.025) * pressure_weight,
        0.04
    )

    psychological_adj = _clip_adjustment(
        (pa["deciding_win"] - pb["deciding_win"]) * 0.025
        + (pa["recent_win"] - pb["recent_win"]) * 0.012,
        0.03
    )

    h2h = matches[
        (
            matches["winner_name"].map(normalize_name).eq(ka)
            & matches["loser_name"].map(normalize_name).eq(kb)
        )
        | (
            matches["winner_name"].map(normalize_name).eq(kb)
            & matches["loser_name"].map(normalize_name).eq(ka)
        )
    ]
    h2h_a_wins = int(h2h["winner_name"].map(normalize_name).eq(ka).sum())
    h2h_total = len(h2h)
    h2h_adj = 0.0
    if h2h_total >= 3:
        h2h_adj = _clip_adjustment(((h2h_a_wins / h2h_total) - 0.5) * 0.025, 0.02)
        matchup_adj += h2h_adj

    adjustments = [
        {
            "factor": "Matchup",
            "adjustment": matchup_adj,
            "explanation": (
                f"{player_a}: {pa['serve_points_won']:.1%} serve points won and "
                f"{pa['return_points_won']:.1%} return points won; "
                f"{player_b}: {pb['serve_points_won']:.1%} and {pb['return_points_won']:.1%}. "
                f"Head-to-head: {h2h_a_wins}-{h2h_total-h2h_a_wins} for {player_a}."
            )
        },
        {
            "factor": "Recent Form",
            "adjustment": form_adj,
            "explanation": (
                f"Database last-10 win rates: {player_a} {pa['recent_win']:.0%}, "
                f"{player_b} {pb['recent_win']:.0%}. Your form ratings: "
                f"{form_a}/10 and {form_b}/10."
            )
        },
        {
            "factor": "Surface & Conditions",
            "adjustment": surface_adj,
            "explanation": (
                f"Two-year {surface} records: {player_a} {pa['surface_win']:.0%}, "
                f"{player_b} {pb['surface_win']:.0%}. Surface Elo: "
                f"{sa:.0f} vs {sb:.0f}."
            )
        },
        {
            "factor": "Fitness & Fatigue",
            "adjustment": fitness_adj,
            "explanation": (
                f"{player_a}: {pa['matches_7']} matches in 7 days, {pa['matches_14']} in 14 days, "
                f"{pa['rest_days']} rest days. {player_b}: {pb['matches_7']}, "
                f"{pb['matches_14']}, and {pb['rest_days']}."
            )
        },
        {
            "factor": "Event & Pressure",
            "adjustment": pressure_adj,
            "explanation": (
                f"Advanced-round win rates: {player_a} {pa['advanced_round_win']:.0%}, "
                f"{player_b} {pb['advanced_round_win']:.0%}. Big-event win rates: "
                f"{pa['big_event_win']:.0%} vs {pb['big_event_win']:.0%}. "
                f"Pressure weighting reflects the selected {round_label}."
            )
        },
        {
            "factor": "Psychological Proxy",
            "adjustment": psychological_adj,
            "explanation": (
                f"Estimated from deciding-match results and recent form—not private mental-state information. "
                f"Deciding-match win rates: {player_a} {pa['deciding_win']:.0%}, "
                f"{player_b} {pb['deciding_win']:.0%}."
            )
        },
    ]

    total_adjustment = sum(x["adjustment"] for x in adjustments)
    final_probability = float(np.clip(base_probability + total_adjustment, 0.05, 0.95))

    sample_size = min(pa["data_matches"], pb["data_matches"])
    data_quality = int(np.clip(round(3 + min(sample_size, 50) / 8), 3, 10))
    if pa["matches"] < 8 or pb["matches"] < 8:
        data_quality = min(data_quality, 5)

    confidence = int(np.clip(round(
        5 + abs(final_probability - 0.5) * 8 + (data_quality - 6) * 0.35
    ), 1, 10))

    return {
        "player_a": player_a,
        "player_b": player_b,
        "tournament": tournament,
        "round": round_label,
        "surface": surface,
        "base_probability": base_probability,
        "final_probability": final_probability,
        "fair_line": american_from_probability(final_probability),
        "total_adjustment": total_adjustment,
        "confidence": confidence,
        "data_quality": data_quality,
        "overall_elo_a": oa,
        "overall_elo_b": ob,
        "surface_elo_a": sa,
        "surface_elo_b": sb,
        "profile_a": pa,
        "profile_b": pb,
        "adjustments": adjustments,
    }
