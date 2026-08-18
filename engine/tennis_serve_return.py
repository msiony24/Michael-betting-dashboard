from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


TOUR_SERVE_POINTS_WON = 0.635
TOUR_RETURN_POINTS_WON = 1.0 - TOUR_SERVE_POINTS_WON
MIN_MATCHES = 3
POINT_SHRINKAGE = 350.0
MAX_PROBABILITY_ADJUSTMENT = 0.032


from .tennis_identity import canonical_player_key


def _key(value: Any) -> str:
    return canonical_player_key(value)


def _perspective(matches: pd.DataFrame, player: str, event_date: date | pd.Timestamp) -> pd.DataFrame:
    event_ts = pd.Timestamp(event_date)
    key = _key(player)
    mask = (
        matches["winner_name"].map(_key).eq(key)
        | matches["loser_name"].map(_key).eq(key)
    ) & (matches["tourney_date"] < event_ts)
    subset = matches.loc[mask].sort_values("tourney_date")
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        won = _key(row.get("winner_name")) == key
        side = "w" if won else "l"
        opp = "l" if won else "w"
        rows.append({
            "date": row.get("tourney_date"),
            "surface": row.get("surface"),
            "svpt": row.get(f"{side}_svpt", np.nan),
            "first_in": row.get(f"{side}_1stIn", np.nan),
            "first_won": row.get(f"{side}_1stWon", np.nan),
            "second_won": row.get(f"{side}_2ndWon", np.nan),
            "aces": row.get(f"{side}_ace", np.nan),
            "double_faults": row.get(f"{side}_df", np.nan),
            "service_games": row.get(f"{side}_SvGms", np.nan),
            "bp_saved": row.get(f"{side}_bpSaved", np.nan),
            "bp_faced": row.get(f"{side}_bpFaced", np.nan),
            "opp_svpt": row.get(f"{opp}_svpt", np.nan),
            "opp_first_won": row.get(f"{opp}_1stWon", np.nan),
            "opp_second_won": row.get(f"{opp}_2ndWon", np.nan),
            "opp_service_games": row.get(f"{opp}_SvGms", np.nan),
            "opp_bp_saved": row.get(f"{opp}_bpSaved", np.nan),
            "opp_bp_faced": row.get(f"{opp}_bpFaced", np.nan),
        })
    return pd.DataFrame(rows)


def _ratio(num: float, den: float) -> float | None:
    if den is None or not np.isfinite(den) or den <= 0:
        return None
    if num is None or not np.isfinite(num):
        return None
    return float(num / den)


def _aggregate(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"matches": 0, "service_points": 0, "return_points": 0}

    serve = rows.dropna(subset=["svpt", "first_won", "second_won"])
    ret = rows.dropna(subset=["opp_svpt", "opp_first_won", "opp_second_won"])

    service_points = float(serve["svpt"].sum()) if not serve.empty else 0.0
    service_points_won = float((serve["first_won"] + serve["second_won"]).sum()) if not serve.empty else 0.0
    return_points = float(ret["opp_svpt"].sum()) if not ret.empty else 0.0
    return_points_won = float((ret["opp_svpt"] - ret["opp_first_won"] - ret["opp_second_won"]).sum()) if not ret.empty else 0.0

    first_rows = rows.dropna(subset=["svpt", "first_in", "first_won"])
    second_rows = rows.dropna(subset=["svpt", "first_in", "second_won"])
    first_in = float(first_rows["first_in"].sum()) if not first_rows.empty else 0.0
    first_won = float(first_rows["first_won"].sum()) if not first_rows.empty else 0.0
    second_attempts = float((second_rows["svpt"] - second_rows["first_in"]).sum()) if not second_rows.empty else 0.0
    second_won = float(second_rows["second_won"].sum()) if not second_rows.empty else 0.0

    ace_rows = rows.dropna(subset=["svpt", "aces"])
    df_rows = rows.dropna(subset=["svpt", "double_faults"])
    ace_den = float(ace_rows["svpt"].sum()) if not ace_rows.empty else 0.0
    ace_num = float(ace_rows["aces"].sum()) if not ace_rows.empty else 0.0
    df_den = float(df_rows["svpt"].sum()) if not df_rows.empty else 0.0
    df_num = float(df_rows["double_faults"].sum()) if not df_rows.empty else 0.0

    hold_rows = rows.dropna(subset=["service_games", "bp_faced", "bp_saved"])
    service_games = float(hold_rows["service_games"].sum()) if not hold_rows.empty else 0.0
    breaks_conceded = float((hold_rows["bp_faced"] - hold_rows["bp_saved"]).clip(lower=0).sum()) if not hold_rows.empty else 0.0

    break_rows = rows.dropna(subset=["opp_service_games", "opp_bp_faced", "opp_bp_saved"])
    return_games = float(break_rows["opp_service_games"].sum()) if not break_rows.empty else 0.0
    breaks_won = float((break_rows["opp_bp_faced"] - break_rows["opp_bp_saved"]).clip(lower=0).sum()) if not break_rows.empty else 0.0

    return {
        "matches": int(len(rows)),
        "serve_matches": int(len(serve)),
        "return_matches": int(len(ret)),
        "service_points": int(service_points),
        "return_points": int(return_points),
        "serve_points_won": _ratio(service_points_won, service_points),
        "return_points_won": _ratio(return_points_won, return_points),
        "first_serve_in": _ratio(first_in, float(first_rows["svpt"].sum()) if not first_rows.empty else 0.0),
        "first_serve_points_won": _ratio(first_won, first_in),
        "second_serve_points_won": _ratio(second_won, second_attempts),
        "ace_rate": _ratio(ace_num, ace_den),
        "double_fault_rate": _ratio(df_num, df_den),
        "hold_rate": _ratio(service_games - breaks_conceded, service_games),
        "break_rate": _ratio(breaks_won, return_games),
    }


def _blend_metric(parts: list[tuple[dict[str, Any], float]], metric: str, count_metric: str, baseline: float) -> tuple[float | None, int]:
    values: list[tuple[float, float]] = []
    total_count = 0
    for profile, weight in parts:
        value = profile.get(metric)
        count = int(profile.get(count_metric, 0) or 0)
        if value is None or count <= 0:
            continue
        total_count += count
        values.append((float(value), float(weight)))
    if not values:
        return None, 0
    weight_sum = sum(w for _, w in values)
    raw = sum(v * w for v, w in values) / weight_sum
    shrink = total_count / (total_count + POINT_SHRINKAGE)
    return float(baseline + (raw - baseline) * shrink), total_count


def serve_return_profile(matches: pd.DataFrame, player: str, event_date: date | pd.Timestamp, surface: str) -> dict[str, Any]:
    rows = _perspective(matches, player, event_date)
    event_ts = pd.Timestamp(event_date)
    one_year = rows[rows["date"] >= event_ts - pd.Timedelta(days=365)]
    recent = rows[rows["date"] >= event_ts - pd.Timedelta(days=90)]
    surface_rows = rows[
        (rows["date"] >= event_ts - pd.Timedelta(days=730))
        & rows["surface"].astype(str).str.casefold().eq(str(surface).casefold())
    ]

    season_profile = _aggregate(one_year)
    recent_profile = _aggregate(recent)
    surface_profile = _aggregate(surface_rows)
    parts = [(season_profile, 0.50), (recent_profile, 0.30), (surface_profile, 0.20)]

    spw, service_points = _blend_metric(parts, "serve_points_won", "service_points", TOUR_SERVE_POINTS_WON)
    rpw, return_points = _blend_metric(parts, "return_points_won", "return_points", TOUR_RETURN_POINTS_WON)

    coverage_matches = max(season_profile.get("serve_matches", 0), season_profile.get("return_matches", 0))
    available = spw is not None and rpw is not None and coverage_matches >= MIN_MATCHES

    return {
        "player": player,
        "surface": surface,
        "available": bool(available),
        "serve_points_won": spw,
        "return_points_won": rpw,
        "service_points": service_points,
        "return_points": return_points,
        "matches_with_stats": int(coverage_matches),
        "season_365": season_profile,
        "recent_90": recent_profile,
        "surface_730": surface_profile,
        "coverage_note": (
            "Verified match-level serve/return statistics available."
            if available else
            "Insufficient verified match-level serve/return statistics; no serve/return adjustment applied."
        ),
    }


def serve_return_matchup_adjustment(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> dict[str, Any]:
    if not profile_a.get("available") or not profile_b.get("available"):
        return {
            "available": False,
            "probability_adjustment_a": 0.0,
            "expected_service_point_win_a": None,
            "expected_service_point_win_b": None,
            "reason": "Serve/return engine excluded because verified statistical coverage is insufficient for one or both players.",
        }

    a_spw = float(profile_a["serve_points_won"])
    a_rpw = float(profile_a["return_points_won"])
    b_spw = float(profile_b["serve_points_won"])
    b_rpw = float(profile_b["return_points_won"])

    expected_a = float(np.clip(TOUR_SERVE_POINTS_WON + (a_spw - TOUR_SERVE_POINTS_WON) - (b_rpw - TOUR_RETURN_POINTS_WON), 0.50, 0.78))
    expected_b = float(np.clip(TOUR_SERVE_POINTS_WON + (b_spw - TOUR_SERVE_POINTS_WON) - (a_rpw - TOUR_RETURN_POINTS_WON), 0.50, 0.78))
    service_edge = expected_a - expected_b

    sample = min(int(profile_a.get("matches_with_stats", 0)), int(profile_b.get("matches_with_stats", 0)))
    sample_scale = float(np.clip(sample / 12.0, 0.35, 1.0))
    adjustment = float(np.clip(service_edge * 0.42 * sample_scale, -MAX_PROBABILITY_ADJUSTMENT, MAX_PROBABILITY_ADJUSTMENT))

    return {
        "available": True,
        "probability_adjustment_a": adjustment,
        "expected_service_point_win_a": expected_a,
        "expected_service_point_win_b": expected_b,
        "sample_scale": sample_scale,
        "reason": (
            f"Matchup-adjusted service-point expectation {expected_a:.1%} vs {expected_b:.1%}; "
            f"sample reliability {sample_scale:.0%}. Adjustment capped at ±{MAX_PROBABILITY_ADJUSTMENT:.1%}."
        ),
    }
