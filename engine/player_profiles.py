from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import math
import re
import unicodedata
from typing import Any

import pandas as pd

try:
    from .api_tennis import APITennisClient, APITennisError
except ImportError:
    from api_tennis import APITennisClient, APITennisError


def canonical_player_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text).casefold()
    tokens = [token for token in text.split() if token not in {"jr", "sr", "ii", "iii", "iv"}]
    tokens = [token for token in tokens if len(token) > 1]
    return " ".join(tokens)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass
class PlayerProfile:
    requested_name: str
    historical_name: str | None = None
    api_player_key: str | None = None
    ranking: int | None = None
    ranking_points: int | None = None
    career_matches: int = 0
    career_wins: int = 0
    career_losses: int = 0
    surface_matches: dict[str, int] = field(default_factory=dict)
    surface_wins: dict[str, int] = field(default_factory=dict)
    grand_slam_matches: int = 0
    masters_matches: int = 0
    tour_finals_matches: int = 0
    atp_tour_matches: int = 0
    challenger_matches: int = 0
    top_10_matches: int = 0
    top_25_matches: int = 0
    top_50_matches: int = 0
    top_100_matches: int = 0
    top_10_record: str = "0-0"
    top_25_record: str = "0-0"
    top_50_record: str = "0-0"
    top_100_record: str = "0-0"
    recent_30_matches: int = 0
    recent_90_matches: int = 0
    last_match_date: str | None = None
    data_sources: list[str] = field(default_factory=list)
    data_flags: list[str] = field(default_factory=list)
    api_source: str = "unavailable"
    api_fetched_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_historical_name(matches: pd.DataFrame, requested_name: str) -> str | None:
    if matches is None or matches.empty:
        return None

    names = pd.concat(
        [matches.get("winner_name", pd.Series(dtype=str)), matches.get("loser_name", pd.Series(dtype=str))]
    ).dropna().astype(str)

    requested_key = canonical_player_key(requested_name)
    exact = names[names.map(canonical_player_key).eq(requested_key)]
    if not exact.empty:
        return str(exact.value_counts().index[0])

    return None


def _player_history(matches: pd.DataFrame, player_name: str, event_date: date) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()

    key = canonical_player_key(player_name)
    winners = matches["winner_name"].map(canonical_player_key)
    losers = matches["loser_name"].map(canonical_player_key)
    history = matches[(winners.eq(key) | losers.eq(key))].copy()

    if "tourney_date" in history:
        history["tourney_date"] = pd.to_datetime(history["tourney_date"], errors="coerce")
        history = history[history["tourney_date"] < pd.Timestamp(event_date)]
        history = history.sort_values("tourney_date")
    return history


def _historical_metrics(history: pd.DataFrame, player_name: str, event_date: date) -> dict[str, Any]:
    if history.empty:
        return {}

    key = canonical_player_key(player_name)
    won = history["winner_name"].map(canonical_player_key).eq(key)
    opponent_rank = pd.Series(index=history.index, dtype=float)
    opponent_rank.loc[won] = pd.to_numeric(history.loc[won, "loser_rank"], errors="coerce")
    opponent_rank.loc[~won] = pd.to_numeric(history.loc[~won, "winner_rank"], errors="coerce")

    metrics: dict[str, Any] = {
        "career_matches": int(len(history)),
        "career_wins": int(won.sum()),
        "career_losses": int((~won).sum()),
        "surface_matches": {},
        "surface_wins": {},
    }

    for surface in ("Hard", "Clay", "Grass", "Carpet"):
        mask = history.get("surface", "").astype(str).str.casefold().eq(surface.casefold())
        metrics["surface_matches"][surface] = int(mask.sum())
        metrics["surface_wins"][surface] = int((mask & won).sum())

    levels = history.get("tourney_level", "").astype(str)
    metrics["grand_slam_matches"] = int(levels.eq("G").sum())
    metrics["masters_matches"] = int(levels.eq("M").sum())
    metrics["tour_finals_matches"] = int(levels.eq("F").sum())
    metrics["challenger_matches"] = int(levels.eq("C").sum())
    metrics["atp_tour_matches"] = int(levels.isin(["A", "G", "M", "F", "D"]).sum())

    for threshold in (10, 25, 50, 100):
        mask = opponent_rank.le(threshold)
        wins = int((mask & won).sum())
        losses = int((mask & ~won).sum())
        metrics[f"top_{threshold}_matches"] = wins + losses
        metrics[f"top_{threshold}_record"] = f"{wins}-{losses}"

    event_ts = pd.Timestamp(event_date)
    dates = pd.to_datetime(history["tourney_date"], errors="coerce")
    metrics["recent_30_matches"] = int((dates >= event_ts - pd.Timedelta(days=30)).sum())
    metrics["recent_90_matches"] = int((dates >= event_ts - pd.Timedelta(days=90)).sum())
    valid_dates = dates.dropna()
    metrics["last_match_date"] = (
        valid_dates.max().date().isoformat() if not valid_dates.empty else None
    )
    return metrics


def _ranking_lookup(rows: list[dict[str, Any]], player_name: str) -> dict[str, Any] | None:
    key = canonical_player_key(player_name)
    for row in rows:
        candidate = (
            row.get("player")
            or row.get("player_name")
            or row.get("standing_player")
            or ""
        )
        if canonical_player_key(candidate) == key:
            return row
    return None


def build_player_profile(
    matches: pd.DataFrame,
    player_name: str,
    event_date: date,
    *,
    api_client: APITennisClient | None = None,
    include_api: bool = True,
    use_store: bool = True,
) -> PlayerProfile:
    """Build a stable profile, preferring the daily store when it is safe to use."""
    if use_store:
        try:
            from .player_intelligence_store import get_stored_profile
            stored = get_stored_profile(player_name, event_date=event_date)
            if stored is not None:
                return stored
        except Exception:
            pass
    historical_name = _find_historical_name(matches, player_name)
    profile = PlayerProfile(
        requested_name=player_name,
        historical_name=historical_name,
    )

    if historical_name:
        history = _player_history(matches, historical_name, event_date)
        for field_name, value in _historical_metrics(
            history, historical_name, event_date
        ).items():
            setattr(profile, field_name, value)
        profile.data_sources.append("historical_database")
    else:
        profile.data_flags.append("historical_player_not_found")

    if include_api:
        client = api_client or APITennisClient()
        try:
            response = client.get_standings("ATP")
            ranking_row = _ranking_lookup(response.result, player_name)
            profile.api_source = response.source
            profile.api_fetched_at = response.fetched_at

            if ranking_row:
                profile.ranking = _safe_int(
                    ranking_row.get("place")
                    or ranking_row.get("ranking")
                    or ranking_row.get("position")
                )
                profile.ranking_points = _safe_int(
                    ranking_row.get("points")
                    or ranking_row.get("player_points")
                )
                raw_key = (
                    ranking_row.get("player_key")
                    or ranking_row.get("standing_player_key")
                    or ranking_row.get("id")
                )
                profile.api_player_key = str(raw_key) if raw_key not in (None, "") else None
                profile.data_sources.append("api_tennis_standings")
            else:
                profile.data_flags.append("api_ranking_not_found")
        except APITennisError:
            profile.data_flags.append("api_unavailable")
        except Exception:
            profile.data_flags.append("api_profile_error")

    if profile.career_matches < 10:
        profile.data_flags.append("very_small_historical_sample")
    elif profile.career_matches < 30:
        profile.data_flags.append("small_historical_sample")

    return profile


def experience_reliability(profile: PlayerProfile) -> float:
    """Return a 0–1 reliability score for downstream confidence calculations."""
    match_component = min(profile.career_matches / 120.0, 1.0)
    surface_total = max(profile.surface_matches.values(), default=0)
    surface_component = min(surface_total / 60.0, 1.0)
    ranking_component = 1.0 if profile.ranking is not None else 0.45
    return float(
        max(
            0.0,
            min(
                1.0,
                0.55 * match_component
                + 0.25 * surface_component
                + 0.20 * ranking_component,
            ),
        )
    )


def compare_experience(
    profile_a: PlayerProfile,
    profile_b: PlayerProfile,
    surface: str,
    *,
    maximum_probability_adjustment: float = 0.04,
) -> dict[str, Any]:
    """Compare experience conservatively and cap the probability impact at ±4%."""
    surface_key = str(surface).title()
    a_surface = profile_a.surface_matches.get(surface_key, 0)
    b_surface = profile_b.surface_matches.get(surface_key, 0)

    def score(profile: PlayerProfile, surface_matches: int) -> float:
        return (
            0.32 * math.log1p(profile.career_matches)
            + 0.24 * math.log1p(surface_matches)
            + 0.18 * math.log1p(profile.grand_slam_matches + profile.masters_matches)
            + 0.16 * math.log1p(profile.top_50_matches)
            + 0.10 * experience_reliability(profile)
        )

    score_a = score(profile_a, a_surface)
    score_b = score(profile_b, b_surface)
    difference = score_a - score_b

    adjustment = maximum_probability_adjustment * math.tanh(difference / 2.5)
    adjustment = max(
        -maximum_probability_adjustment,
        min(maximum_probability_adjustment, adjustment),
    )

    if abs(adjustment) < 0.005:
        advantage = "Even"
    elif adjustment > 0:
        advantage = profile_a.requested_name
    else:
        advantage = profile_b.requested_name

    return {
        "advantage": advantage,
        "score_a": score_a,
        "score_b": score_b,
        "probability_adjustment_a": adjustment,
        "reliability_a": experience_reliability(profile_a),
        "reliability_b": experience_reliability(profile_b),
        "surface_matches_a": a_surface,
        "surface_matches_b": b_surface,
        "maximum_adjustment": maximum_probability_adjustment,
    }
