"""Automatic NFL weather lookup and conservative model adjustments.

Weather is contextual, not a primary driver. Normal conditions create little or
no movement. Severe wind/precipitation and climate-mismatch situations can move
the total, confidence, and (slightly) the side.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import time
from typing import Any

import pandas as pd
import requests

from engine.nfl_fetch import TEAM_ABBR_TO_NAME

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl" / "schedules.csv"
WEATHER_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl" / "weather_cache.json"
MAX_FORECAST_DAYS = 16

# Home-stadium coordinates. Neutral-site games prefer the stadium information in
# the nflverse schedule; known international sites are included below.
TEAM_STADIUMS = {
    "Arizona Cardinals": (33.5276, -112.2626, "State Farm Stadium", "retractable", "hot"),
    "Atlanta Falcons": (33.7554, -84.4008, "Mercedes-Benz Stadium", "retractable", "indoor"),
    "Baltimore Ravens": (39.2780, -76.6227, "M&T Bank Stadium", "outdoor", "temperate"),
    "Buffalo Bills": (42.7738, -78.7868, "Highmark Stadium", "outdoor", "cold"),
    "Carolina Panthers": (35.2258, -80.8528, "Bank of America Stadium", "outdoor", "temperate"),
    "Chicago Bears": (41.8623, -87.6167, "Soldier Field", "outdoor", "cold"),
    "Cincinnati Bengals": (39.0955, -84.5161, "Paycor Stadium", "outdoor", "cold"),
    "Cleveland Browns": (41.5061, -81.6995, "Huntington Bank Field", "outdoor", "cold"),
    "Dallas Cowboys": (32.7473, -97.0945, "AT&T Stadium", "retractable", "indoor"),
    "Denver Broncos": (39.7439, -105.0201, "Empower Field at Mile High", "outdoor", "cold"),
    "Detroit Lions": (42.3400, -83.0456, "Ford Field", "dome", "indoor"),
    "Green Bay Packers": (44.5013, -88.0622, "Lambeau Field", "outdoor", "cold"),
    "Houston Texans": (29.6847, -95.4107, "NRG Stadium", "retractable", "indoor"),
    "Indianapolis Colts": (39.7601, -86.1639, "Lucas Oil Stadium", "retractable", "indoor"),
    "Jacksonville Jaguars": (30.3239, -81.6373, "EverBank Stadium", "outdoor", "hot"),
    "Kansas City Chiefs": (39.0489, -94.4839, "GEHA Field at Arrowhead Stadium", "outdoor", "cold"),
    "Las Vegas Raiders": (36.0908, -115.1830, "Allegiant Stadium", "dome", "indoor"),
    "Los Angeles Chargers": (33.9535, -118.3392, "SoFi Stadium", "covered", "temperate"),
    "Los Angeles Rams": (33.9535, -118.3392, "SoFi Stadium", "covered", "temperate"),
    "Miami Dolphins": (25.9580, -80.2389, "Hard Rock Stadium", "outdoor", "hot"),
    "Minnesota Vikings": (44.9738, -93.2577, "U.S. Bank Stadium", "dome", "indoor"),
    "New England Patriots": (42.0909, -71.2643, "Gillette Stadium", "outdoor", "cold"),
    "New Orleans Saints": (29.9511, -90.0812, "Caesars Superdome", "dome", "indoor"),
    "New York Giants": (40.8135, -74.0745, "MetLife Stadium", "outdoor", "cold"),
    "New York Jets": (40.8135, -74.0745, "MetLife Stadium", "outdoor", "cold"),
    "Philadelphia Eagles": (39.9008, -75.1675, "Lincoln Financial Field", "outdoor", "cold"),
    "Pittsburgh Steelers": (40.4468, -80.0158, "Acrisure Stadium", "outdoor", "cold"),
    "San Francisco 49ers": (37.4030, -121.9700, "Levi's Stadium", "outdoor", "temperate"),
    "Seattle Seahawks": (47.5952, -122.3316, "Lumen Field", "outdoor", "temperate"),
    "Tampa Bay Buccaneers": (27.9759, -82.5033, "Raymond James Stadium", "outdoor", "hot"),
    "Tennessee Titans": (36.1665, -86.7713, "Nissan Stadium", "outdoor", "temperate"),
    "Washington Commanders": (38.9076, -76.8645, "Northwest Stadium", "outdoor", "temperate"),
}

KNOWN_NEUTRAL_STADIUMS = {
    "Melbourne Cricket Ground": (-37.8199, 144.9834, "outdoor"),
    "Wembley Stadium": (51.5560, -0.2796, "outdoor"),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664, "covered"),
    "Deutsche Bank Park": (50.0686, 8.6455, "outdoor"),
    "Estadio Santiago Bernabeu": (40.4531, -3.6883, "retractable"),
}

INDOOR_ROOFS = {"dome", "closed"}


@dataclass(frozen=True)
class WeatherContext:
    available: bool
    source: str
    stadium: str
    venue_type: str
    kickoff_local: str | None
    temperature_f: float | None
    humidity_pct: float | None
    precipitation_in: float | None
    snowfall_in: float | None
    wind_mph: float | None
    gust_mph: float | None
    label: str
    impact: str
    summary: str
    home_margin_adjustment: float
    total_adjustment: float
    confidence_penalty: float
    climate_mismatch: str
    fetched_at_utc: str | None




def _parse_game_day(value: date | str) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _weather_cache_key(*, lat: float, lon: float, day: str, gametime: str) -> str:
    return f"{lat:.4f}|{lon:.4f}|{day}|{gametime}"


def _load_weather_cache(path: Path = WEATHER_CACHE_PATH) -> dict:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_weather_cache(cache: dict, path: Path = WEATHER_CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
        tmp.replace(path)
    except Exception:
        # Cache writes should never be allowed to break an analysis.
        pass


def _cache_ttl_seconds(game_day: date | None) -> int:
    if game_day is None:
        return 3 * 60 * 60
    days_until = (game_day - date.today()).days
    if days_until <= 2:
        return 60 * 60
    if days_until <= 7:
        return 3 * 60 * 60
    return 6 * 60 * 60


def _cached_weather(cache: dict, key: str, *, max_age_seconds: int) -> dict | None:
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    saved_at = float(entry.get("saved_at_epoch", 0) or 0)
    result = entry.get("result")
    if not isinstance(result, dict) or saved_at <= 0:
        return None
    if time.time() - saved_at > max_age_seconds:
        return None
    cached = dict(result)
    cached["source"] = str(cached.get("source") or "Open-Meteo") + " (cached)"
    return cached


def _store_weather_cache(cache: dict, key: str, result: dict) -> None:
    cache[key] = {"saved_at_epoch": time.time(), "result": result}
    # Prevent unbounded cache growth. Keep the newest 100 entries.
    if len(cache) > 100:
        ordered = sorted(
            cache.items(),
            key=lambda item: float((item[1] or {}).get("saved_at_epoch", 0) or 0),
            reverse=True,
        )[:100]
        cache.clear()
        cache.update(dict(ordered))
    _save_weather_cache(cache)


def _clean_roof(value: Any) -> str:
    roof = str(value or "").strip().lower()
    if roof in {"dome", "closed"}:
        return "dome"
    if roof in {"outdoors", "outdoor", "open"}:
        return "outdoor"
    if roof in {"retractable", "retractable roof"}:
        return "retractable"
    return roof or "outdoor"


def _team_abbr(team_name: str) -> str | None:
    for abbr, name in TEAM_ABBR_TO_NAME.items():
        if name == team_name:
            return abbr
    return None


def find_scheduled_game(away_team: str, home_team: str, game_date: date | str, path: Path = SCHEDULE_PATH) -> dict:
    """Find a matching nflverse schedule row, if one exists."""
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    if frame.empty or not {"away_team", "home_team", "gameday"}.issubset(frame.columns):
        return {}
    away_abbr, home_abbr = _team_abbr(away_team), _team_abbr(home_team)
    if not away_abbr or not home_abbr:
        return {}
    day = str(game_date)
    match = frame[
        (frame["away_team"].astype(str) == away_abbr)
        & (frame["home_team"].astype(str) == home_abbr)
        & (frame["gameday"].astype(str) == day)
    ]
    if match.empty:
        return {}
    row = match.iloc[0].to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def _venue_details(home_team: str, schedule_row: dict) -> tuple[float, float, str, str]:
    stadium = str(schedule_row.get("stadium") or "").strip()
    roof = _clean_roof(schedule_row.get("roof"))
    if stadium in KNOWN_NEUTRAL_STADIUMS:
        lat, lon, neutral_roof = KNOWN_NEUTRAL_STADIUMS[stadium]
        return lat, lon, stadium, _clean_roof(roof or neutral_roof)
    lat, lon, default_stadium, default_roof, _ = TEAM_STADIUMS[home_team]
    return lat, lon, stadium or default_stadium, _clean_roof(roof or default_roof)


def _kickoff_parts(game_date: date | str, schedule_row: dict) -> tuple[str, str]:
    day = str(schedule_row.get("gameday") or game_date)
    gametime = str(schedule_row.get("gametime") or "13:00")
    if len(gametime) == 5 and ":" in gametime:
        return day, gametime
    return day, "13:00"


def _nearest_hour_index(times: list[str], target_iso: str) -> int | None:
    if not times:
        return None
    try:
        target = datetime.fromisoformat(target_iso)
        parsed = [datetime.fromisoformat(value) for value in times]
        return min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).total_seconds()))
    except Exception:
        return None


def _impact_from_conditions(*, temp: float, humidity: float, precip: float, snow: float, wind: float, gust: float) -> tuple[str, str, float, float]:
    """Return label, impact, total adjustment, confidence penalty."""
    total = 0.0
    confidence = 0.0
    impacts: list[str] = []

    effective_wind = max(wind, gust * 0.75)
    if effective_wind >= 25:
        total -= 4.0
        confidence += 2.5
        impacts.append("severe wind")
    elif effective_wind >= 20:
        total -= 2.5
        confidence += 1.5
        impacts.append("strong wind")
    elif effective_wind >= 15:
        total -= 1.0
        confidence += 0.5
        impacts.append("moderate wind")

    if snow >= 0.08:
        total -= 1.5
        confidence += 1.0
        impacts.append("snow")
    elif precip >= 0.12:
        total -= 1.0
        confidence += 0.75
        impacts.append("heavy precipitation")
    elif precip >= 0.04:
        total -= 0.5
        confidence += 0.25
        impacts.append("light precipitation")

    if temp <= 15:
        total -= 1.0
        confidence += 0.5
        impacts.append("extreme cold")
    elif temp >= 95 and humidity >= 55:
        confidence += 0.75
        impacts.append("extreme heat/humidity")

    total = max(-6.0, min(1.0, total))
    confidence = max(0.0, min(5.0, confidence))
    if confidence >= 2.5 or total <= -4:
        impact = "High"
    elif confidence >= 0.75 or total <= -1.5:
        impact = "Moderate"
    elif impacts:
        impact = "Low"
    else:
        impact = "None"

    if "severe wind" in impacts:
        label = "High wind"
    elif "snow" in impacts:
        label = "Snow"
    elif "heavy precipitation" in impacts or "light precipitation" in impacts:
        label = "Rain"
    elif "extreme heat/humidity" in impacts:
        label = "Extreme heat"
    elif "extreme cold" in impacts:
        label = "Extreme cold"
    else:
        label = "Normal"
    return label, impact, round(total, 2), round(confidence, 2)


def _climate_mismatch(away_team: str, home_team: str, temp: float, humidity: float) -> tuple[float, str]:
    away_profile = TEAM_STADIUMS.get(away_team, (None, None, None, None, "temperate"))[4]
    home_profile = TEAM_STADIUMS.get(home_team, (None, None, None, None, "temperate"))[4]
    # Only move the side when the environment is genuinely unusual for the visitor.
    if temp >= 90 and humidity >= 60 and away_profile in {"cold", "indoor"} and home_profile == "hot":
        return 0.5, f"Heat/humidity familiarity favors {home_team}."
    if temp >= 96 and humidity >= 55 and away_profile != "hot" and home_profile == "hot":
        return 0.35, f"Extreme heat slightly favors the home environment of {home_team}."
    if temp <= 28 and away_profile in {"hot", "indoor"} and home_profile == "cold":
        return 0.5, f"Cold-weather familiarity favors {home_team}."
    if temp <= 18 and away_profile == "hot" and home_profile in {"cold", "temperate"}:
        return 0.35, f"Extreme cold creates a small familiarity edge for {home_team}."
    return 0.0, "No meaningful climate-familiarity mismatch."


def get_nfl_weather(
    *,
    away_team: str,
    home_team: str,
    game_date: date | str,
    timeout: int = 8,
    session=None,
) -> dict:
    """Return automatic kickoff weather and conservative model adjustments.

    The lookup is intentionally conservative: games outside Open-Meteo's
    supported forecast horizon do not trigger an API request, successful
    forecasts are cached across Streamlit reruns, and a recent cached forecast
    can be reused if the provider is temporarily rate-limited or unavailable.
    """
    schedule_row = find_scheduled_game(away_team, home_team, game_date)
    lat, lon, stadium, venue_type = _venue_details(home_team, schedule_row)
    day, gametime = _kickoff_parts(game_date, schedule_row)
    kickoff_local = f"{day}T{gametime}"

    if venue_type == "dome":
        return asdict(WeatherContext(
            available=True, source="venue", stadium=stadium, venue_type="dome",
            kickoff_local=kickoff_local, temperature_f=None, humidity_pct=None,
            precipitation_in=None, snowfall_in=None, wind_mph=None, gust_mph=None,
            label="Indoor", impact="None", summary="Indoor venue — weather excluded from the model.",
            home_margin_adjustment=0.0, total_adjustment=0.0, confidence_penalty=0.0,
            climate_mismatch="No outdoor weather exposure.", fetched_at_utc=None,
        ))

    game_day = _parse_game_day(day)
    today = date.today()
    if game_day is not None and session is None:
        days_until = (game_day - today).days
        if days_until < 0:
            return asdict(WeatherContext(
                available=False, source="Open-Meteo", stadium=stadium, venue_type=venue_type,
                kickoff_local=kickoff_local, temperature_f=None, humidity_pct=None,
                precipitation_in=None, snowfall_in=None, wind_mph=None, gust_mph=None,
                label="Unavailable", impact="None",
                summary="This game date is in the past, so no live forecast was requested. No weather adjustment applied.",
                home_margin_adjustment=0.0, total_adjustment=0.0, confidence_penalty=0.0,
                climate_mismatch="Not evaluated.", fetched_at_utc=None,
            ))
        if days_until >= MAX_FORECAST_DAYS:
            return asdict(WeatherContext(
                available=False, source="Open-Meteo", stadium=stadium, venue_type=venue_type,
                kickoff_local=kickoff_local, temperature_f=None, humidity_pct=None,
                precipitation_in=None, snowfall_in=None, wind_mph=None, gust_mph=None,
                label="Not yet available", impact="None",
                summary=(
                    f"Kickoff is {days_until} days away. Weather is not requested until the game is "
                    f"inside the {MAX_FORECAST_DAYS}-day forecast window. No weather adjustment applied."
                ),
                home_margin_adjustment=0.0, total_adjustment=0.0, confidence_penalty=0.0,
                climate_mismatch="Not evaluated yet.", fetched_at_utc=None,
            ))

    cache = _load_weather_cache() if session is None else {}
    cache_key = _weather_cache_key(lat=lat, lon=lon, day=day, gametime=gametime)
    if session is None:
        fresh_cached = _cached_weather(cache, cache_key, max_age_seconds=_cache_ttl_seconds(game_day))
        if fresh_cached is not None:
            return fresh_cached

    requester = session or requests
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "start_date": day,
        "end_date": day,
    }
    headers = {"User-Agent": "Macabets/1.0 weather-context"}
    try:
        response = requester.get(OPEN_METEO_URL, params=params, timeout=timeout, headers=headers)
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly") or {}
        index = _nearest_hour_index(hourly.get("time") or [], kickoff_local)
        if index is None:
            raise ValueError("Kickoff is outside the available hourly forecast window.")
        temp = float(hourly["temperature_2m"][index])
        humidity = float(hourly["relative_humidity_2m"][index])
        precip = float(hourly["precipitation"][index] or 0.0)
        snow = float(hourly["snowfall"][index] or 0.0)
        wind = float(hourly["wind_speed_10m"][index] or 0.0)
        gust = float(hourly["wind_gusts_10m"][index] or 0.0)
        label, impact, total_adjustment, confidence_penalty = _impact_from_conditions(
            temp=temp, humidity=humidity, precip=precip, snow=snow, wind=wind, gust=gust,
        )
        side_adjustment, mismatch = _climate_mismatch(away_team, home_team, temp, humidity)
        if side_adjustment:
            confidence_penalty = min(5.0, confidence_penalty + 0.5)
            if impact == "None":
                impact = "Low"
        summary_parts = [f"{temp:.0f}°F", f"wind {wind:.0f} mph"]
        if gust >= wind + 5:
            summary_parts.append(f"gusts {gust:.0f}")
        if snow >= 0.02:
            summary_parts.append("snow")
        elif precip >= 0.04:
            summary_parts.append("rain")
        if side_adjustment:
            summary_parts.append(mismatch.rstrip("."))
        summary = ", ".join(summary_parts) + f" — {impact.lower()} impact."
        result = asdict(WeatherContext(
            available=True, source="Open-Meteo", stadium=stadium, venue_type=venue_type,
            kickoff_local=kickoff_local, temperature_f=round(temp, 1), humidity_pct=round(humidity, 1),
            precipitation_in=round(precip, 3), snowfall_in=round(snow, 3), wind_mph=round(wind, 1),
            gust_mph=round(gust, 1), label=label, impact=impact, summary=summary,
            home_margin_adjustment=round(side_adjustment, 2), total_adjustment=total_adjustment,
            confidence_penalty=round(confidence_penalty, 2), climate_mismatch=mismatch,
            fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        ))
        if session is None:
            _store_weather_cache(cache, cache_key, result)
        return result
    except Exception as exc:
        stale_cached = _cached_weather(cache, cache_key, max_age_seconds=24 * 60 * 60) if session is None else None
        if stale_cached is not None:
            stale_cached["summary"] = (
                str(stale_cached.get("summary") or "Cached weather available.")
                + " Provider refresh failed, so Macabets reused the most recent forecast."
            )
            return stale_cached
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            reason = "provider rate limit reached"
        else:
            reason = str(exc)
        return asdict(WeatherContext(
            available=False, source="Open-Meteo", stadium=stadium, venue_type=venue_type,
            kickoff_local=kickoff_local, temperature_f=None, humidity_pct=None,
            precipitation_in=None, snowfall_in=None, wind_mph=None, gust_mph=None,
            label="Unavailable", impact="None",
            summary=f"Automatic weather unavailable ({reason}). No weather adjustment applied.",
            home_margin_adjustment=0.0, total_adjustment=0.0, confidence_penalty=0.0,
            climate_mismatch="Not evaluated.", fetched_at_utc=None,
        ))

