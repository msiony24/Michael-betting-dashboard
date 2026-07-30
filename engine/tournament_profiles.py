"""Tournament-specific court context for Macabets.

This module is explanation-only in v0.57. Court profiles help describe how a
selected tournament may amplify or reduce player strengths, but they do not
change Elo, simulations, fair odds, ROI, or betting verdicts.
"""
from __future__ import annotations

import re
import unicodedata


def _key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


# Speed is a relative explanation-layer score on a 1-5 scale:
# 1 slow, 2 medium-slow, 3 medium, 4 medium-fast, 5 fast.
# Profiles are intentionally conservative and can later be overridden by year.
TOURNAMENT_PROFILES = {
    "australian open": {
        "surface": "Hard", "speed": 4, "bounce": "medium", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "indian wells": {
        "surface": "Hard", "speed": 2, "bounce": "high", "altitude": "normal",
        "environment": "outdoor", "confidence": "high",
    },
    "miami": {
        "surface": "Hard", "speed": 2, "bounce": "medium-high", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "monte carlo": {
        "surface": "Clay", "speed": 2, "bounce": "high", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "barcelona": {
        "surface": "Clay", "speed": 2, "bounce": "high", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "madrid": {
        "surface": "Clay", "speed": 4, "bounce": "high", "altitude": "high",
        "environment": "outdoor", "confidence": "high",
    },
    "rome": {
        "surface": "Clay", "speed": 2, "bounce": "high", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "roland garros": {
        "surface": "Clay", "speed": 2, "bounce": "high", "altitude": "normal",
        "environment": "outdoor", "confidence": "high",
    },
    "queens club": {
        "surface": "Grass", "speed": 4, "bounce": "low", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "halle": {
        "surface": "Grass", "speed": 4, "bounce": "low", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "wimbledon": {
        "surface": "Grass", "speed": 4, "bounce": "low-medium", "altitude": "normal",
        "environment": "outdoor", "confidence": "high",
    },
    "canada masters": {
        "surface": "Hard", "speed": 3, "bounce": "medium", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "cincinnati": {
        "surface": "Hard", "speed": 4, "bounce": "medium-low", "altitude": "normal",
        "environment": "outdoor", "confidence": "high",
    },
    "us open": {
        "surface": "Hard", "speed": 4, "bounce": "medium", "altitude": "normal",
        "environment": "outdoor", "confidence": "high",
    },
    "shanghai": {
        "surface": "Hard", "speed": 4, "bounce": "medium-low", "altitude": "normal",
        "environment": "outdoor", "confidence": "medium",
    },
    "paris masters": {
        "surface": "Hard", "speed": 4, "bounce": "low-medium", "altitude": "normal",
        "environment": "indoor", "confidence": "medium",
    },
    "atp finals": {
        "surface": "Hard", "speed": 4, "bounce": "medium-low", "altitude": "normal",
        "environment": "indoor", "confidence": "medium",
    },
}


ALIASES = {
    "australian open": "australian open",
    "indian wells masters": "indian wells",
    "bnP paribas open": "indian wells",
    "miami open": "miami",
    "monte carlo masters": "monte carlo",
    "madrid masters": "madrid",
    "mutua madrid open": "madrid",
    "rome masters": "rome",
    "italian open": "rome",
    "french open": "roland garros",
    "roland garros": "roland garros",
    "queens": "queens club",
    "queen s club": "queens club",
    "cinch championships": "queens club",
    "canadian open": "canada masters",
    "montreal": "canada masters",
    "toronto": "canada masters",
    "western southern open": "cincinnati",
    "cincinnati masters": "cincinnati",
    "u s open": "us open",
    "us open": "us open",
    "shanghai masters": "shanghai",
    "paris": "paris masters",
    "paris bercy": "paris masters",
    "tour finals": "atp finals",
    "nitto atp finals": "atp finals",
}
ALIASES = {_key(alias): target for alias, target in ALIASES.items()}


SPEED_LABELS = {
    1: "slow",
    2: "medium-slow",
    3: "medium",
    4: "medium-fast",
    5: "fast",
}


def _surface_fallback(surface: str, environment: str = "Outdoor") -> dict:
    surface_name = str(surface or "Hard").title()
    indoor = "indoor" in str(environment or "").casefold()
    defaults = {
        "Clay": {"speed": 2, "bounce": "high"},
        "Grass": {"speed": 4, "bounce": "low"},
        "Carpet": {"speed": 5, "bounce": "low"},
        "Hard": {"speed": 3, "bounce": "medium"},
    }
    base = defaults.get(surface_name, defaults["Hard"])
    speed = min(5, base["speed"] + (1 if indoor and surface_name == "Hard" else 0))
    return {
        "surface": surface_name,
        "speed": speed,
        "speed_label": SPEED_LABELS[speed],
        "bounce": base["bounce"],
        "altitude": "unknown",
        "environment": "indoor" if indoor else "outdoor",
        "confidence": "fallback",
        "source": "surface_fallback",
        "matched_name": None,
    }


def resolve_tournament_profile(
    tournament: str,
    surface: str,
    environment: str = "Outdoor",
) -> dict:
    """Automatically resolve the selected tournament to its court profile.

    The lookup tolerates provider naming differences and falls back to a broad
    surface profile rather than inventing tournament-specific conditions.
    """
    requested_key = _key(tournament)
    canonical = ALIASES.get(requested_key, requested_key)

    profile = TOURNAMENT_PROFILES.get(canonical)
    if profile is None:
        # Provider names often add sponsors or city labels. Use a conservative
        # contained-name match only when one tournament clearly matches.
        candidates = [name for name in TOURNAMENT_PROFILES if name in requested_key or requested_key in name]
        if len(candidates) == 1:
            canonical = candidates[0]
            profile = TOURNAMENT_PROFILES[canonical]

    if profile is None:
        fallback = _surface_fallback(surface, environment)
        fallback["requested_name"] = tournament
        return fallback

    resolved = dict(profile)
    resolved["speed_label"] = SPEED_LABELS.get(int(resolved["speed"]), "medium")
    resolved["source"] = "tournament_profile"
    resolved["matched_name"] = canonical
    resolved["requested_name"] = tournament
    return resolved


def court_context_sentence(tournament: str, profile: dict) -> str:
    """Return one concise, neutral description for the explanation layer."""
    speed = profile.get("speed_label", "medium")
    surface = str(profile.get("surface", "Hard")).lower()
    bounce = profile.get("bounce", "medium")
    altitude = profile.get("altitude", "normal")
    environment = profile.get("environment", "outdoor")

    details = [f"{speed} {surface} conditions", f"{bounce} bounce"]
    if altitude == "high":
        details.append("high altitude")
    if environment == "indoor":
        details.append("indoor conditions")

    label = tournament or profile.get("matched_name") or "This event"
    return f"{label} is treated as having " + ", ".join(details) + "."
