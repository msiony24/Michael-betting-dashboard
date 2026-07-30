"""Curated ATP player-trait database loader and validator.

This module is explanation-only. It should not modify prediction probabilities,
fair odds, value calculations, or betting recommendations.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "player_traits.json"
VALID_SKILL_VALUES = {1, 2, 3, 4, 5}


class PlayerTraitsError(ValueError):
    """Raised when the player-trait database is malformed."""


def normalize_player_name(name: str) -> str:
    """Normalize player names and aliases for case-insensitive lookup."""
    if not isinstance(name, str):
        raise TypeError("Player name must be a string.")

    normalized = name.strip().casefold()
    normalized = re.sub(r"[.,]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


class PlayerTraitsDatabase:
    """Load, validate, and query the curated ATP player-trait database."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self._data = self._load_json()
        validate_database(self._data)
        self._players: dict[str, dict[str, Any]] = self._data["players"]
        self._lookup = self._build_lookup()

    def _load_json(self) -> dict[str, Any]:
        try:
            with self.database_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise PlayerTraitsError(
                f"Player-trait database not found: {self.database_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PlayerTraitsError(
                f"Invalid JSON in player-trait database: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise PlayerTraitsError("Player-trait database root must be an object.")
        return data

    def _build_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for canonical_name, profile in self._players.items():
            for candidate in [canonical_name, *profile["aliases"]]:
                normalized = normalize_player_name(candidate)
                existing = lookup.get(normalized)
                if existing and existing != canonical_name:
                    raise PlayerTraitsError(
                        f"Duplicate normalized name or alias '{candidate}' is shared by "
                        f"'{existing}' and '{canonical_name}'."
                    )
                lookup[normalized] = canonical_name
        return lookup

    @property
    def version(self) -> str:
        return str(self._data["_metadata"]["version"])

    def canonical_name(self, player_name: str) -> str | None:
        """Return the canonical database name, or None when the player is unknown."""
        return self._lookup.get(normalize_player_name(player_name))

    def exists(self, player_name: str) -> bool:
        return self.canonical_name(player_name) is not None

    def get(self, player_name: str) -> dict[str, Any] | None:
        """Return a defensive copy of a player's profile."""
        canonical = self.canonical_name(player_name)
        if canonical is None:
            return None
        profile = deepcopy(self._players[canonical])
        profile["name"] = canonical
        return profile

    def require(self, player_name: str) -> dict[str, Any]:
        """Return a profile or raise KeyError when the player is unknown."""
        profile = self.get(player_name)
        if profile is None:
            raise KeyError(f"Unknown player: {player_name}")
        return profile

    def all_players(self) -> tuple[str, ...]:
        return tuple(sorted(self._players))

    def rating_label(self, rating: int) -> str:
        """Convert a numeric internal rating to its display label."""
        try:
            return str(self._data["_metadata"]["rating_scale"][str(rating)])
        except KeyError as exc:
            raise ValueError(f"Invalid skill rating: {rating}") from exc


def validate_database(data: Mapping[str, Any]) -> None:
    """Validate the complete database and raise PlayerTraitsError on failure."""
    if "_metadata" not in data or "players" not in data:
        raise PlayerTraitsError("Database must contain '_metadata' and 'players'.")

    metadata = data["_metadata"]
    players = data["players"]
    if not isinstance(metadata, Mapping):
        raise PlayerTraitsError("'_metadata' must be an object.")
    if not isinstance(players, Mapping) or not players:
        raise PlayerTraitsError("'players' must be a non-empty object.")

    required_metadata = {
        "version",
        "rating_scale",
        "valid_playing_styles",
        "valid_court_positions",
        "valid_rally_preferences",
        "valid_signature_traits",
        "required_skills",
    }
    missing_metadata = required_metadata - set(metadata)
    if missing_metadata:
        raise PlayerTraitsError(
            f"Metadata is missing required fields: {sorted(missing_metadata)}"
        )

    required_skills = set(metadata["required_skills"])
    valid_styles = set(metadata["valid_playing_styles"])
    valid_positions = set(metadata["valid_court_positions"])
    valid_rallies = set(metadata["valid_rally_preferences"])
    valid_traits = set(metadata["valid_signature_traits"])

    seen_names: dict[str, str] = {}
    for player_name, profile in players.items():
        _validate_player_profile(
            player_name=player_name,
            profile=profile,
            required_skills=required_skills,
            valid_styles=valid_styles,
            valid_positions=valid_positions,
            valid_rallies=valid_rallies,
            valid_traits=valid_traits,
        )

        for candidate in [player_name, *profile["aliases"]]:
            normalized = normalize_player_name(candidate)
            if normalized in seen_names:
                raise PlayerTraitsError(
                    f"Duplicate player name or alias '{candidate}' conflicts with "
                    f"'{seen_names[normalized]}'."
                )
            seen_names[normalized] = player_name


def _validate_player_profile(
    *,
    player_name: str,
    profile: Any,
    required_skills: set[str],
    valid_styles: set[str],
    valid_positions: set[str],
    valid_rallies: set[str],
    valid_traits: set[str],
) -> None:
    if not isinstance(player_name, str) or not player_name.strip():
        raise PlayerTraitsError("Each player must have a non-empty string name.")
    if not isinstance(profile, Mapping):
        raise PlayerTraitsError(f"Profile for '{player_name}' must be an object.")

    required_fields = {
        "aliases",
        "playing_style",
        "court_position",
        "preferred_rally",
        "skills",
        "signature_traits",
    }
    missing_fields = required_fields - set(profile)
    if missing_fields:
        raise PlayerTraitsError(
            f"Profile for '{player_name}' is missing: {sorted(missing_fields)}"
        )

    aliases = profile["aliases"]
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise PlayerTraitsError(
            f"Aliases for '{player_name}' must be a list of non-empty strings."
        )

    if profile["playing_style"] not in valid_styles:
        raise PlayerTraitsError(
            f"Invalid playing style for '{player_name}': {profile['playing_style']}"
        )
    if profile["court_position"] not in valid_positions:
        raise PlayerTraitsError(
            f"Invalid court position for '{player_name}': {profile['court_position']}"
        )
    if profile["preferred_rally"] not in valid_rallies:
        raise PlayerTraitsError(
            f"Invalid rally preference for '{player_name}': "
            f"{profile['preferred_rally']}"
        )

    skills = profile["skills"]
    if not isinstance(skills, Mapping):
        raise PlayerTraitsError(f"Skills for '{player_name}' must be an object.")
    actual_skills = set(skills)
    if actual_skills != required_skills:
        missing = required_skills - actual_skills
        extra = actual_skills - required_skills
        raise PlayerTraitsError(
            f"Invalid skill keys for '{player_name}'. Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for skill, rating in skills.items():
        if type(rating) is not int or rating not in VALID_SKILL_VALUES:
            raise PlayerTraitsError(
                f"Skill '{skill}' for '{player_name}' must be an integer from 1 to 5."
            )

    traits = profile["signature_traits"]
    if not isinstance(traits, list) or not traits:
        raise PlayerTraitsError(
            f"Signature traits for '{player_name}' must be a non-empty list."
        )
    if len(traits) != len(set(traits)):
        raise PlayerTraitsError(
            f"Signature traits for '{player_name}' contain duplicates."
        )
    invalid_traits = set(traits) - valid_traits
    if invalid_traits:
        raise PlayerTraitsError(
            f"Invalid signature traits for '{player_name}': {sorted(invalid_traits)}"
        )
