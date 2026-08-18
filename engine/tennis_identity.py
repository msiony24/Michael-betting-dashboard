from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import re
import unicodedata

import pandas as pd


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
IDENTITY_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "tennis_player_ids.csv"


def _tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[A-Za-z]+", text.casefold())
    return [token for token in tokens if token not in _SUFFIXES]


def normalized_name(value: Any) -> str:
    return " ".join(_tokens(value))


def _normalized_player_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def player_name_signature(value: Any) -> tuple[str, str]:
    """Return a provider-tolerant surname and first-initial signature.

    Historical feeds use forms such as Auger-Aliassime F. and Cerundolo J.M.,
    while live feeds use full names. Trailing initials are peeled off before the
    surname is selected, fixing compound provider names.
    """
    tokens = _tokens(value)
    if not tokens:
        return "", ""

    first_initial_index = len(tokens)
    while first_initial_index > 0 and len(tokens[first_initial_index - 1]) == 1:
        first_initial_index -= 1

    if 0 < first_initial_index < len(tokens):
        surname = tokens[first_initial_index - 1]
        first_initial = tokens[first_initial_index]
        return surname, first_initial

    if len(tokens) >= 2:
        return tokens[-1], tokens[0][0]
    return tokens[0], ""


def _build_alias_map(registry: pd.DataFrame | None) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    if registry is None or registry.empty or "player_key" not in registry:
        return aliases

    for _, row in registry.iterrows():
        player_key = _normalized_player_key(row.get("player_key"))
        if not player_key:
            continue
        for column in ("alias", "canonical_name"):
            name = normalized_name(row.get(column))
            if not name:
                continue
            aliases.setdefault(name, set()).add(player_key)
    return aliases


def _registry_signature() -> tuple[int, int] | None:
    try:
        stat = IDENTITY_REGISTRY_PATH.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return None


@lru_cache(maxsize=4)
def _default_alias_map(signature: tuple[int, int] | None) -> dict[str, set[str]]:
    if signature is None:
        return {}
    try:
        registry = pd.read_csv(IDENTITY_REGISTRY_PATH, dtype={"player_key": str})
    except Exception:
        return {}
    return _build_alias_map(registry)


def provider_player_key(value: Any, registry: pd.DataFrame | None = None) -> str | None:
    """Return one unambiguous API-Tennis player key for a known alias."""
    alias_map = _build_alias_map(registry) if registry is not None else _default_alias_map(_registry_signature())
    candidates = alias_map.get(normalized_name(value), set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def canonical_player_key(value: Any, registry: pd.DataFrame | None = None) -> str:
    """Return the stable identity key used by tennis-history subsystems.

    API-Tennis player IDs are preferred whenever the alias registry can resolve a
    name unambiguously. Older players without provider IDs retain the surname and
    first-initial fallback so historical coverage remains intact.
    """
    player_key = provider_player_key(value, registry=registry)
    if player_key:
        return f"api:{player_key}"

    surname, initial = player_name_signature(value)
    if surname and initial:
        return f"name:{surname}|{initial}"
    return f"name:{normalized_name(value)}"


def resolve_player_name(
    matches: pd.DataFrame,
    requested_name: str,
    registry: pd.DataFrame | None = None,
) -> tuple[str | None, dict]:
    """Resolve a requested player to a display name in the historical database."""
    if matches is None or matches.empty:
        return None, {"requested": requested_name, "resolved": None, "method": "not_found"}

    names = pd.concat([
        matches.get("winner_name", pd.Series(dtype=str)),
        matches.get("loser_name", pd.Series(dtype=str)),
    ]).dropna().astype(str)
    if names.empty:
        return None, {"requested": requested_name, "resolved": None, "method": "not_found"}

    counts = names.value_counts()
    unique_names = counts.index.to_series()
    requested_normalized = normalized_name(requested_name)
    requested_key = canonical_player_key(requested_name, registry=registry)

    key_matches = unique_names[
        unique_names.map(lambda value: canonical_player_key(value, registry=registry)).eq(requested_key)
    ]
    if key_matches.empty:
        return None, {
            "requested": requested_name,
            "resolved": None,
            "method": "not_found",
            "identity_key": requested_key,
            "player_key": requested_key.removeprefix("api:") if requested_key.startswith("api:") else None,
        }

    exact = key_matches[key_matches.map(normalized_name).eq(requested_normalized)]
    if not exact.empty:
        resolved = str(exact.iloc[0])
        method = "exact"
    else:
        ranked = sorted(
            ((str(name), int(counts.get(name, 0))) for name in key_matches.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        resolved = ranked[0][0]
        method = "provider_player_id" if requested_key.startswith("api:") else "surname_initial"

    return resolved, {
        "requested": requested_name,
        "resolved": resolved,
        "method": method,
        "identity_key": requested_key,
        "player_key": requested_key.removeprefix("api:") if requested_key.startswith("api:") else None,
        "aliases": [str(name) for name in key_matches.tolist()],
    }
