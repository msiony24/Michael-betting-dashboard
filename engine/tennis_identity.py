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


@lru_cache(maxsize=1)
def _runtime_alias_map() -> dict[str, set[str]]:
    """Load the on-disk registry once per app process.

    Streamlit restarts the process when deployed files change, so repeatedly
    stat-ing the CSV during every historical row lookup only adds filesystem
    overhead without improving correctness.
    """
    return _default_alias_map(_registry_signature())


def _alias_map(registry: pd.DataFrame | None = None) -> dict[str, set[str]]:
    """Return the registry alias map without rebuilding it for every lookup."""
    if registry is not None:
        return _build_alias_map(registry)
    return _runtime_alias_map()


def _canonical_player_key_from_alias_map(value: Any, alias_map: dict[str, set[str]]) -> str:
    candidates = alias_map.get(normalized_name(value), set())
    if len(candidates) == 1:
        return f"api:{next(iter(candidates))}"

    surname, initial = player_name_signature(value)
    if surname and initial:
        return f"name:{surname}|{initial}"
    return f"name:{normalized_name(value)}"


def provider_player_key(value: Any, registry: pd.DataFrame | None = None) -> str | None:
    """Return one unambiguous API-Tennis player key for a known alias."""
    candidates = _alias_map(registry).get(normalized_name(value), set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def canonical_player_key(value: Any, registry: pd.DataFrame | None = None) -> str:
    """Return the stable identity key used by tennis-history subsystems.

    API-Tennis player IDs are preferred whenever the alias registry can resolve a
    name unambiguously. Older players without provider IDs retain the surname and
    first-initial fallback so historical coverage remains intact.
    """
    return _canonical_player_key_from_alias_map(value, _alias_map(registry))


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
    alias_map = _alias_map(registry)
    requested_key = _canonical_player_key_from_alias_map(requested_name, alias_map)
    requested_player_key = (
        requested_key.removeprefix("api:") if requested_key.startswith("api:") else None
    )
    requested_signature = player_name_signature(requested_name)

    def _matches_requested_identity(value: Any) -> bool:
        # Normal path: both names reduce to the same unambiguous canonical key.
        if _canonical_player_key_from_alias_map(value, alias_map) == requested_key:
            return True

        # Historical ATP feeds often abbreviate names (for example,
        # ``Nakashima B.``). That abbreviated alias can legitimately be shared by
        # more than one provider player ID, so canonical_player_key must remain
        # conservative and cannot assign it globally. When the requested full
        # name resolves to one verified provider ID, however, we can bridge that
        # request to a historical alias that explicitly contains the same ID and
        # has the same surname/first-initial signature.
        if requested_player_key:
            historical_candidates = alias_map.get(normalized_name(value), set())
            if (
                requested_player_key in historical_candidates
                and player_name_signature(value) == requested_signature
            ):
                return True
        return False

    key_matches = unique_names[unique_names.map(_matches_requested_identity)]

    # Some live feeds provide a full name that has not yet been written into the
    # provider-ID registry, while the historical feed already contains the
    # abbreviated form (for example ``Juan Manuel Cerundolo`` vs
    # ``Cerundolo J.M.``). In that case, allow a surname/first-initial bridge
    # only when every historical name with that signature collapses to one
    # canonical identity. This preserves coverage without guessing across
    # genuinely ambiguous players who share the same surname and initial.
    if key_matches.empty and requested_signature != ("", ""):
        signature_matches = unique_names[
            unique_names.map(lambda value: player_name_signature(value) == requested_signature)
        ]
        if not signature_matches.empty:
            signature_keys = {
                _canonical_player_key_from_alias_map(value, alias_map)
                for value in signature_matches.tolist()
            }
            if len(signature_keys) == 1:
                key_matches = signature_matches

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
        resolved_key = _canonical_player_key_from_alias_map(resolved, alias_map)
        method = (
            "provider_player_id"
            if requested_key.startswith("api:")
            else "unique_signature" if resolved_key.startswith("api:")
            else "surname_initial"
        )

    return resolved, {
        "requested": requested_name,
        "resolved": resolved,
        "method": method,
        "identity_key": requested_key,
        "player_key": requested_key.removeprefix("api:") if requested_key.startswith("api:") else None,
        "aliases": [str(name) for name in key_matches.tolist()],
    }
