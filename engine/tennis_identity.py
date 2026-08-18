from __future__ import annotations

from typing import Any
import re
import unicodedata

import pandas as pd


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[A-Za-z]+", text.casefold())
    return [token for token in tokens if token not in _SUFFIXES]


def normalized_name(value: Any) -> str:
    return " ".join(_tokens(value))


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


def canonical_player_key(value: Any) -> str:
    """Return the shared identity key used by tennis-history subsystems."""
    surname, initial = player_name_signature(value)
    if surname and initial:
        return f"{surname}|{initial}"
    return normalized_name(value)


def resolve_player_name(matches: pd.DataFrame, requested_name: str) -> tuple[str | None, dict]:
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
    requested_key = canonical_player_key(requested_name)

    key_matches = unique_names[unique_names.map(canonical_player_key).eq(requested_key)]
    if key_matches.empty:
        return None, {
            "requested": requested_name,
            "resolved": None,
            "method": "not_found",
            "identity_key": requested_key,
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
        method = "surname_initial"

    return resolved, {
        "requested": requested_name,
        "resolved": resolved,
        "method": method,
        "identity_key": requested_key,
        "aliases": [str(name) for name in key_matches.tolist()],
    }
