"""Handedness splits for the Macabets tennis model and evidence layer.

The handedness resolver is deliberately identity-safe. Historical tennis feeds often
store players as ``Surname F.`` while other sources use full names. A generic
canonicalizer that removes one-letter initials can collapse different players with
the same surname (for example ``Zverev A.`` and ``Zverev M.``). This module keeps
surname + first-initial information intact and refuses ambiguous lookups.

Unknown players remain unknown; the model never imputes a hand from nationality,
style, or name.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import re
import unicodedata

import numpy as np
import pandas as pd

from .tennis import norm, player_name_signature, resolve_player_name


DEFAULT_HANDEDNESS_FILE = Path(__file__).resolve().parent.parent / "data" / "atp_player_handedness.csv"


def normalize_hand(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in {"l", "left", "left-handed", "left handed"}:
        return "Left"
    if text in {"r", "right", "right-handed", "right handed"}:
        return "Right"
    return None


def _exact_name_key(value: Any) -> str:
    """Normalize a name without discarding initials.

    ``Zverev A.`` -> ``zverev a`` and ``Zverev M.`` -> ``zverev m``.
    This is intentionally different from the broader tennis canonicalizer.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text).casefold()
    tokens = [token for token in text.split() if token not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def _identity_signature(value: Any) -> tuple[str, str]:
    surname, initial = player_name_signature(str(value or ""))
    return str(surname or ""), str(initial or "")


def _add_consistent(mapping: dict[Any, str | None], key: Any, hand: str) -> None:
    """Add a lookup only when every occurrence of the key agrees on hand."""
    if not key or key == ("", ""):
        return
    if key not in mapping:
        mapping[key] = hand
    elif mapping[key] != hand:
        # Ambiguous identity: fail closed rather than assigning the wrong hand.
        mapping[key] = None


@lru_cache(maxsize=4)
def _load_alias_registry(path_text: str = str(DEFAULT_HANDEDNESS_FILE)) -> dict[str, dict[Any, str | None]]:
    path = Path(path_text)
    if not path.exists():
        return {"exact": {}, "signature": {}}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {"exact": {}, "signature": {}}

    exact: dict[str, str | None] = {}
    signature: dict[tuple[str, str], str | None] = {}
    for _, row in frame.iterrows():
        hand = normalize_hand(row.get("hand"))
        if not hand:
            continue
        for raw_name in (row.get("alias"), row.get("resolved_player")):
            name = str(raw_name or "").strip()
            if not name:
                continue
            _add_consistent(exact, _exact_name_key(name), hand)
            _add_consistent(signature, _identity_signature(name), hand)

    return {"exact": exact, "signature": signature}


def player_hand(player: str, *, manual_hand: Any = None, path: str | Path = DEFAULT_HANDEDNESS_FILE) -> str | None:
    """Return verified Left/Right hand, preferring an explicit current-match value.

    Resolution order:
    1. explicit current-match value;
    2. exact normalized alias/full name (initials preserved);
    3. surname + first-initial signature, but only if that signature is unambiguous.
    """
    manual = normalize_hand(manual_hand)
    if manual:
        return manual

    registry = _load_alias_registry(str(path))
    exact = registry["exact"].get(_exact_name_key(player))
    if exact in {"Left", "Right"}:
        return exact

    by_signature = registry["signature"].get(_identity_signature(player))
    return by_signature if by_signature in {"Left", "Right"} else None


def _row_identity_mask(series: pd.Series, target: str) -> pd.Series:
    """Match a resolved historical player without collapsing surname initials."""
    target_norm = norm(target)
    exact = series.astype(str).map(norm).eq(target_norm)
    if exact.any():
        return exact

    target_signature = _identity_signature(target)
    if target_signature != ("", ""):
        return series.astype(str).map(_identity_signature).eq(target_signature)
    return exact


def _player_rows(matches: pd.DataFrame, player: str, event_date: Any) -> tuple[pd.DataFrame, str]:
    resolved, _ = resolve_player_name(matches, player)
    target = resolved or player
    event_ts = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_ts):
        event_ts = pd.Timestamp.today().normalize()

    player_mask = _row_identity_mask(matches["winner_name"], target) | _row_identity_mask(matches["loser_name"], target)
    mask = player_mask & (matches["tourney_date"] < pd.Timestamp(event_ts))
    return matches.loc[mask].sort_values("tourney_date", ascending=False).copy(), target


def _decorate_with_opponent_hand(rows: pd.DataFrame, player_target: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    out["won"] = _row_identity_mask(out["winner_name"], player_target)
    out["opponent"] = np.where(out["won"], out["loser_name"], out["winner_name"])
    out["opponent_hand"] = out["opponent"].map(player_hand)
    return out


def _record(rows: pd.DataFrame) -> dict[str, Any]:
    known = rows[rows["opponent_hand"].isin(["Left", "Right"])].copy() if "opponent_hand" in rows else rows.iloc[0:0]
    wins = int(known["won"].sum()) if not known.empty else 0
    total = int(len(known))
    return {
        "wins": wins,
        "losses": total - wins,
        "matches": total,
        "win_rate": float(wins / total) if total else None,
    }


def handedness_record_splits(matches: pd.DataFrame, player: str, event_date: Any, surface: str = "") -> dict[str, Any]:
    """Build verified historical records vs lefties/righties for model + Challenge."""
    rows, target = _player_rows(matches, player, event_date)
    decorated = _decorate_with_opponent_hand(rows, target)
    event_ts = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_ts):
        event_ts = pd.Timestamp.today().normalize()
    season_start = pd.Timestamp(year=event_ts.year, month=1, day=1)
    recent_cutoff = pd.Timestamp(event_ts) - pd.Timedelta(days=365)
    surface_key = str(surface or "").strip().casefold()

    def subset(hand: str, scope: pd.DataFrame) -> dict[str, Any]:
        return _record(scope[scope["opponent_hand"] == hand])

    season = decorated[decorated["tourney_date"] >= season_start]
    recent = decorated[decorated["tourney_date"] >= recent_cutoff]
    surface_rows = decorated[
        decorated["surface"].astype(str).str.casefold().eq(surface_key)
    ] if surface_key else decorated.iloc[0:0]

    known = decorated[decorated["opponent_hand"].isin(["Left", "Right"])]
    total = int(len(decorated))
    known_count = int(len(known))
    return {
        "player": player,
        "coverage": float(known_count / total) if total else 0.0,
        "known_opponent_hands": known_count,
        "total_matches": total,
        "career": {"vs_left": subset("Left", decorated), "vs_right": subset("Right", decorated)},
        "season": {"vs_left": subset("Left", season), "vs_right": subset("Right", season)},
        "last_365_days": {"vs_left": subset("Left", recent), "vs_right": subset("Right", recent)},
        "surface": str(surface or "Unknown"),
        "surface_split": {"vs_left": subset("Left", surface_rows), "vs_right": subset("Right", surface_rows)},
    }


def handedness_matchup_profile(
    matches: pd.DataFrame,
    player: str,
    opponent_hand: Any,
    event_date: Any,
    surface: str = "",
) -> dict[str, Any]:
    """Return a conservative probability signal for performance vs opponent hand.

    The signal is shrunk toward the player's own baseline and capped. Small samples
    therefore have little or no impact, while repeatable handedness-specific results
    can move the final probability modestly.
    """
    target_hand = normalize_hand(opponent_hand)
    splits = handedness_record_splits(matches, player, event_date, surface)
    if not target_hand:
        return {"available": False, "opponent_hand": None, "adjustment": 0.0, "splits": splits}

    key = "vs_left" if target_hand == "Left" else "vs_right"
    preferred = splits["surface_split"][key]
    recent = splits["last_365_days"][key]
    career = splits["career"][key]

    # Prefer surface-specific history only once it has a useful sample. Otherwise use
    # the current-year-ish signal, then career. No single 2-0 or 3-0 split can dominate.
    selected = preferred if preferred["matches"] >= 8 else recent if recent["matches"] >= 8 else career
    n = int(selected["matches"])
    if n == 0 or selected["win_rate"] is None:
        return {"available": False, "opponent_hand": target_hand, "adjustment": 0.0, "splits": splits}

    all_left = splits["career"]["vs_left"]
    all_right = splits["career"]["vs_right"]
    base_wins = all_left["wins"] + all_right["wins"]
    base_matches = all_left["matches"] + all_right["matches"]
    baseline = float(base_wins / base_matches) if base_matches else 0.5

    # Equivalent to a 12-match prior at the player's own baseline.
    shrunk_rate = float((selected["wins"] + 12.0 * baseline) / (n + 12.0))
    performance_edge = shrunk_rate - baseline
    sample_reliability = float(np.clip((n - 3) / 17.0, 0.0, 1.0))
    coverage_reliability = float(np.clip(splits["coverage"] / 0.80, 0.0, 1.0))
    adjustment = float(np.clip(performance_edge * 0.18 * sample_reliability * coverage_reliability, -0.02, 0.02))

    return {
        "available": True,
        "opponent_hand": target_hand,
        "selected_record": selected,
        "baseline_win_rate": baseline,
        "shrunk_win_rate": shrunk_rate,
        "sample_reliability": sample_reliability,
        "coverage_reliability": coverage_reliability,
        "adjustment": adjustment,
        "splits": splits,
    }
