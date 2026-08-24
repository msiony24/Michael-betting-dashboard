from __future__ import annotations

from typing import Any

import pandas as pd

from .tennis_identity import canonical_player_key


def tennis_pair_key(player_a: Any, player_b: Any) -> tuple[str, str]:
    return tuple(sorted((canonical_player_key(player_a), canonical_player_key(player_b))))


def merge_tennis_schedule_with_market(
    schedule: pd.DataFrame | None,
    market_slate: pd.DataFrame | None,
) -> pd.DataFrame:
    """Keep API-Tennis as the schedule and prefer market-feed prices when available.

    ``schedule`` may already contain API-Tennis fallback prices joined by the API
    match key. ``market_slate`` contains The Odds API prices, which use unrelated
    event IDs, so matches are aligned by canonical player pair and nearest start
    time. A market-only event is retained rather than silently discarded.
    """
    if schedule is None or schedule.empty:
        return market_slate.copy() if market_slate is not None else pd.DataFrame()

    merged = schedule.copy()
    if market_slate is None or market_slate.empty:
        return merged

    market_by_pair: dict[tuple[str, str], list[tuple[Any, pd.Series]]] = {}
    for idx, row in market_slate.iterrows():
        key = tennis_pair_key(row.get("participant_a"), row.get("participant_b"))
        market_by_pair.setdefault(key, []).append((idx, row))

    matched_market_indices: set[Any] = set()
    for idx, row in merged.iterrows():
        key = tennis_pair_key(row.get("participant_a"), row.get("participant_b"))
        candidates = market_by_pair.get(key, [])
        if not candidates:
            continue

        def _distance(candidate: tuple[Any, pd.Series]) -> float:
            try:
                return abs((candidate[1].get("start_time") - row.get("start_time")).total_seconds())
            except Exception:
                return float("inf")

        market_idx, market_row = min(candidates, key=_distance)
        matched_market_indices.add(market_idx)

        # The Odds API is the primary source. Participant order can differ
        # between providers, so map each quoted side back to the schedule name.
        for market_side, market_book in (("a", "book_a"), ("b", "book_b")):
            value = market_row.get(f"odds_{market_side}")
            if pd.isna(value):
                continue
            market_name = market_row.get(f"participant_{market_side}")
            if canonical_player_key(market_name) == canonical_player_key(row.get("participant_a")):
                target_side = "a"
            elif canonical_player_key(market_name) == canonical_player_key(row.get("participant_b")):
                target_side = "b"
            else:
                continue
            merged.at[idx, f"odds_{target_side}"] = value
            merged.at[idx, f"book_{target_side}"] = market_row.get(market_book, "—")

    unmatched = market_slate.loc[~market_slate.index.isin(matched_market_indices)].copy()
    if not unmatched.empty:
        merged = pd.concat([merged, unmatched], ignore_index=True)

    sort_columns = [column for column in ("start_time", "sport", "participant_a") if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns)
    return merged.reset_index(drop=True)
