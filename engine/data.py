from __future__ import annotations

from pathlib import Path

import pandas as pd

# Streamlit is only needed for its caching decorator below -- everything else
# in this file is pure logic that should be importable and testable without
# Streamlit installed at all. Falling back to a no-op decorator when it isn't
# available keeps that true, at the cost of losing cross-rerun caching only
# in a non-Streamlit context (tests, scripts) where that caching was never
# doing anything useful anyway.
try:
    import streamlit as st
    _cache_data = st.cache_data
except ImportError:
    def _cache_data(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _match_file_signature() -> tuple[tuple[str, int, int], ...]:
    """Return a cheap fingerprint that changes whenever an ATP CSV is replaced."""
    signature: list[tuple[str, int, int]] = []
    for path in sorted(DATA_DIR.glob("atp_matches_*.csv")):
        try:
            stat = path.stat()
            signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            # If a file disappears during a deploy/update, the next rerun will rebuild
            # a different signature. Do not keep a stale cached dataset because of it.
            continue
    return tuple(signature)


@_cache_data(show_spinner=False)
def _load_matches_cached(
    file_signature: tuple[tuple[str, int, int], ...],
) -> tuple[pd.DataFrame, list[str]]:
    """Load ATP history for one exact on-disk file version.

    ``file_signature`` is intentionally unused inside the loader. It is part of the
    cache key so a GitHub data refresh invalidates Streamlit's cached dataframe as
    soon as the deployed CSV files change.
    """
    del file_signature
    files = sorted(DATA_DIR.glob("atp_matches_*.csv"))

    if not files:
        raise RuntimeError(
            "The local tennis database has not been created yet. "
            "Run the GitHub Action named 'Update Macabets Tennis Data', "
            "then reboot the Streamlit app."
        )

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for path in files:
        try:
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = path.name
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not frames:
        raise RuntimeError("The local tennis database files could not be read.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    matches = matches.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    matches["surface"] = matches.get("surface", "").fillna("Unknown")
    matches["tourney_name"] = matches.get("tourney_name", "").fillna("Unknown")
    matches["tourney_level"] = matches.get("tourney_level", "").fillna("")
    matches["round"] = matches.get("round", "").fillna("")
    matches["score"] = matches.get("score", "").fillna("")

    numeric_columns = [
        "winner_rank", "loser_rank", "winner_age", "loser_age",
        "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
        "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon",
        "w_2ndWon", "l_2ndWon", "w_SvGms", "l_SvGms",
        "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
    ]
    for column in numeric_columns:
        if column in matches:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    return matches, errors


def load_matches() -> tuple[pd.DataFrame, list[str]]:
    """Load the newest local ATP database without requiring a manual cache clear."""
    signature = _match_file_signature()
    if not signature:
        raise RuntimeError(
            "The local tennis database has not been created yet. "
            "Run the GitHub Action named 'Update Macabets Tennis Data', "
            "then reboot the Streamlit app."
        )
    return _load_matches_cached(signature)
