
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data(show_spinner=False)
def load_matches() -> tuple[pd.DataFrame, list[str]]:
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
