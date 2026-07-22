
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable
import warnings

import certifi
import pandas as pd
import requests
import streamlit as st
import urllib3


DATA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
CACHE_DIR = Path(".macabets_cache")
YEARS = list(range(max(2021, date.today().year - 5), date.today().year + 1))


def _download_text(url: str) -> str:
    headers = {"User-Agent": "Macabets/1.0"}
    errors: list[str] = []

    # Normal verified request.
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            verify=certifi.where(),
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        errors.append(f"verified request: {exc}")

    # Some Streamlit Cloud environments have an incomplete certificate chain.
    # GitHub raw content is still HTTPS; this fallback bypasses local certificate
    # validation only after the verified request has failed.
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        errors.append(f"certificate fallback: {exc}")

    raise RuntimeError(" | ".join(errors))


def _read_year(year: int) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"atp_matches_{year}.csv"

    if cache_file.exists() and cache_file.stat().st_size > 1000:
        return pd.read_csv(cache_file, low_memory=False)

    url = f"{DATA_BASE}/atp_matches_{year}.csv"
    text = _download_text(url)
    cache_file.write_text(text, encoding="utf-8")
    return pd.read_csv(StringIO(text), low_memory=False)


@st.cache_data(ttl=21600, show_spinner=False)
def load_matches() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for year in YEARS:
        try:
            frame = _read_year(year)
            frame["source_year"] = year
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{year}: {exc}")

    if not frames:
        raise RuntimeError(
            "Macabets could not load any ATP match files. "
            "Use Reboot app once, then try again."
        )

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
