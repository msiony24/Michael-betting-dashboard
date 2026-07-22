from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2021
SOURCE_TEMPLATE = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"


def normalize_round(value: object) -> str:
    mapping = {
        "1st Round": "R128",
        "2nd Round": "R64",
        "3rd Round": "R32",
        "4th Round": "R16",
        "Round Robin": "RR",
        "Quarterfinals": "QF",
        "Quarterfinal": "QF",
        "Semifinals": "SF",
        "Semifinal": "SF",
        "The Final": "F",
        "Final": "F",
    }
    text = str(value).strip()
    return mapping.get(text, text)


def normalize_level(value: object) -> str:
    text = str(value).strip().lower()
    if "grand slam" in text:
        return "G"
    if "masters" in text:
        return "M"
    if "atp250" in text or "atp 250" in text:
        return "A"
    if "atp500" in text or "atp 500" in text:
        return "A"
    if "masters cup" in text or "tour finals" in text:
        return "F"
    return "A"


def build_score(row: pd.Series) -> str:
    parts: list[str] = []
    for number in range(1, 6):
        w_col = f"W{number}"
        l_col = f"L{number}"
        if w_col not in row.index or l_col not in row.index:
            continue
        w = row.get(w_col)
        l = row.get(l_col)
        if pd.isna(w) or pd.isna(l):
            continue
        try:
            parts.append(f"{int(float(w))}-{int(float(l))}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts)


def convert_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frame.copy()

    required = ["Date", "Tournament", "Surface", "Winner", "Loser"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{year} workbook is missing required columns: {', '.join(missing)}"
        )

    dates = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=True)

    output = pd.DataFrame({
        "tourney_date": dates.dt.strftime("%Y%m%d"),
        "tourney_name": frame["Tournament"].astype(str).str.strip(),
        "surface": frame["Surface"].astype(str).str.strip().str.title(),
        "tourney_level": frame.get("Series", "ATP").map(normalize_level)
            if "Series" in frame.columns else "A",
        "round": frame.get("Round", "").map(normalize_round)
            if "Round" in frame.columns else "",
        "winner_name": frame["Winner"].astype(str).str.strip(),
        "loser_name": frame["Loser"].astype(str).str.strip(),
        "winner_rank": pd.to_numeric(frame.get("WRank"), errors="coerce"),
        "loser_rank": pd.to_numeric(frame.get("LRank"), errors="coerce"),
        "score": frame.apply(build_score, axis=1),
    })

    # The replacement source does not consistently provide point-by-point serve totals.
    # These columns are retained so the existing Macabets engine can use safe defaults.
    for column in [
        "winner_age", "loser_age",
        "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
        "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon",
        "w_2ndWon", "l_2ndWon", "w_SvGms", "l_SvGms",
        "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
    ]:
        output[column] = pd.NA

    output = output.dropna(
        subset=["tourney_date", "winner_name", "loser_name"]
    )
    output = output[
        (output["winner_name"] != "")
        & (output["loser_name"] != "")
        & (output["winner_name"].str.lower() != "nan")
        & (output["loser_name"].str.lower() != "nan")
    ]
    return output


def download_year(year: int) -> pd.DataFrame:
    url = SOURCE_TEMPLATE.format(year=year)
    print(f"Downloading ATP {year}: {url}")

    response = requests.get(
        url,
        timeout=180,
        headers={"User-Agent": "Macabets personal tennis analytics"},
    )
    response.raise_for_status()

    if len(response.content) < 1000:
        raise RuntimeError(f"Downloaded workbook was unexpectedly small: {url}")

    workbook = pd.ExcelFile(BytesIO(response.content))
    sheet = str(year) if str(year) in workbook.sheet_names else workbook.sheet_names[0]
    raw = pd.read_excel(workbook, sheet_name=sheet)
    return convert_year(raw, year)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for year in range(START_YEAR, date.today().year + 1):
        try:
            converted = download_year(year)
            destination = DATA_DIR / f"atp_matches_{year}.csv"
            converted.to_csv(destination, index=False)
            print(f"Saved {destination.name}: {len(converted):,} matches")
        except Exception as exc:
            failures.append(f"{year}: {exc}")

    if failures:
        print("\nSome seasons failed:")
        for failure in failures:
            print(failure)
        return 1

    print("\nMacabets ATP database update completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
