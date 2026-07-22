
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import requests


BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2021


def download(url: str, destination: Path) -> None:
    response = requests.get(
        url,
        timeout=90,
        headers={"User-Agent": "Macabets personal analytics"},
    )
    response.raise_for_status()
    if len(response.content) < 500:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")
    destination.write_bytes(response.content)
    print(f"Saved {destination.name}: {len(response.content):,} bytes")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_year = date.today().year

    failed: list[str] = []
    for year in range(START_YEAR, current_year + 1):
        filename = f"atp_matches_{year}.csv"
        try:
            download(f"{BASE}/{filename}", DATA_DIR / filename)
        except Exception as exc:
            failed.append(f"{filename}: {exc}")

    if failed:
        print("\nSome downloads failed:")
        for item in failed:
            print(item)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
