from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import requests


BASE_URL = "https://github.com/JeffSackmann/tennis_atp/raw/refs/heads/master"
DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2021


def download_file(url: str, destination: Path) -> None:
    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent": "Macabets-Tennis-Analytics/1.0",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    response.raise_for_status()

    if len(response.content) < 500:
        raise RuntimeError(
            f"Downloaded file was unexpectedly small: {url}"
        )

    destination.write_bytes(response.content)
    print(
        f"Saved {destination.name}: "
        f"{len(response.content):,} bytes"
    )


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_year = date.today().year
    failures: list[str] = []

    for year in range(START_YEAR, current_year + 1):
        filename = f"atp_matches_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        destination = DATA_DIR / filename

        try:
            print(f"Downloading {url}")
            download_file(url, destination)
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    if failures:
        print("\nSome downloads failed:")
        for failure in failures:
            print(failure)
        return 1

    print("\nATP database update completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
