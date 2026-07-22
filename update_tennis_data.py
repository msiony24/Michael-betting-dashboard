from __future__ import annotations

from datetime import date
from pathlib import Path

import requests


API_BASE = "https://api.github.com/repos/JeffSackmann/tennis_atp/contents"
DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2021


def download_file(filename: str, destination: Path) -> None:
    url = f"{API_BASE}/{filename}"

    response = requests.get(
        url,
        params={"ref": "master"},
        timeout=120,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Macabets-Tennis-Analytics",
        },
    )

    response.raise_for_status()

    if len(response.content) < 500:
        raise RuntimeError(
            f"Downloaded file was unexpectedly small: {filename}"
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
        destination = DATA_DIR / filename

        try:
            print(f"Downloading {filename}")
            download_file(filename, destination)
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
