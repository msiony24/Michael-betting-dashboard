"""Refresh the automated Macabets NFL data foundation from nflverse.

Usage:
    python update_nfl_data.py
    python update_nfl_data.py --season 2026
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from engine.nfl_foundation import refresh_nfl_foundation


MIN_SUPPORTED_SEASON = 1999


def default_season() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=default_season())
    parser.add_argument("--data-dir", default="data/nfl")
    args = parser.parse_args()

    if args.season < MIN_SUPPORTED_SEASON:
        raise ValueError(f"NFL season must be {MIN_SUPPORTED_SEASON} or later.")

    result = refresh_nfl_foundation(args.season, data_dir=Path(args.data_dir))
    print(
        f"NFL foundation updated at {result.updated_at_utc}: "
        f"{result.available_count}/{len(result.datasets)} datasets available."
    )
    if result.performance_season != result.requested_season:
        print(
            f"Play-by-play for {result.requested_season} was not available; "
            f"team performance ratings use {result.performance_season}."
        )
    for item in result.datasets:
        state = f"{item.rows} rows" if item.available else f"unavailable: {item.error}"
        print(f"- {item.name}: {state}")


if __name__ == "__main__":
    main()
