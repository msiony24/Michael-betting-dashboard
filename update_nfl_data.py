"""Refresh Macabets NFL performance data from nflverse.

Usage:
    python update_nfl_data.py
    python update_nfl_data.py --season 2025
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from engine.nfl_fetch import fetch_and_build


MIN_SUPPORTED_SEASON = 1999


def default_season() -> int:
    today = date.today()
    # NFL league-year convention: after March, use the current calendar year.
    return today.year if today.month >= 3 else today.year - 1


def fetch_with_latest_available_season(requested_season: int, output_path: Path):
    """Use the requested season when available, otherwise fall back automatically.

    nflverse/nflreadpy can lag the current league year before play-by-play data is
    published. When that specific availability error occurs, try earlier seasons
    one at a time. Other errors still fail normally so genuine data or code issues
    are not hidden.
    """
    requested_season = int(requested_season)
    if requested_season < MIN_SUPPORTED_SEASON:
        raise ValueError(
            f"NFL season must be {MIN_SUPPORTED_SEASON} or later; received {requested_season}."
        )

    for season in range(requested_season, MIN_SUPPORTED_SEASON - 1, -1):
        try:
            result = fetch_and_build(season, output_path)
            if season != requested_season:
                print(
                    f"Requested NFL season {requested_season} is not available yet. "
                    f"Automatically used the latest available season: {season}."
                )
            return result
        except ValueError as exc:
            message = str(exc)
            unavailable_season_error = (
                "Season must be between" in message
                or "season must be between" in message.lower()
            )
            if unavailable_season_error and season > MIN_SUPPORTED_SEASON:
                print(f"NFL season {season} is unavailable; trying {season - 1}.")
                continue
            raise

    raise RuntimeError("No supported NFL season could be loaded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=default_season())
    parser.add_argument("--output", default="data/nfl/team_snapshot.csv")
    args = parser.parse_args()

    result = fetch_with_latest_available_season(args.season, Path(args.output))
    print(
        f"Saved {result.rows} NFL team rows for {result.season} to {result.output_path} "
        f"at {result.fetched_at_utc}."
    )


if __name__ == "__main__":
    main()
