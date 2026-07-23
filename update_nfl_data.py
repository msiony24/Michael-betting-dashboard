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


def default_season() -> int:
    today = date.today()
    # NFL league-year convention: after March, use the current calendar year.
    return today.year if today.month >= 3 else today.year - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=default_season())
    parser.add_argument("--output", default="data/nfl/team_snapshot.csv")
    args = parser.parse_args()
    result = fetch_and_build(args.season, Path(args.output))
    print(
        f"Saved {result.rows} NFL team rows for {result.season} to {result.output_path} "
        f"at {result.fetched_at_utc}."
    )


if __name__ == "__main__":
    main()
