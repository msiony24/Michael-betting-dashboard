"""Refresh Madden 27 from the authoritative launch-ratings workbook and rebuild Macabets ratings."""
from __future__ import annotations

from pathlib import Path

from engine.madden_launch_ratings_importer import DEFAULT_WORKBOOK, import_launch_ratings
from engine.madden_team_builder import build_and_save_team_ratings
from engine.nfl_rating_engine import build_and_save_ratings
from engine.madden_personnel_audit import build_personnel_audit


def main() -> None:
    print("\n=== Macabets Madden NFL 27 Audited Refresh ===\n")
    if not Path(DEFAULT_WORKBOOK).exists():
        raise RuntimeError(
            "Authoritative Madden 27 workbook is missing: data/madden_27_launch_ratings.xlsx. "
            "Macabets will not fall back to the stale EA endpoint automatically."
        )
    players = import_launch_ratings()
    print(f"Imported authoritative Madden 27 launch ratings: {len(players)} players / {players['team'].nunique()} teams")
    teams = build_and_save_team_ratings()
    print(f"Built raw Madden team-unit ratings: {len(teams)} teams")
    status = build_and_save_ratings()
    print(f"Built audited Macabets player ratings: {status['players_rated']} players / {status['teams_rated']} teams")
    audit = build_personnel_audit()
    print("Personnel audit:", audit["summary"])
    print("\nThe stale EA ratings endpoint is intentionally bypassed while this workbook is the source of truth.")


if __name__ == "__main__":
    main()
