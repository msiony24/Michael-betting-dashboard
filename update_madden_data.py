"""
Update Madden NFL 27 ratings, enrich them with current NFL roster identity,
then rebuild Macabets team-unit ratings.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from engine.madden_ratings_loader import (
    DEFAULT_METADATA_PATH,
    download_and_save_madden_ratings,
)
from engine.madden_roster_join import enrich_and_save_madden_players
from engine.madden_team_builder import build_and_save_team_ratings


def main():
    print("\n=== Macabets Madden NFL 27 Update ===\n")

    ea_players = download_and_save_madden_ratings()
    print(f"EA Madden records normalized: {len(ea_players)}")

    season = date.today().year
    enriched, join_report = enrich_and_save_madden_players(season=season)
    print("\n=== NFL roster identity join ===")
    print(json.dumps(join_report, indent=2))

    team_ratings = build_and_save_team_ratings()
    print(f"\nBuilt Madden unit ratings for {len(team_ratings)} NFL teams.")

    # Add roster-join metrics to metadata.
    metadata_path = Path(DEFAULT_METADATA_PATH)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata["roster_identity_source"] = "nflverse rosters"
    metadata["roster_season"] = season
    metadata["roster_join"] = join_report
    temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp.replace(metadata_path)

    print("\nFiles created/updated:")
    print("  data/madden_27_players.csv       <- final enriched Madden database")
    print("  data/madden_27_raw.json")
    print("  data/madden_27_metadata.json")
    print("  data/madden_27_team_ratings.json")
    print(f"  data/nfl/roster_{season}.csv")
    print("\nPrediction influence remains OFF until roster validation is complete.")


if __name__ == "__main__":
    main()
