"""Update the local Madden NFL 26 database."""

from engine.madden_ratings_loader import download_and_save_madden_ratings
from engine.madden_team_builder import build_and_save_team_ratings


def main():
    print("\n=== Macabets Madden NFL Update ===\n")
    players = download_and_save_madden_ratings()
    resolved = players[players["team"].astype(str).str.strip().ne("") & players["position"].astype(str).str.strip().ne("")]
    print(f"Downloaded {len(players)} player records.")
    print(f"Matched {len(resolved)} players to a team and position.")
    if len(resolved) == 0:
        raise RuntimeError(
            "Madden ratings downloaded, but no players could be matched to the NFL roster. "
            "Run Update Macabets NFL Data first so data/nfl/weekly_rosters.csv exists."
        )

    team_ratings = build_and_save_team_ratings()
    print(f"Built ratings for {len(team_ratings)} NFL teams.")
    print("\nFiles created:")
    print("  data/madden_26_players.csv")
    print("  data/madden_26_raw.json")
    print("  data/madden_26_metadata.json")
    print("  data/madden_26_team_ratings.json")
    print("\nUpdate complete.")


if __name__ == "__main__":
    main()
