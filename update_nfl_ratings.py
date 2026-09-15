"""Build Macabets automated NFL player, unit, and team ratings."""
from engine.nfl_rating_engine import build_and_save_ratings


def main() -> None:
    status = build_and_save_ratings()
    print("\n=== Macabets NFL Rating Engine ===")
    print(f"Players rated: {status['players_rated']}")
    print(f"Teams rated: {status['teams_rated']}")
    print(f"Players with nflverse performance: {status['players_with_performance_data']}")
    fallback = status.get("fallback_players", {})
    print(f"Depth-chart players added from prior-year Madden: {fallback.get('prior_year_madden', 0)}")
    print(f"Depth-chart players added at replacement level: {fallback.get('replacement_level', 0)}")
    print(f"Depth-chart players rated manually: {fallback.get('manual', 0)}")
    manual_file = fallback.get("manual_file") or {}
    if not manual_file.get("found"):
        print(f"  Manual ratings file not found at {manual_file.get('path', 'data/nfl/manual_fallback_ratings.csv')}")
    else:
        entries = manual_file.get("entries") or []
        print(f"  Manual ratings file: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} read")
        for entry in entries:
            print(f"    {entry['player_name']} ({entry['team']}) {entry['overall']:g}: {entry.get('status', 'not checked')}")
        for problem in manual_file.get("problems") or []:
            print(f"    Problem: {problem}")
    for label, key in (("Prior-year Madden", "prior_year_madden_players"),
                       ("Replacement level", "replacement_level_players"),
                       ("Manual", "manual_players")):
        names = fallback.get(key) or []
        if names:
            print(f"  {label}: " + "; ".join(names))
    print(f"Rating model: {status.get('engine_version')} (these ratings feed NFL predictions).")


if __name__ == "__main__":
    main()
