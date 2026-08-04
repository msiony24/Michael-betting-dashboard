"""Build Macabets automated NFL player, unit, and team ratings."""
from engine.nfl_rating_engine import build_and_save_ratings


def main() -> None:
    status = build_and_save_ratings()
    print("\n=== Macabets NFL Rating Engine ===")
    print(f"Players rated: {status['players_rated']}")
    print(f"Teams rated: {status['teams_rated']}")
    print(f"Players with nflverse performance: {status['players_with_performance_data']}")
    print("Prediction influence remains OFF until validation is complete.")


if __name__ == "__main__":
    main()
