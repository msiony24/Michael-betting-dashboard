"""Refresh Macabets NFL injury/availability data from Sleeper."""
from engine.nfl_availability import refresh_sleeper_availability


def main() -> None:
    status = refresh_sleeper_availability()
    print("\n=== Macabets NFL Availability ===")
    print(f"Source: {status['source']}")
    print(f"Players: {status['players']}")
    print(f"Teams: {status['teams']}")
    print(f"Definitively unavailable: {status['definitively_unavailable']}")
    print(f"Questionable/Doubtful: {status['uncertain']}")
    print(f"Updated: {status['updated_at_utc']}")


if __name__ == "__main__":
    main()
