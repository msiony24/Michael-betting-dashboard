from __future__ import annotations

import json
import sys

from engine.nfl_player_weekly import refresh_player_weekly_stats


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 and str(sys.argv[1]).strip() else None
    result = refresh_player_weekly_stats(season)
    print("\n=== NFL Player Weekly Stats ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
