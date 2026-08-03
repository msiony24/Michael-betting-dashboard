"""
Download one raw page from EA's Madden ratings API for schema inspection.

This script does not parse or transform the response.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path


EA_API_BASE = "https://drop-api.ea.com/rating/madden-nfl"
OUTPUT_PATH = Path("madden_api_sample.json")


def main() -> None:
    query = urllib.parse.urlencode({"limit": 5, "offset": 0})
    request = urllib.request.Request(
        f"{EA_API_BASE}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 Macabets/0.61",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.ea.com/games/madden-nfl/ratings",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Saved raw EA response to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
