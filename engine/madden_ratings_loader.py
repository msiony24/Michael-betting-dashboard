import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EA_API_BASE = "https://drop-api.ea.com/rating/madden-nfl"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "madden_27_players.json"

def fetch_madden_players(limit=100, offset=0):
    params = urllib.parse.urlencode({
        "limit": limit,
        "offset": offset,
    })

    request = urllib.request.Request(
        f"{EA_API_BASE}?{params}",
        headers={
            "User-Agent": "Macabets/0.59",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"EA ratings API returned HTTP {exc.code}: {detail[:300]}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to the EA ratings API: {exc.reason}"
        ) from exc

def inspect_first_page():
    payload = fetch_madden_players(limit=5, offset=0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("EA data downloaded successfully.")
    print(f"Saved test response to: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_first_page()
