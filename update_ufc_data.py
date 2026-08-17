from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import shutil

import pandas as pd

from engine.ufc_data import FetchConfig, fetch_completed_events, fetch_fight_history, source_status
from engine.ufc_ratings import build_fighter_ratings


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ufc"
FIGHTS_PATH = DATA_DIR / "ufc_fight_history.csv"
RATINGS_PATH = DATA_DIR / "fighter_ratings.csv"
STATUS_PATH = DATA_DIR / "refresh_status.json"

# A long history is important for recursive opponent quality. UFCStats has event
# results back to the early UFC era, but starting in 2010 keeps refresh time sane
# while covering the modern roster and the opponents who shape current ratings.
DEFAULT_SINCE = date(2010, 1, 1)


def _write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = FetchConfig()

    print("Fetching completed UFC events from UFCStats...")
    events = fetch_completed_events(config=config)
    if events.empty:
        raise RuntimeError("UFCStats returned no completed events.")

    print(f"Found {len(events):,} completed UFC events. Building fight history since {DEFAULT_SINCE}...")
    fights = fetch_fight_history(events, since=DEFAULT_SINCE, config=config)
    if fights.empty:
        raise RuntimeError("UFCStats returned no fight history.")

    print("Building opponent-adjusted Macabets fighter ratings...")
    ratings = build_fighter_ratings(fights)
    if ratings.empty:
        raise RuntimeError("UFC rating build produced no fighters.")

    _write_atomic_csv(fights, FIGHTS_PATH)
    _write_atomic_csv(ratings, RATINGS_PATH)

    status = source_status(fights)
    status.update(
        {
            "rating_model": "Macabets UFC Strength v0.1",
            "history_start": DEFAULT_SINCE.isoformat(),
            "rated_fighters": int(len(ratings)),
            "active_pool_fighters": int(ratings["active_pool"].sum()),
            "notes": "UFCStats results + opponent-adjusted Elo backbone. Detailed per-round performance and external benchmark calibration are the next layer.",
        }
    )
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved {len(fights):,} fighter-fight rows -> {FIGHTS_PATH}")
    print(f"Saved {len(ratings):,} fighter ratings -> {RATINGS_PATH}")
    print("Top active fighters by Macabets rating:")
    print(
        ratings.loc[ratings["active_pool"], ["fighter", "division", "division_rank", "strength_score", "macabets_rating"]]
        .sort_values("macabets_rating", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
