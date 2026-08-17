from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from engine.ufc_data import (
    FetchConfig,
    UFCStatsError,
    fetch_completed_events,
    fetch_fight_history,
    normalize_division,
    source_status,
)
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

# Public, tracked UFCStats seed maintained by the mma-ai project. This is only a
# bootstrap/fallback source when UFCStats refuses traffic from GitHub-hosted
# runners. Live UFCStats remains the preferred source whenever it is reachable.
MIRROR_COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/DanMcInerney/mma-ai/"
    "main/data/raw/ufcstats/competitions.csv"
)


def _write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_cached_fights() -> pd.DataFrame:
    if not FIGHTS_PATH.exists() or FIGHTS_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(FIGHTS_PATH)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()
    return frame if not frame.empty else pd.DataFrame()


def _first_int(value: object) -> int:
    text = "" if value is None else str(value)
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else 0


def _sum_round_stat(row: pd.Series, fighter_prefix: str, stat: str) -> int:
    total = 0
    for round_no in range(1, 6):
        column = f"{fighter_prefix}_rd{round_no}_{stat}"
        if column in row.index:
            total += _first_int(row.get(column))
    return total


def _opposite_result(result: str) -> str:
    result = result.strip().upper()
    if result == "W":
        return "L"
    if result == "L":
        return "W"
    return result


def _synthetic_fight_id(event_url: str, player1: str, player2: str) -> str:
    key = f"{event_url}|{player1}|{player2}".encode("utf-8")
    return "mirror:" + hashlib.sha1(key).hexdigest()


def _load_github_mirror_history() -> pd.DataFrame:
    """Build Macabets' fighter-fight grain from the public UFCStats seed mirror."""
    print("Trying GitHub-hosted UFCStats mirror bootstrap...")
    raw = pd.read_csv(MIRROR_COMPETITIONS_URL, low_memory=False)
    if raw.empty:
        return pd.DataFrame()

    required = {"player1", "player2", "event_date", "weightclass"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(
            "UFCStats mirror schema changed; missing columns: " + ", ".join(sorted(missing))
        )

    parsed_dates = pd.to_datetime(raw["event_date"], errors="coerce")
    raw = raw.loc[parsed_dates.dt.date >= DEFAULT_SINCE].copy()
    if raw.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, fight in raw.iterrows():
        player1 = str(fight.get("player1", "") or "").strip()
        player2 = str(fight.get("player2", "") or "").strip()
        if not player1 or not player2:
            continue

        event_url = str(fight.get("event_url", "") or "").strip()
        fight_id = _synthetic_fight_id(event_url, player1, player2)
        result1 = str(fight.get("result", "") or "").strip().upper()
        result2 = _opposite_result(result1)
        event_date = pd.to_datetime(fight.get("event_date"), errors="coerce")
        event_date_text = event_date.date().isoformat() if pd.notna(event_date) else ""
        division = normalize_division(fight.get("weightclass", ""))

        common = {
            "event_name": str(fight.get("event_name", "") or "").strip(),
            "event_date": event_date_text,
            "location": str(fight.get("event_location", "") or "").strip(),
            "fight_url": fight_id,
            "division": division,
            "method": str(fight.get("method", "") or "").strip(),
            "round": _first_int(fight.get("round")),
            "time": str(fight.get("time", "") or "").strip(),
        }

        rows.append(
            {
                **common,
                "fighter": player1,
                "fighter_url": str(fight.get("player1_url", "") or "").strip(),
                "opponent": player2,
                "opponent_url": str(fight.get("player2_url", "") or "").strip(),
                "result": result1,
                "kd": _sum_round_stat(fight, "p1", "KD"),
                "sig_str": _sum_round_stat(fight, "p1", "Sig_str"),
                "td": _sum_round_stat(fight, "p1", "Td"),
                "sub_att": _sum_round_stat(fight, "p1", "Sub_att"),
            }
        )
        rows.append(
            {
                **common,
                "fighter": player2,
                "fighter_url": str(fight.get("player2_url", "") or "").strip(),
                "opponent": player1,
                "opponent_url": str(fight.get("player1_url", "") or "").strip(),
                "result": result2,
                "kd": _sum_round_stat(fight, "p2", "KD"),
                "sig_str": _sum_round_stat(fight, "p2", "Sig_str"),
                "td": _sum_round_stat(fight, "p2", "Td"),
                "sub_att": _sum_round_stat(fight, "p2", "Sub_att"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.drop_duplicates(subset=["fight_url", "fighter"], keep="last")
        .sort_values(["event_date", "fight_url", "fighter"])
        .reset_index(drop=True)
    )


def _get_fight_history(config: FetchConfig) -> tuple[pd.DataFrame, str, str]:
    """Return fights, data mode, and any live-source error message."""
    live_error = ""
    try:
        print("Fetching completed UFC events from UFCStats...")
        events = fetch_completed_events(config=config)
        if events.empty:
            raise UFCStatsError("UFCStats returned no completed events.")

        print(
            f"Found {len(events):,} completed UFC events. "
            f"Building fight history since {DEFAULT_SINCE}..."
        )
        fights = fetch_fight_history(events, since=DEFAULT_SINCE, config=config)
        if fights.empty:
            raise UFCStatsError("UFCStats returned no fight history.")
        return fights, "fresh", ""
    except Exception as exc:
        live_error = f"{type(exc).__name__}: {exc}"
        print(f"Live UFCStats refresh unavailable: {live_error}")

    cached = _load_cached_fights()
    if not cached.empty:
        print(
            f"Using last-good cached UFC dataset ({len(cached):,} fighter-fight rows) "
            "instead of failing the workflow."
        )
        return cached, "cached_fallback", live_error

    try:
        mirror = _load_github_mirror_history()
        if not mirror.empty:
            print(
                f"Bootstrapped {len(mirror):,} fighter-fight rows from the "
                "GitHub-hosted UFCStats mirror."
            )
            return mirror, "mirror_bootstrap", live_error
    except Exception as exc:
        mirror_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "Live UFCStats failed, no last-good cache exists, and the GitHub mirror "
            f"bootstrap also failed. Live error: {live_error}. Mirror error: {mirror_error}"
        ) from exc

    raise RuntimeError(
        "Live UFCStats failed and neither a last-good cache nor mirror bootstrap was available. "
        f"Live error: {live_error}"
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = FetchConfig()

    fights, data_mode, live_error = _get_fight_history(config)

    print("Building opponent-adjusted Macabets fighter ratings...")
    ratings = build_fighter_ratings(fights)
    if ratings.empty:
        raise RuntimeError("UFC rating build produced no fighters.")

    # Fresh and mirror-bootstrap runs establish/replace the last-good snapshot.
    # Cached fallback simply re-writes the same validated snapshot atomically.
    _write_atomic_csv(fights, FIGHTS_PATH)
    _write_atomic_csv(ratings, RATINGS_PATH)

    status = source_status(fights)
    status.update(
        {
            "rating_model": "Macabets UFC Strength v0.1",
            "history_start": DEFAULT_SINCE.isoformat(),
            "rated_fighters": int(len(ratings)),
            "active_pool_fighters": int(ratings["active_pool"].sum()),
            "data_mode": data_mode,
            "live_source_error": live_error,
            "notes": (
                "UFCStats results + opponent-adjusted Elo backbone. Live UFCStats is preferred; "
                "last-good cache is preserved when the site refuses GitHub-hosted runners, with a "
                "public GitHub UFCStats mirror used only to bootstrap an empty repository."
            ),
        }
    )
    if data_mode == "mirror_bootstrap":
        status["source"] = "UFCStats mirror (GitHub seed)"
        status["source_url"] = MIRROR_COMPETITIONS_URL
    elif data_mode == "cached_fallback":
        status["source"] = "UFCStats last-good cache"

    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Data mode: {data_mode}")
    print(f"Saved {len(fights):,} fighter-fight rows -> {FIGHTS_PATH}")
    print(f"Saved {len(ratings):,} fighter ratings -> {RATINGS_PATH}")
    print("Top active fighters by Macabets rating:")
    print(
        ratings.loc[
            ratings["active_pool"],
            ["fighter", "division", "division_rank", "strength_score", "macabets_rating"],
        ]
        .sort_values("macabets_rating", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
