from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


UFCSTATS_BASE = "http://ufcstats.com"
COMPLETED_EVENTS_URL = f"{UFCSTATS_BASE}/statistics/events/completed?page=all"
USER_AGENT = "Macabets UFC analytics (personal research)"

DIVISION_LIMITS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}


class UFCStatsError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchConfig:
    timeout_seconds: int = 30
    attempts: int = 4
    pause_seconds: float = 0.25


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _to_int(value: object) -> int | None:
    text = _clean_text(value)
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def _parse_date(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def _get(session: requests.Session, url: str, config: FetchConfig) -> str:
    error: Exception | None = None
    for attempt in range(1, config.attempts + 1):
        try:
            response = session.get(
                url,
                timeout=config.timeout_seconds,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            if len(response.text) < 100:
                raise UFCStatsError(f"UFCStats returned an unexpectedly small page: {url}")
            return response.text
        except (requests.RequestException, UFCStatsError) as exc:
            error = exc
            if attempt < config.attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise UFCStatsError(f"Unable to fetch UFCStats after {config.attempts} attempts: {url}") from error


def parse_completed_events(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []

    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        if not link:
            continue
        event_name = _clean_text(link.get_text(" ", strip=True))
        event_url = _clean_text(link.get("href"))
        if not event_name or not event_url:
            continue

        date_node = row.select_one("span.b-statistics__date")
        cells = row.select("td.b-statistics__table-col")
        location = _clean_text(cells[-1].get_text(" ", strip=True)) if cells else ""
        event_date = _parse_date(date_node.get_text(" ", strip=True) if date_node else "")
        rows.append(
            {
                "event_name": event_name,
                "event_date": event_date,
                "location": location,
                "event_url": urljoin(UFCSTATS_BASE, event_url),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["event_name", "event_date", "location", "event_url"])
    frame = frame.drop_duplicates(subset=["event_url"]).sort_values("event_date").reset_index(drop=True)
    return frame


def parse_event_fights(html: str, event: dict[str, object]) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []

    fight_rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
    if not fight_rows:
        fight_rows = [r for r in soup.select("tr.b-fight-details__table-row") if r.get("data-link")]

    for row in fight_rows:
        fight_url = _clean_text(row.get("data-link"))
        cells = row.select("td.b-fight-details__table-col")
        if len(cells) < 10:
            continue

        wl = [_clean_text(p.get_text(" ", strip=True)) for p in cells[0].select("p")]
        fighters = [_clean_text(p.get_text(" ", strip=True)) for p in cells[1].select("p")]
        fighter_links = [urljoin(UFCSTATS_BASE, _clean_text(a.get("href"))) for a in cells[1].select("a")]
        kd = [_to_int(p.get_text(" ", strip=True)) for p in cells[2].select("p")]
        sig_str = [_to_int(p.get_text(" ", strip=True)) for p in cells[3].select("p")]
        td = [_to_int(p.get_text(" ", strip=True)) for p in cells[4].select("p")]
        sub = [_to_int(p.get_text(" ", strip=True)) for p in cells[5].select("p")]

        if len(fighters) != 2:
            continue

        method = _clean_text(cells[7].get_text(" ", strip=True))
        round_number = _to_int(cells[8].get_text(" ", strip=True))
        finish_time = _clean_text(cells[9].get_text(" ", strip=True))
        division = _clean_text(cells[6].get_text(" ", strip=True))

        # UFCStats lists the winner first for normal completed bouts. Preserve the
        # explicit W/L markers so draws/no-contests can be represented safely.
        for idx in (0, 1):
            opponent_idx = 1 - idx
            result = wl[idx].upper() if idx < len(wl) else ""
            rows.append(
                {
                    "event_name": _clean_text(event.get("event_name")),
                    "event_date": _clean_text(event.get("event_date")),
                    "location": _clean_text(event.get("location")),
                    "fight_url": urljoin(UFCSTATS_BASE, fight_url),
                    "fighter": fighters[idx],
                    "fighter_url": fighter_links[idx] if idx < len(fighter_links) else "",
                    "opponent": fighters[opponent_idx],
                    "opponent_url": fighter_links[opponent_idx] if opponent_idx < len(fighter_links) else "",
                    "result": result,
                    "division": division,
                    "kd": kd[idx] if idx < len(kd) else None,
                    "sig_str": sig_str[idx] if idx < len(sig_str) else None,
                    "td": td[idx] if idx < len(td) else None,
                    "sub_att": sub[idx] if idx < len(sub) else None,
                    "method": method,
                    "round": round_number,
                    "time": finish_time,
                }
            )

    return pd.DataFrame(rows)


def normalize_division(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return "Unknown"

    lower = text.casefold()
    if "catch weight" in lower or "catchweight" in lower:
        return "Catch Weight"

    gender = "Women’s" if "women" in lower else "Men’s"
    for name in DIVISION_LIMITS:
        if name.casefold() in lower:
            if name == "Strawweight":
                gender = "Women’s" if "women" in lower else "Men’s"
            return f"{gender} {name}"
    return text


def fetch_completed_events(
    session: requests.Session | None = None,
    config: FetchConfig | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    config = config or FetchConfig()
    html = _get(session, COMPLETED_EVENTS_URL, config)
    return parse_completed_events(html)


def fetch_fight_history(
    events: pd.DataFrame,
    *,
    since: date | None = None,
    session: requests.Session | None = None,
    config: FetchConfig | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    config = config or FetchConfig()
    frames: list[pd.DataFrame] = []

    working = events.copy()
    if since is not None and not working.empty:
        dates = pd.to_datetime(working["event_date"], errors="coerce").dt.date
        working = working.loc[dates >= since]

    for _, event in working.iterrows():
        html = _get(session, str(event["event_url"]), config)
        frame = parse_event_fights(html, event.to_dict())
        if not frame.empty:
            frames.append(frame)
        if config.pause_seconds > 0:
            time.sleep(config.pause_seconds)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    output["division"] = output["division"].map(normalize_division)
    output = output.drop_duplicates(subset=["fight_url", "fighter"], keep="last")
    return output.sort_values(["event_date", "fight_url", "fighter"]).reset_index(drop=True)


def source_status(fights: pd.DataFrame) -> dict[str, object]:
    latest = ""
    if fights is not None and not fights.empty and "event_date" in fights:
        parsed = pd.to_datetime(fights["event_date"], errors="coerce")
        if parsed.notna().any():
            latest = parsed.max().date().isoformat()
    return {
        "source": "UFCStats",
        "source_url": COMPLETED_EVENTS_URL,
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_event_date": latest,
        "fighter_fight_rows": int(len(fights)) if fights is not None else 0,
        "unique_fights": int(fights["fight_url"].nunique()) if fights is not None and not fights.empty else 0,
        "unique_fighters": int(fights["fighter"].nunique()) if fights is not None and not fights.empty else 0,
    }
