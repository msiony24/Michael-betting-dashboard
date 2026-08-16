from __future__ import annotations

from datetime import date, timedelta
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .result_settlement import event_participants_match, names_compatible


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
API_TENNIS_BASE = "https://api.api-tennis.com/tennis/"


class ProviderError(RuntimeError):
    pass


def _get_json(url: str, *, timeout: int = 30) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Macabets result settlement/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Provider returned HTTP {exc.code}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach provider: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider returned invalid JSON") from exc


class OddsApiClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or os.environ.get("THE_ODDS_API_KEY", "")).strip()
        if not self.api_key:
            raise ProviderError("THE_ODDS_API_KEY is not configured")

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        query = {"apiKey": self.api_key, **(params or {})}
        return f"{ODDS_API_BASE}{path}?{urllib.parse.urlencode(query)}"

    def sports(self) -> list[dict[str, Any]]:
        payload = _get_json(self._url("/sports", {"all": "true"}))
        return payload if isinstance(payload, list) else []

    def events(self, sport_key: str) -> list[dict[str, Any]]:
        payload = _get_json(
            self._url(
                f"/sports/{urllib.parse.quote(str(sport_key), safe='')}/events",
                {"dateFormat": "iso"},
            )
        )
        return payload if isinstance(payload, list) else []

    def event_odds(
        self,
        sport_key: str,
        event_id: str,
        *,
        markets: str = "h2h",
    ) -> dict[str, Any]:
        payload = _get_json(
            self._url(
                f"/sports/{urllib.parse.quote(str(sport_key), safe='')}/events/"
                f"{urllib.parse.quote(str(event_id), safe='')}/odds",
                {
                    "regions": "us",
                    "markets": markets,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
        )
        return payload if isinstance(payload, dict) else {}

    def scores(self, sport_key: str, *, days_from: int = 3) -> list[dict[str, Any]]:
        payload = _get_json(
            self._url(
                f"/sports/{urllib.parse.quote(str(sport_key), safe='')}/scores/",
                {"daysFrom": max(1, min(int(days_from), 3)), "dateFormat": "iso"},
            )
        )
        return payload if isinstance(payload, list) else []

    def tennis_sport_keys(self) -> list[str]:
        keys: list[str] = []
        for item in self.sports():
            if not item.get("active", True):
                continue
            searchable = " ".join(
                str(item.get(field, ""))
                for field in ("key", "group", "title", "description")
            ).casefold()
            key = str(item.get("key", "")).strip()
            if key and "tennis" in searchable and ("atp" in searchable or "wta" in searchable):
                keys.append(key)
        return sorted(set(keys))

    def find_event(
        self,
        *,
        sport_key: str,
        participant_a: str,
        participant_b: str,
    ) -> dict[str, Any] | None:
        matches = []
        for event in self.events(sport_key):
            if event_participants_match(
                participant_a,
                participant_b,
                event.get("away_team"),
                event.get("home_team"),
            ):
                matches.append(event)
        return matches[0] if len(matches) == 1 else None

    def find_tennis_event(
        self,
        *,
        participant_a: str,
        participant_b: str,
    ) -> tuple[str, dict[str, Any]] | None:
        matches: list[tuple[str, dict[str, Any]]] = []
        for sport_key in self.tennis_sport_keys():
            for event in self.events(sport_key):
                if event_participants_match(
                    participant_a,
                    participant_b,
                    event.get("away_team"),
                    event.get("home_team"),
                ):
                    matches.append((sport_key, event))
        return matches[0] if len(matches) == 1 else None


class APITennisSettlementClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = (
            api_key
            or os.environ.get("API_TENNIS_KEY", "")
            or os.environ.get("API_TENNIS_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ProviderError("API_TENNIS_KEY is not configured")

    def _request(self, method: str, **params: Any) -> Any:
        query = urllib.parse.urlencode(
            {"method": method, "APIkey": self.api_key, **params}
        )
        payload = _get_json(f"{API_TENNIS_BASE}?{query}")
        if not isinstance(payload, dict) or int(payload.get("success", 0)) != 1:
            detail = payload.get("error") if isinstance(payload, dict) else "invalid payload"
            raise ProviderError(f"API Tennis {method} failed: {detail}")
        return payload.get("result")

    def fixtures(
        self,
        date_start: date,
        date_stop: date,
        *,
        timezone_name: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        result = self._request(
            "get_fixtures",
            date_start=date_start.isoformat(),
            date_stop=date_stop.isoformat(),
            timezone=timezone_name,
        )
        return result if isinstance(result, list) else []

    def fixture_by_event_id(
        self,
        event_id: str,
        *,
        event_date: date,
    ) -> dict[str, Any] | None:
        # The documented fixture method is date-based, so fetch a tight window and
        # then use event_key as the stable identity.
        for row in self.fixtures(event_date - timedelta(days=1), event_date + timedelta(days=1)):
            if str(row.get("event_key", "")) == str(event_id):
                return row
        return None

    def find_fixture(
        self,
        *,
        participant_a: str,
        participant_b: str,
        event_date: date,
        tournament: str = "",
    ) -> dict[str, Any] | None:
        candidates: list[tuple[int, dict[str, Any]]] = []
        for row in self.fixtures(event_date - timedelta(days=1), event_date + timedelta(days=1)):
            if not event_participants_match(
                participant_a,
                participant_b,
                row.get("event_first_player"),
                row.get("event_second_player"),
            ):
                continue
            score = 10
            provider_tournament = str(row.get("tournament_name") or "")
            if tournament and provider_tournament:
                # Tournament match is supporting evidence only. Player identity is
                # still required, and ambiguity is rejected below.
                if tournament.casefold() in provider_tournament.casefold() or provider_tournament.casefold() in tournament.casefold():
                    score += 2
            if str(row.get("event_date") or "") == event_date.isoformat():
                score += 1
            candidates.append((score, row))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        return candidates[0][1]

    @staticmethod
    def actual_winner(fixture: dict[str, Any]) -> str | None:
        winner = str(fixture.get("event_winner") or "").strip().casefold()
        if winner == "first player":
            return str(fixture.get("event_first_player") or "").strip() or None
        if winner == "second player":
            return str(fixture.get("event_second_player") or "").strip() or None
        # Some provider payloads may return the actual name rather than side label.
        first = str(fixture.get("event_first_player") or "").strip()
        second = str(fixture.get("event_second_player") or "").strip()
        raw = str(fixture.get("event_winner") or "").strip()
        if raw and (names_compatible(raw, first) or names_compatible(raw, second)):
            return first if names_compatible(raw, first) else second
        return None
