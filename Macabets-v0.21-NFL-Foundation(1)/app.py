
import io
import math
import json
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo
from datetime import date, datetime, time, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from engine.data import load_matches
    from engine.tennis import (
        analyze as analyze_tennis_match,
        player_names as tennis_player_names,
        tournament_names as tennis_tournament_names,
        tournament_surface as tennis_tournament_surface,
        tournament_category as tennis_tournament_category,
    )
    TENNIS_ENGINE_AVAILABLE = True
    TENNIS_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    TENNIS_ENGINE_AVAILABLE = False
    TENNIS_ENGINE_IMPORT_ERROR = str(exc)

try:
    from engine.nfl import analyze_nfl_matchup
    from engine.nfl_data import NFL_TEAMS
    NFL_ENGINE_AVAILABLE = True
    NFL_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    NFL_ENGINE_AVAILABLE = False
    NFL_ENGINE_IMPORT_ERROR = str(exc)

APP_VERSION = "Macabets v0.21 — NFL Foundation"
BUILD_DATE = "July 23, 2026"

st.set_page_config(
    page_title="Macabets",
    page_icon="📊",
    layout="wide",
)

SPORTS = ["NFL", "College Football", "NBA", "Tennis", "UFC", "Boxing"]
STATUSES = ["Pending", "Won", "Lost", "Void", "Cashed Out"]
BET_TYPES = ["Moneyline", "Spread", "Total", "Prop", "Parlay", "Live"]
ANALYSIS_COLUMNS = [
    "analysis_id", "created_at", "match_date", "tournament", "surface", "round",
    "player_a", "player_b", "market_odds_a", "model_probability_a", "fair_odds_a",
    "market_odds_b", "no_vig_probability_a", "no_vig_edge", "decision",
    "minimum_acceptable_odds_a", "estimated_roi", "confidence", "prediction",
    "upset_path", "biggest_risk", "assumptions", "notes", "result",
    "closing_odds_a", "prediction_correct",
    "closing_line_value", "review", "lesson"
]

DEFAULT_COLUMNS = [
    "id", "date", "sport", "event", "selection", "bet_type", "odds",
    "stake", "target_profit", "status", "result_profit", "book",
    "confidence", "notes"
]


SLATE_COLUMNS = [
    "slate_id", "match_date", "tournament", "surface", "round",
    "player_a", "player_b", "market_odds_a", "market_odds_b",
    "model_probability_a", "confidence", "notes"
]


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
EASTERN_TZ = ZoneInfo("America/New_York")


def _odds_api_key():
    """Read the API key safely from Streamlit secrets without exposing it."""
    try:
        return str(st.secrets.get("THE_ODDS_API_KEY", "")).strip()
    except Exception:
        return ""


def _api_get_json(path, params):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{ODDS_API_BASE}{path}?{query}",
        headers={"User-Agent": "Macabets/0.20"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Odds API returned HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the Odds API: {exc.reason}") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_active_sports(api_key):
    payload, headers = _api_get_json("/sports", {"apiKey": api_key, "all": "true"})
    return payload, {
        "remaining": headers.get("x-requests-remaining", "—"),
        "used": headers.get("x-requests-used", "—"),
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_sport_odds(api_key, sport_key):
    payload, headers = _api_get_json(
        f"/sports/{sport_key}/odds",
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )
    return payload, {
        "remaining": headers.get("x-requests-remaining", "—"),
        "used": headers.get("x-requests-used", "—"),
    }


def _best_h2h_prices(event):
    best = {}
    source = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", "")).strip()
                price = outcome.get("price")
                if not name or price is None:
                    continue
                try:
                    price = int(price)
                except (TypeError, ValueError):
                    continue
                # For American odds, the numerically larger price is always more favorable.
                if name not in best or price > best[name]:
                    best[name] = price
                    source[name] = bookmaker.get("title", bookmaker.get("key", "Sportsbook"))
    return best, source


def discover_active_tennis_sports(active_sports):
    """Return active ATP/WTA sport keys reported by The Odds API.

    The API can vary the tournament titles while keeping ATP/WTA in the key,
    description, group, or title. Checking all fields makes the Daily Slate
    resilient to those naming changes.
    """
    discovered = []
    seen_keys = set()
    for item in active_sports or []:
        key = str(item.get("key", "")).strip()
        if not key or key in seen_keys or not item.get("active", True):
            continue
        searchable = " ".join(
            str(item.get(field, "")) for field in ("key", "group", "title", "description")
        ).lower()
        is_tennis = "tennis" in searchable
        is_main_tour = "atp" in searchable or "wta" in searchable
        if is_tennis and is_main_tour:
            discovered.append(item)
            seen_keys.add(key)
    return sorted(discovered, key=lambda item: str(item.get("title", item.get("key", ""))))


def combine_tennis_slate(api_key, tennis_items):
    """Fetch every active ATP/WTA market without one failed tour breaking the slate."""
    frames = []
    errors = []
    remaining = "—"
    used = "—"
    for item in tennis_items:
        event_title = str(item.get("title") or item.get("description") or item.get("key") or "Tennis")
        try:
            api_events, usage = fetch_sport_odds(api_key, item["key"])
            remaining = usage.get("remaining", remaining)
            used = usage.get("used", used)
            frame = normalize_api_slate(api_events, event_title)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{event_title}: {exc}")

    if not frames:
        return pd.DataFrame(), {"remaining": remaining, "used": used}, errors

    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [column for column in ["event_id", "start_time", "participant_a", "participant_b"] if column in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")
    combined = combined.sort_values(["start_time", "sport", "participant_a"]).reset_index(drop=True)
    return combined, {"remaining": remaining, "used": used}, errors


def normalize_api_slate(events, sport_title):
    rows = []
    today_eastern = datetime.now(EASTERN_TZ).date()
    for event in events:
        try:
            start_utc = datetime.fromisoformat(str(event.get("commence_time", "")).replace("Z", "+00:00"))
            start_et = start_utc.astimezone(EASTERN_TZ)
        except (TypeError, ValueError):
            continue
        if start_et.date() != today_eastern:
            continue

        home = str(event.get("home_team", "")).strip()
        away = str(event.get("away_team", "")).strip()
        if not home or not away:
            continue
        prices, sources = _best_h2h_prices(event)
        rows.append({
            "event_id": str(event.get("id", "")),
            "start_time": start_et,
            "time_et": start_et.strftime("%-I:%M %p"),
            "sport": sport_title,
            "participant_a": away,
            "participant_b": home,
            "odds_a": prices.get(away),
            "odds_b": prices.get(home),
            "book_a": sources.get(away, "—"),
            "book_b": sources.get(home, "—"),
        })
    return pd.DataFrame(rows).sort_values("start_time") if rows else pd.DataFrame()


def money(value):
    return f"${value:,.2f}"


def safe_int(value, default: int = 0) -> int:
    """Convert blank, missing, or numeric-looking values safely."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _plain_factor_sentence(factor_name, player, opponent, reason):
    """Translate model factors into direct matchup language without rating jargon."""
    templates = {
        "Context-weighted matchup": (
            f"{player} has the more favorable serve-and-return profile for this surface and match format."
        ),
        "Context-weighted recent form": (
            f"{player} enters with the stronger recent results and is converting more of the matches expected of them."
        ),
        "Opponent strength": (
            f"{player}'s recent form has been tested against the stronger level of opposition."
        ),
        "Surface": (
            f"{player} has produced the better recent results on this surface."
        ),
        "Surface transition": (
            f"{player} appears better adapted to the current surface and has had the cleaner transition into this event."
        ),
        "Style matchup": (
            f"{player}'s playing style creates a favorable tactical matchup against {opponent}."
        ),
        "Injury / retirement risk": (
            f"{player} carries the cleaner health profile entering the match."
        ),
        "Tournament motivation": (
            f"The tournament context gives {player} the stronger motivation profile."
        ),
        "Draw context": (
            f"The surrounding draw and event situation is slightly more favorable for {player}."
        ),
        "Event pressure": (
            f"{player} has handled comparable rounds and higher-pressure matches more reliably."
        ),
        "Deciding-match history": (
            f"{player} has been more dependable when matches extend into a deciding set."
        ),
    }
    return templates.get(factor_name, f"{player} holds an advantage in {factor_name.lower()}.")


def build_head_to_head_summary(matches, player_a, player_b, current_surface):
    """Summarize prior meetings using common ATP/WTA dataset column names.

    The helper is deliberately defensive so the app keeps working when data
    providers use slightly different names for winner, loser, event, or date.
    """
    if matches is None or matches.empty:
        return {"meetings": 0, "wins_a": 0, "wins_b": 0, "surface_meetings": 0,
                "surface_wins_a": 0, "surface_wins_b": 0, "last_meeting": None}

    def first_column(options):
        return next((name for name in options if name in matches.columns), None)

    winner_col = first_column(["winner_name", "winner", "Winner", "w_name"])
    loser_col = first_column(["loser_name", "loser", "Loser", "l_name"])
    surface_col = first_column(["surface", "Surface"])
    date_col = first_column(["tourney_date", "match_date", "date", "Date"])
    event_col = first_column(["tourney_name", "tournament", "event", "Tournament"])
    score_col = first_column(["score", "Score"])
    round_col = first_column(["round", "Round"])

    if not winner_col or not loser_col:
        return {"meetings": 0, "wins_a": 0, "wins_b": 0, "surface_meetings": 0,
                "surface_wins_a": 0, "surface_wins_b": 0, "last_meeting": None}

    winner = matches[winner_col].astype(str).str.strip()
    loser = matches[loser_col].astype(str).str.strip()
    pair_mask = ((winner == player_a) & (loser == player_b)) | ((winner == player_b) & (loser == player_a))
    meetings = matches.loc[pair_mask].copy()

    if meetings.empty:
        return {"meetings": 0, "wins_a": 0, "wins_b": 0, "surface_meetings": 0,
                "surface_wins_a": 0, "surface_wins_b": 0, "last_meeting": None}

    meetings["_winner"] = meetings[winner_col].astype(str).str.strip()
    wins_a = int((meetings["_winner"] == player_a).sum())
    wins_b = int((meetings["_winner"] == player_b).sum())

    if surface_col:
        surface_mask = meetings[surface_col].astype(str).str.casefold() == str(current_surface).casefold()
        surface_meetings = meetings.loc[surface_mask]
    else:
        surface_meetings = meetings.iloc[0:0]

    surface_wins_a = int((surface_meetings["_winner"] == player_a).sum())
    surface_wins_b = int((surface_meetings["_winner"] == player_b).sum())

    if date_col:
        raw_dates = meetings[date_col]
        numeric_dates = pd.to_numeric(raw_dates, errors="coerce")
        parsed_numeric = pd.to_datetime(numeric_dates.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        parsed_general = pd.to_datetime(raw_dates, errors="coerce")
        meetings["_parsed_date"] = parsed_numeric.fillna(parsed_general)
        meetings = meetings.sort_values("_parsed_date", ascending=False, na_position="last")

    latest = meetings.iloc[0]
    latest_date = latest.get("_parsed_date")
    if pd.notna(latest_date):
        latest_date = pd.Timestamp(latest_date).date().isoformat()
    else:
        latest_date = "Date unavailable"

    details = []
    if event_col and str(latest.get(event_col, "")).strip() not in {"", "nan", "None"}:
        details.append(str(latest.get(event_col)).strip())
    if round_col and str(latest.get(round_col, "")).strip() not in {"", "nan", "None"}:
        details.append(str(latest.get(round_col)).strip())

    score = ""
    if score_col and str(latest.get(score_col, "")).strip() not in {"", "nan", "None"}:
        score = str(latest.get(score_col)).strip()

    return {
        "meetings": int(len(meetings)),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "surface_meetings": int(len(surface_meetings)),
        "surface_wins_a": surface_wins_a,
        "surface_wins_b": surface_wins_b,
        "last_meeting": {
            "date": latest_date,
            "winner": str(latest["_winner"]),
            "event": " — ".join(details) if details else "Event unavailable",
            "score": score,
        },
    }


def render_head_to_head_summary(matches, player_a, player_b, current_surface):
    """Render a compact, decision-useful H2H card in the match analysis."""
    h2h = build_head_to_head_summary(matches, player_a, player_b, current_surface)
    st.markdown("#### Head-to-Head Summary")

    if h2h["meetings"] == 0:
        st.info("No previous meetings were found in the available Macabets match data.")
        return

    h1, h2, h3 = st.columns(3)
    h1.metric("Overall meetings", h2h["meetings"])
    h2.metric(f"{player_a} H2H wins", h2h["wins_a"])
    h3.metric(f"{player_b} H2H wins", h2h["wins_b"])

    s1, s2, s3 = st.columns(3)
    s1.metric(f"Meetings on {current_surface}", h2h["surface_meetings"])
    s2.metric(f"{player_a} {current_surface} wins", h2h["surface_wins_a"])
    s3.metric(f"{player_b} {current_surface} wins", h2h["surface_wins_b"])

    last = h2h["last_meeting"]
    score_text = f" Score: {last['score']}." if last.get("score") else ""
    st.caption(
        f"Last meeting: {last['winner']} won on {last['date']} at {last['event']}.{score_text}"
    )


def build_matchup_analysis(result, selected_player=None):
    """Create a neutral, data-grounded explanation for both players and the selected bet."""
    player_a = result["player_a"]
    player_b = result["player_b"]
    factors = [
        {
            "name": str(item.get("name", "Matchup factor")),
            "impact_a": float(item.get("impact", 0.0)),
            "reason": str(item.get("reason", "")),
        }
        for item in result.get("factors", [])
        if str(item.get("name", "")).strip() != "Fatigue 2.0"
    ]

    def side_rows(is_a):
        rows = []
        player = player_a if is_a else player_b
        opponent = player_b if is_a else player_a
        for item in factors:
            impact = item["impact_a"] if is_a else -item["impact_a"]
            rows.append({
                "name": item["name"],
                "impact": impact,
                "sentence": _plain_factor_sentence(
                    item["name"], player, opponent, item["reason"]
                ),
            })
        return rows

    def winning_case(is_a):
        player = player_a if is_a else player_b
        opponent = player_b if is_a else player_a
        rows = side_rows(is_a)
        positives = sorted(
            [row for row in rows if row["impact"] > 0.001],
            key=lambda row: row["impact"],
            reverse=True,
        )[:3]
        if not positives:
            positives = sorted(rows, key=lambda row: row["impact"], reverse=True)[:2]
        points = [row["sentence"] for row in positives]
        style = result.get("playing_style_a" if is_a else "playing_style_b", {}).get("label")
        if style and all("style" not in row["name"].lower() for row in positives):
            points.append(
                f"As a {style.lower()}, {player}'s clearest path is to impose that pattern before {opponent} can settle into preferred rallies."
            )
        return points[:3]

    analysis = {
        "player_a_reasons": winning_case(True),
        "player_b_reasons": winning_case(False),
    }

    if selected_player in {player_a, player_b}:
        selected_is_a = selected_player == player_a
        opponent = player_b if selected_is_a else player_a
        rows = side_rows(selected_is_a)
        support = sorted(rows, key=lambda row: row["impact"], reverse=True)[0]
        risks = sorted([row for row in rows if row["impact"] < -0.001], key=lambda row: row["impact"])
        risk = risks[0] if risks else sorted(rows, key=lambda row: abs(row["impact"]))[0]

        risk_paths = {
            "Context-weighted matchup": f"{opponent} can win if they consistently attack the weaker serve or return pattern and prevent {selected_player} from controlling first-strike points.",
            "Context-weighted recent form": f"The bet is vulnerable if {selected_player}'s recent form proves temporary and {opponent} starts cleaner than the recent results suggest.",
            "Opponent strength": f"There is a risk that {selected_player}'s recent record has not prepared them for the level {opponent} brings in this matchup.",
            "Surface": f"The largest danger is that {opponent} settles into the surface faster and turns the match into the type of points where {selected_player} has been less reliable.",
            "Surface transition": f"If {selected_player} struggles with timing or movement early, {opponent} can build scoreboard pressure before the adjustment arrives.",
            "Style matchup": f"{opponent}'s style can disrupt {selected_player}'s preferred patterns and force them to win through a less comfortable plan B.",
            "Injury / retirement risk": f"Any physical limitation could reduce {selected_player}'s serve, movement, or ability to sustain their level across the full match.",
            "Tournament motivation": f"The concern is that {opponent} treats this event as the higher-priority opportunity and competes with greater urgency in the key moments.",
            "Draw context": f"External event context may make {selected_player}'s position less comfortable than the headline matchup suggests.",
            "Event pressure": f"The bet becomes vulnerable if {selected_player} tightens in the important games and {opponent} handles the occasion more cleanly.",
            "Deciding-match history": f"If the match reaches a deciding set, the historical late-match profile favors {opponent}."
        }
        analysis.update({
            "supporting_factor": support["sentence"],
            "biggest_risk": risk["sentence"].replace(selected_player, opponent, 1)
                if risk["impact"] >= 0 else risk["sentence"],
            "loss_path": risk_paths.get(
                risk["name"],
                f"{opponent} can win by neutralizing {selected_player}'s primary advantage and extending the match into less favorable patterns."
            ),
            "risk_factor_name": risk["name"],
        })
    return analysis


def american_to_decimal(odds):
    if odds == 0:
        return 1.0
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)


def probability_to_american(probability):
    """Convert a win probability (0-1) to fair American odds."""
    probability = min(max(float(probability), 0.0001), 0.9999)
    if probability >= 0.5:
        return -round(100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def format_american(odds):
    odds = int(round(odds))
    return f"+{odds}" if odds > 0 else str(odds)


def fair_line_probability(scores_favorite, scores_opponent, weights, confidence):
    """Create a first-pass fair probability from a weighted matchup scorecard.

    The confidence input shrinks uncertain estimates toward 50%, preventing
    low-information matchups from producing extreme prices.
    """
    weighted_difference = sum(
        weights[key] * (scores_favorite[key] - scores_opponent[key])
        for key in weights
    )
    raw_probability = 1 / (1 + math.exp(-0.45 * weighted_difference))
    confidence_factor = min(max(confidence / 10, 0.1), 1.0)
    adjusted_probability = 0.5 + (raw_probability - 0.5) * confidence_factor
    return min(max(adjusted_probability, 0.02), 0.98), weighted_difference


def implied_probability(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.0


def no_vig_probabilities(odds_a, odds_b):
    """Remove sportsbook margin from a two-sided moneyline market."""
    raw_a = implied_probability(int(odds_a))
    raw_b = implied_probability(int(odds_b))
    total = raw_a + raw_b
    if total <= 0:
        return 0.5, 0.5, 0.0
    return raw_a / total, raw_b / total, total - 1


def minimum_acceptable_odds(model_probability, required_roi=0.02):
    """Worst American price that still preserves the required expected ROI."""
    probability = min(max(float(model_probability), 0.0001), 0.9999)
    required_decimal = (1 + required_roi) / probability
    if required_decimal <= 1:
        return -10000
    if required_decimal >= 2:
        return round((required_decimal - 1) * 100)
    return -round(100 / (required_decimal - 1))


def decision_label(expected_roi, confidence):
    """
    Grade the offered price independently from model confidence.

    Confidence affects the strength/caution attached to the recommendation,
    but it must not turn a clearly positive-value price into WATCH or PASS.
    """
    expected_roi = float(expected_roi)
    confidence = int(confidence)

    if expected_roi >= 0.05:
        if confidence >= 7:
            return "BET", "The offered price shows meaningful expected value with strong model confidence."
        if confidence >= 5:
            return "BET", "The offered price shows meaningful expected value, but model confidence is moderate."
        return "BET", "The offered price shows meaningful expected value, but model confidence is low; treat the fair line cautiously."

    if expected_roi >= 0.02:
        return "WATCH", "The offered price shows a smaller positive edge that does not yet reach the full BET threshold."

    return "PASS", "The current price does not provide enough model-supported value."


def build_matchup_brief(player_a, player_b, scores_a, scores_b, weights):
    contributions = []
    for factor, weight in weights.items():
        difference = scores_a[factor] - scores_b[factor]
        contributions.append((factor, difference * weight, difference))
    contributions.sort(key=lambda item: item[1], reverse=True)

    strengths = [item for item in contributions if item[1] > 0][:3]
    risks = sorted([item for item in contributions if item[1] < 0], key=lambda item: item[1])[:2]

    strength_text = (
        "; ".join(f"{factor} ({raw_diff:+.1f})" for factor, _, raw_diff in strengths)
        if strengths else "no clear category-level advantage"
    )
    risk_text = (
        "; ".join(f"{factor} ({raw_diff:+.1f})" for factor, _, raw_diff in risks)
        if risks else "no major scorecard disadvantage"
    )

    return (
        f"{player_a} grades best in {strength_text}. "
        f"The clearest concerns relative to {player_b} are {risk_text}. "
        "This summary reflects the current pre-match scorecard and should be revised if injury, "
        "weather, scheduling, or market information changes."
    )


def stake_to_win(odds, target):
    if odds < 0:
        return target * abs(odds) / 100
    if odds > 0:
        return target * 100 / odds
    return 0.0


def potential_profit(odds, stake):
    if odds < 0:
        return stake * 100 / abs(odds)
    if odds > 0:
        return stake * odds / 100
    return 0.0


def kelly_fraction(model_prob, odds):
    dec = american_to_decimal(odds)
    b = dec - 1
    q = 1 - model_prob
    if b <= 0:
        return 0.0
    return max(0.0, (b * model_prob - q) / b)


def empty_bets():
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


def normalize_bets(df):
    clean = df.copy()
    for col in DEFAULT_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[DEFAULT_COLUMNS]
    numeric = ["odds", "stake", "target_profit", "result_profit", "confidence"]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    clean["status"] = clean["status"].replace("", "Pending")
    return clean


def empty_analyses():
    return pd.DataFrame(columns=ANALYSIS_COLUMNS)


def normalize_analyses(df):
    """Keep old archive exports compatible as the archive gains new fields."""
    clean = df.copy()
    for col in ANALYSIS_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[ANALYSIS_COLUMNS]

    numeric = [
        "analysis_id", "market_odds_a", "market_odds_b", "model_probability_a",
        "fair_odds_a", "no_vig_probability_a", "no_vig_edge",
        "minimum_acceptable_odds_a", "estimated_roi", "confidence",
        "closing_odds_a", "closing_line_value"
    ]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["analysis_id"] = clean["analysis_id"].fillna(0).astype(int)
    clean["result"] = clean["result"].replace("", "Pending").fillna("Pending")
    clean["prediction_correct"] = clean["prediction_correct"].fillna("")
    return clean


def closing_line_value(model_probability, closing_odds):
    """Expected ROI at the closing price; positive means the model still beat close."""
    if pd.isna(closing_odds) or float(closing_odds) == 0:
        return float("nan")
    decimal = american_to_decimal(int(closing_odds))
    probability = float(model_probability)
    return probability * (decimal - 1) - (1 - probability)


def empty_slate():
    return pd.DataFrame(columns=SLATE_COLUMNS)


def normalize_slate(df):
    clean = df.copy()
    for col in SLATE_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[SLATE_COLUMNS]

    numeric = [
        "slate_id", "market_odds_a", "market_odds_b",
        "model_probability_a", "confidence"
    ]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["slate_id"] = clean["slate_id"].fillna(0).astype(int)
    clean["model_probability_a"] = clean["model_probability_a"].fillna(0.5).clip(0.01, 0.99)
    clean["confidence"] = clean["confidence"].fillna(5).clip(1, 10)
    return clean


def score_daily_slate(df):
    scored = normalize_slate(df)
    if scored.empty:
        return scored

    market_a = scored["market_odds_a"].apply(
        lambda value: implied_probability(int(value)) if pd.notna(value) and value != 0 else np.nan
    )
    market_b = scored["market_odds_b"].apply(
        lambda value: implied_probability(int(value)) if pd.notna(value) and value != 0 else np.nan
    )
    totals = market_a + market_b

    scored["no_vig_probability_a"] = np.where(
        totals > 0, market_a / totals, market_a
    )
    scored["sportsbook_hold"] = totals - 1
    scored["fair_odds_a"] = scored["model_probability_a"].apply(probability_to_american)
    scored["no_vig_edge"] = scored["model_probability_a"] - scored["no_vig_probability_a"]
    scored["estimated_roi"] = scored.apply(
        lambda row: (
            row["model_probability_a"] * (american_to_decimal(int(row["market_odds_a"])) - 1)
            - (1 - row["model_probability_a"])
        )
        if pd.notna(row["market_odds_a"]) and row["market_odds_a"] != 0
        else np.nan,
        axis=1,
    )
    scored["decision"] = scored.apply(
        lambda row: decision_label(
            float(row["estimated_roi"]) if pd.notna(row["estimated_roi"]) else -1,
            int(row["confidence"]),
        )[0],
        axis=1,
    )
    scored["minimum_acceptable_odds_a"] = scored["model_probability_a"].apply(
        lambda probability: minimum_acceptable_odds(probability, required_roi=0.02)
    )

    # A transparent ranking score: edge is primary, confidence and market disagreement are secondary.
    scored["opportunity_score"] = (
        scored["estimated_roi"].fillna(-1) * 100 * 0.60
        + scored["confidence"] * 0.25
        + scored["no_vig_edge"].fillna(0) * 100 * 0.15
    )
    return scored.sort_values(
        ["opportunity_score", "confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)


if "pending_fair_line_prefill" in st.session_state:
    pending = st.session_state.pop("pending_fair_line_prefill")
    for key, value in pending.items():
        st.session_state[key] = value


if "bets" not in st.session_state:
    st.session_state.bets = empty_bets()

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100000.0

if "target_profit" not in st.session_state:
    st.session_state.target_profit = 10000.0

if "analyses" not in st.session_state:
    st.session_state.analyses = empty_analyses()


if "daily_slate" not in st.session_state:
    st.session_state.daily_slate = empty_slate()


st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color: #777; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

title_col, version_col = st.columns([4, 1])
with title_col:
    st.title("Macabets")
    st.caption("Favorite-focused bet tracking, matchup analysis and bankroll risk control.")
with version_col:
    st.markdown(
        f"""
        <div style="
            margin-top: 0.65rem;
            padding: 0.55rem 0.75rem;
            border: 1px solid #d8d8d8;
            border-radius: 0.55rem;
            text-align: center;
            font-weight: 600;
        ">
            {APP_VERSION}<br>
            <span style="font-size: 0.78rem; font-weight: 400;">{BUILD_DATE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Core Settings")
    st.session_state.bankroll = st.number_input(
        "Starting bankroll",
        min_value=0.0,
        value=float(st.session_state.bankroll),
        step=1000.0,
    )
    st.session_state.target_profit = st.number_input(
        "Default target profit",
        min_value=1.0,
        value=float(st.session_state.target_profit),
        step=500.0,
    )
    st.divider()
    st.subheader("Restore / Import")
    uploaded = st.file_uploader("Upload a prior bets CSV", type=["csv"], key="bets_restore")
    if uploaded is not None:
        try:
            imported = normalize_bets(pd.read_csv(uploaded))
            st.session_state.bets = imported
            st.success(f"Loaded {len(imported)} bets.")
        except Exception as exc:
            st.error(f"Could not load bets CSV: {exc}")

    analysis_upload = st.file_uploader(
        "Upload a prior analysis archive CSV",
        type=["csv"],
        key="analysis_restore",
    )
    if analysis_upload is not None:
        try:
            imported_analyses = normalize_analyses(pd.read_csv(analysis_upload))
            st.session_state.analyses = imported_analyses
            st.success(f"Loaded {len(imported_analyses)} archived analyses.")
        except Exception as exc:
            st.error(f"Could not load analysis CSV: {exc}")

bets = normalize_bets(st.session_state.bets)
settled = bets[bets["status"].isin(["Won", "Lost", "Void", "Cashed Out"])]
pending = bets[bets["status"] == "Pending"]
net_profit = float(settled["result_profit"].sum()) if not settled.empty else 0.0
current_bankroll = st.session_state.bankroll + net_profit
total_staked = float(settled["stake"].sum()) if not settled.empty else 0.0
roi = net_profit / total_staked if total_staked else 0.0
decisions = settled[settled["status"].isin(["Won", "Lost"])]
wins = int((decisions["status"] == "Won").sum()) if not decisions.empty else 0
win_rate = wins / len(decisions) if len(decisions) else 0.0
pending_exposure = float(pending["stake"].sum()) if not pending.empty else 0.0

tabs = st.tabs([
    "Dashboard", "Analysis Engine", "Bets",
    "Daily Slate", "Archive", "Settings"
])

with tabs[0]:
    with st.expander("What's New in Macabets v0.21", expanded=True):
        st.markdown(
            """
            - Added a separate NFL Foundation analysis engine without altering the tennis engine
            - NFL fair spread, win probability, projected score, fair moneylines and market comparison
            - Transparent manual matchup scorecard while automated NFL data pipelines are still under construction
            - Added one-click tennis analysis directly from the Automatic Daily Slate
            - Daily Slate matchups now prefill and automatically run the Tennis Analysis Engine
            - Tennis always appears as a Daily Slate option, with feed diagnostics and API quota details
            - Head-to-Head Summary: overall record, current-surface record and most recent meeting
            - Removed 7-day workload and rest metrics because the available data was not reliable enough
            - Surface Transition Engine: recent exposure and adaptation to the current surface
            - Opponent Style Matchups with automatic or manual style tags
            - Injury and retirement-risk context
            - Tournament motivation: home event, defending points, priority and ranking pressure
            - Draw-pressure context with deliberately limited model impact
            - Every new factor appears in the probability-impact breakdown
            """
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Bankroll", money(current_bankroll), money(net_profit))
    c2.metric("Pending Exposure", money(pending_exposure))
    c3.metric("Settled ROI", f"{roi:.1%}")
    c4.metric("Win Rate", f"{win_rate:.1%}")
    c5.metric("Bets Logged", f"{len(bets)}")

    if current_bankroll > 0:
        exposure_pct = pending_exposure / current_bankroll
        if exposure_pct >= 0.25:
            st.error(f"Pending exposure is {exposure_pct:.1%} of bankroll. This is a major concentration risk.")
        elif exposure_pct >= 0.15:
            st.warning(f"Pending exposure is {exposure_pct:.1%} of bankroll. Proceed carefully.")
        elif exposure_pct > 0:
            st.info(f"Pending exposure is {exposure_pct:.1%} of bankroll.")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recent Bets")
        if bets.empty:
            st.write("No bets logged yet.")
        else:
            view = bets.sort_values("id", ascending=False).head(10)
            st.dataframe(
                view[["date", "sport", "event", "selection", "odds", "stake", "status", "result_profit"]],
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.subheader("Profit by Sport")
        if settled.empty:
            st.write("Settle bets to populate this chart.")
        else:
            profit_sport = settled.groupby("sport", as_index=False)["result_profit"].sum()
            fig, ax = plt.subplots()
            ax.bar(profit_sport["sport"], profit_sport["result_profit"])
            ax.axhline(0, linewidth=1)
            ax.set_ylabel("Profit / Loss ($)")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with tabs[1]:
    analysis_tabs = st.tabs(["Tennis Analysis", "NFL Analysis", "Outcome Simulator"])

    with analysis_tabs[0]:
        st.subheader("Analysis Engine — Tennis")
        st.caption(
            "Select the matchup and event context. Macabets builds the probability from "
            "historical ATP results, Elo, surface performance, form, serve/return data, "
            "and event-pressure history."
        )

        if not TENNIS_ENGINE_AVAILABLE:
            st.error(
                "The tennis engine could not be imported. Confirm that engine/data.py and "
                f"engine/tennis.py are in the repository. Import error: {TENNIS_ENGINE_IMPORT_ERROR}"
            )
        else:
            try:
                matches, data_errors = load_matches()
            except Exception as exc:
                matches = pd.DataFrame()
                data_errors = []
                st.error(str(exc))
                st.info(
                    "Run update_tennis_data.py or the GitHub Action named "
                    "'Update Macabets Tennis Data', then reboot the Streamlit app."
                )

            if not matches.empty:
                if data_errors:
                    with st.expander("Data-file warnings"):
                        for item in data_errors:
                            st.warning(item)

                players = tennis_player_names(matches)
                tournaments = tennis_tournament_names(matches)

                existing_a = str(st.session_state.get("fle_favorite", "")).strip()
                existing_b = str(st.session_state.get("fle_opponent", "")).strip()
                existing_tournament = str(st.session_state.get("fle_tournament", "")).strip()

                player_a_options = players.copy()
                player_b_options = players.copy()
                if existing_a and existing_a not in player_a_options:
                    player_a_options.insert(0, existing_a)
                if existing_b and existing_b not in player_b_options:
                    player_b_options.insert(0, existing_b)
                if not player_a_options:
                    player_a_options = ["Player A"]
                if not player_b_options:
                    player_b_options = ["Player B"]

                tournament_options = tournaments.copy()
                if existing_tournament and existing_tournament not in tournament_options:
                    tournament_options.insert(0, existing_tournament)
                if not tournament_options:
                    tournament_options = ["Montreal"]

                meta1, meta2, meta3, meta4 = st.columns(4)
                match_date = meta1.date_input("Match date", value=date.today(), key="auto_match_date")
                tournament = meta2.selectbox(
                    "Tournament",
                    tournament_options,
                    index=(
                        tournament_options.index(existing_tournament)
                        if existing_tournament in tournament_options else 0
                    ),
                    key="auto_tournament",
                )

                detected_surface = tennis_tournament_surface(matches, tournament)
                surface_options = ["Hard", "Clay", "Grass", "Carpet"]
                surface = meta3.selectbox(
                    "Surface",
                    surface_options,
                    index=surface_options.index(detected_surface) if detected_surface in surface_options else 0,
                    key="auto_surface",
                )
                round_name = meta4.selectbox(
                    "Round",
                    ["Qualifying", "R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
                    key="auto_round",
                )

                detected_category = tennis_tournament_category(matches, tournament)
                category_options = [
                    "Grand Slam", "Masters 1000", "ATP 500", "ATP 250",
                    "Challenger", "Tour Finals", "Davis Cup"
                ]
                c1, c2, c3 = st.columns(3)
                tournament_category = c1.selectbox(
                    "Event category",
                    category_options,
                    index=(
                        category_options.index(detected_category)
                        if detected_category in category_options else 3
                    ),
                    key="auto_tournament_category",
                    help="Macabets infers this from the event, but you can correct it.",
                )
                environment = c2.selectbox(
                    "Environment",
                    ["Outdoor", "Indoor"],
                    key="auto_environment",
                )
                inferred_format = (
                    "Best of 5"
                    if tournament_category == "Grand Slam" and round_name != "Qualifying"
                    else "Best of 3"
                )
                match_format = c3.selectbox(
                    "Match format",
                    ["Best of 3", "Best of 5"],
                    index=0 if inferred_format == "Best of 3" else 1,
                    key="auto_match_format",
                )

                with st.expander("Advanced match context — Tier 1 & 2", expanded=False):
                    st.caption(
                        "Leave uncertain fields at their neutral defaults. Manual context should "
                        "only be entered when the information is known."
                    )

                    st.markdown("##### Playing style and handedness")
                    s1, s2, s3, s4 = st.columns(4)
                    style_options = [
                        "Auto", "Big Server", "Elite Returner", "Aggressive All-Court",
                        "Counterpuncher", "Balanced Baseliner"
                    ]
                    manual_style_a = s1.selectbox(
                        "Player A style", style_options, key="auto_style_a"
                    )
                    manual_style_b = s2.selectbox(
                        "Player B style", style_options, key="auto_style_b"
                    )
                    handedness_a = s3.selectbox(
                        "Player A hand", ["Right", "Left"], key="auto_hand_a"
                    )
                    handedness_b = s4.selectbox(
                        "Player B hand", ["Right", "Left"], key="auto_hand_b"
                    )

                    st.markdown("##### Health")
                    h1, h2 = st.columns(2)
                    injury_options = [
                        "Clear", "Minor concern", "Recent medical timeout",
                        "Returning from layoff", "Recent retirement", "Significant concern"
                    ]
                    injury_status_a = h1.selectbox(
                        "Player A health", injury_options, key="auto_injury_a"
                    )
                    injury_status_b = h2.selectbox(
                        "Player B health", injury_options, key="auto_injury_b"
                    )

                    # Workload, rest, travel and short-turnaround inputs are intentionally
                    # disabled until Macabets has a sufficiently reliable scheduling source.
                    travel_load_a = "None"
                    travel_load_b = "None"
                    late_finish_a = False
                    late_finish_b = False

                    st.markdown("##### Motivation and tournament context")
                    mca1, mca2 = st.columns(2)
                    with mca1:
                        st.markdown("**Player A**")
                        home_event_a = st.checkbox("Home-country event", key="auto_home_a")
                        defending_status_a = st.selectbox(
                            "Defending status",
                            ["None", "Defending meaningful points", "Defending title/final"],
                            key="auto_defending_a",
                        )
                        priority_a = st.selectbox(
                            "Event priority", ["Low", "Normal", "High"],
                            index=1, key="auto_priority_a"
                        )
                        ranking_pressure_a = st.selectbox(
                            "Ranking pressure", ["None", "Moderate", "High"],
                            key="auto_rank_pressure_a",
                        )
                        draw_pressure_a = st.selectbox(
                            "Forward draw", ["Favorable", "Normal", "Difficult"],
                            index=1, key="auto_draw_a"
                        )

                    with mca2:
                        st.markdown("**Player B**")
                        home_event_b = st.checkbox("Home-country event", key="auto_home_b")
                        defending_status_b = st.selectbox(
                            "Defending status",
                            ["None", "Defending meaningful points", "Defending title/final"],
                            key="auto_defending_b",
                        )
                        priority_b = st.selectbox(
                            "Event priority", ["Low", "Normal", "High"],
                            index=1, key="auto_priority_b"
                        )
                        ranking_pressure_b = st.selectbox(
                            "Ranking pressure", ["None", "Moderate", "High"],
                            key="auto_rank_pressure_b",
                        )
                        draw_pressure_b = st.selectbox(
                            "Forward draw", ["Favorable", "Normal", "Difficult"],
                            index=1, key="auto_draw_b"
                        )

                p1, p2 = st.columns(2)
                default_a_index = player_a_options.index(existing_a) if existing_a in player_a_options else 0
                default_b_index = player_b_options.index(existing_b) if existing_b in player_b_options else min(1, len(player_b_options) - 1)
                player_a = p1.selectbox(
                    "Player A",
                    player_a_options,
                    index=default_a_index,
                    key="auto_player_a",
                )
                player_b = p2.selectbox(
                    "Player B",
                    player_b_options,
                    index=default_b_index,
                    key="auto_player_b",
                )

                bet_side_options = [
                    "Just analyze",
                    f"{player_a} — Player A",
                    f"{player_b} — Player B",
                ]
                considering_bet = st.radio(
                    "Who are you considering betting on?",
                    bet_side_options,
                    horizontal=True,
                    key="auto_considering_bet",
                    help=(
                        "This does not influence the model. It only tells Macabets which "
                        "market position you want evaluated."
                    ),
                )

                o1, o2, o3 = st.columns(3)
                market_odds_a = o1.number_input(
                    f"Sportsbook odds — {player_a}",
                    value=safe_int(st.session_state.get("fle_market_a", -180), -180),
                    step=5,
                    key="auto_market_a",
                )
                market_odds_b = o2.number_input(
                    f"Sportsbook odds — {player_b}",
                    value=safe_int(st.session_state.get("fle_market_b", 155), 155),
                    step=5,
                    key="auto_market_b",
                )
                simulations = o3.selectbox(
                    "Simulations",
                    [5000, 10000, 20000, 50000],
                    index=2,
                    key="auto_simulations",
                )

                analyze_disabled = player_a == player_b
                if analyze_disabled:
                    st.warning("Select two different players.")

                auto_analysis_requested = bool(
                    st.session_state.pop("run_analysis_from_daily_slate", False)
                )
                manual_analysis_requested = st.button(
                    "Analyze Match",
                    type="primary",
                    use_container_width=True,
                    disabled=analyze_disabled,
                )

                if manual_analysis_requested or (auto_analysis_requested and not analyze_disabled):
                    with st.spinner("Macabets is analyzing the matchup..."):
                        try:
                            st.session_state.automatic_match_result = analyze_tennis_match(
                                matches=matches,
                                player_a=player_a,
                                player_b=player_b,
                                tournament=tournament,
                                round_label=round_name,
                                surface=surface,
                                event_date=match_date,
                                simulations=int(simulations),
                                tournament_category_label=tournament_category,
                                environment=environment,
                                match_format=match_format,
                                style_a=manual_style_a,
                                style_b=manual_style_b,
                                handedness_a=handedness_a,
                                handedness_b=handedness_b,
                                injury_status_a=injury_status_a,
                                injury_status_b=injury_status_b,
                                travel_load_a=travel_load_a,
                                travel_load_b=travel_load_b,
                                late_finish_a=late_finish_a,
                                late_finish_b=late_finish_b,
                                home_event_a=home_event_a,
                                home_event_b=home_event_b,
                                defending_status_a=defending_status_a,
                                defending_status_b=defending_status_b,
                                priority_a=priority_a,
                                priority_b=priority_b,
                                ranking_pressure_a=ranking_pressure_a,
                                ranking_pressure_b=ranking_pressure_b,
                                draw_pressure_a=draw_pressure_a,
                                draw_pressure_b=draw_pressure_b,
                            )
                            st.session_state.automatic_match_market = {
                                "market_odds_a": safe_int(market_odds_a, -180),
                                "market_odds_b": safe_int(market_odds_b, 155),
                                "match_date": match_date.isoformat(),
                                "considering_bet": considering_bet,
                                "tournament_category": tournament_category,
                                "environment": environment,
                                "match_format": match_format,
                            }
                            if auto_analysis_requested:
                                st.session_state["daily_slate_analysis_ready"] = (
                                    f"Analysis completed for {player_a} vs {player_b}. "
                                    "Open the Analysis Engine tab to review the full result."
                                )
                        except Exception as exc:
                            st.session_state.pop("automatic_match_result", None)
                            st.error(f"Analysis failed: {exc}")
                            st.exception(exc)

                result = st.session_state.get("automatic_match_result")
                market_snapshot = st.session_state.get("automatic_match_market", {})

                if result:
                    analyzed_a = result["player_a"]
                    analyzed_b = result["player_b"]
                    listed_a = safe_int(market_snapshot.get("market_odds_a", market_odds_a), safe_int(market_odds_a, -180))
                    listed_b = safe_int(market_snapshot.get("market_odds_b", market_odds_b), safe_int(market_odds_b, 155))

                    model_probability = float(result["win_probability"])
                    probability_b = 1 - model_probability
                    fair_odds = int(result["fair_line"])
                    fair_odds_b = probability_to_american(probability_b)
                    no_vig_a, no_vig_b, sportsbook_hold = no_vig_probabilities(listed_a, listed_b)

                    roi_a = (
                        model_probability * (american_to_decimal(listed_a) - 1)
                        - (1 - model_probability)
                    )
                    roi_b = (
                        probability_b * (american_to_decimal(listed_b) - 1)
                        - (1 - probability_b)
                    )
                    edge_a = model_probability - no_vig_a
                    edge_b = probability_b - no_vig_b
                    confidence = int(result["confidence"])

                    considered_snapshot = str(
                        market_snapshot.get("considering_bet", "Just analyze")
                    )
                    if "Player A" in considered_snapshot:
                        considered_player = analyzed_a
                        considered_probability = model_probability
                        considered_fair_odds = fair_odds
                        considered_market_odds = listed_a
                        considered_no_vig = no_vig_a
                        considered_edge = edge_a
                        considered_roi = roi_a
                        opposite_player = analyzed_b
                        opposite_roi = roi_b
                    elif "Player B" in considered_snapshot:
                        considered_player = analyzed_b
                        considered_probability = probability_b
                        considered_fair_odds = fair_odds_b
                        considered_market_odds = listed_b
                        considered_no_vig = no_vig_b
                        considered_edge = edge_b
                        considered_roi = roi_b
                        opposite_player = analyzed_a
                        opposite_roi = roi_a
                    else:
                        considered_player = None
                        considered_probability = None
                        considered_fair_odds = None
                        considered_market_odds = None
                        considered_no_vig = None
                        considered_edge = None
                        considered_roi = None
                        opposite_player = None
                        opposite_roi = None

                    minimum_price_a = minimum_acceptable_odds(
                        model_probability, required_roi=0.02
                    )
                    minimum_price_b = minimum_acceptable_odds(
                        probability_b, required_roi=0.02
                    )
                    if considered_player == analyzed_a:
                        minimum_price = minimum_price_a
                    elif considered_player == analyzed_b:
                        minimum_price = minimum_price_b
                    else:
                        minimum_price = None

                    if considered_player:
                        decision, decision_reason = decision_label(
                            considered_roi, confidence
                        )
                    else:
                        decision = "ANALYZE"
                        decision_reason = (
                            "No betting side was selected. Macabets is showing the matchup "
                            "objectively."
                        )

                    # Preserve Player A fields for the existing archive structure.
                    no_vig_edge = edge_a
                    expected_roi = roi_a

                    st.divider()
                    st.markdown(f"### {analyzed_a} vs {analyzed_b}")

                    st.markdown("#### Match Context")
                    cx1, cx2, cx3, cx4, cx5 = st.columns(5)
                    cx1.metric("Category", result.get("tournament_category", "—"))
                    cx2.metric("Round", result.get("round", "—"))
                    cx3.metric("Surface", result.get("surface", "—"))
                    cx4.metric("Environment", result.get("environment", "—"))
                    cx5.metric("Format", result.get("match_format", "—"))

                    render_head_to_head_summary(
                        matches, analyzed_a, analyzed_b, result.get("surface", surface)
                    )

                    st.markdown("#### Objective Match Price")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(
                        f"{analyzed_a} probability",
                        f"{model_probability:.1%}",
                        f"Fair {format_american(fair_odds)}",
                    )
                    m2.metric(
                        f"{analyzed_b} probability",
                        f"{probability_b:.1%}",
                        f"Fair {format_american(fair_odds_b)}",
                    )
                    m3.metric("Sportsbook hold", f"{sportsbook_hold:.1%}")
                    m4.metric("Model confidence", f"{confidence}/10")

                    if considered_player:
                        st.markdown(f"#### Your Considered Bet: {considered_player}")
                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric(
                            "Your market price",
                            format_american(considered_market_odds),
                        )
                        n2.metric(
                            "Macabets fair price",
                            format_american(considered_fair_odds),
                        )
                        n3.metric(
                            "Edge vs no-vig market",
                            f"{considered_edge:+.1%}",
                        )
                        n4.metric(
                            "Expected ROI",
                            f"{considered_roi:+.1%}",
                        )

                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Decision", decision)
                        d2.metric(
                            "Recommendation strength",
                            (
                                "Strong"
                                if confidence >= 7 and int(result["data_quality"]) >= 7
                                else "Moderate"
                                if confidence >= 5 and int(result["data_quality"]) >= 5
                                else "Low"
                            ),
                        )
                        d3.metric(
                            "Minimum acceptable price",
                            format_american(minimum_price),
                        )
                        d4.metric("Data quality", f"{int(result['data_quality'])}/10")

                        if decision == "BET":
                            caution = (
                                " Data quality is limited, so this is a low-confidence value signal."
                                if int(result["data_quality"]) < 5 or confidence < 5
                                else ""
                            )
                            st.success(
                                f"BET: Macabets prices {considered_player} at "
                                f"{format_american(considered_fair_odds)} versus your available "
                                f"price of {format_american(considered_market_odds)}. "
                                f"Estimated ROI is {considered_roi:+.1%}. "
                                f"Your price is also better than the minimum acceptable price of "
                                f"{format_american(minimum_price)}.{caution}"
                            )
                        elif decision == "WATCH":
                            st.warning(
                                f"WATCH: {decision_reason} Macabets needs approximately "
                                f"{format_american(minimum_price)} or better for a 2% expected return."
                            )
                        else:
                            st.error(
                                f"PASS: {format_american(considered_market_odds)} is too expensive "
                                f"relative to Macabets' fair price of "
                                f"{format_american(considered_fair_odds)}. "
                                f"A price near {format_american(minimum_price)} or better is required."
                            )

                        if opposite_roi > considered_roi:
                            if opposite_roi > 0:
                                st.info(
                                    f"Macabets currently sees more value on {opposite_player} "
                                    f"({opposite_roi:+.1%} estimated ROI) than on your considered side."
                                )
                            else:
                                st.caption(
                                    f"The opposite side grades better than your considered bet, "
                                    f"but it still does not show positive expected value "
                                    f"({opposite_roi:+.1%})."
                                )
                    else:
                        st.info(
                            "No betting side selected. Macabets has priced both players objectively. "
                            "Select a player above and analyze again to receive a direct BET / WATCH / "
                            "PASS evaluation."
                        )
                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric(f"{analyzed_a} ROI", f"{roi_a:+.1%}")
                        n2.metric(f"{analyzed_b} ROI", f"{roi_b:+.1%}")
                        n3.metric(f"{analyzed_a} no-vig edge", f"{edge_a:+.1%}")
                        n4.metric(f"{analyzed_b} no-vig edge", f"{edge_b:+.1%}")

                    matchup_analysis = build_matchup_analysis(result, considered_player)
                    st.markdown("#### Macabets Matchup Analysis")
                    st.caption(
                        "A plain-English explanation generated from the same matchup data used by the Analysis Engine. "
                        "It does not change the probability or recommendation."
                    )
                    why_a, why_b = st.columns(2)
                    with why_a:
                        st.markdown(f"**Why {analyzed_a} can win**")
                        for point in matchup_analysis.get("player_a_reasons", []):
                            st.markdown(f"- {point}")
                    with why_b:
                        st.markdown(f"**Why {analyzed_b} can win**")
                        for point in matchup_analysis.get("player_b_reasons", []):
                            st.markdown(f"- {point}")

                    if considered_player:
                        st.markdown(f"**Bet-specific analysis: {considered_player}**")
                        ba1, ba2 = st.columns(2)
                        with ba1:
                            st.success(
                                "**Strongest reason to back the bet**\n\n"
                                + matchup_analysis.get(
                                    "supporting_factor",
                                    "The selected player holds the stronger overall matchup profile."
                                )
                            )
                        with ba2:
                            st.warning(
                                "**Biggest risk to the bet**\n\n"
                                + matchup_analysis.get(
                                    "biggest_risk",
                                    "The opponent has a credible path to disrupt the preferred match pattern."
                                )
                            )
                        st.info(
                            "**Most realistic loss path:** "
                            + matchup_analysis.get(
                                "loss_path",
                                f"{opposite_player} extends the match and prevents {considered_player} from imposing the expected advantage."
                            )
                        )

                    # Build a decision-focused explanation from the same neutral model factors.
                    raw_factors = []
                    for factor in result["factors"]:
                        if str(factor.get("name", "")).strip() == "Fatigue 2.0":
                            continue
                        raw_factors.append({
                            "name": str(factor["name"]),
                            "impact_a": float(factor["impact"]),
                            "reason": str(factor["reason"]),
                        })

                    if considered_player:
                        considered_is_a = considered_player == analyzed_a
                        considered_factor_rows = []
                        for factor in raw_factors:
                            side_impact = (
                                factor["impact_a"]
                                if considered_is_a
                                else -factor["impact_a"]
                            )
                            considered_factor_rows.append({
                                "name": factor["name"],
                                "impact": side_impact,
                                "reason": factor["reason"],
                            })

                        support_factors = sorted(
                            [f for f in considered_factor_rows if f["impact"] > 0],
                            key=lambda item: item["impact"],
                            reverse=True,
                        )[:3]
                        opposition_factors = sorted(
                            [f for f in considered_factor_rows if f["impact"] < 0],
                            key=lambda item: item["impact"],
                        )[:3]

                        model_favors_considered = considered_probability > 0.50
                        market_favors_considered = considered_no_vig > 0.50
                        has_positive_value = considered_roi > 0
                        meaningful_value = considered_roi >= 0.05
                        price_gap = considered_probability - considered_no_vig

                        if model_favors_considered and not has_positive_value:
                            verdict_type = "Good player, bad price"
                            verdict_text = (
                                f"Macabets expects {considered_player} to win more often than lose, "
                                f"but {format_american(considered_market_odds)} is too expensive. "
                                f"The player and the bet are not the same decision."
                            )
                        elif not model_favors_considered and has_positive_value:
                            verdict_type = "Underdog value"
                            verdict_text = (
                                f"Macabets does not make {considered_player} the most likely winner, "
                                f"but the offered price is large enough to create positive expected value."
                            )
                        elif model_favors_considered and meaningful_value:
                            verdict_type = "Player and price align"
                            verdict_text = (
                                f"Macabets favors {considered_player} in the matchup and also believes "
                                f"your price is better than the model's fair price."
                            )
                        elif has_positive_value:
                            verdict_type = "Small price advantage"
                            verdict_text = (
                                f"The offered price is slightly better than Macabets' fair value, "
                                f"but the margin is not yet strong enough for a full BET grade."
                            )
                        else:
                            verdict_type = "No betting advantage"
                            verdict_text = (
                                f"Macabets does not see enough compensation at "
                                f"{format_american(considered_market_odds)} for the matchup risk."
                            )

                        st.markdown("#### Why Macabets Gave This Decision")
                        v1, v2, v3 = st.columns(3)
                        v1.metric(
                            "Player outlook",
                            "Favored" if model_favors_considered else "Underdog",
                            f"{considered_probability:.1%} win probability",
                        )
                        v2.metric(
                            "Price outlook",
                            "Positive value" if has_positive_value else "Negative value",
                            f"{price_gap:+.1%} vs no-vig market",
                        )
                        v3.metric("Bet diagnosis", verdict_type)

                        if decision == "BET":
                            st.success(verdict_text)
                        elif decision == "WATCH":
                            st.warning(verdict_text)
                        else:
                            st.error(verdict_text)

                        reason_col_a, reason_col_b = st.columns(2)
                        with reason_col_a:
                            st.markdown(f"**What supports {considered_player}**")
                            if support_factors:
                                for factor in support_factors:
                                    st.markdown(
                                        f"- **{factor['name']}** "
                                        f"({factor['impact']:+.1%}): {factor['reason']}"
                                    )
                            else:
                                st.caption(
                                    "The current model does not identify a meaningful statistical "
                                    "factor supporting this side."
                                )

                        with reason_col_b:
                            st.markdown(f"**What works against {considered_player}**")
                            if opposition_factors:
                                for factor in opposition_factors:
                                    st.markdown(
                                        f"- **{factor['name']}** "
                                        f"({factor['impact']:+.1%}): {factor['reason']}"
                                    )
                            else:
                                st.caption(
                                    "The current model does not identify a meaningful statistical "
                                    "factor working against this side."
                                )

                        if confidence < 6:
                            st.warning(
                                "Model confidence is limited. Treat the fair line as less stable "
                                "until the data sample or matchup context improves."
                            )
                        elif int(result["data_quality"]) < 6:
                            st.warning(
                                "The recommendation is being made with limited data quality. "
                                "The calculated edge may be less reliable than the headline number."
                            )

                    st.markdown("#### Tier 1 & 2 Context")
                    fp_a = result.get("fatigue_profile_a", {})
                    fp_b = result.get("fatigue_profile_b", {})
                    tr_a = result.get("surface_transition_a", {})
                    tr_b = result.get("surface_transition_b", {})
                    ps_a = result.get("playing_style_a", {})
                    ps_b = result.get("playing_style_b", {})

                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.markdown(f"**{analyzed_a}**")
                        x1, x2, x3 = st.columns(3)
                        x1.metric("Style", ps_a.get("label", "—"))
                        x2.metric("Surface adaptation", f"{tr_a.get('adaptation_score', .5):.0%}")
                        x3.metric("Recent surface matches", tr_a.get("matches_current_surface_30", 0))
                        x4, x5 = st.columns(2)
                        x4.metric("Health", result.get("injury_status_a", "Clear"))
                        x5.metric("Hand", result.get("handedness_a", "—"))

                    with tc2:
                        st.markdown(f"**{analyzed_b}**")
                        y1, y2, y3 = st.columns(3)
                        y1.metric("Style", ps_b.get("label", "—"))
                        y2.metric("Surface adaptation", f"{tr_b.get('adaptation_score', .5):.0%}")
                        y3.metric("Recent surface matches", tr_b.get("matches_current_surface_30", 0))
                        y4, y5 = st.columns(2)
                        y4.metric("Health", result.get("injury_status_b", "Clear"))
                        y5.metric("Hand", result.get("handedness_b", "—"))

                    st.caption(
                        "Surface-transition data is combined with any manual health context "
                        "entered before analysis. Neutral defaults create no adjustment."
                    )

                    st.markdown("#### Opponent Strength Index")
                    osa = result.get("opponent_strength_a", {})
                    osb = result.get("opponent_strength_b", {})

                    if osa and osb:
                        osi1, osi2 = st.columns(2)

                        with osi1:
                            st.markdown(f"**{analyzed_a} — recent opposition**")
                            a1, a2, a3 = st.columns(3)
                            a1.metric("Strength score", f"{osa.get('strength_score', 0.5):.0%}")
                            a2.metric("Average opponent Elo", f"{osa.get('avg_opponent_elo', 1500):.0f}")
                            avg_rank_a = osa.get("avg_opponent_rank")
                            a3.metric(
                                "Average opponent rank",
                                f"{avg_rank_a:.0f}" if avg_rank_a is not None else "N/A",
                            )
                            a4, a5, a6 = st.columns(3)
                            a4.metric("Top-50 record", osa.get("top_50_record", "0-0"))
                            a5.metric("Top-100 record", osa.get("top_100_record", "0-0"))
                            a6.metric("Quality form", f"{osa.get('quality_form', 0.5):.0%}")

                        with osi2:
                            st.markdown(f"**{analyzed_b} — recent opposition**")
                            b1, b2, b3 = st.columns(3)
                            b1.metric("Strength score", f"{osb.get('strength_score', 0.5):.0%}")
                            b2.metric("Average opponent Elo", f"{osb.get('avg_opponent_elo', 1500):.0f}")
                            avg_rank_b = osb.get("avg_opponent_rank")
                            b3.metric(
                                "Average opponent rank",
                                f"{avg_rank_b:.0f}" if avg_rank_b is not None else "N/A",
                            )
                            b4, b5, b6 = st.columns(3)
                            b4.metric("Top-50 record", osb.get("top_50_record", "0-0"))
                            b5.metric("Top-100 record", osb.get("top_100_record", "0-0"))
                            b6.metric("Quality form", f"{osb.get('quality_form', 0.5):.0%}")

                        st.caption(
                            "This score combines recent opponent Elo, opponent ranking, and "
                            "the quality of the player's results. It directly changes the fair line."
                        )

                    st.markdown("#### Context Engine Weights")
                    context_weights_result = result.get("context_weights", {})
                    if context_weights_result:
                        cw1, cw2, cw3, cw4 = st.columns(4)
                        cw1.metric(
                            "Base Elo mix",
                            f"{context_weights_result.get('overall_elo', 0):.0%} overall",
                            f"{context_weights_result.get('surface_elo', 0):.0%} surface",
                        )
                        cw2.metric(
                            "Serve / return",
                            f"{context_weights_result.get('serve', 1):.2f}x serve",
                            f"{context_weights_result.get('return', 1):.2f}x return",
                        )
                        cw3.metric(
                            "Form / fatigue",
                            f"{context_weights_result.get('form', 1):.2f}x form",
                            f"{context_weights_result.get('fatigue', 1):.2f}x fatigue",
                        )
                        cw4.metric(
                            "Pressure",
                            f"{context_weights_result.get('pressure', 1):.2f}x",
                            f"{context_weights_result.get('deciding', 1):.2f}x deciding",
                        )
                        st.caption(
                            "These weights are selected before the player comparison. They depend "
                            "only on the match context and do not change based on the side you want to bet."
                        )

                    st.markdown("#### Model Foundation")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric(
                        f"{analyzed_a} overall Elo",
                        f"{result['overall_elo'][0]:.0f}",
                    )
                    e2.metric(
                        f"{analyzed_b} overall Elo",
                        f"{result['overall_elo'][1]:.0f}",
                    )
                    e3.metric(
                        f"{analyzed_a} surface Elo",
                        f"{result['surface_elo'][0]:.0f}",
                    )
                    e4.metric(
                        f"{analyzed_b} surface Elo",
                        f"{result['surface_elo'][1]:.0f}",
                    )

                    factor_rows = []
                    for factor in raw_factors:
                        impact = factor["impact_a"]
                        factor_rows.append({
                            "Factor": factor["name"],
                            "Probability impact": impact,
                            "Direction": analyzed_a if impact > 0 else analyzed_b if impact < 0 else "Neutral",
                            "Explanation": factor["reason"],
                        })
                    factor_df = pd.DataFrame(factor_rows).sort_values(
                        "Probability impact",
                        ascending=False,
                    )
                    st.markdown("#### Why Macabets Made This Line")
                    st.dataframe(
                        factor_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Probability impact": st.column_config.NumberColumn(format="%+.1%%")
                        },
                    )

                    profile_a = result["profile_a"]
                    profile_b = result["profile_b"]
                    profile_df = pd.DataFrame([
                        {
                            "Player": analyzed_a,
                            "Rank": profile_a["rank"],
                            "Last-10 win rate": profile_a["recent_win"],
                            f"{result['surface']} win rate": profile_a["surface_win"],
                            "Serve points won": profile_a["serve_points_won"],
                            "Return points won": profile_a["return_points_won"],
                            "Historical sample": profile_a["sample"],
                        },
                        {
                            "Player": analyzed_b,
                            "Rank": profile_b["rank"],
                            "Last-10 win rate": profile_b["recent_win"],
                            f"{result['surface']} win rate": profile_b["surface_win"],
                            "Serve points won": profile_b["serve_points_won"],
                            "Return points won": profile_b["return_points_won"],
                            "Historical sample": profile_b["sample"],
                        },
                    ])
                    st.markdown("#### Player Profiles")
                    st.dataframe(
                        profile_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Last-10 win rate": st.column_config.NumberColumn(format="%.1%%"),
                            f"{result['surface']} win rate": st.column_config.NumberColumn(format="%.1%%"),
                            "Serve points won": st.column_config.NumberColumn(format="%.1%%"),
                            "Return points won": st.column_config.NumberColumn(format="%.1%%"),
                        },
                    )

                    simulation = result["simulation"]
                    st.markdown("#### Outcome Simulation")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric(f"{analyzed_a} wins", f"{simulation['win_probability']:.1%}")
                    s2.metric(f"{analyzed_a} straight sets", f"{simulation['straight_sets_a']:.1%}")
                    s3.metric(f"{analyzed_b} straight sets", f"{simulation['straight_sets_b']:.1%}")
                    s4.metric("Deciding set", f"{simulation['deciding_set']:.1%}")

                    score_df = pd.DataFrame(
                        [
                            {"Set score": score, "Probability": probability}
                            for score, probability in simulation["set_scores"].items()
                        ]
                    ).sort_values("Probability", ascending=False)
                    st.dataframe(
                        score_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Probability": st.column_config.NumberColumn(format="%.1%%")
                        },
                    )

                    st.markdown("#### Pre-Match Decision Record")
                    d1, d2 = st.columns(2)
                    prediction = d1.text_area(
                        "Why Player A wins",
                        value=(
                            f"Macabets gives {analyzed_a} a {model_probability:.1%} win probability, "
                            f"with a fair line of {format_american(fair_odds)}."
                        ),
                        key="auto_prediction",
                    )
                    upset_path = d2.text_area(
                        "Why Player B wins",
                        key="auto_upset_path",
                    )
                    d3, d4 = st.columns(2)
                    biggest_risk = d3.text_area("Biggest risk", key="auto_biggest_risk")
                    assumptions = d4.text_area("Key assumptions", key="auto_assumptions")
                    analysis_notes = st.text_area("Additional notes", key="auto_analysis_notes")

                    if st.button(
                        "Save Automatic Analysis",
                        type="primary",
                        use_container_width=True,
                    ):
                        analyses = st.session_state.analyses.copy()
                        next_analysis_id = int(analyses["analysis_id"].max()) + 1 if not analyses.empty else 1
                        row = {
                            "analysis_id": next_analysis_id,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "match_date": str(market_snapshot.get("match_date", date.today().isoformat())),
                            "tournament": result["tournament"],
                            "surface": result["surface"],
                            "round": result["round"],
                            "player_a": analyzed_a,
                            "player_b": analyzed_b,
                            "market_odds_a": listed_a,
                            "market_odds_b": listed_b,
                            "model_probability_a": model_probability,
                            "fair_odds_a": fair_odds,
                            "no_vig_probability_a": no_vig_a,
                            "no_vig_edge": no_vig_edge,
                            "decision": decision,
                            "minimum_acceptable_odds_a": minimum_price_a,
                            "estimated_roi": expected_roi,
                            "confidence": confidence,
                            "prediction": prediction.strip(),
                            "upset_path": upset_path.strip(),
                            "biggest_risk": biggest_risk.strip(),
                            "assumptions": assumptions.strip(),
                            "notes": (
                                (
                                    f"Considering bet: {considered_player} at "
                                    f"{format_american(considered_market_odds)}. "
                                    f"Side-specific decision: {decision}. "
                                    f"Side-specific estimated ROI: {considered_roi:+.1%}. "
                                    if considered_player else
                                    "No betting side selected. "
                                )
                                + analysis_notes.strip()
                            ).strip(),
                            "result": "Pending",
                            "closing_odds_a": np.nan,
                            "prediction_correct": "",
                            "closing_line_value": np.nan,
                            "review": "",
                            "lesson": "",
                        }
                        st.session_state.analyses = normalize_analyses(
                            pd.concat([analyses, pd.DataFrame([row])], ignore_index=True)
                        )
                        st.success("Automatic analysis saved to the archive.")

                    st.caption(
                        f"Model base probability: {result['base_probability']:.1%}. "
                        f"Final pre-simulation model: {result['model_probability']:.1%}. "
                        f"Simulation count: {simulation['simulations']:,}."
                    )

    with analysis_tabs[1]:
        st.subheader("Analysis Engine — NFL Foundation")
        st.caption(
            "Build an independent fair spread and projected score from explicit matchup grades. "
            "This foundation does not yet fetch automated NFL ratings, injuries or statistics."
        )

        if not NFL_ENGINE_AVAILABLE:
            st.error(f"The NFL engine could not be imported: {NFL_ENGINE_IMPORT_ERROR}")
        else:
            n1, n2, n3, n4 = st.columns(4)
            nfl_away = n1.selectbox("Away team", NFL_TEAMS, key="nfl_away")
            default_home = 1 if len(NFL_TEAMS) > 1 else 0
            nfl_home = n2.selectbox("Home team", NFL_TEAMS, index=default_home, key="nfl_home")
            nfl_spread_home = n3.number_input(
                "Market spread — home team", value=-3.0, step=0.5, key="nfl_market_spread_home",
                help="Enter -3 when the home team is favored by 3; enter +3 when it is an underdog by 3."
            )
            nfl_total = n4.number_input("Market total", value=44.5, step=0.5, key="nfl_market_total")

            c1, c2, c3 = st.columns(3)
            venue_context = c1.selectbox(
                "Venue context",
                ["Standard home field", "Strong home field", "Weak home field", "Neutral site"],
                key="nfl_venue_context",
            )
            weather_total_adjustment = c2.number_input(
                "Weather total adjustment", value=0.0, step=0.5, min_value=-10.0, max_value=10.0,
                key="nfl_weather_total_adjustment",
                help="Use a negative value for wind, rain, snow or other conditions expected to suppress scoring."
            )
            data_quality = c3.slider("Current data quality", 1, 10, 4, key="nfl_data_quality")

            st.markdown("#### Matchup scorecard")
            st.caption("Grade each side from 1–10 using the best information currently available. Leave uncertain categories near 5.")
            headers = st.columns([2, 1, 1])
            headers[0].markdown("**Category**")
            headers[1].markdown(f"**{nfl_home}**")
            headers[2].markdown(f"**{nfl_away}**")

            categories = [
                ("Team quality", "team_quality"),
                ("Quarterback", "quarterback"),
                ("Matchup", "matchup"),
                ("Coaching", "coaching"),
                ("Situational context", "situational"),
            ]
            nfl_scores = {}
            for label, key in categories:
                row = st.columns([2, 1, 1])
                row[0].write(label)
                nfl_scores[f"{key}_home"] = row[1].slider(
                    f"{label} — {nfl_home}", 1.0, 10.0, 5.0, 0.5, key=f"nfl_{key}_home", label_visibility="collapsed"
                )
                nfl_scores[f"{key}_away"] = row[2].slider(
                    f"{label} — {nfl_away}", 1.0, 10.0, 5.0, 0.5, key=f"nfl_{key}_away", label_visibility="collapsed"
                )

            i1, i2 = st.columns(2)
            injury_home = i1.number_input(
                f"Injury adjustment — {nfl_home} (points)", value=0.0, step=0.5, min_value=-10.0, max_value=10.0, key="nfl_injury_home"
            )
            injury_away = i2.number_input(
                f"Injury adjustment — {nfl_away} (points)", value=0.0, step=0.5, min_value=-10.0, max_value=10.0, key="nfl_injury_away"
            )

            nfl_disabled = nfl_home == nfl_away
            if nfl_disabled:
                st.warning("Select two different NFL teams.")

            if st.button("Analyze NFL Matchup", type="primary", use_container_width=True, disabled=nfl_disabled):
                try:
                    st.session_state.nfl_result = analyze_nfl_matchup(
                        away_team=nfl_away, home_team=nfl_home, market_spread_home=nfl_spread_home,
                        market_total=nfl_total, venue_context=venue_context,
                        team_quality_home=nfl_scores["team_quality_home"], team_quality_away=nfl_scores["team_quality_away"],
                        quarterback_home=nfl_scores["quarterback_home"], quarterback_away=nfl_scores["quarterback_away"],
                        matchup_home=nfl_scores["matchup_home"], matchup_away=nfl_scores["matchup_away"],
                        coaching_home=nfl_scores["coaching_home"], coaching_away=nfl_scores["coaching_away"],
                        situational_home=nfl_scores["situational_home"], situational_away=nfl_scores["situational_away"],
                        injury_adjustment_home=injury_home, injury_adjustment_away=injury_away,
                        weather_total_adjustment=weather_total_adjustment, data_quality=data_quality,
                    )
                except Exception as exc:
                    st.session_state.pop("nfl_result", None)
                    st.error(f"NFL analysis failed: {exc}")

            nfl_result = st.session_state.get("nfl_result")
            if nfl_result:
                st.divider()
                st.markdown(f"### {nfl_result['away_team']} at {nfl_result['home_team']}")
                st.warning(nfl_result["foundation_warning"])
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Macabets fair spread", f"{nfl_result['home_team']} {nfl_result['fair_spread_home']:+.1f}")
                r2.metric("Projected score", f"{nfl_result['home_team']} {nfl_result['projected_home_score']:.1f} – {nfl_result['away_team']} {nfl_result['projected_away_score']:.1f}")
                r3.metric("Fair total", f"{nfl_result['fair_total']:.1f}")
                r4.metric("Confidence", f"{nfl_result['confidence']}/10")

                p1, p2, p3, p4 = st.columns(4)
                p1.metric(f"{nfl_result['home_team']} win probability", f"{nfl_result['home_win_probability']:.1%}")
                p2.metric(f"{nfl_result['away_team']} win probability", f"{nfl_result['away_win_probability']:.1%}")
                p3.metric(f"Fair ML — {nfl_result['home_team']}", format_american(nfl_result['fair_moneyline_home']))
                p4.metric(f"Fair ML — {nfl_result['away_team']}", format_american(nfl_result['fair_moneyline_away']))

                st.markdown(f"#### Recommendation: {nfl_result['recommendation']}")
                st.write(nfl_result["recommendation_reason"])
                st.write(nfl_result["projected_script"])
                st.caption(f"Upset risk: {nfl_result['upset_risk']}. Data quality: {nfl_result['data_quality']}/10.")

                factor_frame = pd.DataFrame(nfl_result["factor_rows"])
                st.markdown("#### Probability drivers")
                st.dataframe(factor_frame, use_container_width=True, hide_index=True)


    with analysis_tabs[2]:
        st.subheader("Outcome Simulator")
        r1, r2, r3, r4 = st.columns(4)
        sim_bankroll = r1.number_input("Simulation bankroll", min_value=100.0, value=float(current_bankroll), step=1000.0)
        sim_odds = r2.number_input("Odds per bet", value=-250, step=5)
        model_prob_pct = r3.slider("Your estimated true win probability", 1.0, 99.0, 75.0, 0.5)
        number_bets = r4.number_input("Number of bets", min_value=1, max_value=500, value=50, step=1)

        s1, s2, s3 = st.columns(3)
        target_each = s1.number_input("Target profit per bet", min_value=1.0, value=float(st.session_state.target_profit), step=500.0)
        simulations = s2.number_input("Simulation runs", min_value=100, max_value=20000, value=3000, step=100)
        staking = s3.selectbox("Staking method", ["Target-profit stake", "Flat % bankroll", "Quarter Kelly"])
        flat_pct = st.slider("Flat stake %", 0.5, 25.0, 5.0, 0.5) / 100

        true_prob = model_prob_pct / 100
        implied = implied_probability(int(sim_odds))
        edge = true_prob - implied
        full_kelly = kelly_fraction(true_prob, int(sim_odds))
        quarter_kelly = full_kelly / 4

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Implied probability", f"{implied:.1%}")
        q2.metric("Estimated edge", f"{edge:.1%}")
        q3.metric("Full Kelly", f"{full_kelly:.1%}")
        q4.metric("Quarter Kelly", f"{quarter_kelly:.1%}")

        if edge <= 0:
            st.error("Your estimated probability does not beat the sportsbook's implied probability.")
        elif sim_odds <= -500:
            st.warning("The favorite may win often, but the payoff structure creates severe loss-recovery risk.")

        if st.button("Run Simulation", type="primary", use_container_width=True):
            rng = np.random.default_rng()
            paths = np.zeros((int(simulations), int(number_bets) + 1))
            paths[:, 0] = sim_bankroll
            ruined = np.zeros(int(simulations), dtype=bool)

            for run in range(int(simulations)):
                bank = sim_bankroll
                for n in range(1, int(number_bets) + 1):
                    if staking == "Target-profit stake":
                        stake_n = stake_to_win(int(sim_odds), target_each)
                    elif staking == "Flat % bankroll":
                        stake_n = bank * flat_pct
                    else:
                        stake_n = bank * quarter_kelly
                    stake_n = min(stake_n, bank)
                    if stake_n <= 0:
                        paths[run, n:] = bank
                        ruined[run] = True
                        break
                    if rng.random() < true_prob:
                        bank += potential_profit(int(sim_odds), stake_n)
                    else:
                        bank -= stake_n
                    paths[run, n] = bank
                    if bank <= 0:
                        paths[run, n:] = 0
                        ruined[run] = True
                        break

            ending = paths[:, -1]
            median_path = np.median(paths, axis=0)
            p10_path = np.percentile(paths, 10, axis=0)
            p90_path = np.percentile(paths, 90, axis=0)
            probability_profit = float(np.mean(ending > sim_bankroll))
            probability_loss_half = float(np.mean(ending <= sim_bankroll * 0.5))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median ending bankroll", money(float(np.median(ending))))
            m2.metric("Chance of profit", f"{probability_profit:.1%}")
            m3.metric("Chance of losing 50%+", f"{probability_loss_half:.1%}")
            m4.metric("5th percentile finish", money(float(np.percentile(ending, 5))))

            fig, ax = plt.subplots()
            x = np.arange(int(number_bets) + 1)
            ax.plot(x, median_path, label="Median")
            ax.fill_between(x, p10_path, p90_path, alpha=0.2, label="10th–90th percentile")
            ax.axhline(sim_bankroll, linewidth=1)
            ax.set_xlabel("Bet number")
            ax.set_ylabel("Bankroll ($)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with tabs[2]:
    bet_tabs = st.tabs(["New Bet", "Bet Ledger", "Performance"])

    with bet_tabs[0]:
        st.subheader("Enter a Bet")
        a, b, c = st.columns(3)
        sport = a.selectbox("Sport", SPORTS)
        event_date = b.date_input("Event date", value=date.today())
        bet_type = c.selectbox("Bet type", BET_TYPES)

        event = st.text_input("Event / matchup", placeholder="Bills vs Jets")
        selection = st.text_input("Selection", placeholder="Bills moneyline")

        d, e, f = st.columns(3)
        odds = d.number_input("American odds", value=-200, step=5)
        target_profit = e.number_input(
            "Target profit",
            min_value=1.0,
            value=float(st.session_state.target_profit),
            step=500.0,
        )
        suggested_stake = stake_to_win(int(odds), target_profit)
        stake_mode = f.radio("Stake method", ["Risk enough to win target", "Enter my own stake"], horizontal=True)

        if stake_mode == "Risk enough to win target":
            stake = suggested_stake
            f.metric("Required stake", money(stake))
        else:
            stake = f.number_input("Stake", min_value=0.0, value=1000.0, step=100.0)

        p_profit = potential_profit(int(odds), stake)
        implied = implied_probability(int(odds))
        g, h, i = st.columns(3)
        g.metric("Potential profit", money(p_profit))
        h.metric("Implied probability", f"{implied:.1%}")
        i.metric("Total return", money(stake + p_profit))

        j, k, l = st.columns(3)
        book = j.text_input("Sportsbook", placeholder="Optional")
        confidence = k.slider("Confidence", 1, 10, 7)
        status = l.selectbox("Initial status", STATUSES, index=0)
        notes = st.text_area("Notes / thesis")

        stake_pct = stake / current_bankroll if current_bankroll > 0 else 0
        if odds <= -500:
            st.warning("Very heavy favorite: one loss can erase several wins. Confirm the price is justified.")
        if stake_pct >= 0.20:
            st.error(f"This stake is {stake_pct:.1%} of the current bankroll.")
        elif stake_pct >= 0.10:
            st.warning(f"This stake is {stake_pct:.1%} of the current bankroll.")
        elif stake_pct > 0:
            st.caption(f"Stake size: {stake_pct:.1%} of current bankroll.")

        if st.button("Add Bet", type="primary", use_container_width=True):
            if not event.strip() or not selection.strip():
                st.error("Enter both the event and selection.")
            elif stake <= 0:
                st.error("Stake must be greater than zero.")
            else:
                next_id = int(bets["id"].max()) + 1 if not bets.empty else 1
                initial_result = 0.0
                if status == "Won":
                    initial_result = p_profit
                elif status == "Lost":
                    initial_result = -stake
                row = {
                    "id": next_id,
                    "date": event_date.isoformat(),
                    "sport": sport,
                    "event": event.strip(),
                    "selection": selection.strip(),
                    "bet_type": bet_type,
                    "odds": int(odds),
                    "stake": float(stake),
                    "target_profit": float(target_profit),
                    "status": status,
                    "result_profit": float(initial_result),
                    "book": book.strip(),
                    "confidence": confidence,
                    "notes": notes.strip(),
                }
                st.session_state.bets = pd.concat([bets, pd.DataFrame([row])], ignore_index=True)
                st.success("Bet added. Open Bet Ledger to settle or edit it.")

    with bet_tabs[1]:
        st.subheader("Bet Ledger")
        if bets.empty:
            st.write("No bets have been added.")
        else:
            filter_col1, filter_col2 = st.columns(2)
            sport_filter = filter_col1.multiselect("Filter sport", SPORTS, default=SPORTS)
            status_filter = filter_col2.multiselect("Filter status", STATUSES, default=STATUSES)
            filtered = bets[bets["sport"].isin(sport_filter) & bets["status"].isin(status_filter)]
            st.dataframe(filtered, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Settle or Update a Bet")
            selected_id = st.selectbox(
                "Bet ID",
                bets["id"].astype(int).tolist(),
                format_func=lambda x: f"#{x} — {bets.loc[bets['id'] == x, 'selection'].iloc[0]}",
            )
            selected = bets[bets["id"] == selected_id].iloc[0]
            u1, u2, u3 = st.columns(3)
            new_status = u1.selectbox(
                "Status",
                STATUSES,
                index=STATUSES.index(selected["status"]) if selected["status"] in STATUSES else 0,
            )
            default_result = float(selected["result_profit"])
            if new_status == "Won":
                default_result = potential_profit(int(selected["odds"]), float(selected["stake"]))
            elif new_status == "Lost":
                default_result = -float(selected["stake"])
            elif new_status == "Void":
                default_result = 0.0
            result_profit = u2.number_input("Net result", value=float(default_result), step=100.0)
            updated_notes = u3.text_input("Updated note", value=str(selected["notes"]))

            col_save, col_delete = st.columns(2)
            if col_save.button("Save Update", type="primary", use_container_width=True):
                idx = st.session_state.bets.index[st.session_state.bets["id"] == selected_id][0]
                st.session_state.bets.at[idx, "status"] = new_status
                st.session_state.bets.at[idx, "result_profit"] = float(result_profit)
                st.session_state.bets.at[idx, "notes"] = updated_notes
                st.success("Bet updated.")
                st.rerun()

            if col_delete.button("Delete Bet", use_container_width=True):
                st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != selected_id].reset_index(drop=True)
                st.success("Bet deleted.")
                st.rerun()

    with bet_tabs[2]:
        st.subheader("Performance Analysis")
        if settled.empty:
            st.write("Settle bets to populate performance analytics.")
        else:
            perf = settled.copy()
            perf["odds_band"] = pd.cut(
                perf["odds"],
                bins=[-10000, -500, -350, -250, -180, -110, 0, 10000],
                labels=["≤ -500", "-499 to -350", "-349 to -250", "-249 to -180", "-179 to -110", "Even/+ odds", "Other"],
                include_lowest=True,
            )
            p1, p2 = st.columns(2)
            with p1:
                by_sport = perf.groupby("sport").agg(
                    bets=("id", "count"),
                    staked=("stake", "sum"),
                    profit=("result_profit", "sum"),
                )
                by_sport["roi"] = by_sport["profit"] / by_sport["staked"].replace(0, np.nan)
                st.markdown("#### By Sport")
                st.dataframe(by_sport.reset_index(), use_container_width=True, hide_index=True)
            with p2:
                by_band = perf.groupby("odds_band", observed=False).agg(
                    bets=("id", "count"),
                    staked=("stake", "sum"),
                    profit=("result_profit", "sum"),
                )
                by_band["roi"] = by_band["profit"] / by_band["staked"].replace(0, np.nan)
                st.markdown("#### By Odds Range")
                st.dataframe(by_band.reset_index(), use_container_width=True, hide_index=True)

            ordered = perf.sort_values(["date", "id"]).copy()
            ordered["cumulative_profit"] = ordered["result_profit"].cumsum()
            ordered["bankroll_curve"] = st.session_state.bankroll + ordered["cumulative_profit"]
            fig, ax = plt.subplots()
            ax.plot(range(1, len(ordered) + 1), ordered["bankroll_curve"], marker="o")
            ax.axhline(st.session_state.bankroll, linewidth=1)
            ax.set_xlabel("Settled bet")
            ax.set_ylabel("Bankroll ($)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with tabs[3]:
    st.subheader("Daily Slate")
    analysis_ready_message = st.session_state.pop("daily_slate_analysis_ready", None)
    if analysis_ready_message:
        st.success(analysis_ready_message)
    st.caption(
        "Automatically load today's market card without changing the existing manual slate or Analysis Engine."
    )

    st.markdown("### Automatic Slate Preview")
    api_key = _odds_api_key()
    if not api_key:
        st.info(
            "Automatic loading is ready but inactive. Add THE_ODDS_API_KEY to Streamlit Secrets; "
            "the key is never stored in GitHub or displayed in the app."
        )
    else:
        try:
            active_sports, sports_usage = fetch_active_sports(api_key)
            fixed_choices = {
                "MLB": "baseball_mlb",
                "WNBA": "basketball_wnba",
                "NFL": "americanfootball_nfl",
                "College Football": "americanfootball_ncaaf",
                "NBA": "basketball_nba",
            }
            tennis_items = discover_active_tennis_sports(active_sports)
            tennis_like_items = [
                item for item in (active_sports or [])
                if "tennis" in " ".join(
                    str(item.get(field, "")) for field in ("key", "group", "title", "description")
                ).lower()
            ]
            available_choices = {
                "Tennis — All ATP & WTA": "__all_tennis__",
                **{
                    label: key for label, key in fixed_choices.items()
                    if any(item.get("key") == key and item.get("active", True) for item in active_sports)
                },
            }

            if len(available_choices) == 1 and not tennis_items:
                st.caption("No other configured leagues are active right now, but Tennis diagnostics remain available.")

            if available_choices:
                auto_col1, auto_col2 = st.columns([2, 1])
                selected_label = auto_col1.selectbox(
                    "Sport",
                    list(available_choices.keys()),
                    key="automatic_slate_sport",
                )
                refresh = auto_col2.button("Refresh Slate", use_container_width=True)
                if refresh:
                    fetch_sport_odds.clear()

                with st.spinner("Loading today's market slate..."):
                    tennis_load_errors = []
                    if available_choices[selected_label] == "__all_tennis__":
                        if tennis_items:
                            automatic_slate, usage, tennis_load_errors = combine_tennis_slate(api_key, tennis_items)
                        else:
                            automatic_slate = pd.DataFrame()
                            usage = sports_usage
                    else:
                        api_events, usage = fetch_sport_odds(api_key, available_choices[selected_label])
                        automatic_slate = normalize_api_slate(api_events, selected_label)

                if available_choices[selected_label] == "__all_tennis__":
                    with st.expander("Tennis API diagnostics", expanded=automatic_slate.empty):
                        st.write(f"Active ATP/WTA feeds discovered: **{len(tennis_items)}**")
                        if tennis_items:
                            for item in tennis_items:
                                st.caption(
                                    f"{item.get('title', item.get('key', 'Tennis'))} — `{item.get('key', 'unknown')}`"
                                )
                        elif tennis_like_items:
                            st.caption("The API returned tennis-related entries, but none matched an active ATP/WTA feed:")
                            for item in tennis_like_items:
                                st.caption(
                                    f"{item.get('title', item.get('key', 'Tennis'))} — `{item.get('key', 'unknown')}` "
                                    f"(active={item.get('active', 'unknown')})"
                                )
                        else:
                            st.caption("The /sports endpoint returned no tennis-related sport keys.")
                        st.caption(
                            f"API requests remaining: {sports_usage.get('remaining', '—')} | "
                            f"requests used: {sports_usage.get('used', '—')}"
                        )

                if tennis_load_errors:
                    st.warning(
                        f"Loaded the available tennis card, but {len(tennis_load_errors)} tournament feed(s) failed."
                    )
                    with st.expander("Tennis feed details", expanded=False):
                        for message in tennis_load_errors:
                            st.caption(message)

                if automatic_slate.empty:
                    if available_choices[selected_label] == "__all_tennis__" and not tennis_items:
                        st.info(
                            "Tennis is enabled in Macabets, but The Odds API is not currently reporting an active "
                            "ATP or WTA feed for this account. Open the diagnostics above to confirm the returned keys and quota."
                        )
                    else:
                        st.info(f"No {selected_label} events with US moneyline odds are scheduled today.")
                else:
                    if available_choices[selected_label] == "__all_tennis__":
                        tournament_count = automatic_slate["sport"].nunique()
                        st.success(
                            f"Loaded {len(automatic_slate)} ATP/WTA matches across {tournament_count} active tournament(s)."
                        )
                    st.caption(
                        f"Best available US moneyline shown for each side. API requests remaining: {usage['remaining']}."
                    )
                    automatic_display = automatic_slate[
                        ["time_et", "sport", "participant_a", "odds_a", "book_a", "participant_b", "odds_b", "book_b"]
                    ].copy()
                    automatic_display.columns = [
                        "Time (ET)", "League", "Participant A", "Best Odds A", "Book A",
                        "Participant B", "Best Odds B", "Book B"
                    ]
                    st.dataframe(automatic_display, use_container_width=True, hide_index=True)

                    st.markdown("#### Send an Event to the Existing Manual Slate")
                    event_options = automatic_slate.index.tolist()
                    selected_event_index = st.selectbox(
                        "Automatic event",
                        event_options,
                        format_func=lambda idx: (
                            f"{automatic_slate.loc[idx, 'time_et']} — "
                            f"{automatic_slate.loc[idx, 'participant_a']} vs {automatic_slate.loc[idx, 'participant_b']}"
                        ),
                        key="automatic_slate_event",
                    )
                    selected_event = automatic_slate.loc[selected_event_index]
                    is_tennis_event = available_choices[selected_label] == "__all_tennis__"
                    analyze_col, add_col = st.columns(2)

                    if analyze_col.button(
                        "Analyze This Tennis Match",
                        type="primary",
                        use_container_width=True,
                        disabled=not is_tennis_event,
                    ):
                        if pd.isna(selected_event["odds_a"]) or pd.isna(selected_event["odds_b"]):
                            st.error("Both sides need moneyline odds before this matchup can be analyzed.")
                        else:
                            tournament_name = str(selected_event["sport"])
                            tournament_lower = tournament_name.lower()
                            if any(name in tournament_lower for name in ["wimbledon"]):
                                inferred_surface = "Grass"
                            elif any(name in tournament_lower for name in [
                                "french open", "roland garros", "monte carlo", "madrid",
                                "rome", "italian open", "barcelona", "hamburg", "kitzbuhel",
                                "umag", "bastad", "geneva", "estoril", "munich"
                            ]):
                                inferred_surface = "Clay"
                            else:
                                inferred_surface = "Hard"

                            if any(name in tournament_lower for name in [
                                "australian open", "french open", "roland garros",
                                "wimbledon", "us open"
                            ]):
                                inferred_category = "Grand Slam"
                            elif any(token in tournament_lower for token in ["masters", "1000"]):
                                inferred_category = "Masters 1000"
                            elif "500" in tournament_lower:
                                inferred_category = "ATP 500"
                            else:
                                inferred_category = "ATP 250"

                            inferred_format = (
                                "Best of 5"
                                if inferred_category == "Grand Slam" and "WTA" not in tournament_name.upper()
                                else "Best of 3"
                            )
                            event_date = selected_event["start_time"].date()
                            player_a_name = str(selected_event["participant_a"])
                            player_b_name = str(selected_event["participant_b"])
                            odds_a_value = int(selected_event["odds_a"])
                            odds_b_value = int(selected_event["odds_b"])

                            st.session_state.pending_fair_line_prefill = {
                                "fle_date": event_date,
                                "fle_tournament": tournament_name,
                                "fle_surface": inferred_surface,
                                "fle_round": "R32",
                                "fle_favorite": player_a_name,
                                "fle_opponent": player_b_name,
                                "fle_market_a": odds_a_value,
                                "fle_market_b": odds_b_value,
                                "auto_match_date": event_date,
                                "auto_tournament": tournament_name,
                                "auto_surface": inferred_surface,
                                "auto_round": "R32",
                                "auto_tournament_category": inferred_category,
                                "auto_environment": "Outdoor",
                                "auto_match_format": inferred_format,
                                "auto_player_a": player_a_name,
                                "auto_player_b": player_b_name,
                                "auto_considering_bet": "Just analyze",
                                "auto_market_a": odds_a_value,
                                "auto_market_b": odds_b_value,
                                "auto_simulations": 20000,
                            }
                            st.session_state.run_analysis_from_daily_slate = True
                            st.rerun()

                    if add_col.button("Add Event to Manual Slate", use_container_width=True):
                        if pd.isna(selected_event["odds_a"]) or pd.isna(selected_event["odds_b"]):
                            st.error("Both sides need moneyline odds before this event can be added.")
                        else:
                            slate = normalize_slate(st.session_state.daily_slate)
                            next_id = int(slate["slate_id"].max()) + 1 if not slate.empty else 1
                            implied_a, implied_b, _ = no_vig_probabilities(
                                int(selected_event["odds_a"]), int(selected_event["odds_b"])
                            )
                            row = {
                                "slate_id": next_id,
                                "match_date": selected_event["start_time"].date().isoformat(),
                                "tournament": str(selected_event["sport"]),
                                "surface": "Unverified" if "ATP" in str(selected_event["sport"]) or "WTA" in str(selected_event["sport"]) else "N/A",
                                "round": "Unverified" if "ATP" in str(selected_event["sport"]) or "WTA" in str(selected_event["sport"]) else "Game",
                                "player_a": str(selected_event["participant_a"]),
                                "player_b": str(selected_event["participant_b"]),
                                "market_odds_a": int(selected_event["odds_a"]),
                                "market_odds_b": int(selected_event["odds_b"]),
                                "model_probability_a": float(implied_a),
                                "confidence": 1,
                                "notes": (
                                    "Imported automatically from market odds. Model probability has not been run yet. "
                                    "Do not treat the slate grade as a Macabets recommendation until analyzed."
                                ),
                            }
                            st.session_state.daily_slate = normalize_slate(
                                pd.concat([slate, pd.DataFrame([row])], ignore_index=True)
                            )
                            st.success("Event added safely to the existing manual slate.")
                            st.rerun()
        except Exception as exc:
            st.error(f"Automatic slate could not load: {exc}")
            st.caption("The manual Daily Slate below remains fully available and unaffected.")

    st.divider()
    st.markdown("### Manual Slate and Ranking")
    st.caption(
        "This is the existing slate workflow. It remains independent from the automatic preview."
    )

    with st.expander("Add Matchup Manually", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        slate_date = s1.date_input("Match date", value=date.today(), key="slate_date")
        slate_tournament = s2.text_input("Tournament", placeholder="Montreal", key="slate_tournament")
        slate_surface = s3.selectbox(
            "Surface", ["Hard", "Clay", "Grass", "Indoor Hard"], key="slate_surface"
        )
        slate_round = s4.selectbox(
            "Round",
            ["R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
            key="slate_round",
        )

        p1, p2 = st.columns(2)
        slate_player_a = p1.text_input("Player A", key="slate_player_a")
        slate_player_b = p2.text_input("Player B", key="slate_player_b")

        o1, o2, o3, o4 = st.columns(4)
        slate_odds_a = o1.number_input(
            "Sportsbook odds — A", value=-150, step=5, key="slate_odds_a"
        )
        slate_odds_b = o2.number_input(
            "Sportsbook odds — B", value=130, step=5, key="slate_odds_b"
        )
        slate_probability = o3.slider(
            "Macabets probability — A",
            1.0, 99.0, 60.0, 0.5,
            key="slate_probability",
        ) / 100
        slate_confidence = o4.slider(
            "Data confidence", 1, 10, 6, key="slate_confidence"
        )
        slate_notes = st.text_input(
            "Quick note",
            placeholder="Example: Better hard-court form, but fatigue needs verification.",
            key="slate_notes",
        )

        if st.button("Add to Daily Slate", type="primary", use_container_width=True):
            if not slate_player_a.strip() or not slate_player_b.strip():
                st.error("Enter both players.")
            else:
                slate = normalize_slate(st.session_state.daily_slate)
                next_id = int(slate["slate_id"].max()) + 1 if not slate.empty else 1
                row = {
                    "slate_id": next_id,
                    "match_date": slate_date.isoformat(),
                    "tournament": slate_tournament.strip(),
                    "surface": slate_surface,
                    "round": slate_round,
                    "player_a": slate_player_a.strip(),
                    "player_b": slate_player_b.strip(),
                    "market_odds_a": int(slate_odds_a),
                    "market_odds_b": int(slate_odds_b),
                    "model_probability_a": float(slate_probability),
                    "confidence": int(slate_confidence),
                    "notes": slate_notes.strip(),
                }
                st.session_state.daily_slate = normalize_slate(
                    pd.concat([slate, pd.DataFrame([row])], ignore_index=True)
                )
                st.success("Matchup added to the Daily Slate.")
                st.rerun()

    st.divider()
    slate_upload = st.file_uploader(
        "Import Daily Slate CSV",
        type=["csv"],
        key="daily_slate_upload",
    )
    if slate_upload is not None:
        try:
            st.session_state.daily_slate = normalize_slate(pd.read_csv(slate_upload))
            st.success(f"Loaded {len(st.session_state.daily_slate)} slate matchups.")
        except Exception as exc:
            st.error(f"Could not load Daily Slate CSV: {exc}")

    slate = score_daily_slate(st.session_state.daily_slate)

    if slate.empty:
        st.info("No matchups have been added to today's slate.")
    else:
        f1, f2, f3 = st.columns(3)
        tournament_options = sorted(
            [value for value in slate["tournament"].dropna().astype(str).unique() if value]
        )
        tournament_filter = f1.multiselect(
            "Tournament",
            tournament_options,
            default=tournament_options,
        )
        surface_options = sorted(slate["surface"].dropna().astype(str).unique().tolist())
        surface_filter = f2.multiselect(
            "Surface",
            surface_options,
            default=surface_options,
        )
        decision_filter = f3.multiselect(
            "Decision",
            ["BET", "WATCH", "PASS"],
            default=["BET", "WATCH", "PASS"],
        )

        filtered_slate = slate[
            slate["tournament"].astype(str).isin(tournament_filter)
            & slate["surface"].astype(str).isin(surface_filter)
            & slate["decision"].isin(decision_filter)
        ].copy()

        bet_count = int((slate["decision"] == "BET").sum())
        watch_count = int((slate["decision"] == "WATCH").sum())
        pass_count = int((slate["decision"] == "PASS").sum())
        best_score = float(slate["opportunity_score"].max())

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Slate Matches", len(slate))
        q2.metric("Bet Candidates", bet_count)
        q3.metric("Watch List", watch_count)
        q4.metric("Top Opportunity Score", f"{best_score:.1f}")

        display = filtered_slate[
            [
                "slate_id", "match_date", "tournament", "round", "player_a", "player_b",
                "market_odds_a", "market_odds_b", "model_probability_a",
                "fair_odds_a", "no_vig_edge", "estimated_roi", "confidence",
                "decision", "opportunity_score"
            ]
        ].copy()
        display["model_probability_a"] = display["model_probability_a"].map(lambda x: f"{x:.1%}")
        display["no_vig_edge"] = display["no_vig_edge"].map(lambda x: f"{x:+.1%}")
        display["estimated_roi"] = display["estimated_roi"].map(lambda x: f"{x:+.1%}")
        display["opportunity_score"] = display["opportunity_score"].map(lambda x: f"{x:.1f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### Open a Matchup for Deep Analysis")
        selected_slate_id = st.selectbox(
            "Slate matchup",
            slate["slate_id"].astype(int).tolist(),
            format_func=lambda x: (
                f"#{x} — "
                f"{slate.loc[slate['slate_id'] == x, 'player_a'].iloc[0]} vs "
                f"{slate.loc[slate['slate_id'] == x, 'player_b'].iloc[0]}"
            ),
        )
        selected_slate = slate[slate["slate_id"] == selected_slate_id].iloc[0]

        a1, a2, a3 = st.columns(3)
        a1.metric("Decision", selected_slate["decision"])
        a2.metric("Expected ROI", f"{selected_slate['estimated_roi']:+.1%}")
        a3.metric("Fair line — A", format_american(selected_slate["fair_odds_a"]))

        load_col, delete_col = st.columns(2)
        if load_col.button(
            "Load into Analysis Engine",
            type="primary",
            use_container_width=True,
        ):
            surface_values = ["Hard", "Clay", "Grass", "Indoor Hard"]
            round_values = ["R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"]
            st.session_state.pending_fair_line_prefill = {
                "fle_date": date.fromisoformat(str(selected_slate["match_date"])),
                "fle_tournament": str(selected_slate["tournament"]),
                "fle_surface": (
                    str(selected_slate["surface"])
                    if str(selected_slate["surface"]) in surface_values else "Hard"
                ),
                "fle_round": (
                    str(selected_slate["round"])
                    if str(selected_slate["round"]) in round_values else "R32"
                ),
                "fle_favorite": str(selected_slate["player_a"]),
                "fle_opponent": str(selected_slate["player_b"]),
                "fle_market_a": int(selected_slate["market_odds_a"]),
                "fle_market_b": int(selected_slate["market_odds_b"]),
            }
            st.success("Matchup loaded. Open the Analysis Engine tab.")
            st.rerun()

        if delete_col.button("Remove from Slate", use_container_width=True):
            st.session_state.daily_slate = st.session_state.daily_slate[
                st.session_state.daily_slate["slate_id"] != selected_slate_id
            ].reset_index(drop=True)
            st.success("Matchup removed.")
            st.rerun()

        export_slate = normalize_slate(st.session_state.daily_slate)
        st.download_button(
            "Download Daily Slate CSV",
            export_slate.to_csv(index=False).encode("utf-8"),
            f"macabets_daily_slate_{date.today().isoformat()}.csv",
            "text/csv",
            use_container_width=True,
        )

with tabs[4]:
    archive_tabs = st.tabs(["Analysis Archive", "Matchup Lab"])

    with archive_tabs[0]:
        st.subheader("Analysis Archive")
        analyses = normalize_analyses(st.session_state.analyses)

        if analyses.empty:
            st.write("No pre-match analyses saved yet.")
        else:
            completed = analyses[analyses["result"].isin(["Player A Won", "Player B Won"])]
            completed_count = len(completed)
            correct_count = int((completed["prediction_correct"].astype(str) == "Yes").sum())
            accuracy = correct_count / completed_count if completed_count else 0.0
            avg_roi = pd.to_numeric(analyses["estimated_roi"], errors="coerce").mean()
            avg_clv = pd.to_numeric(completed["closing_line_value"], errors="coerce").mean()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Analyses Logged", len(analyses))
            k2.metric("Completed Reviews", completed_count)
            k3.metric("Prediction Accuracy", f"{accuracy:.1%}" if completed_count else "—")
            k4.metric("Average Closing Edge", f"{avg_clv:.1%}" if pd.notna(avg_clv) else "—")

            display_cols = [
                "analysis_id", "match_date", "tournament", "round", "player_a", "player_b",
                "market_odds_a", "fair_odds_a", "model_probability_a", "no_vig_edge",
                "decision", "estimated_roi", "confidence", "result", "prediction_correct"
            ]
            archive_view = analyses[display_cols].sort_values("analysis_id", ascending=False)
            st.dataframe(archive_view, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Post-Match Review")
            selected_analysis_id = st.selectbox(
                "Analysis ID",
                analyses["analysis_id"].astype(int).sort_values(ascending=False).tolist(),
                format_func=lambda x: (
                    f"#{x} — "
                    f"{analyses.loc[analyses['analysis_id'] == x, 'player_a'].iloc[0]} vs "
                    f"{analyses.loc[analyses['analysis_id'] == x, 'player_b'].iloc[0]}"
                ),
            )
            selected_analysis = analyses[analyses["analysis_id"] == selected_analysis_id].iloc[0]

            st.markdown(
                f"**Original call:** {selected_analysis['player_a']} "
                f"{format_american(selected_analysis['market_odds_a']) if pd.notna(selected_analysis['market_odds_a']) else ''}  |  "
                f"Macabets {selected_analysis['model_probability_a']:.1%} "
                f"({format_american(selected_analysis['fair_odds_a']) if pd.notna(selected_analysis['fair_odds_a']) else '—'})"
            )

            a1, a2, a3 = st.columns(3)
            result_options = ["Pending", "Player A Won", "Player B Won", "Void"]
            current_result = selected_analysis["result"] if selected_analysis["result"] in result_options else "Pending"
            result = a1.selectbox("Result", result_options, index=result_options.index(current_result))
            closing_default = (
                int(selected_analysis["closing_odds_a"])
                if pd.notna(selected_analysis["closing_odds_a"]) and float(selected_analysis["closing_odds_a"]) != 0
                else int(selected_analysis["market_odds_a"])
            )
            closing_odds = a2.number_input(
                "Closing odds on Player A",
                value=closing_default,
                step=5,
            )

            predicted_a = float(selected_analysis["model_probability_a"]) >= 0.5
            actual_a = result == "Player A Won"
            auto_correct = "Yes" if result in ["Player A Won", "Player B Won"] and predicted_a == actual_a else (
                "No" if result in ["Player A Won", "Player B Won"] else ""
            )
            correctness_options = ["", "Yes", "No"]
            saved_correctness = str(selected_analysis["prediction_correct"])
            default_correctness = saved_correctness if saved_correctness in correctness_options and saved_correctness else auto_correct
            prediction_correct = a3.selectbox(
                "Was the prediction correct?",
                correctness_options,
                index=correctness_options.index(default_correctness) if default_correctness in correctness_options else 0,
            )

            review = st.text_area(
                "What happened?",
                value=str(selected_analysis["review"]),
                placeholder="Describe how the match actually unfolded—not just the final score.",
            )
            lesson = st.text_area(
                "What should Macabets learn?",
                value=str(selected_analysis["lesson"]),
                placeholder="Identify whether the model, assumptions, context, or price assessment needs adjustment.",
            )

            calculated_clv = closing_line_value(selected_analysis["model_probability_a"], closing_odds)
            if pd.notna(calculated_clv):
                st.caption(f"Model edge at closing price: {calculated_clv:+.1%}")

            save_col, delete_col = st.columns(2)
            if save_col.button("Save Post-Match Review", type="primary", use_container_width=True):
                idx = st.session_state.analyses.index[
                    st.session_state.analyses["analysis_id"] == selected_analysis_id
                ][0]
                st.session_state.analyses.at[idx, "result"] = result
                st.session_state.analyses.at[idx, "closing_odds_a"] = int(closing_odds)
                st.session_state.analyses.at[idx, "prediction_correct"] = prediction_correct
                st.session_state.analyses.at[idx, "closing_line_value"] = float(calculated_clv)
                st.session_state.analyses.at[idx, "review"] = review.strip()
                st.session_state.analyses.at[idx, "lesson"] = lesson.strip()
                st.success("Post-match review saved.")
                st.rerun()

            if delete_col.button("Delete Analysis", use_container_width=True):
                st.session_state.analyses = st.session_state.analyses[
                    st.session_state.analyses["analysis_id"] != selected_analysis_id
                ].reset_index(drop=True)
                st.success("Analysis deleted.")
                st.rerun()

            csv = normalize_analyses(st.session_state.analyses).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Analysis Archive CSV",
                csv,
                f"macabets_analysis_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True,
            )

    with archive_tabs[1]:
        st.subheader("Matchup Lab")
        sport_lab = st.selectbox("Choose sport", SPORTS, key="lab_sport")

        if sport_lab in ["NFL", "College Football"]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Favorite")
                fav = st.text_input("Favorite team")
                fav_form = st.text_area("Recent form")
                fav_off = st.text_area("Offensive strengths / weaknesses")
                fav_def = st.text_area("Defensive strengths / weaknesses")
                fav_inj = st.text_area("Injuries / availability")
            with c2:
                st.markdown("#### Opponent")
                dog = st.text_input("Opponent")
                dog_form = st.text_area("Opponent recent form")
                dog_off = st.text_area("Opponent offensive profile")
                dog_def = st.text_area("Opponent defensive profile")
                situational = st.text_area("Venue, travel, rest, weather, rivalry")
            st.text_area("Where does the favorite have the clearest matchup advantage?")
            st.text_area("How can the opponent realistically upset the favorite?")
            st.slider("Overall confidence", 1, 10, 7, key="football_conf")

        elif sport_lab == "NBA":
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite team")
                st.text_area("Last 5–10 games")
                st.text_area("Offensive matchup")
                st.text_area("Defensive matchup")
                st.text_area("Injuries / minutes restrictions")
            with c2:
                st.text_input("Opponent")
                st.text_area("Opponent last 5–10 games")
                st.text_area("Pace and shot profile")
                st.text_area("Rest / back-to-back / travel")
                st.text_area("Rebounding and turnover matchup")
            st.text_area("Upset path and late-game risk")
            st.slider("Overall confidence", 1, 10, 7, key="nba_conf")

        elif sport_lab == "Tennis":
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite player")
                st.text_area("Recent form and workload")
                st.text_area("Serve / return advantages")
                st.text_area("Surface and conditions")
                st.text_area("Fitness / injury concerns")
            with c2:
                st.text_input("Opponent player")
                st.text_area("Opponent form")
                st.text_area("Opponent's upset weapons")
                st.text_area("Head-to-head context")
                st.text_area("Travel / scheduling / fatigue")
            st.text_area("How does the favorite lose this match?")
            st.slider("Overall confidence", 1, 10, 7, key="tennis_conf")

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite fighter")
                st.text_area("Recent form and quality of opposition")
                st.text_area("Power at this weight")
                st.text_area("Chin durability")
                st.text_area("Wrestling / grappling / clinch profile")
                st.text_area("Injuries, layoff, weight cut")
            with c2:
                st.text_input("Opponent fighter")
                st.text_area("Opponent recent form")
                st.text_area("Opponent power and finishing threat")
                st.text_area("Opponent chin and recovery")
                st.text_area("Opponent technical advantages")
                st.text_area("Age, mileage and camp changes")
            st.text_area("Favorite's clearest path to victory")
            st.text_area("Opponent's most realistic upset path")
            st.slider("Overall confidence", 1, 10, 7, key="combat_conf")

        st.info("This page is a structured research worksheet. It does not automatically fetch current injuries, odds or statistics yet.")

with tabs[5]:
    st.subheader("Settings")
    st.caption("Bankroll, target-profit and restore controls remain in the sidebar.")
    st.divider()
    st.subheader("Backup and Export")
    st.warning(
        "Streamlit Community Cloud may restart the app and clear temporary session data. "
        "Download your CSV after updates and upload it again when needed."
    )
    csv_data = bets.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download All Bets CSV",
        data=csv_data,
        file_name=f"michael_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    analysis_csv_data = normalize_analyses(st.session_state.analyses).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Analysis Archive CSV",
        data=analysis_csv_data,
        file_name=f"macabets_analysis_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Current Bet Data")
    st.dataframe(bets, use_container_width=True, hide_index=True)

    st.markdown("#### Current Analysis Data")
    st.dataframe(normalize_analyses(st.session_state.analyses), use_container_width=True, hide_index=True)
