
import io
import math
import json
import html
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo
from datetime import date, datetime, time, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis_store import (
    create_analysis as db_create_analysis,
    delete_analysis as db_delete_analysis,
    is_configured as analysis_db_configured,
    list_analyses as db_list_analyses,
    update_analysis as db_update_analysis,
)
from engine.analysis_log_metrics import (
    group_rows_by_day,
    summarize_rows,
)

try:
    from engine.challenge_macabets import challenge_macabets, ChallengeMacabetsError
    CHALLENGE_MACABETS_AVAILABLE = True
    CHALLENGE_MACABETS_IMPORT_ERROR = ""
except Exception as exc:
    CHALLENGE_MACABETS_AVAILABLE = False
    CHALLENGE_MACABETS_IMPORT_ERROR = str(exc)

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
    from engine.nfl import analyze as analyze_nfl_match
    from engine.nfl_data import (
        NFL_TEAMS, NFL_TEAM_RATINGS, NFL_DATA_STATUS, TEAM_RATING_WEIGHTS,
        VENUE_TYPES, WEATHER_OPTIONS,
    )
    from engine.nfl_ratings_loader import load_all_team_ratings
    from engine.nfl_weather import find_scheduled_game, get_nfl_weather

    NFL_QUALITY_RATINGS = load_all_team_ratings()
    NFL_ENGINE_AVAILABLE = True
    NFL_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    NFL_QUALITY_RATINGS = {}
    NFL_ENGINE_AVAILABLE = False
    NFL_ENGINE_IMPORT_ERROR = str(exc)

APP_VERSION = "Macabets v0.62 — Live Debate"
BUILD_DATE = "July 31, 2026"

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

NFL_PROFILE_WEIGHTS = {
    "quarterback": 0.20,
    "offense": 0.12,
    "defense": 0.12,
    "coaching": 0.10,
    "offensive_line": 0.10,
    "defensive_line": 0.10,
    "skill_positions": 0.08,
    "secondary": 0.08,
    "special_teams": 0.04,
    "continuity": 0.06,
}


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


def _openai_api_key():
    """Read the OpenAI key from Streamlit Secrets without exposing it."""
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _openai_challenge_model():
    """Allow the challenge model to be changed without editing application code."""
    try:
        return str(st.secrets.get("OPENAI_CHALLENGE_MODEL", "gpt-5-mini")).strip() or "gpt-5-mini"
    except Exception:
        return "gpt-5-mini"


def _challenge_match_key(sport, participant_a, participant_b, event_date, tournament=""):
    return "|".join(
        str(value or "").strip().lower()
        for value in (sport, participant_a, participant_b, event_date, tournament)
    )


def _challenge_state(match_key):
    states = st.session_state.setdefault("macabets_challenge_states", {})
    return states.setdefault(
        match_key,
        {
            "messages": [],
            "pending_revision": None,
            "applied_revision": None,
            "debate_revision": None,
            "turns": [],
        },
    )


def _render_challenge_macabets(match_key, context):
    """Render a matchup-only live debate with Macabets and optional final revision."""
    state = _challenge_state(match_key)
    with st.expander("Challenge Macabets", expanded=False):
        st.caption(
            "Ask Macabets about the matchup or challenge its analysis. Research questions are read-only; "
            "only verified evidence or a substantive betting challenge can move the live debate position. "
            "Nothing becomes official until you finalize a revision."
        )

        if not CHALLENGE_MACABETS_AVAILABLE:
            st.error(f"Challenge layer unavailable: {CHALLENGE_MACABETS_IMPORT_ERROR}")
            return
        if not _openai_api_key():
            st.warning(
                "Challenge Macabets needs an OpenAI API key in Streamlit Secrets. "
                "Add OPENAI_API_KEY, then reboot the app."
            )
            return

        messages = state.get("messages", [])
        for message in messages:
            role = message.get("role", "assistant")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(message.get("content", ""))

        debate = state.get("debate_revision") or state.get("pending_revision")
        original = context.get("original_opinion", {})
        if debate:
            st.markdown("**Live debate position**")
            d1, d2, d3 = st.columns(3)
            player_a = context.get("player_a", "Player A")
            debate_prob = float(debate.get("proposed_probability_a", context.get("current_opinion", {}).get("probability_a", 0.5)))
            debate_conf = int(debate.get("proposed_confidence", context.get("current_opinion", {}).get("confidence", 50)))
            debate_verdict = str(debate.get("proposed_verdict", context.get("current_opinion", {}).get("verdict", "Pass")))
            d1.metric(
                f"{player_a} win probability",
                f"{debate_prob:.1%}",
                f"{(debate_prob - float(original.get('probability_a', debate_prob))) * 100:+.1f} pts vs original",
            )
            d2.metric(
                "Confidence",
                f"{debate_conf}/100",
                f"{debate_conf - int(original.get('confidence', debate_conf)):+d} vs original",
            )
            d3.metric("Verdict", debate_verdict)

            agree_points = debate.get("agree_points") or []
            pushback_points = debate.get("pushback_points") or []
            if agree_points or pushback_points:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**What changed my view**")
                    if agree_points:
                        for point in agree_points:
                            st.markdown(f"- {point}")
                    else:
                        st.caption("No part of the challenge has materially changed the model's view yet.")
                with c2:
                    st.markdown("**What I still push back on**")
                    if pushback_points:
                        for point in pushback_points:
                            st.markdown(f"- {point}")
                    else:
                        st.caption("No major pushback remains on the current argument.")

            if debate.get("question_to_user"):
                st.info(f"**Macabets asks:** {debate.get('question_to_user')}")
            if debate.get("revision_summary"):
                st.caption(str(debate.get("revision_summary")))
            if debate.get("uses_unverified_user_claim"):
                st.warning(
                    "Part of the current debate position relies on information you supplied that "
                    "Macabets could not verify from its matchup data."
                )

            finalize_col, original_col = st.columns(2)
            if finalize_col.button(
                "Finalize Revised Analysis",
                key=f"apply_challenge_{match_key}",
                type="primary",
                use_container_width=True,
            ):
                applied = dict(debate)
                state["applied_revision"] = applied
                # Keep a second, stable pointer to the finalized matchup revision.
                # This prevents a Streamlit rerun/widget-state refresh from making the
                # report fall back to the original model opinion.
                st.session_state["macabets_active_tennis_challenge"] = {
                    "match_key": match_key,
                    "revision": applied,
                }
                analysis_meta = st.session_state.get("macabets_tennis_analysis_records", {}).get(match_key, {})
                analysis_id = analysis_meta.get("analysis_id")
                if not analysis_id:
                    analysis_id, recovered_snapshot = _find_analysis_record_for_challenge(context)
                    if analysis_id:
                        analysis_meta = {
                            "analysis_id": analysis_id,
                            "analysis_snapshot": recovered_snapshot or {},
                        }
                        st.session_state.setdefault("macabets_tennis_analysis_records", {})[match_key] = analysis_meta
                if analysis_id:
                    try:
                        updated_row = _apply_revision_to_analysis(
                            analysis_id,
                            context,
                            applied,
                            base_snapshot=analysis_meta.get("analysis_snapshot"),
                        )
                        if updated_row:
                            analysis_meta["analysis_snapshot"] = updated_row.get("analysis_snapshot") or analysis_meta.get("analysis_snapshot")
                            st.session_state.setdefault("macabets_tennis_analysis_records", {})[match_key] = analysis_meta
                            st.session_state["analysis_log_last_saved"] = (
                                f"Revised analysis saved: {context.get('event_name') or context.get('player_a', '') + ' vs ' + context.get('player_b', '')}"
                            )
                        else:
                            st.session_state["analysis_log_warning"] = (
                                "Challenge was finalized, but the Analysis Log update returned no saved row."
                            )
                    except Exception as exc:
                        st.session_state["analysis_log_warning"] = (
                            f"Challenge finalized in the report, but the Analysis Log could not be updated: {exc}"
                        )
                else:
                    st.session_state["analysis_log_warning"] = (
                        "Challenge finalized in the report, but Macabets could not locate the saved Analysis Log record."
                    )
                state["pending_revision"] = None
                state["debate_revision"] = None
                st.rerun()
            if original_col.button(
                "Keep Original Analysis",
                key=f"discard_challenge_{match_key}",
                use_container_width=True,
            ):
                state["pending_revision"] = None
                state["debate_revision"] = None
                st.rerun()

        with st.form(key=f"challenge_form_{match_key}", clear_on_submit=True):
            user_message = st.text_area(
                "Ask or challenge Macabets",
                placeholder=(
                    "Ask a factual question or challenge the analysis. Example: Who have they played recently? "
                    "Or: I still favor him to win, but his reliability is too poor for a Strong Bet."
                ),
                height=100,
            )
            submitted = st.form_submit_button(
                "Send to Macabets", type="primary", use_container_width=True
            )

        if submitted and user_message.strip():
            with st.spinner("Macabets is reconsidering the matchup..."):
                try:
                    debate_context = dict(context)
                    current = dict(context.get("current_opinion", {}))
                    prior_debate = state.get("debate_revision")
                    if prior_debate:
                        current["probability_a"] = float(prior_debate.get("proposed_probability_a", current.get("probability_a", 0.5)))
                        current["confidence"] = int(prior_debate.get("proposed_confidence", current.get("confidence", 50)))
                        current["verdict"] = str(prior_debate.get("proposed_verdict", current.get("verdict", "Pass")))
                    debate_context["current_opinion"] = current

                    response = challenge_macabets(
                        api_key=_openai_api_key(),
                        model=_openai_challenge_model(),
                        matchup_context=debate_context,
                        conversation=messages,
                        user_message=user_message,
                    )
                    messages.append({"role": "user", "content": user_message.strip()})
                    messages.append({"role": "assistant", "content": response["reply"]})
                    state["messages"] = messages[-24:]
                    state.setdefault("turns", []).append(
                        {
                            "user": user_message.strip(),
                            "stance": response.get("stance"),
                            "intent": response.get("message_intent"),
                            "category": response.get("adjustment_category"),
                            "adjustment_reason": response.get("adjustment_reason"),
                            "probability_a": response.get("proposed_probability_a"),
                            "confidence": response.get("proposed_confidence"),
                            "verdict": response.get("proposed_verdict"),
                        }
                    )
                    state["turns"] = state["turns"][-12:]
                    # Informational/research turns are read-only and must not replace
                    # an existing live debate position. Only a response that actually
                    # proposes an allowed model change becomes the new debate state.
                    if response.get("should_offer_apply"):
                        state["debate_revision"] = response
                        state["pending_revision"] = response
                    st.rerun()
                except ChallengeMacabetsError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Challenge Macabets failed: {exc}")

        reset_col, status_col = st.columns([1, 2])
        if reset_col.button(
            "Reset Debate",
            key=f"reset_challenge_{match_key}",
            use_container_width=True,
        ):
            state["messages"] = []
            state["pending_revision"] = None
            state["debate_revision"] = None
            state["applied_revision"] = None
            active_pointer = st.session_state.get("macabets_active_tennis_challenge")
            if isinstance(active_pointer, dict) and active_pointer.get("match_key") == match_key:
                st.session_state.pop("macabets_active_tennis_challenge", None)
            state["turns"] = []
            st.rerun()
        if state.get("applied_revision"):
            status_col.success("A matchup-only challenged revision is currently official for this analysis.")
        elif state.get("debate_revision"):
            status_col.caption("The debate position is temporary. The original analysis remains official until you finalize it.")
        else:
            status_col.caption("The original Macabets analysis remains official.")


def _api_get_json(path, params):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{ODDS_API_BASE}{path}?{query}",
        headers={"User-Agent": "Macabets/0.21"},
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


def tennis_confidence_meter(result):
    """Combine model stability, data quality, sample size and context clarity."""
    model_score = min(max(float(result.get("confidence", 5)) * 10, 0), 100)
    data_score = min(max(float(result.get("data_quality", 5)) * 10, 0), 100)

    samples = []
    for profile_key in ("profile_a", "profile_b"):
        try:
            sample_value = float(result.get(profile_key, {}).get("sample", 0))
            samples.append(sample_value if math.isfinite(sample_value) else 0.0)
        except (TypeError, ValueError):
            samples.append(0.0)
    minimum_sample = min(samples) if samples else 0.0
    sample_score = min(max(minimum_sample / 40.0 * 100.0, 0), 100)

    health_penalties = {
        "Clear": 0,
        "Minor concern": 10,
        "Recent medical timeout": 18,
        "Returning from layoff": 22,
        "Recent retirement": 28,
        "Significant concern": 35,
    }
    health_a = str(result.get("injury_status_a", "Clear"))
    health_b = str(result.get("injury_status_b", "Clear"))
    context_penalty = min(
        health_penalties.get(health_a, 10) + health_penalties.get(health_b, 10),
        60,
    )
    context_score = 100 - context_penalty

    overall = round(
        model_score * 0.35
        + data_score * 0.35
        + sample_score * 0.20
        + context_score * 0.10
    )
    if overall >= 85:
        band = "High"
    elif overall >= 70:
        band = "Solid"
    elif overall >= 55:
        band = "Moderate"
    else:
        band = "Low"

    return {
        "overall": int(min(max(overall, 0), 100)),
        "band": band,
        "model": round(model_score),
        "data": round(data_score),
        "sample": round(sample_score),
        "context": round(context_score),
        "minimum_sample": int(minimum_sample),
    }


def tennis_bet_confidence(analysis_confidence, edge, expected_roi):
    """Grade confidence in a specific price without changing the match analysis."""
    positive_edge = max(float(edge), 0.0)
    positive_roi = max(float(expected_roi), 0.0)
    edge_score = min(positive_edge / 0.10 * 100.0, 100.0)
    roi_score = min(positive_roi / 0.10 * 100.0, 100.0)
    score = (
        float(analysis_confidence) * 0.55
        + edge_score * 0.20
        + roi_score * 0.25
    )

    if expected_roi < 0.02:
        score = min(score, 49)
    elif expected_roi < 0.05:
        score = min(score, 69)
    score = int(round(min(max(score, 0), 100)))

    if score >= 80:
        band = "High"
    elif score >= 65:
        band = "Solid"
    elif score >= 50:
        band = "Cautious"
    else:
        band = "Low / Pass"
    return {"overall": score, "band": band}


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



def tennis_matchup_context(h2h, player_a, player_b, base_probability_a):
    """Apply a deliberately capped H2H adjustment and return display context.

    Same-surface history is preferred when at least three meetings exist. The
    feature requires four overall meetings before changing probability, caps the
    move at six percentage points, and mainly reduces confidence when historical
    matchup evidence conflicts with the base model.
    """
    base_probability_a = float(base_probability_a)
    meetings = int(h2h.get("meetings", 0) or 0)
    surface_meetings = int(h2h.get("surface_meetings", 0) or 0)

    if surface_meetings >= 3:
        sample = surface_meetings
        wins_a = int(h2h.get("surface_wins_a", 0) or 0)
        wins_b = int(h2h.get("surface_wins_b", 0) or 0)
        scope = "on this surface"
    else:
        sample = meetings
        wins_a = int(h2h.get("wins_a", 0) or 0)
        wins_b = int(h2h.get("wins_b", 0) or 0)
        scope = "overall"

    neutral = {
        "active": False,
        "scope": scope,
        "sample": sample,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "leader": None,
        "leader_rate": 0.5,
        "adjustment_a": 0.0,
        "adjusted_probability_a": base_probability_a,
        "confidence_penalty": 0,
        "severity": "None",
        "message": "No reliable opponent-specific matchup adjustment was applied.",
    }
    if meetings < 4 or sample < 3 or sample <= 0 or wins_a == wins_b:
        return neutral

    leader = player_a if wins_a > wins_b else player_b
    leader_wins = max(wins_a, wins_b)
    leader_rate = leader_wins / sample
    if leader_rate < 0.67:
        return neutral

    # Shrink aggressively so H2H refines current-strength analysis rather than replacing it.
    dominance = (leader_rate - 0.50) / 0.50
    sample_strength = min(sample / 8.0, 1.0)
    raw_move = 0.06 * dominance * sample_strength
    adjustment_a = raw_move if leader == player_a else -raw_move
    adjustment_a = max(-0.06, min(0.06, adjustment_a))
    adjusted_a = max(0.05, min(0.95, base_probability_a + adjustment_a))

    base_leader = player_a if base_probability_a >= 0.50 else player_b
    conflict = leader != base_leader
    confidence_penalty = 0
    if conflict:
        confidence_penalty = min(12, 4 + round(abs(adjustment_a) * 100))
    elif abs(adjustment_a) >= 0.04:
        confidence_penalty = 3

    if leader_rate >= 0.80 and sample >= 5:
        severity = "Strong"
    elif leader_rate >= 0.70:
        severity = "Meaningful"
    else:
        severity = "Modest"

    conflict_text = (
        "This conflicts with the base player-strength model, so confidence is reduced."
        if conflict else
        "This supports the base model but remains a secondary, capped input."
    )
    message = (
        f"{leader} leads the relevant head-to-head {leader_wins}-{sample - leader_wins} "
        f"{scope}. {conflict_text}"
    )
    return {
        "active": True,
        "scope": scope,
        "sample": sample,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "leader": leader,
        "leader_rate": leader_rate,
        "adjustment_a": adjustment_a,
        "adjusted_probability_a": adjusted_a,
        "confidence_penalty": confidence_penalty,
        "severity": severity,
        "message": message,
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


def nfl_profile_summary(ratings):
    """Build transparent composite grades from the saved NFL team profile."""
    clean = {
        category: float(ratings.get(category, 50.0))
        for category in NFL_PROFILE_WEIGHTS
    }
    overall = sum(clean[category] * weight for category, weight in NFL_PROFILE_WEIGHTS.items())
    overall += float(ratings.get("injury_adjustment", 0.0))
    overall += float(ratings.get("rookie_adjustment", 0.0))
    return {
        "overall": max(0.0, min(100.0, overall)),
        "offense": np.mean([
            clean["quarterback"],
            clean["offense"],
            clean["offensive_line"],
            clean["skill_positions"],
        ]),
        "defense": np.mean([
            clean["defense"],
            clean["defensive_line"],
            clean["secondary"],
        ]),
        "coaching": clean["coaching"],
        "special_teams": clean["special_teams"],
        "continuity": clean["continuity"],
        "units": clean,
    }


def nfl_grade_band(grade):
    if grade >= 88:
        return "Elite"
    if grade >= 82:
        return "Strong"
    if grade >= 76:
        return "Above Average"
    if grade >= 70:
        return "Average"
    if grade >= 64:
        return "Below Average"
    return "Weak"


def build_nfl_category_verdicts(away_team, home_team, team_ratings):
    """Turn saved team grades into direct, matchup-specific category verdicts."""
    away = team_ratings.get(away_team, {})
    home = team_ratings.get(home_team, {})

    def grade(profile, category, default=50.0):
        return float(profile.get(category, default))

    category_formulas = [
        (
            "Quarterback",
            grade(away, "quarterback"),
            grade(home, "quarterback"),
        ),
        (
            "Running Game",
            (
                grade(away, "offense") * 0.45
                + grade(away, "offensive_line") * 0.35
                + grade(away, "skill_positions") * 0.20
            ),
            (
                grade(home, "offense") * 0.45
                + grade(home, "offensive_line") * 0.35
                + grade(home, "skill_positions") * 0.20
            ),
        ),
        (
            "Receiving Weapons",
            (
                grade(away, "skill_positions") * 0.70
                + grade(away, "offense") * 0.30
            ),
            (
                grade(home, "skill_positions") * 0.70
                + grade(home, "offense") * 0.30
            ),
        ),
        (
            "Offensive Line",
            grade(away, "offensive_line"),
            grade(home, "offensive_line"),
        ),
        (
            "Defensive Front",
            grade(away, "defensive_line"),
            grade(home, "defensive_line"),
        ),
        (
            "Secondary",
            grade(away, "secondary"),
            grade(home, "secondary"),
        ),
        (
            "Overall Defense",
            grade(away, "defense"),
            grade(home, "defense"),
        ),
        (
            "Coaching",
            grade(away, "coaching"),
            grade(home, "coaching"),
        ),
        (
            "Special Teams",
            grade(away, "special_teams"),
            grade(home, "special_teams"),
        ),
        (
            "Depth / Continuity",
            grade(away, "continuity"),
            grade(home, "continuity"),
        ),
    ]

    rows = []
    wins = {away_team: 0, home_team: 0, "Even": 0}
    for category, away_grade, home_grade in category_formulas:
        difference = away_grade - home_grade
        gap = abs(difference)
        if gap <= 1.5:
            winner = "Even"
            strength = "Even"
        else:
            winner = away_team if difference > 0 else home_team
            if gap < 4:
                strength = "Slight"
            elif gap < 8:
                strength = "Clear"
            else:
                strength = "Major"
        wins[winner] += 1
        rows.append({
            "Category": category,
            "Advantage": winner,
            "Strength": strength,
            "Rating Gap": round(gap, 1),
            away_team: round(away_grade, 1),
            home_team: round(home_grade, 1),
        })

    verdicts = pd.DataFrame(rows)
    decisive = verdicts[verdicts["Advantage"] != "Even"].sort_values(
        "Rating Gap",
        ascending=False,
    )
    strongest = decisive.iloc[0].to_dict() if not decisive.empty else None

    if wins[away_team] > wins[home_team]:
        category_leader = away_team
    elif wins[home_team] > wins[away_team]:
        category_leader = home_team
    else:
        category_leader = "Even"

    return verdicts, wins, strongest, category_leader


def build_nfl_explanation_report(nfl_result, projected_winner, category_verdicts, price_report):
    """Build concise, matchup-specific NFL explanations from existing model output."""
    away_team = str(nfl_result["away_team"])
    home_team = str(nfl_result["home_team"])
    opponent = home_team if projected_winner == away_team else away_team

    winner_rows = category_verdicts[
        category_verdicts["Advantage"] == projected_winner
    ].sort_values("Rating Gap", ascending=False)
    opponent_rows = category_verdicts[
        category_verdicts["Advantage"] == opponent
    ].sort_values("Rating Gap", ascending=False)

    def category_sentence(row, team, other):
        category = str(row["Category"])
        strength = str(row["Strength"]).lower()
        templates = {
            "Quarterback": f"{team} owns the {strength} quarterback advantage, the most important individual edge in the matchup.",
            "Running Game": f"{team} has the stronger projected rushing structure and should be better positioned to stay on schedule.",
            "Receiving Weapons": f"{team} has the more dangerous receiving group and a better chance to create explosive plays.",
            "Offensive Line": f"{team}'s offensive line projects to protect more reliably and create the cleaner down-to-down environment.",
            "Defensive Front": f"{team}'s defensive front can disrupt {other}'s timing and create pressure without constant blitzing.",
            "Secondary": f"{team} has the stronger secondary and is better equipped to limit explosive passes.",
            "Overall Defense": f"{team} carries the more dependable overall defensive profile.",
            "Coaching": f"{team} has the coaching edge in adjustments, clock management and late-game decisions.",
            "Special Teams": f"{team} has the special-teams advantage and the better chance to win hidden-yardage situations.",
            "Depth / Continuity": f"{team} has the stronger depth and continuity profile.",
        }
        return templates.get(category, f"{team} has a {strength} advantage in {category.lower()}.")

    key_advantages = [
        category_sentence(row, projected_winner, opponent)
        for _, row in winner_rows.head(4).iterrows()
    ] or [f"{projected_winner} holds the stronger overall model profile, although no single category creates major separation."]

    why_away = list(nfl_result.get("why_away_can_win", []))[:3]
    why_home = list(nfl_result.get("why_home_can_win", []))[:3]

    if not why_away:
        rows = category_verdicts[category_verdicts["Advantage"] == away_team].sort_values("Rating Gap", ascending=False)
        why_away = [category_sentence(row, away_team, home_team) for _, row in rows.head(3).iterrows()]
    if not why_home:
        rows = category_verdicts[category_verdicts["Advantage"] == home_team].sort_values("Rating Gap", ascending=False)
        why_home = [category_sentence(row, home_team, away_team) for _, row in rows.head(3).iterrows()]

    why_away = why_away or [f"{away_team} can win by controlling turnovers and outperforming the model in high-leverage situations."]
    why_home = why_home or [f"{home_team} can win by using home field, creating short fields and forcing a higher-variance game."]

    risks = []
    biggest_risk = str(nfl_result.get("biggest_risk", "")).strip()
    if biggest_risk:
        risks.append(biggest_risk)
    for item in list(nfl_result.get("swing_factors", [])):
        item = str(item).strip()
        if item and item not in risks:
            risks.append(item)
    for _, row in opponent_rows.head(2).iterrows():
        item = category_sentence(row, opponent, projected_winner)
        if item not in risks:
            risks.append(item)
    risks = risks[:4] or ["Turnovers, injuries or explosive plays could overturn the projected edge."]

    names = [str(row["Category"]).lower() for _, row in winner_rows.head(3).iterrows()]
    advantage_phrase = (
        names[0] if len(names) == 1
        else ", ".join(names[:-1]) + f" and {names[-1]}" if names
        else "overall team quality"
    )

    price_sentence = {
        "Strong Value": "The market is giving Macabets a meaningful price advantage.",
        "Good Value": "The current price offers a useful cushion relative to fair value.",
        "Slight Value": "The price contains a modest edge, but not a major market mistake.",
        "Fair Price": "The market is close to Macabets' fair number.",
        "Premium": "The projected winner is a little expensive, but the premium is understandable for the more likely winner.",
        "Overpriced": "Macabets still prefers the winner, but the market price has removed most of the betting appeal.",
    }.get(price_report["quality"], "The market price should be weighed separately from the predicted winner.")

    take = (
        f"Macabets makes {projected_winner} the more likely winner because of the stronger "
        f"{advantage_phrase} profile. {price_sentence} "
        f"The clearest upset path for {opponent} is to create disruption early and force the game "
        f"away from Macabets' expected script. Verdict: {price_report['recommendation']}."
    )

    game_script = str(nfl_result.get("game_script", "")).strip() or (
        f"Macabets expects {projected_winner} to establish the more stable offense, play from a favorable "
        f"scoreboard position and force {opponent} into higher-variance situations. If {opponent} avoids "
        f"turnovers and creates early explosive plays, the game can remain close into the fourth quarter."
    )

    return {
        "take": take,
        "key_advantages": key_advantages,
        "risks": risks,
        "why_away": why_away[:3],
        "why_home": why_home[:3],
        "game_script": game_script,
    }


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


def estimated_nfl_cover_probability(point_edge, margin_std=13.86):
    """Estimate cover probability from the gap between market and fair spread."""
    z_score = float(point_edge) / float(margin_std)
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


def required_nfl_spread_edge(american_odds, required_roi=0.05, margin_std=13.86):
    """Point edge required to reach a target ROI at the entered spread price."""
    decimal_price = american_to_decimal(int(american_odds))
    target_probability = min(max((1 + required_roi) / decimal_price, 0.0001), 0.9999)
    low, high = -30.0, 30.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if estimated_nfl_cover_probability(midpoint, margin_std) < target_probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def moneyline_price_quality(model_probability, market_odds, confidence_score):
    """Separate the mathematical price assessment from the flexible final verdict."""
    probability = min(max(float(model_probability), 0.0001), 0.9999)
    market_odds = int(market_odds)
    confidence_score = float(confidence_score)
    if confidence_score <= 10:
        confidence_score *= 10

    expected_roi = probability * american_to_decimal(market_odds) - 1

    if expected_roi >= 0.15:
        quality = "Very Underpriced"
    elif expected_roi >= 0.04:
        quality = "Underpriced"
    elif expected_roi >= -0.02:
        quality = "Fair"
    elif expected_roi >= -0.07:
        quality = "Premium"
    elif expected_roi >= -0.12:
        quality = "Overpriced"
    else:
        quality = "Very Overpriced"

    # A Strong Bet must clear BOTH the value test and a minimum win-probability
    # conviction floor. A large pricing edge alone cannot turn a modest favorite
    # into Macabets' strongest recommendation.
    if expected_roi >= 0.08 and confidence_score >= 75 and probability >= 0.65:
        verdict = "Strong Bet"
    elif expected_roi >= 0.025 and confidence_score >= 62:
        verdict = "Worth Betting"
    elif expected_roi >= -0.015 and confidence_score >= 82:
        verdict = "Worth Betting"
    elif expected_roi >= -0.05 and confidence_score >= 88:
        verdict = "Worth Betting"
    elif expected_roi >= -0.075 and confidence_score >= 78:
        verdict = "Lean"
    elif expected_roi <= -0.12 or (expected_roi <= -0.08 and confidence_score < 78):
        verdict = "Complete Pass"
    else:
        verdict = "Pass"

    return {
        "expected_roi": expected_roi,
        "quality": quality,
        "price_assessment": quality,
        "verdict": verdict,
        # Backward compatibility for older display code and the existing DB column.
        "recommendation": verdict,
    }




PRICE_ASSESSMENT_DEFINITIONS = {
    "Very Underpriced": "The market is offering a much better price than Macabets' fair line.",
    "Underpriced": "The market is offering a better price than Macabets' fair line.",
    "Fair": "The market price is close to Macabets' projected fair line.",
    "Premium": "The price is a little expensive, but the premium is understandable for the more likely winner.",
    "Overpriced": "The market is charging more than Macabets believes is justified.",
    "Very Overpriced": "The market price is far beyond Macabets' fair line and leaves very little betting value.",
}

VERDICT_DEFINITIONS = {
    "Strong Bet": "One of the strongest betting opportunities identified by Macabets.",
    "Worth Betting": "The combination of price and confidence is strong enough to justify a wager.",
    "Lean": "There is a slight betting case, but not enough for a full recommendation.",
    "Pass": "Macabets has a preferred winner, but the current price does not justify a wager.",
    "Complete Pass": "Stay away because the price is too poor, confidence is too low, or uncertainty is too high.",
}

LEGACY_PRICE_LABELS = {
    "Significantly Underpriced": "Very Underpriced",
    "Fairly Priced": "Fair",
    "Slightly Overpriced": "Premium",
    "Significantly Overpriced": "Very Overpriced",
}


def normalize_price_assessment(label):
    label = str(label or "").strip()
    return LEGACY_PRICE_LABELS.get(label, label or "—")


def render_price_verdict_guide():
    """Show compact definitions without taking permanent space in the Analysis Log."""
    with st.expander("Price Assessment & Verdict Guide", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Price Assessment")
            for label, definition in PRICE_ASSESSMENT_DEFINITIONS.items():
                st.markdown(f"**{label}** — {definition}")
        with right:
            st.markdown("#### Verdict")
            for label, definition in VERDICT_DEFINITIONS.items():
                st.markdown(f"**{label}** — {definition}")


def decision_label(expected_roi, confidence):
    """Compatibility wrapper returning the new verdict language."""
    expected_roi = float(expected_roi)
    confidence_score = float(confidence)
    if confidence_score <= 10:
        confidence_score *= 10

    # This compatibility helper does not know the underlying model probability,
    # so it must never manufacture a Strong Bet from ROI + confidence alone.
    # The main moneyline_price_quality() function is the source of truth for that
    # verdict because it can enforce the win-probability floor.
    if expected_roi >= 0.08 and confidence_score >= 70:
        verdict = "Worth Betting"
    elif expected_roi >= 0.025 and confidence_score >= 62:
        verdict = "Worth Betting"
    elif expected_roi >= -0.015 and confidence_score >= 82:
        verdict = "Worth Betting"
    elif expected_roi >= -0.05 and confidence_score >= 88:
        verdict = "Worth Betting"
    elif expected_roi >= -0.075 and confidence_score >= 78:
        verdict = "Lean"
    elif expected_roi <= -0.12 or (expected_roi <= -0.08 and confidence_score < 78):
        verdict = "Complete Pass"
    else:
        verdict = "Pass"

    reason = (
        f"Macabets' final verdict is {verdict.lower()} after weighing the offered price "
        f"against the model edge and {confidence_score:.0f}/100 prediction confidence."
    )
    return verdict, reason


def nfl_bottom_line(team, probability, quality, verdict, confidence_band):
    likely = f"Macabets expects {team} to win and assigns a {probability:.1%} win probability."
    if quality in {"Very Underpriced", "Underpriced"}:
        price = "The current moneyline is favorable relative to Macabets' fair price."
    elif quality == "Fair":
        price = "The market is close to Macabets' fair price, so the case rests more on conviction than value."
    elif quality == "Premium":
        price = "The projected winner is a little expensive, but the premium is understandable and can remain playable when conviction is high."
    elif quality == "Overpriced":
        price = "The projected winner is expensive, so Macabets requires stronger conviction before accepting the number."
    else:
        price = "The projected winner is priced well beyond Macabets' fair number."
    return f"{likely} {price} Verdict: {verdict}. Prediction confidence: {confidence_band}."


def _analysis_market_line(row):
    """Return the actual market moneyline attached to the logged prediction."""
    prediction = str(row.get("prediction", ""))
    if prediction and prediction == str(row.get("participant_a", "")):
        odds = row.get("market_odds_a")
    elif prediction and prediction == str(row.get("participant_b", "")):
        odds = row.get("market_odds_b")
    else:
        odds = row.get("market_line")
    try:
        return format_american(float(odds)) if odds is not None else "—"
    except (TypeError, ValueError):
        return str(odds) if odds not in (None, "") else "—"


def _analysis_pricing_report(row):
    """Recalculate pricing with current thresholds, including older saved analyses."""
    snapshot = row.get("analysis_snapshot") or {}
    explicit_assessment = None
    explicit_verdict = None
    saved_report = {}
    if isinstance(snapshot, dict):
        explicit_assessment = snapshot.get("price_assessment")
        explicit_verdict = snapshot.get("verdict")
        saved_report = snapshot.get("price_report") or {}
        if isinstance(saved_report, dict):
            explicit_assessment = explicit_assessment or saved_report.get("price_assessment") or saved_report.get("quality")
            explicit_verdict = explicit_verdict or saved_report.get("verdict") or saved_report.get("recommendation")

    # A finalized Challenge Macabets revision is the official analysis.
    # Do not recalculate its user-approved verdict back to the pre-challenge label.
    if isinstance(snapshot, dict) and snapshot.get("challenge_revision_applied") and explicit_verdict:
        return {
            "price_assessment": normalize_price_assessment(explicit_assessment) if explicit_assessment else "—",
            "verdict": str(explicit_verdict),
            "expected_roi": saved_report.get("expected_roi") if isinstance(saved_report, dict) else None,
        }

    # Always rebuild the assessment from the stored probability/fair line and
    # market line so threshold changes also update existing Analysis Log rows.
    # Saved labels are used only when the underlying pricing inputs are missing.

    # Legacy rows may not contain predicted_probability. Rebuild it from the
    # stored fair American line when necessary.
    probability = row.get("predicted_probability")
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = None

    if probability is None or not 0 < probability < 1:
        fair_line = row.get("fair_line")
        try:
            fair_odds = int(float(str(fair_line).replace("+", "").replace(",", "").strip()))
            probability = american_to_implied(fair_odds)
        except (TypeError, ValueError):
            probability = None

    confidence = row.get("confidence") or 0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0

    actual_line = _analysis_market_line(row)
    try:
        market_odds = int(float(str(actual_line).replace("+", "").replace(",", "").strip()))
    except (TypeError, ValueError):
        market_odds = None

    if probability is not None and market_odds is not None:
        report = moneyline_price_quality(probability, market_odds, confidence)
        return {
            "price_assessment": normalize_price_assessment(report["price_assessment"]),
            "verdict": report["verdict"],
            "expected_roi": report.get("expected_roi"),
        }

    if explicit_assessment or explicit_verdict:
        return {
            "price_assessment": normalize_price_assessment(explicit_assessment) if explicit_assessment else "—",
            "verdict": str(explicit_verdict) if explicit_verdict else "—",
            "expected_roi": saved_report.get("expected_roi") if isinstance(saved_report, dict) else None,
        }

    return {"price_assessment": "—", "verdict": "—", "expected_roi": None}


def _analysis_price_assessment(row):
    return _analysis_pricing_report(row)["price_assessment"]


def _analysis_verdict(row):
    """Show the new verdict language and replace legacy recommendation strings."""
    return _analysis_pricing_report(row)["verdict"]


def _analysis_price_verdict_explanation(row):
    """Explain the selected log entry using its own market line, fair line, confidence and labels."""
    pricing = _analysis_pricing_report(row)
    assessment = pricing.get("price_assessment", "—")
    verdict = pricing.get("verdict", "—")
    prediction = str(row.get("prediction") or "The predicted winner")
    actual_line = _analysis_market_line(row)
    fair_line = str(row.get("fair_line") or "—")

    assessment_text = PRICE_ASSESSMENT_DEFINITIONS.get(assessment, "This label compares the market line with Macabets' fair line.")
    verdict_text = VERDICT_DEFINITIONS.get(verdict, "This verdict weighs both price and prediction confidence.")

    if assessment == "Premium":
        specific = (
            f"{prediction} is priced at {actual_line} versus a Macabets fair line of {fair_line}. "
            "The favorite is somewhat pricey, but Macabets can understand paying the premium for the more likely winner."
        )
    elif assessment in {"Overpriced", "Very Overpriced"}:
        specific = (
            f"{prediction} is priced at {actual_line} versus a Macabets fair line of {fair_line}. "
            "The offered price is meaningfully more expensive than the model's estimate."
        )
    elif assessment in {"Underpriced", "Very Underpriced"}:
        specific = (
            f"{prediction} is priced at {actual_line} versus a Macabets fair line of {fair_line}. "
            "The market is offering a more favorable number than the model believes is fair."
        )
    else:
        specific = (
            f"{prediction} is priced at {actual_line} versus a Macabets fair line of {fair_line}. "
            "The market and model are close enough that price alone does not create a major edge."
        )

    return specific, assessment_text, verdict_text


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


def _analysis_event_token(sport, inputs):
    """Create one stable token per explicit analysis click to prevent Streamlit duplicates."""
    raw = json.dumps(inputs, sort_keys=True, default=str)
    return f"{sport.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{abs(hash(raw))}"


def _save_universal_analysis(record):
    """Write one frozen analysis snapshot to the permanent Analysis Log."""
    if not analysis_db_configured():
        st.session_state["analysis_log_warning"] = (
            "Analysis completed, but permanent logging is not configured yet. "
            "Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets."
        )
        return None
    try:
        saved = db_create_analysis(record)
        st.session_state["analysis_log_last_saved"] = record.get("event_name", "Analysis")
        return saved
    except Exception as exc:
        st.session_state["analysis_log_warning"] = f"Analysis completed, but could not be saved: {exc}"
        return None


def _find_analysis_record_for_challenge(context):
    """Recover the most recent saved tennis analysis when Streamlit session metadata is missing."""
    if not analysis_db_configured():
        return None, None
    player_a = str(context.get("player_a") or "").strip()
    player_b = str(context.get("player_b") or "").strip()
    event_name = str(context.get("event_name") or f"{player_a} vs {player_b}").strip()
    event_date = str(context.get("event_date") or "").strip()
    try:
        rows = db_list_analyses(250, sport="Tennis")
    except Exception:
        return None, None
    for row in rows:
        row_event = str(row.get("event_name") or "").strip()
        row_a = str(row.get("participant_a") or "").strip()
        row_b = str(row.get("participant_b") or "").strip()
        same_match = (row_event == event_name) or ({row_a, row_b} == {player_a, player_b} and player_a and player_b)
        if not same_match:
            continue
        if event_date and str(row.get("event_date") or "").strip() not in {"", event_date}:
            continue
        return row.get("id"), row.get("analysis_snapshot") or {}
    return None, None


def _apply_revision_to_analysis(analysis_id, context, revision, base_snapshot=None):
    """Make a finalized Challenge Macabets revision the one official saved analysis."""
    if not analysis_id or not analysis_db_configured():
        return None

    player_a = str(context.get("player_a") or "Player A")
    player_b = str(context.get("player_b") or "Player B")
    market = context.get("market") or {}

    try:
        probability_a = min(max(float(revision.get("proposed_probability_a")), 0.01), 0.99)
    except (TypeError, ValueError):
        probability_a = float((context.get("current_opinion") or {}).get("probability_a", 0.5))
    probability_b = 1.0 - probability_a

    try:
        confidence = int(round(float(revision.get("proposed_confidence"))))
    except (TypeError, ValueError):
        confidence = int((context.get("current_opinion") or {}).get("confidence", 50))
    confidence = min(max(confidence, 0), 100)

    verdict = str(
        revision.get("proposed_verdict")
        or (context.get("current_opinion") or {}).get("verdict")
        or "Pass"
    )

    if probability_a >= probability_b:
        prediction = player_a
        predicted_probability = probability_a
        market_odds = market.get("odds_a")
    else:
        prediction = player_b
        predicted_probability = probability_b
        market_odds = market.get("odds_b")

    fair_odds_a = probability_to_american(probability_a)
    fair_odds_b = probability_to_american(probability_b)
    fair_line = format_american(fair_odds_a if prediction == player_a else fair_odds_b)

    try:
        pricing = moneyline_price_quality(predicted_probability, int(market_odds), confidence)
        price_assessment = pricing.get("price_assessment", "—")
    except (TypeError, ValueError):
        pricing = {}
        price_assessment = (context.get("current_opinion") or {}).get("price_assessment", "—")

    snapshot = dict(base_snapshot or {})
    snapshot.update({
        "probability_a": probability_a,
        "probability_b": probability_b,
        "fair_odds_a": fair_odds_a,
        "fair_odds_b": fair_odds_b,
        "price_assessment": price_assessment,
        "verdict": verdict,
        "challenge_revision_applied": True,
        "revision_source": "User-approved challenge",
        "challenge_revision": dict(revision),
    })
    prior_confidence = snapshot.get("analysis_confidence")
    if isinstance(prior_confidence, dict):
        updated_confidence = dict(prior_confidence)
        updated_confidence["overall"] = confidence
        snapshot["analysis_confidence"] = updated_confidence
    else:
        snapshot["analysis_confidence"] = {"overall": confidence}
    if pricing:
        snapshot["price_report"] = {**pricing, "verdict": verdict, "recommendation": verdict}

    changes = {
        "prediction": prediction,
        "predicted_probability": predicted_probability,
        "fair_line": fair_line,
        "confidence": confidence,
        "recommendation": verdict,
        "analysis_snapshot": snapshot,
    }
    updated = db_update_analysis(str(analysis_id), changes)
    if updated:
        st.session_state["analysis_log_last_saved"] = context.get("event_name") or f"{player_a} vs {player_b}"
    return updated


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
.macabets-edge-card {
    border: 1px solid #d9dee7;
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin: 0.8rem 0 1rem 0;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}
.macabets-edge-top {
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    gap: 1.2rem;
    align-items: center;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e7eaf0;
}
.macabets-confidence-label {font-size: .78rem; font-weight: 700; letter-spacing: .08em; color: #667085;}
.macabets-confidence-value {font-size: 3.35rem; line-height: 1; font-weight: 800; color: #172033; margin: .3rem 0;}
.macabets-verdict {font-size: 1.05rem; font-weight: 800; color: #172033;}
.macabets-play-label {font-size: .78rem; color: #667085; margin-bottom: .25rem;}
.macabets-play-value {font-size: 1.65rem; line-height: 1.2; font-weight: 750; color: #172033;}
.macabets-edge-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .85rem 1rem;
    padding: 1rem 0;
}
.macabets-edge-item {min-width: 0;}
.macabets-edge-label {font-size: .76rem; color: #667085; margin-bottom: .2rem;}
.macabets-edge-value {font-size: 1.08rem; line-height: 1.3; font-weight: 700; color: #172033; overflow-wrap: anywhere;}
.macabets-score-row {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 1rem;
    padding-top: .15rem;
}
.macabets-score-box {border-top: 1px solid #e7eaf0; padding-top: .9rem;}
.macabets-projected-score {font-size: 1.65rem; font-weight: 800; color: #172033;}
.macabets-probability {font-size: 1rem; font-weight: 650; color: #172033; line-height: 1.55;}
@media (max-width: 900px) {
    .macabets-edge-top, .macabets-score-row {grid-template-columns: 1fr;}
    .macabets-edge-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
}
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
    "Daily Slate", "Archive", "Settings", "Information"
])

# Streamlit does not expose a native API for selecting a top-level tab. When an
# Analysis Log entry is reopened, this small client-side bridge selects the
# Analysis Engine tab after the rerun so the user lands directly on the matchup.
if st.session_state.pop("open_analysis_engine_tab", False):
    components.html(
        """
        <script>
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        const target = Array.from(tabs).find(
            (tab) => tab.textContent.trim().startsWith('Analysis Engine')
        );
        if (target) {
            target.click();
            target.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
        </script>
        """,
        height=0,
    )

with tabs[0]:
    with st.expander("What's New in Macabets v0.21", expanded=True):
        st.markdown(
            """
            - Added the Macabets NFL v0.1 foundation workspace
            - Added all 32 NFL teams, game context and Vegas market inputs
            - Added a stable NFL report structure for fair lines, projected scores, win probability, confidence, upset risk and game scripts
            - NFL v0.1 is explicitly market-derived and will not claim an independent betting edge before the Team Quality Engine exists
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
        reopened_notice = st.session_state.pop("reopened_analysis_notice", None)
        if reopened_notice:
            st.success(reopened_notice)
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
                    new_challenge_key = _challenge_match_key(
                        "Tennis", player_a, player_b, match_date.isoformat(), tournament
                    )
                    st.session_state.setdefault("macabets_challenge_states", {}).pop(
                        new_challenge_key, None
                    )
                    # A fresh analysis always starts from the untouched model output.
                    st.session_state.pop("macabets_active_tennis_challenge", None)
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
                            suppress_log = bool(
                                st.session_state.pop("suppress_next_tennis_log", False)
                            )
                            if suppress_log:
                                # Clear any stale pending token so the reopened analysis is
                                # displayed but never inserted as a duplicate log entry.
                                st.session_state.pop("tennis_analysis_log_pending", None)
                            else:
                                st.session_state["tennis_analysis_log_pending"] = _analysis_event_token(
                                    "Tennis",
                                    {
                                        "player_a": player_a, "player_b": player_b,
                                        "match_date": match_date.isoformat(),
                                        "market_odds_a": int(market_odds_a),
                                        "market_odds_b": int(market_odds_b),
                                    },
                                )
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

                    validation = result.get("data_validation", {})
                    if validation:
                        with st.expander("Data Validation — verified inputs", expanded=False):
                            st.success(
                                "Both players were matched to the historical database and passed "
                                "the minimum data requirements. No neutral placeholder profile was used."
                            )
                            validation_rows = []
                            for side_key, label in (("player_a", analyzed_a), ("player_b", analyzed_b)):
                                item = validation.get(side_key, {})
                                validation_rows.append({
                                    "Player": label,
                                    "Database name": item.get("resolved") or "Not found",
                                    "Match method": item.get("method", "—"),
                                    "Historical matches": item.get("historical_matches", 0),
                                    "Two-year sample": item.get("two_year_sample", 0),
                                    "Surface sample": item.get("surface_sample", 0),
                                    "Serve-stat matches": item.get("serve_sample", 0),
                                    "Return-stat matches": item.get("return_sample", 0),
                                    "Overall Elo": "Found" if item.get("overall_elo_found") else "Missing",
                                    "Surface Elo": "Found" if item.get("surface_elo_found") else "Overall Elo fallback",
                                    "Warnings": ", ".join(item.get("flags", [])) or "None",
                                })
                            st.dataframe(pd.DataFrame(validation_rows), use_container_width=True, hide_index=True)

                    intelligence_a = result.get("player_intelligence_a", {})
                    intelligence_b = result.get("player_intelligence_b", {})
                    experience_engine = result.get("experience_engine", {})
                    if intelligence_a or intelligence_b:
                        with st.expander("Player Intelligence — v0.54", expanded=False):
                            profile_rows = []
                            for label, item, reliability in (
                                (analyzed_a, intelligence_a, experience_engine.get("reliability_a", 0)),
                                (analyzed_b, intelligence_b, experience_engine.get("reliability_b", 0)),
                            ):
                                profile_rows.append({
                                    "Player": label,
                                    "Live ranking": item.get("ranking") or "Unavailable",
                                    "Career matches": item.get("career_matches", 0),
                                    f"{result.get('surface', surface)} matches": item.get("surface_matches", {}).get(result.get("surface", surface), 0),
                                    "Grand Slam matches": item.get("grand_slam_matches", 0),
                                    "Masters matches": item.get("masters_matches", 0),
                                    "Top-10 record": item.get("top_10_record", "0-0"),
                                    "Top-50 record": item.get("top_50_record", "0-0"),
                                    "Experience reliability": f"{float(reliability):.0%}",
                                    "API status": item.get("api_source", "unavailable"),
                                    "Warnings": ", ".join(item.get("data_flags", [])) or "None",
                                })
                            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)
                            adjustment = float(experience_engine.get("probability_adjustment_a", 0.0))
                            advantage = experience_engine.get("advantage", "Even")
                            st.caption(
                                f"Experience advantage: {advantage}. Probability adjustment to "
                                f"{analyzed_a}: {adjustment:+.1%}. This input is capped at ±4%."
                            )
                    listed_a = safe_int(market_snapshot.get("market_odds_a", market_odds_a), safe_int(market_odds_a, -180))
                    listed_b = safe_int(market_snapshot.get("market_odds_b", market_odds_b), safe_int(market_odds_b, 155))

                    base_model_probability = float(result["win_probability"])
                    h2h_context = build_head_to_head_summary(
                        matches, analyzed_a, analyzed_b, result.get("surface", surface)
                    )
                    matchup_context = tennis_matchup_context(
                        h2h_context, analyzed_a, analyzed_b, base_model_probability
                    )
                    model_probability = float(matchup_context["adjusted_probability_a"])
                    original_model_probability = model_probability
                    challenge_match_key = _challenge_match_key(
                        "Tennis",
                        analyzed_a,
                        analyzed_b,
                        market_snapshot.get("match_date", match_date.isoformat()),
                        result.get("tournament", tournament),
                    )
                    challenge_state = _challenge_state(challenge_match_key)
                    active_challenge = challenge_state.get("applied_revision")
                    active_pointer = st.session_state.get("macabets_active_tennis_challenge")
                    if (
                        isinstance(active_pointer, dict)
                        and active_pointer.get("match_key") == challenge_match_key
                        and isinstance(active_pointer.get("revision"), dict)
                    ):
                        active_challenge = active_pointer["revision"]
                        # Heal the per-match state as well so both stores agree.
                        challenge_state["applied_revision"] = dict(active_challenge)
                    if active_challenge:
                        model_probability = min(
                            max(float(active_challenge.get("proposed_probability_a", model_probability)), 0.05),
                            0.95,
                        )
                    probability_b = 1 - model_probability
                    fair_odds = probability_to_american(model_probability)
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
                    if active_challenge:
                        confidence = int(
                            min(
                                max(active_challenge.get("proposed_confidence", confidence), 0),
                                100,
                            )
                        )

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

                    analysis_confidence = tennis_confidence_meter(result)
                    original_analysis_confidence = dict(analysis_confidence)
                    if matchup_context["confidence_penalty"]:
                        analysis_confidence["overall"] = max(
                            0,
                            analysis_confidence["overall"] - matchup_context["confidence_penalty"],
                        )
                        if analysis_confidence["overall"] >= 85:
                            analysis_confidence["band"] = "High"
                        elif analysis_confidence["overall"] >= 70:
                            analysis_confidence["band"] = "Solid"
                        elif analysis_confidence["overall"] >= 55:
                            analysis_confidence["band"] = "Moderate"
                        else:
                            analysis_confidence["band"] = "Low"
                    original_effective_confidence = int(analysis_confidence["overall"])
                    if active_challenge:
                        analysis_confidence["overall"] = int(
                            min(
                                max(active_challenge.get("proposed_confidence", analysis_confidence["overall"]), 0),
                                100,
                            )
                        )
                        if analysis_confidence["overall"] >= 85:
                            analysis_confidence["band"] = "High"
                        elif analysis_confidence["overall"] >= 70:
                            analysis_confidence["band"] = "Solid"
                        elif analysis_confidence["overall"] >= 55:
                            analysis_confidence["band"] = "Moderate"
                        else:
                            analysis_confidence["band"] = "Low"
                    bet_confidence = (
                        tennis_bet_confidence(
                            analysis_confidence["overall"],
                            considered_edge,
                            considered_roi,
                        )
                        if considered_player
                        else None
                    )

                    # Preserve Player A fields for the existing archive structure.
                    no_vig_edge = edge_a
                    expected_roi = roi_a

                    tennis_log_token = st.session_state.pop("tennis_analysis_log_pending", None)
                    if tennis_log_token:
                        projected_winner_for_log = analyzed_a if model_probability >= probability_b else analyzed_b
                        projected_probability_for_log = max(model_probability, probability_b)
                        projected_market_odds_for_log = listed_a if projected_winner_for_log == analyzed_a else listed_b
                        projected_confidence_for_log = analysis_confidence["overall"]
                        projected_price_report_for_log = moneyline_price_quality(
                            projected_probability_for_log,
                            projected_market_odds_for_log,
                            projected_confidence_for_log,
                        )
                        recommendation_for_log = projected_price_report_for_log["verdict"]
                        tennis_inputs = {
                            **market_snapshot,
                            "player_a": analyzed_a, "player_b": analyzed_b,
                            "tournament": result.get("tournament", tournament),
                            "round": result.get("round", round_name),
                            "surface": result.get("surface", surface),
                            "simulations": int(simulations),
                        }
                        tennis_snapshot = {
                            "engine_result": result,
                            "head_to_head": h2h_context,
                            "matchup_context": matchup_context,
                            "probability_a": model_probability,
                            "probability_b": probability_b,
                            "fair_odds_a": fair_odds,
                            "fair_odds_b": fair_odds_b,
                            "no_vig_probability_a": no_vig_a,
                            "no_vig_probability_b": no_vig_b,
                            "roi_a": roi_a, "roi_b": roi_b,
                            "analysis_confidence": analysis_confidence,
                            "bet_confidence": bet_confidence,
                            "selected_bet_decision": decision,
                            "selected_bet_reason": decision_reason,
                            "projected_winner_market_line": projected_market_odds_for_log,
                            "price_assessment": projected_price_report_for_log["price_assessment"],
                            "verdict": projected_price_report_for_log["verdict"],
                        }
                        saved_tennis_analysis = _save_universal_analysis({
                            "client_event_id": tennis_log_token,
                            "event_date": market_snapshot.get("match_date"),
                            "sport": "Tennis",
                            "model_version": "Macabets Tennis v0.47",
                            "event_name": f"{analyzed_a} vs {analyzed_b}",
                            "participant_a": analyzed_a, "participant_b": analyzed_b,
                            "market_type": "Moneyline",
                            "market_odds_a": listed_a, "market_odds_b": listed_b,
                            "prediction": projected_winner_for_log,
                            "predicted_probability": projected_probability_for_log,
                            "fair_line": format_american(
                                fair_odds if projected_winner_for_log == analyzed_a else fair_odds_b
                            ),
                            "confidence": analysis_confidence["overall"],
                            "recommendation": recommendation_for_log,
                            "status": "Pending",
                            "input_snapshot": tennis_inputs,
                            "analysis_snapshot": tennis_snapshot,
                        })
                        if saved_tennis_analysis and saved_tennis_analysis.get("id"):
                            st.session_state.setdefault("macabets_tennis_analysis_records", {})[challenge_match_key] = {
                                "analysis_id": saved_tennis_analysis.get("id"),
                                "analysis_snapshot": tennis_snapshot,
                                "client_event_id": tennis_log_token,
                            }

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

                    # Decision summary: separate the likely winner from the quality of the price.
                    projected_winner = analyzed_a if model_probability >= probability_b else analyzed_b
                    projected_winner_probability = max(model_probability, probability_b)
                    projected_winner_fair_odds = fair_odds if projected_winner == analyzed_a else fair_odds_b

                    winner_market_odds = listed_a if projected_winner == analyzed_a else listed_b
                    projected_price_report = moneyline_price_quality(
                        projected_winner_probability,
                        winner_market_odds,
                        analysis_confidence["overall"],
                    )
                    if active_challenge and active_challenge.get("proposed_verdict"):
                        projected_price_report["verdict"] = str(active_challenge["proposed_verdict"])
                        projected_price_report["recommendation"] = str(active_challenge["proposed_verdict"])
                    price_assessment = projected_price_report["price_assessment"]

                    original_probability_b = 1 - original_model_probability
                    original_projected_winner = (
                        analyzed_a if original_model_probability >= original_probability_b else analyzed_b
                    )
                    original_projected_probability = max(original_model_probability, original_probability_b)
                    original_market_odds = (
                        listed_a if original_projected_winner == analyzed_a else listed_b
                    )
                    original_price_report = moneyline_price_quality(
                        original_projected_probability,
                        original_market_odds,
                        original_effective_confidence,
                    )

                    st.markdown("#### Macabets Verdict")
                    verdict1, verdict2, verdict3, verdict4 = st.columns(4)
                    verdict1.metric("Projected Winner", projected_winner)
                    verdict2.metric(
                        "Win Probability",
                        f"{projected_winner_probability:.1%}",
                        f"Fair {format_american(projected_winner_fair_odds)}",
                    )
                    verdict3.metric("Verdict", projected_price_report["verdict"])
                    verdict4.metric(
                        "Price Assessment",
                        price_assessment,
                        f"Market {format_american(winner_market_odds)}",
                    )
                    st.info(
                        f"Macabets picks {projected_winner}. The {format_american(winner_market_odds)} "
                        f"market line is {price_assessment.lower()}, producing a final verdict of "
                        f"{projected_price_report['verdict']}."
                    )
                    if active_challenge:
                        st.caption(
                            "Challenge revision applied for this matchup only. "
                            f"Original: {original_projected_winner} — "
                            f"{original_projected_probability:.1%}, "
                            f"{original_effective_confidence}/100 confidence, "
                            f"{original_price_report['verdict']}."
                        )

                    challenge_context = {
                        "sport": "Tennis",
                        "event_name": f"{analyzed_a} vs {analyzed_b}",
                        "event_date": market_snapshot.get("match_date"),
                        "player_a": analyzed_a,
                        "player_b": analyzed_b,
                        "tournament": result.get("tournament", tournament),
                        "round": result.get("round", round_name),
                        "surface": result.get("surface", surface),
                        "environment": result.get("environment", environment),
                        "match_format": result.get("match_format", match_format),
                        "market": {
                            "odds_a": listed_a,
                            "odds_b": listed_b,
                            "no_vig_probability_a": round(no_vig_a, 4),
                            "no_vig_probability_b": round(no_vig_b, 4),
                        },
                        "original_opinion": {
                            "projected_winner": original_projected_winner,
                            "probability_a": round(original_model_probability, 4),
                            "confidence": original_effective_confidence,
                            "verdict": original_price_report["verdict"],
                            "fair_odds_a": probability_to_american(original_model_probability),
                            "fair_odds_b": probability_to_american(original_probability_b),
                        },
                        "current_opinion": {
                            "projected_winner": projected_winner,
                            "probability_a": round(model_probability, 4),
                            "confidence": int(analysis_confidence["overall"]),
                            "verdict": projected_price_report["verdict"],
                            "fair_odds_a": fair_odds,
                            "fair_odds_b": fair_odds_b,
                            "price_assessment": price_assessment,
                        },
                        "head_to_head": h2h_context,
                        "matchup_context": matchup_context,
                        "match_intelligence": result.get("match_intelligence", {}),
                        # Deterministic model evidence: Challenge Macabets should reason from
                        # the same recent-resume and fatigue inputs as the prediction engine
                        # before reaching for web search or constructing a narrative.
                        "recent_resume_comparison": result.get("recent_resume_comparison", {}),
                        "fatigue_profile_a": result.get("fatigue_profile_a", {}),
                        "fatigue_profile_b": result.get("fatigue_profile_b", {}),
                        "fatigue_resilience_a": result.get("fatigue_resilience_a", 0.0),
                        "fatigue_resilience_b": result.get("fatigue_resilience_b", 0.0),
                        "player_intelligence_a": intelligence_a,
                        "player_intelligence_b": intelligence_b,
                        "factors": [
                            {
                                "name": str(factor.get("name", "")),
                                "impact": float(factor.get("impact", 0.0)),
                                "reason": str(factor.get("reason", "")),
                            }
                            for factor in result.get("factors", [])[:12]
                        ],
                    }
                    _render_challenge_macabets(challenge_match_key, challenge_context)

                    st.markdown("#### Matchup Context")
                    st.markdown("**Macabets Take**")
                    if matchup_context["active"]:
                        st.info(matchup_context["message"])
                    elif h2h_context["meetings"] == 0:
                        st.info(
                            "These players have no previous meetings in the available Macabets data. "
                            "The evaluation is driven by current form, surface performance, playing style "
                            "and overall player strength."
                        )
                    elif h2h_context["meetings"] < 4:
                        st.info(
                            f"These players have met only {h2h_context['meetings']} time"
                            f"{'s' if h2h_context['meetings'] != 1 else ''}, so Macabets places very little weight "
                            "on the head-to-head record. Current form and underlying performance remain "
                            "the primary drivers of the prediction."
                        )
                    else:
                        st.info(
                            "The available head-to-head history does not show a strong enough persistent "
                            "matchup advantage to materially influence the evaluation. Macabets therefore "
                            "relies primarily on current form, surface performance and overall player strength."
                        )

                    match_intelligence = result.get("match_intelligence", {})
                    if match_intelligence:
                        st.markdown("#### Matchup Stability & Volatility")
                        intel1, intel2, intel3 = st.columns(3)
                        intel1.metric(
                            "Matchup Stability",
                            f"{match_intelligence.get('stability_score', 0)}/100",
                            match_intelligence.get("stability_band", "—"),
                        )
                        intel2.metric(
                            "Volatility",
                            f"{match_intelligence.get('volatility_score', 0)}/100",
                            match_intelligence.get("volatility_band", "—"),
                        )
                        intel3.metric(
                            "Factor Consensus",
                            f"{match_intelligence.get('factor_consensus', 0):.0%}",
                            f"{match_intelligence.get('supporting_factors', 0)} support / "
                            f"{match_intelligence.get('opposing_factors', 0)} oppose",
                        )
                        st.caption(
                            "Stability measures how repeatable the projected edge appears. Volatility measures "
                            "how easily tiebreaks, close-set variance, health, fatigue or conflicting matchup "
                            "signals could disrupt the prediction. Neither score considers the sportsbook price."
                        )

                        st.markdown("**Primary volatility drivers**")
                        for driver in match_intelligence.get("drivers", []):
                            st.markdown(f"- {driver.capitalize()}")

                        st.markdown(f"#### Upset Paths for {match_intelligence.get('underdog', 'the underdog')}")
                        for path in match_intelligence.get("upset_paths", []):
                            st.markdown(f"- {path}")

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
                    m4.metric(
                        "Analysis Confidence",
                        f"{analysis_confidence['overall']}/100",
                        analysis_confidence["band"],
                    )

                    st.markdown("#### Macabets vs. Market")

                    comparison = pd.DataFrame(
                        {
                            "Player": [analyzed_a, analyzed_b],
                            "Market": [f"{no_vig_a:.1%}", f"{no_vig_b:.1%}"],
                            "Macabets": [f"{model_probability:.1%}", f"{probability_b:.1%}"],
                        }
                    )
                    st.dataframe(
                        comparison,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if abs(no_vig_edge) < 0.03:
                        st.info(
                            "Macabets is largely in agreement with the betting market on this matchup."
                        )
                    else:
                        model_favored_player = analyzed_a if no_vig_edge > 0 else analyzed_b
                        market_favored_player = analyzed_b if no_vig_edge > 0 else analyzed_a
                        st.info(
                            f"Macabets is substantially more bullish on {model_favored_player} than the "
                            f"betting market. The market is comparatively higher on {market_favored_player}. "
                            f"The difference is {abs(no_vig_edge):.1%}."
                        )

                    st.markdown("#### Confidence Meters")
                    confidence_col1, confidence_col2 = st.columns(2)
                    with confidence_col1:
                        st.markdown("**Confidence in Macabets’ Analysis**")
                        st.progress(analysis_confidence["overall"] / 100)
                        st.write(
                            f"{analysis_confidence['overall']}/100 — "
                            f"{analysis_confidence['band']}"
                        )
                        st.caption(
                            "Combines model stability, data quality, the smaller historical "
                            f"sample ({analysis_confidence['minimum_sample']} matches) and "
                            "health/context clarity."
                        )

                    with confidence_col2:
                        if considered_player and bet_confidence:
                            st.markdown(
                                f"**Confidence in Your {considered_player} Bet**"
                            )
                            st.progress(bet_confidence["overall"] / 100)
                            st.write(
                                f"{bet_confidence['overall']}/100 — "
                                f"{bet_confidence['band']}"
                            )
                            st.caption(
                                "Combines analysis confidence with the selected player’s "
                                "edge and expected return at the available sportsbook price."
                            )
                        else:
                            st.markdown("**Confidence in a Specific Bet**")
                            st.info(
                                "Select a player before analyzing to receive a separate "
                                "bet-confidence score."
                            )

                    matchup_analysis = build_matchup_analysis(result, considered_player)

                    # Compact decision summary for the moneyline evaluator.
                    if considered_player:
                        st.markdown(f"#### Moneyline Evaluation: {considered_player}")
                        bet1, bet2, bet3, bet4 = st.columns(4)
                        bet1.metric("Market Price", format_american(considered_market_odds))
                        bet2.metric("Macabets Fair Price", format_american(considered_fair_odds))
                        bet3.metric("Expected ROI", f"{considered_roi:+.1%}")
                        bet4.metric("Decision", decision)

                        if decision == "BET":
                            st.success(decision_reason)
                        elif decision == "WATCH":
                            st.warning(decision_reason)
                        else:
                            st.info(decision_reason)

                    # Show only the strongest decision-useful advantages.
                    raw_factors = [
                        {
                            "name": str(factor.get("name", "Matchup factor")),
                            "impact_a": float(factor.get("impact", 0.0)),
                            "reason": str(factor.get("reason", "")),
                        }
                        for factor in result.get("factors", [])
                        if str(factor.get("name", "")).strip() != "Fatigue 2.0"
                    ]

                    st.markdown("#### Decisive Factors")
                    advantage_rows = []
                    for factor in raw_factors:
                        impact = factor["impact_a"]
                        if abs(impact) < 0.001:
                            continue
                        leader = analyzed_a if impact > 0 else analyzed_b
                        advantage_rows.append((abs(impact), leader, factor["name"]))
                    advantage_rows.sort(reverse=True)

                    if advantage_rows:
                        for _, leader, factor_name in advantage_rows[:6]:
                            st.markdown(f"- **{leader}:** {factor_name}")
                    else:
                        st.caption("Macabets does not identify a clear matchup advantage for either player.")

                    st.markdown("#### Why Each Player Can Win")
                    why_a, why_b = st.columns(2)
                    with why_a:
                        st.markdown(f"**Why {analyzed_a} can win**")
                        for point in matchup_analysis.get("player_a_reasons", []):
                            st.markdown(f"- {point}")
                    with why_b:
                        st.markdown(f"**Why {analyzed_b} can win**")
                        for point in matchup_analysis.get("player_b_reasons", []):
                            st.markdown(f"- {point}")

                    simulation = result["simulation"]

                    # Reconcile the simulation with the final matchup-adjusted verdict.
                    # Preserve the simulation's conditional set-score distribution for each
                    # player, while forcing each side's exact scores to sum to the same
                    # headline win probability shown everywhere else in the report.
                    raw_set_scores = {
                        str(score): float(probability)
                        for score, probability in simulation.get("set_scores", {}).items()
                    }
                    raw_a_total = 0.0
                    raw_b_total = 0.0
                    parsed_scores = {}

                    for raw_score, probability in raw_set_scores.items():
                        try:
                            a_sets, b_sets = (
                                int(value) for value in raw_score.split("-", 1)
                            )
                        except (TypeError, ValueError):
                            continue
                        parsed_scores[raw_score] = (a_sets, b_sets)
                        if a_sets > b_sets:
                            raw_a_total += probability
                        elif b_sets > a_sets:
                            raw_b_total += probability

                    synchronized_set_scores = {}
                    for raw_score, probability in raw_set_scores.items():
                        parsed = parsed_scores.get(raw_score)
                        if parsed is None:
                            synchronized_set_scores[raw_score] = probability
                            continue

                        a_sets, b_sets = parsed
                        if a_sets > b_sets:
                            synchronized_set_scores[raw_score] = (
                                probability / raw_a_total * model_probability
                                if raw_a_total > 0 else 0.0
                            )
                        elif b_sets > a_sets:
                            synchronized_set_scores[raw_score] = (
                                probability / raw_b_total * probability_b
                                if raw_b_total > 0 else 0.0
                            )
                        else:
                            synchronized_set_scores[raw_score] = 0.0

                    straight_sets_a = sum(
                        probability
                        for raw_score, probability in synchronized_set_scores.items()
                        if raw_score in parsed_scores
                        and parsed_scores[raw_score][0] > parsed_scores[raw_score][1]
                        and parsed_scores[raw_score][1] == 0
                    )
                    straight_sets_b = sum(
                        probability
                        for raw_score, probability in synchronized_set_scores.items()
                        if raw_score in parsed_scores
                        and parsed_scores[raw_score][1] > parsed_scores[raw_score][0]
                        and parsed_scores[raw_score][0] == 0
                    )
                    deciding_set_probability = sum(
                        probability
                        for raw_score, probability in synchronized_set_scores.items()
                        if raw_score in parsed_scores
                        and abs(parsed_scores[raw_score][0] - parsed_scores[raw_score][1]) == 1
                        and max(parsed_scores[raw_score]) >= 2
                    )

                    st.markdown("#### Outcome Simulation")
                    win_col_a, win_col_b = st.columns(2)
                    win_col_a.metric(f"{analyzed_a} wins", f"{model_probability:.1%}")
                    win_col_b.metric(f"{analyzed_b} wins", f"{probability_b:.1%}")

                    s1, s2, s3 = st.columns(3)
                    s1.metric(f"{analyzed_a} straight sets", f"{straight_sets_a:.1%}")
                    s2.metric(f"{analyzed_b} straight sets", f"{straight_sets_b:.1%}")
                    s3.metric("Deciding set", f"{deciding_set_probability:.1%}")

                    st.markdown("#### Exact Set Score")
                    set_score_results = []
                    for raw_score, probability in synchronized_set_scores.items():
                        try:
                            a_sets, b_sets = (int(value) for value in raw_score.split("-", 1))
                        except (TypeError, ValueError):
                            # Defensive fallback in case the simulation format changes later.
                            set_score_results.append({
                                "label": str(raw_score),
                                "probability": probability,
                                "winner_order": 2,
                                "loser_sets": 99,
                            })
                            continue

                        if a_sets > b_sets:
                            winner = analyzed_a
                            winner_sets, loser_sets = a_sets, b_sets
                            winner_order = 0
                        else:
                            winner = analyzed_b
                            winner_sets, loser_sets = b_sets, a_sets
                            winner_order = 1

                        set_score_results.append({
                            "label": f"{winner} wins {winner_sets}-{loser_sets}",
                            "probability": probability,
                            "winner_order": winner_order,
                            "loser_sets": loser_sets,
                        })

                    # Keep each player's possible wins together and show the most decisive score first.
                    set_score_results.sort(
                        key=lambda item: (item["winner_order"], item["loser_sets"])
                    )

                    cards_per_row = 4 if len(set_score_results) <= 4 else 3
                    for row_start in range(0, len(set_score_results), cards_per_row):
                        row_results = set_score_results[row_start:row_start + cards_per_row]
                        score_columns = st.columns(len(row_results))
                        for column, score_result in zip(score_columns, row_results):
                            column.metric(
                                score_result["label"],
                                f"{score_result['probability']:.1%}",
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
        st.subheader("NFL Matchup Analysis")
        st.caption(
            "Compare the Macabets fair line with Vegas, evaluate a specific wager and review "
            "the matchup's clearest category advantages."
        )

        if NFL_ENGINE_AVAILABLE:
            with st.expander("Model information", expanded=False):
                if NFL_DATA_STATUS.get("available"):
                    foundation_count = NFL_DATA_STATUS.get("foundation_available_datasets", 0)
                    foundation_total = NFL_DATA_STATUS.get("foundation_total_datasets", 0)
                    foundation_text = (
                        f" NFL foundation: {foundation_count}/{foundation_total} sources refreshed."
                        if foundation_total else ""
                    )
                    st.success(
                        f"Performance ratings active: {NFL_DATA_STATUS.get('teams', 0)} teams "
                        f"loaded from {NFL_DATA_STATUS.get('data_source', 'nflverse')}. "
                        f"Season {NFL_DATA_STATUS.get('season', '—')}; updated "
                        f"{NFL_DATA_STATUS.get('updated_at_utc', 'unknown')}."
                        f"{foundation_text}"
                    )
                else:
                    st.warning(
                        "Starter ratings are active because no generated NFL performance snapshot "
                        "is present. Run the Update Macabets NFL Data workflow to refresh them."
                    )
                st.markdown(
                    "**Offense:** EPA/play, success rate, explosive-play rate and turnover rate.  "
                    "\n**Defense:** EPA allowed, success rate allowed, explosive plays allowed and takeaways.  "
                    "\n**Quarterback:** team passing EPA/dropback, passing success rate and CPOE.  "
                    "\n**Strength of schedule:** average opponent net EPA faced.  "
                    "\n**Special teams:** special-teams EPA.  "
                    "\n**Coaching:** remains a transparent manual prior until a defensible coaching model is added."
                )

        if NFL_QUALITY_RATINGS:
            with st.expander("NFL Team Profiles", expanded=False):
                st.caption(
                    "Review every team by unit, identify its strongest and weakest areas, and compare "
                    "two teams before opening the full game analysis."
                )
                profile_teams = sorted(NFL_QUALITY_RATINGS)
                profile_col1, profile_col2 = st.columns(2)
                profile_team = profile_col1.selectbox(
                    "Team profile",
                    profile_teams,
                    key="nfl_profile_team",
                )
                comparison_options = ["No comparison"] + [
                    team for team in profile_teams if team != profile_team
                ]
                comparison_team = profile_col2.selectbox(
                    "Compare with",
                    comparison_options,
                    key="nfl_profile_comparison",
                )

                selected_ratings = NFL_QUALITY_RATINGS[profile_team]
                selected_profile = nfl_profile_summary(selected_ratings)
                profile1, profile2, profile3, profile4, profile5 = st.columns(5)
                profile1.metric(
                    "Overall Profile",
                    f"{selected_profile['overall']:.1f}",
                    nfl_grade_band(selected_profile["overall"]),
                )
                profile2.metric("Offense", f"{selected_profile['offense']:.1f}")
                profile3.metric("Defense", f"{selected_profile['defense']:.1f}")
                profile4.metric("Coaching", f"{selected_profile['coaching']:.1f}")
                profile5.metric("Continuity", f"{selected_profile['continuity']:.1f}")

                unit_labels = {
                    "quarterback": "Quarterback",
                    "offense": "Overall Offense",
                    "offensive_line": "Offensive Line",
                    "skill_positions": "Skill Positions",
                    "defense": "Overall Defense",
                    "defensive_line": "Defensive Line",
                    "secondary": "Secondary",
                    "coaching": "Coaching",
                    "special_teams": "Special Teams",
                    "continuity": "Continuity",
                }
                profile_rows = []
                comparison_profile = None
                if comparison_team != "No comparison":
                    comparison_profile = nfl_profile_summary(
                        NFL_QUALITY_RATINGS[comparison_team]
                    )

                for category, label in unit_labels.items():
                    selected_grade = selected_profile["units"][category]
                    row = {
                        "Unit": label,
                        profile_team: round(selected_grade, 1),
                        "Grade": nfl_grade_band(selected_grade),
                    }
                    if comparison_profile:
                        comparison_grade = comparison_profile["units"][category]
                        row[comparison_team] = round(comparison_grade, 1)
                        row["Difference"] = round(selected_grade - comparison_grade, 1)
                    profile_rows.append(row)

                st.dataframe(
                    pd.DataFrame(profile_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                ranked_units = sorted(
                    selected_profile["units"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                strength_col, weakness_col, modifier_col = st.columns(3)
                with strength_col:
                    st.markdown("**Core strengths**")
                    for category, grade in ranked_units[:3]:
                        st.write(f"{unit_labels[category]}: {grade:.0f}")
                with weakness_col:
                    st.markdown("**Areas to attack**")
                    for category, grade in reversed(ranked_units[-3:]):
                        st.write(f"{unit_labels[category]}: {grade:.0f}")
                with modifier_col:
                    st.markdown("**Current modifiers**")
                    st.write(
                        f"Injury adjustment: "
                        f"{float(selected_ratings.get('injury_adjustment', 0.0)):+.1f}"
                    )
                    st.write(
                        f"Rookie adjustment: "
                        f"{float(selected_ratings.get('rookie_adjustment', 0.0)):+.1f}"
                    )
                    st.write(f"Special teams: {selected_profile['special_teams']:.0f}")

                if comparison_profile:
                    overall_gap = (
                        selected_profile["overall"] - comparison_profile["overall"]
                    )
                    if abs(overall_gap) < 0.5:
                        comparison_read = "The overall profiles are effectively even."
                    else:
                        stronger_team = profile_team if overall_gap > 0 else comparison_team
                        comparison_read = (
                            f"{stronger_team} owns the stronger overall profile by "
                            f"{abs(overall_gap):.1f} points."
                        )
                    st.info(comparison_read)

        if not NFL_ENGINE_AVAILABLE:
            st.error(
                "The NFL engine could not be imported. Confirm that engine/nfl.py, engine/nfl_data.py and "
                f"engine/confidence.py are in the repository. Import error: {NFL_ENGINE_IMPORT_ERROR}"
            )
        else:
            nfl_top1, nfl_top2, nfl_top3, nfl_top4 = st.columns(4)
            nfl_date = nfl_top1.date_input("Game date", value=date.today(), key="nfl_game_date")
            nfl_week = nfl_top2.number_input("Week", min_value=1, max_value=22, value=1, step=1, key="nfl_week")
            away_team = nfl_top3.selectbox("Away team", NFL_TEAMS, index=2, key="nfl_away_team")
            home_options = [team for team in NFL_TEAMS if team != away_team]
            home_team = nfl_top4.selectbox("Home team", home_options, index=min(4, len(home_options)-1), key="nfl_home_team")

            st.markdown("#### Market")
            market1, market2, market3, market4 = st.columns(4)
            market_spread_home = market1.number_input(
                f"{home_team} spread", value=-3.0, step=0.5, format="%.1f", key="nfl_spread_home",
                help="Enter the home-team line. Example: -3.5 means the home team is favored by 3.5."
            )
            market_ml_away = market2.number_input(
                f"{away_team} moneyline", value=140, step=5, key="nfl_ml_away"
            )
            market_ml_home = market3.number_input(
                f"{home_team} moneyline", value=-165, step=5, key="nfl_ml_home"
            )
            market_total = market4.number_input("Game total", min_value=1.0, value=45.5, step=0.5, format="%.1f", key="nfl_total")
            
            st.markdown("#### Game context")
            scheduled_game = find_scheduled_game(away_team, home_team, nfl_date)
            scheduled_roof = str(scheduled_game.get("roof") or "").lower()
            if scheduled_roof in {"dome", "closed"}:
                auto_venue_type = "Dome"
            elif "retract" in scheduled_roof:
                auto_venue_type = "Retractable roof"
            else:
                auto_venue_type = "Outdoor"

            context1, context2, context3, context4 = st.columns(4)
            context1.metric("Venue", auto_venue_type)
            context2.metric("Weather", "Automatic")
            neutral_site = context3.checkbox("Neutral site", value=False, key="nfl_neutral_site")
            home_field_points = context4.number_input("Home-field points", min_value=0.0, max_value=4.0, value=1.7, step=0.1, key="nfl_hfa")

            with st.expander("Weather / venue override", expanded=False):
                st.caption(
                    "Macabets automatically uses the scheduled stadium and kickoff forecast. "
                    "Only use this override if a roof decision or unusual condition is not reflected correctly."
                )
                manual_weather_override = st.checkbox(
                    "Use manual weather override", value=False, key="nfl_manual_weather_override"
                )
                override1, override2 = st.columns(2)
                manual_venue_type = override1.selectbox(
                    "Venue type", VENUE_TYPES,
                    index=VENUE_TYPES.index(auto_venue_type) if auto_venue_type in VENUE_TYPES else 0,
                    disabled=not manual_weather_override, key="nfl_venue_type_override"
                )
                manual_weather = override2.selectbox(
                    "Weather", WEATHER_OPTIONS, disabled=not manual_weather_override, key="nfl_weather_override"
                )

            st.markdown("#### Bet consideration")
            nfl_considered_side = st.radio(
                "Who are you considering betting on?",
                ["Just analyze", away_team, home_team],
                horizontal=True,
                key="nfl_considered_side",
                help=(
                    "Your selection does not influence the Macabets fair line. "
                    "It only determines which wager is evaluated."
                ),
            )
            bet_input1, bet_input2 = st.columns(2)
            nfl_considered_market = bet_input1.selectbox(
                "Wager type",
                ["Spread", "Moneyline"],
                disabled=nfl_considered_side == "Just analyze",
                key="nfl_considered_market",
            )
            nfl_spread_price = bet_input2.number_input(
                "Spread price",
                value=-110,
                step=5,
                disabled=(
                    nfl_considered_side == "Just analyze"
                    or nfl_considered_market != "Spread"
                ),
                key="nfl_spread_price",
                help="Enter the American odds attached to the spread, such as -110.",
            )
            
            # Start with the existing NFL data-pipeline ratings, then layer the
            # new Team Quality Engine ratings on top wherever the categories match.
            # This keeps the current NFL report stable while making the new 32-team
            # ratings file the primary source for the model inputs.
            away_overrides = dict(NFL_TEAM_RATINGS[away_team])
            home_overrides = dict(NFL_TEAM_RATINGS[home_team])
            away_quality_source = NFL_QUALITY_RATINGS.get(away_team, {})
            home_quality_source = NFL_QUALITY_RATINGS.get(home_team, {})

            for category in TEAM_RATING_WEIGHTS:
                if category in away_quality_source:
                    away_overrides[category] = float(away_quality_source[category])
                if category in home_quality_source:
                    home_overrides[category] = float(home_quality_source[category])

            with st.expander("Advanced rating adjustments", expanded=False):
                st.caption(
                    "The saved team ratings load automatically. Override a number only when current "
                    "injuries or roster news materially change a team."
                )
                away_col, home_col = st.columns(2)
                with away_col:
                    st.markdown(f"#### {away_team}")
                    for category in TEAM_RATING_WEIGHTS:
                        away_overrides[category] = st.number_input(
                            category.replace("_", " ").title(), min_value=0.0, max_value=100.0,
                            value=float(away_overrides[category]), step=1.0,
                            key=f"nfl_away_{away_team}_{category}"
                        )
                with home_col:
                    st.markdown(f"#### {home_team}")
                    for category in TEAM_RATING_WEIGHTS:
                        home_overrides[category] = st.number_input(
                            category.replace("_", " ").title(), min_value=0.0, max_value=100.0,
                            value=float(home_overrides[category]), step=1.0,
                            key=f"nfl_home_{home_team}_{category}"
                        )

            run_nfl = st.button("Generate NFL Report", type="primary", use_container_width=True, key="run_nfl_analysis")
            if run_nfl:
                try:
                    if manual_weather_override:
                        venue_type = manual_venue_type
                        weather = manual_weather
                        weather_context = {
                            "available": True,
                            "source": "Manual override",
                            "stadium": str(scheduled_game.get("stadium") or home_team),
                            "venue_type": venue_type.lower(),
                            "label": weather,
                            "impact": "Manual",
                            "summary": f"Manual override: {weather} at a {venue_type.lower()} venue.",
                            "home_margin_adjustment": 0.0,
                            "total_adjustment": 0.0,
                            "confidence_penalty": 0.0,
                            "climate_mismatch": "Manual override does not add an automatic side adjustment.",
                        }
                    else:
                        weather_context = get_nfl_weather(
                            away_team=away_team, home_team=home_team, game_date=nfl_date
                        )
                        venue_lookup = str(weather_context.get("venue_type") or "outdoor").lower()
                        venue_type = (
                            "Dome" if venue_lookup == "dome"
                            else "Retractable roof" if "retract" in venue_lookup
                            else "Outdoor"
                        )
                        weather = str(weather_context.get("label") or "Normal")

                    nfl_result = analyze_nfl_match(
                        away_team=away_team,
                        home_team=home_team,
                        market_spread_home=market_spread_home,
                        market_moneyline_away=market_ml_away,
                        market_moneyline_home=market_ml_home,
                        market_total=market_total,
                        venue_type=venue_type,
                        weather=weather,
                        neutral_site=neutral_site,
                        away_rating_overrides=away_overrides,
                        home_rating_overrides=home_overrides,
                        home_field_points=home_field_points,
                        weather_context=weather_context,
                    )
                    st.session_state["nfl_weather_context"] = weather_context
                    st.session_state.nfl_result = nfl_result
                    st.session_state["nfl_analysis_log_pending"] = _analysis_event_token(
                        "NFL",
                        {
                            "away_team": away_team, "home_team": home_team,
                            "game_date": nfl_date.isoformat(),
                            "market_spread_home": float(market_spread_home),
                            "market_total": float(market_total),
                        },
                    )
                except Exception as exc:
                    st.error(f"Could not generate the NFL report: {exc}")

            nfl_result = st.session_state.get("nfl_result")
            if nfl_result:
                fair_home_spread = float(nfl_result["fair_spread_home"])
                entered_market_home_spread = float(market_spread_home)
                spread_difference = fair_home_spread - entered_market_home_spread
                fair_away_moneyline = int(
                    nfl_result.get(
                        "fair_moneyline_away",
                        probability_to_american(nfl_result["away_win_probability"]),
                    )
                )

                if fair_home_spread < -0.05:
                    model_favorite = nfl_result["home_team"]
                elif fair_home_spread > 0.05:
                    model_favorite = nfl_result["away_team"]
                else:
                    model_favorite = "Pick'em"

                if spread_difference > 0.50:
                    value_team = nfl_result["away_team"]
                    value_spread = -entered_market_home_spread
                    spread_value_text = f"{value_team} {value_spread:+.1f}"
                    market_direction = nfl_result["away_team"]
                elif spread_difference < -0.50:
                    value_team = nfl_result["home_team"]
                    value_spread = entered_market_home_spread
                    spread_value_text = f"{value_team} {value_spread:+.1f}"
                    market_direction = nfl_result["home_team"]
                else:
                    value_team = None
                    spread_value_text = "No material spread edge"
                    market_direction = "neither side"

                if fair_home_spread < -0.05:
                    fair_line_text = f"{nfl_result['home_team']} {fair_home_spread:+.1f}"
                elif fair_home_spread > 0.05:
                    fair_line_text = f"{nfl_result['away_team']} {-fair_home_spread:+.1f}"
                else:
                    fair_line_text = "Pick'em"

                if entered_market_home_spread < -0.05:
                    vegas_line_text = f"{nfl_result['home_team']} {entered_market_home_spread:+.1f}"
                elif entered_market_home_spread > 0.05:
                    vegas_line_text = (
                        f"{nfl_result['away_team']} {-entered_market_home_spread:+.1f}"
                    )
                else:
                    vegas_line_text = "Pick'em"

                if value_team:
                    edge_text = f"{value_team} by {abs(spread_difference):.1f} pts"
                else:
                    edge_text = f"No edge ({abs(spread_difference):.1f} pts)"

                st.markdown(f"### {nfl_result['away_team']} at {nfl_result['home_team']} — Week {int(nfl_week)}")
                st.caption(
                    f"Game date: {nfl_date.strftime('%B %-d, %Y')} · "
                    f"{len(NFL_QUALITY_RATINGS)} team profiles loaded"
                )

                projected_nfl_winner = (
                    nfl_result["away_team"]
                    if nfl_result["away_win_probability"] >= nfl_result["home_win_probability"]
                    else nfl_result["home_team"]
                )
                projected_nfl_probability = max(
                    float(nfl_result["away_win_probability"]),
                    float(nfl_result["home_win_probability"]),
                )
                projected_nfl_score_side = (
                    f"{nfl_result['away_team']} {nfl_result['projected_away_score']:.1f}"
                    if projected_nfl_winner == nfl_result["away_team"]
                    else f"{nfl_result['home_team']} {nfl_result['projected_home_score']:.1f}"
                )
                if value_team and abs(spread_difference) >= 2.0:
                    nfl_best_bet = f"{value_team} — BETTABLE EDGE"
                elif value_team and abs(spread_difference) > 0.5:
                    nfl_best_bet = f"{value_team} — LEAN"
                else:
                    nfl_best_bet = "NO CLEAR BET"

                # Decision-first moneyline report
                winner_market_ml = int(market_ml_away) if projected_nfl_winner == nfl_result["away_team"] else int(market_ml_home)
                winner_fair_ml = fair_away_moneyline if projected_nfl_winner == nfl_result["away_team"] else int(nfl_result["fair_moneyline_home"])
                price_report = moneyline_price_quality(projected_nfl_probability, winner_market_ml, nfl_result["confidence"])

                nfl_log_token = st.session_state.pop("nfl_analysis_log_pending", None)
                if nfl_log_token:
                    nfl_inputs = {
                        "away_team": away_team, "home_team": home_team,
                        "game_date": nfl_date.isoformat(), "week": int(nfl_week),
                        "market_spread_home": float(market_spread_home),
                        "market_moneyline_away": int(market_ml_away),
                        "market_moneyline_home": int(market_ml_home),
                        "market_total": float(market_total),
                        "venue_type": venue_type, "weather": weather,
                        "neutral_site": bool(neutral_site),
                        "home_field_points": float(home_field_points),
                        "considered_side": nfl_considered_side,
                        "considered_market": nfl_considered_market,
                        "spread_price": int(nfl_spread_price),
                        "away_rating_overrides": away_overrides,
                        "home_rating_overrides": home_overrides,
                    }
                    nfl_snapshot = {
                        "engine_result": nfl_result,
                        "fair_line_text": fair_line_text,
                        "vegas_line_text": vegas_line_text,
                        "spread_edge_text": edge_text,
                        "projected_winner": projected_nfl_winner,
                        "projected_winner_probability": projected_nfl_probability,
                        "winner_market_moneyline": winner_market_ml,
                        "winner_fair_moneyline": winner_fair_ml,
                        "price_report": price_report,
                        "price_assessment": price_report["price_assessment"],
                        "verdict": price_report["verdict"],
                    }
                    _save_universal_analysis({
                        "client_event_id": nfl_log_token,
                        "event_date": nfl_date.isoformat(),
                        "sport": "NFL",
                        "model_version": "Macabets NFL v0.23 / App v0.47",
                        "event_name": f"{away_team} at {home_team}",
                        "participant_a": away_team, "participant_b": home_team,
                        "market_type": nfl_considered_market if nfl_considered_side != "Just analyze" else "Analysis",
                        "market_line": float(market_spread_home),
                        "market_odds_a": int(market_ml_away),
                        "market_odds_b": int(market_ml_home),
                        "prediction": projected_nfl_winner,
                        "predicted_probability": projected_nfl_probability,
                        "fair_line": format_american(winner_fair_ml),
                        "confidence": float(nfl_result["confidence"]),
                        "recommendation": price_report["recommendation"],
                        "status": "Pending",
                        "input_snapshot": nfl_inputs,
                        "analysis_snapshot": nfl_snapshot,
                    })

                category_verdicts, category_wins, strongest_edge, category_leader = build_nfl_category_verdicts(
                    nfl_result["away_team"], nfl_result["home_team"], NFL_QUALITY_RATINGS
                )
                explanation_report = build_nfl_explanation_report(
                    nfl_result,
                    projected_nfl_winner,
                    category_verdicts,
                    price_report,
                )

                spread_value_points = abs(spread_difference)
                if spread_value_points >= 5.0:
                    spread_value_label = "Strong spread value"
                elif spread_value_points >= 2.0:
                    spread_value_label = "Playable spread value"
                elif spread_value_points > 0.5:
                    spread_value_label = "Slight spread value"
                else:
                    spread_value_label = "No meaningful spread value"

                away = nfl_result["away_team"]
                home = nfl_result["home_team"]
                away_score = round(nfl_result["projected_away_score"])
                home_score = round(nfl_result["projected_home_score"])
                if away_score >= home_score:
                    winner, winner_score = away, away_score
                    loser, loser_score = home, home_score
                else:
                    winner, winner_score = home, home_score
                    loser, loser_score = away, away_score
                margin = winner_score - loser_score
                if margin >= 7:
                    outlook = "Comfortable win"
                elif margin >= 3:
                    outlook = "Competitive win"
                else:
                    outlook = "Toss-up"

                decision_sentence = explanation_report["take"]
                recommendation_text = (
                    spread_value_text
                    if price_report["verdict"].lower() != "pass" and value_team
                    else "PASS"
                )

                st.markdown("## Macabets Recommendation")
                st.markdown(
                    f"""
                    <div class="macabets-edge-card">
                        <div class="macabets-edge-top">
                            <div>
                                <div class="macabets-confidence-label">CONFIDENCE</div>
                                <div class="macabets-confidence-value">{nfl_result['confidence']:.0f}<span style="font-size:1.35rem;font-weight:650;">/100</span></div>
                                <div class="macabets-verdict">{html.escape(str(price_report['verdict']).upper())} · {html.escape(str(nfl_result['confidence_band']))}</div>
                            </div>
                            <div>
                                <div class="macabets-play-label">RECOMMENDED PLAY</div>
                                <div class="macabets-play-value">{html.escape(recommendation_text)}</div>
                            </div>
                        </div>
                        <div class="macabets-edge-grid">
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Market Line</div>
                                <div class="macabets-edge-value">{html.escape(vegas_line_text)}</div>
                            </div>
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Macabets Fair Line</div>
                                <div class="macabets-edge-value">{html.escape(fair_line_text)}</div>
                            </div>
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Market Edge</div>
                                <div class="macabets-edge-value">{html.escape(edge_text)}</div>
                            </div>
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Projected Winner</div>
                                <div class="macabets-edge-value">{html.escape(projected_nfl_winner)}</div>
                            </div>
                        </div>
                        <div class="macabets-score-row">
                            <div class="macabets-score-box">
                                <div class="macabets-edge-label">Projected Score</div>
                                <div class="macabets-projected-score">{html.escape(away)} {away_score} – {html.escape(home)} {home_score}</div>
                                <div class="small-note">Projected margin: {html.escape(winner)} by {margin} · {html.escape(outlook)}</div>
                            </div>
                            <div class="macabets-score-box">
                                <div class="macabets-edge-label">Win Probability</div>
                                <div class="macabets-probability">
                                    {html.escape(away)} {nfl_result['away_win_probability']:.1%}<br>
                                    {html.escape(home)} {nfl_result['home_win_probability']:.1%}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if price_report["verdict"].lower() == "pass":
                    st.info(decision_sentence)
                else:
                    st.success(decision_sentence)

                st.caption(
                    "Market edge is the difference between the Macabets fair spread and the entered market spread. "
                    "It does not mean the listed team is projected to win by that amount."
                )

                st.markdown("### Why Macabets Sees It This Way")
                factor_col, risk_col = st.columns(2)
                with factor_col:
                    st.markdown("**Decisive factors**")
                    for item in explanation_report["key_advantages"][:4]:
                        st.markdown(f"- {item}")
                with risk_col:
                    st.markdown("**Risk factors**")
                    for item in explanation_report["risks"][:4]:
                        st.markdown(f"- {item}")

                st.markdown("### Expected Game Script")
                st.write(explanation_report["game_script"])

                if nfl_considered_side != "Just analyze":
                    if nfl_considered_side == nfl_result["away_team"]:
                        considered_probability = float(nfl_result["away_win_probability"])
                        considered_fair_moneyline = fair_away_moneyline
                        considered_market_moneyline = int(market_ml_away)
                        considered_fair_spread = -fair_home_spread
                        considered_market_spread = -entered_market_home_spread
                        no_vig_away, _, _ = no_vig_probabilities(
                            int(market_ml_away), int(market_ml_home)
                        )
                        considered_no_vig_probability = no_vig_away
                    else:
                        considered_probability = float(nfl_result["home_win_probability"])
                        considered_fair_moneyline = int(nfl_result["fair_moneyline_home"])
                        considered_market_moneyline = int(market_ml_home)
                        considered_fair_spread = fair_home_spread
                        considered_market_spread = entered_market_home_spread
                        _, no_vig_home, _ = no_vig_probabilities(
                            int(market_ml_away), int(market_ml_home)
                        )
                        considered_no_vig_probability = no_vig_home

                    if nfl_considered_market == "Moneyline":
                        considered_price = considered_market_moneyline
                        considered_roi = (
                            considered_probability
                            * (american_to_decimal(considered_price) - 1)
                            - (1 - considered_probability)
                        )
                        bet_threshold = minimum_acceptable_odds(
                            considered_probability,
                            required_roi=0.05,
                        )
                        threshold_text = format_american(bet_threshold)
                        current_wager_text = (
                            f"{nfl_considered_side} ML "
                            f"{format_american(considered_market_moneyline)}"
                        )
                        fair_wager_text = format_american(considered_fair_moneyline)
                        wager_edge_text = (
                            f"{considered_probability - considered_no_vig_probability:+.1%}"
                        )
                        edge_label = "Probability edge"
                        threshold_label = "BET price or better"
                    else:
                        considered_price = int(nfl_spread_price)
                        considered_point_edge = (
                            considered_market_spread - considered_fair_spread
                        )
                        considered_cover_probability = estimated_nfl_cover_probability(
                            considered_point_edge
                        )
                        considered_roi = (
                            considered_cover_probability
                            * american_to_decimal(considered_price)
                            - 1
                        )
                        required_point_edge = required_nfl_spread_edge(
                            considered_price,
                            required_roi=0.05,
                        )
                        bet_threshold = considered_fair_spread + required_point_edge
                        threshold_text = (
                            f"{nfl_considered_side} {bet_threshold:+.1f} or better"
                        )
                        current_wager_text = (
                            f"{nfl_considered_side} {considered_market_spread:+.1f} "
                            f"({format_american(considered_price)})"
                        )
                        fair_wager_text = (
                            f"{nfl_considered_side} {considered_fair_spread:+.1f}"
                        )
                        wager_edge_text = f"{considered_point_edge:+.1f} points"
                        edge_label = "Spread edge"
                        threshold_label = "BET line"

                    nfl_bet_decision = "BET" if considered_roi >= 0.05 else "PASS"

                    st.markdown(f"#### Your Considered Bet: {nfl_considered_side}")
                    bet1, bet2, bet3, bet4, bet5 = st.columns(5)
                    bet1.metric("Decision", nfl_bet_decision)
                    bet2.metric("Your Wager", current_wager_text)
                    bet3.metric("Macabets Fair", fair_wager_text)
                    bet4.metric(edge_label, wager_edge_text)
                    bet5.metric("Estimated ROI", f"{considered_roi:+.1%}")

                    threshold1, threshold2 = st.columns([2, 3])
                    threshold1.metric(threshold_label, threshold_text)
                    with threshold2:
                        if nfl_bet_decision == "BET":
                            st.success(
                                f"BETTABLE EDGE: The available {nfl_considered_market.lower()} clears "
                                "Macabets' 5% expected-return threshold."
                            )
                        else:
                            st.info(
                                f"PRICE ASSESSMENT: The available {nfl_considered_market.lower()} does not "
                                f"clear the 5% expected-return threshold. Macabets needs {threshold_text}. "
                                "This price assessment does not reverse the projected winner."
                            )

                    if nfl_considered_market == "Spread":
                        st.caption(
                            "Spread ROI is estimated from the difference between the Macabets fair "
                            "line and the entered market line using a 13.86-point NFL scoring-margin "
                            "distribution. Treat this as a transparent first-pass estimate until the "
                            "NFL engine adds a full margin simulation."
                        )
                else:
                    st.caption(
                        "Select a team under Bet consideration and generate the report to receive "
                        "a direct BET or PASS decision."
                    )

                active_weather = nfl_result.get("weather_context") or st.session_state.get("nfl_weather_context", {})
                if active_weather:
                    st.markdown("### Weather")
                    weather1, weather2, weather3 = st.columns([3, 1, 1])
                    weather1.write(str(active_weather.get("summary") or "No weather adjustment."))
                    weather2.metric("Impact", str(active_weather.get("impact") or "None"))
                    total_weather_move = float(active_weather.get("total_adjustment", 0.0) or 0.0)
                    weather3.metric("Total Adj.", f"{total_weather_move:+.1f}")
                    if not active_weather.get("available", False):
                        st.caption("Weather data was unavailable, so Macabets applied no weather adjustment.")
                    elif float(active_weather.get("home_margin_adjustment", 0.0) or 0.0):
                        st.caption(str(active_weather.get("climate_mismatch") or ""))

                st.markdown("### Matchup Advantages")
                category_verdicts, category_wins, strongest_edge, category_leader = (
                    build_nfl_category_verdicts(
                        nfl_result["away_team"],
                        nfl_result["home_team"],
                        NFL_QUALITY_RATINGS,
                    )
                )

                away_wins = category_wins[nfl_result["away_team"]]
                home_wins = category_wins[nfl_result["home_team"]]
                cat1, cat2, cat3 = st.columns(3)
                cat1.metric("Category Leader", category_leader)
                cat2.metric("Category Score", f"{away_wins}-{home_wins}")
                cat3.metric("Even Categories", category_wins["Even"])

                advantage_cols = st.columns(2)
                visible_rows = category_verdicts[["Category", "Advantage", "Strength"]].to_dict("records")
                for idx, row in enumerate(visible_rows):
                    with advantage_cols[idx % 2]:
                        leader = row["Advantage"]
                        strength = row["Strength"]
                        symbol = "⚪" if leader == "Even" else "🟢"
                        st.markdown(
                            f"{symbol} **{row['Category']}** — {leader}<br>"
                            f"<span style='color:#6b7280;font-size:0.9rem'>{strength} edge</span>",
                            unsafe_allow_html=True,
                        )

                if strongest_edge:
                    st.success(
                        f"Strongest matchup edge: {strongest_edge['Advantage']} in "
                        f"{strongest_edge['Category']} ({strongest_edge['Strength'].lower()})."
                    )

                with st.expander("Detailed category ratings", expanded=False):
                    st.dataframe(
                        category_verdicts,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rating Gap": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )
                    st.caption(
                        "These are provisional composite ratings. Position-specific player data will replace them when verified live inputs are connected."
                    )

                with st.expander("Technical fair-line model audit", expanded=False):
                    audit1, audit2, audit3 = st.columns(3)
                    audit1.metric(
                        f"{nfl_result['away_team']} Power Rating",
                        f"{nfl_result['away_power_rating']:+.2f}",
                    )
                    audit2.metric(
                        f"{nfl_result['home_team']} Power Rating",
                        f"{nfl_result['home_power_rating']:+.2f}",
                    )
                    audit3.metric(
                        "Home-Field Adjustment",
                        f"{nfl_result['home_field_points']:+.1f}",
                    )
                    st.dataframe(
                        pd.DataFrame(nfl_result["rating_breakdown"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander("Game setting and expected script", expanded=False):
                    context1, context2 = st.columns(2)
                    with context1:
                        st.markdown("**Game setting**")
                        st.write(f"Week: {int(nfl_week)}")
                        st.write(f"Venue: {venue_type}")
                        st.write(f"Weather: {weather}")
                        if active_weather.get("source"):
                            st.write(f"Weather source: {active_weather.get('source')}")
                        st.write(f"Neutral site: {'Yes' if neutral_site else 'No'}")
                    with context2:
                        st.markdown("**Model interpretation**")
                        st.write(f"Macabets favorite: {model_favorite}")
                        st.write(f"Spread-value side: {spread_value_text}")
                        st.write(
                            f"Home-field adjustment: "
                            f"{nfl_result['home_field_points']:+.1f} points"
                        )
                        if abs(float(active_weather.get("total_adjustment", 0.0) or 0.0)) >= 0.05:
                            st.write(
                                f"Projected total: market anchor {float(market_total):.1f}, "
                                f"weather adjustment {float(active_weather.get('total_adjustment', 0.0)):+.1f}"
                            )
                        else:
                            st.write("Projected total: market-anchored")

                    st.markdown("**Expected game script**")
                    st.write(nfl_result["game_script"])

                matchup_brain = nfl_result.get("matchup_brain", {})
                if matchup_brain:
                    st.markdown("### NFL Brain")
                    decision_framework = matchup_brain.get("decision_framework", {})
                    questions = decision_framework.get("questions", []) if decision_framework else []
                    if questions:
                        for item in questions:
                            answer = item.get("answer", "Waiting for verified data")
                            if answer in {"Insufficient current data", "Unable to determine"}:
                                answer = "Waiting for verified data"
                                confidence = "Not available"
                                reasoning = (
                                    "Macabets does not yet have enough verified current NFL information to answer this responsibly. "
                                    "It will not guess or rely on outdated inputs."
                                )
                            else:
                                confidence = item.get("readiness_label", "Medium")
                                reasoning = item.get("reason", "")
                            st.markdown(f"**{item.get('number')}. {item.get('question')}**")
                            st.markdown(f"**Answer:** {answer}")
                            st.write(reasoning)
                            st.caption(f"Confidence: {confidence}")

                        if all(
                            item.get("answer") in {"Insufficient current data", "Unable to determine"}
                            for item in questions
                        ):
                            football_summary = (
                                "The NFL Brain is ready, but verified current roster, player, injury and performance data is not yet connected. "
                                "Macabets is intentionally withholding matchup conclusions until those inputs are available."
                            )
                        else:
                            football_summary = matchup_brain.get("summary", "")
                        st.markdown("**Football Summary**")
                        st.info(football_summary)

                    with st.expander("Technical Details", expanded=False):
                        st.write(matchup_brain.get("summary", ""))
                        st.write(f"Status: {matchup_brain.get('status', 'unknown')}")
                        if decision_framework:
                            st.write(
                                f"Questions ready: {int(decision_framework.get('ready_questions', 0))}/8"
                            )
                            for item in questions:
                                st.markdown(f"**Question {item.get('number')}**")
                                st.write(f"Readiness: {item.get('readiness_label', 'Unknown')} ({int(item.get('readiness_score', 0))}/100)")
                                missing_required = item.get("missing_required", [])
                                missing_optional = item.get("missing_optional", [])
                                if missing_required:
                                    st.write("Missing required: " + "; ".join(missing_required))
                                if missing_optional:
                                    st.write("Missing helpful: " + "; ".join(missing_optional))
                                st.write(item.get("refusal_rule", ""))
                        for limitation in matchup_brain.get("limitations", []):
                            st.caption(f"Data boundary: {limitation}")

                with st.expander("Supporting arguments, swing factors and risks", expanded=False):
                    home_path, away_path = st.columns(2)
                    with home_path:
                        st.markdown(f"**Why {nfl_result['home_team']} can win**")
                        for reason in nfl_result["why_home_can_win"]:
                            st.markdown(f"- {reason}")
                    with away_path:
                        st.markdown(f"**Why {nfl_result['away_team']} can win**")
                        for reason in nfl_result["why_away_can_win"]:
                            st.markdown(f"- {reason}")

                    swing_col, risk_col = st.columns(2)
                    with swing_col:
                        st.markdown("**Biggest swing factors**")
                        for factor in nfl_result["swing_factors"]:
                            st.markdown(f"- {factor}")
                    with risk_col:
                        st.markdown("**Biggest risk**")
                        st.write(nfl_result["biggest_risk"])
                        st.markdown("**Conditions that invalidate the report**")
                        for condition in nfl_result["invalidation_conditions"]:
                            st.markdown(f"- {condition}")

                with st.expander("Supporting spread comparison", expanded=False):
                    if value_team:
                        st.write(
                            f"Vegas lists {nfl_result['home_team']} at {entered_market_home_spread:+.1f}. "
                            f"Macabets makes the fair home line {fair_home_spread:+.1f}, a "
                            f"{abs(spread_difference):.1f}-point difference toward {market_direction}. "
                            f"At the entered market line, the potential spread-value side is {spread_value_text}."
                        )
                    else:
                        st.write(
                            f"Vegas lists {nfl_result['home_team']} at {entered_market_home_spread:+.1f}, "
                            f"while Macabets makes the fair home line {fair_home_spread:+.1f}. "
                            f"The {abs(spread_difference):.1f}-point difference is not large enough to create a material spread edge."
                        )

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
    archive_tabs = st.tabs(["Analysis Log", "Legacy Tennis Archive", "Matchup Lab"])

    with archive_tabs[0]:
        st.subheader("Universal Analysis Log")
        st.caption("Every Tennis and NFL analysis is saved automatically as a frozen snapshot.")
        render_price_verdict_guide()

        warning = st.session_state.pop("analysis_log_warning", None)
        if warning:
            st.warning(warning)
        saved_name = st.session_state.pop("analysis_log_last_saved", None)
        if saved_name:
            st.success(f"Saved to Analysis Log: {saved_name}")

        if not analysis_db_configured():
            st.error("Permanent storage is not configured yet.")
            st.code(
                'SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"\n'
                'SUPABASE_KEY = "YOUR-SUPABASE-ANON-KEY"',
                language="toml",
            )
            st.info("Run the included supabase_analysis_log.sql file once in the Supabase SQL Editor, then add these two values to Streamlit secrets.")
        else:
            # Load the complete archive first so the report card is not affected by display filters.
            try:
                all_universal_rows = db_list_analyses(5000)
            except Exception as exc:
                all_universal_rows = []
                st.error(f"Could not load the permanent Analysis Log: {exc}")

            def _prediction_result(row):
                status = str(row.get("status", "Pending"))
                if status == "Won":
                    return "Correct"
                if status == "Lost":
                    return "Incorrect"
                return status if status in {"Pending", "Push", "Void"} else "Pending"

            completed_rows = [row for row in all_universal_rows if _prediction_result(row) in {"Correct", "Incorrect"}]
            correct_rows = [row for row in completed_rows if _prediction_result(row) == "Correct"]
            incorrect_rows = [row for row in completed_rows if _prediction_result(row) == "Incorrect"]
            pending_rows = [row for row in all_universal_rows if _prediction_result(row) == "Pending"]
            overall_accuracy = len(correct_rows) / len(completed_rows) if completed_rows else None

            st.markdown("### Prediction Report Card")
            report1, report2, report3, report4, report5 = st.columns(5)
            report1.metric("Prediction Accuracy", f"{overall_accuracy:.1%}" if overall_accuracy is not None else "—")
            report2.metric("Graded", len(completed_rows))
            report3.metric("Correct", len(correct_rows))
            report4.metric("Incorrect", len(incorrect_rows))
            report5.metric("Pending", len(pending_rows))

            overall_summary = summarize_rows(
                all_universal_rows,
                verdict_getter=_analysis_verdict,
            )
            overall_record = f"{overall_summary['correct']}-{overall_summary['incorrect']}"
            bet_summary = overall_summary["bet"]
            pass_summary = overall_summary["pass"]
            split1, split2, split3 = st.columns(3)
            split1.metric(
                "Overall Prediction Record",
                overall_record,
                f"{overall_summary['accuracy']:.1%}" if overall_summary["accuracy"] is not None else "No graded predictions",
            )
            split2.metric(
                "BET Predictions",
                f"{bet_summary['correct']}-{bet_summary['incorrect']}",
                f"{bet_summary['pending']} pending · {bet_summary['total']} total",
            )
            split3.metric(
                "PASS Predictions",
                f"{pass_summary['correct']}-{pass_summary['incorrect']}",
                f"{pass_summary['pending']} pending · {pass_summary['total']} total",
            )
            st.caption(
                "Every analysis counts toward prediction accuracy. BET and PASS are tracked separately because a pass still contains a predicted winner."
            )

            sport_cards = []
            for report_sport in ("Tennis", "NFL"):
                sport_completed = [row for row in completed_rows if str(row.get("sport", "")) == report_sport]
                sport_correct = sum(_prediction_result(row) == "Correct" for row in sport_completed)
                sport_accuracy = sport_correct / len(sport_completed) if sport_completed else None
                sport_cards.append((report_sport, sport_correct, len(sport_completed) - sport_correct, sport_accuracy))

            tennis_card, nfl_card = st.columns(2)
            for card, (report_sport, sport_correct, sport_incorrect, sport_accuracy) in zip(
                (tennis_card, nfl_card), sport_cards
            ):
                record = f"{sport_correct}-{sport_incorrect}" if sport_correct + sport_incorrect else "0-0"
                card.metric(
                    report_sport,
                    f"{sport_accuracy:.1%}" if sport_accuracy is not None else "—",
                    f"Record: {record}",
                )

            with st.expander("Confidence calibration", expanded=False):
                confidence_bands = [(90, 100), (80, 89), (70, 79), (60, 69), (0, 59)]
                calibration_rows = []
                for low, high in confidence_bands:
                    band_rows = []
                    for row in completed_rows:
                        try:
                            confidence_value = float(row.get("confidence"))
                        except (TypeError, ValueError):
                            continue
                        if confidence_value <= 10:
                            confidence_value *= 10
                        if low <= confidence_value <= high:
                            band_rows.append(row)
                    band_correct = sum(_prediction_result(row) == "Correct" for row in band_rows)
                    band_incorrect = len(band_rows) - band_correct
                    calibration_rows.append({
                        "Confidence": f"{low}–{high}" if low else "Below 60",
                        "Record": f"{band_correct}-{band_incorrect}",
                        "Graded": len(band_rows),
                        "Actual Accuracy": band_correct / len(band_rows) if band_rows else None,
                    })
                st.dataframe(
                    pd.DataFrame(calibration_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Actual Accuracy": st.column_config.NumberColumn(format="%.1%%")},
                )
                st.caption("Pending, Push and Void entries are excluded. Confidence is tested against actual prediction accuracy.")

            st.divider()
            st.markdown("### Browse by Day")
            rows_by_day = group_rows_by_day(all_universal_rows)
            available_days = list(rows_by_day.keys())
            if available_days:
                selected_day = st.selectbox(
                    "Choose analysis date",
                    available_days,
                    format_func=lambda value: pd.to_datetime(value).strftime("%A, %B %-d, %Y")
                    if value else "Unknown date",
                    key="analysis_log_selected_day",
                    help="Only analyses from this date will appear below.",
                )
                selected_day_rows = rows_by_day.get(selected_day, [])
                day_summary = summarize_rows(
                    selected_day_rows,
                    verdict_getter=_analysis_verdict,
                )
                day_bet = day_summary["bet"]
                day_pass = day_summary["pass"]
                day1, day2, day3, day4, day5 = st.columns(5)
                day1.metric("Day Record", f"{day_summary['correct']}-{day_summary['incorrect']}")
                day2.metric(
                    "Day Accuracy",
                    f"{day_summary['accuracy']:.1%}" if day_summary["accuracy"] is not None else "—",
                )
                day3.metric("BET Record", f"{day_bet['correct']}-{day_bet['incorrect']}")
                day4.metric("PASS Record", f"{day_pass['correct']}-{day_pass['incorrect']}")
                day5.metric("Pending", day_summary["pending"])
                st.caption(
                    f"Showing {day_summary['total']} analyses saved for the selected date. Open any one below to mark it Correct, Incorrect or Pending."
                )
            else:
                selected_day = None
                selected_day_rows = []

            st.divider()
            filter1, filter2, filter3 = st.columns([1, 1, 2])
            sport_filter = filter1.selectbox("Sport", ["All", "Tennis", "NFL"], key="analysis_log_sport_filter")
            result_filter = filter2.selectbox(
                "Prediction Result", ["All", "Pending", "Correct", "Incorrect", "Push", "Void"],
                key="analysis_log_status_filter",
            )
            search_filter = filter3.text_input("Search", placeholder="Player, team, event, price assessment or verdict", key="analysis_log_search")

            universal_rows = list(selected_day_rows if selected_day is not None else all_universal_rows)
            if sport_filter != "All":
                universal_rows = [row for row in universal_rows if str(row.get("sport", "")) == sport_filter]
            if result_filter != "All":
                universal_rows = [row for row in universal_rows if _prediction_result(row) == result_filter]

            if search_filter.strip():
                q = search_filter.strip().casefold()
                universal_rows = [
                    row for row in universal_rows
                    if q in (
                        " ".join(str(row.get(field, "")) for field in (
                            "event_name", "participant_a", "participant_b", "prediction",
                            "recommendation", "sport", "model_version"
                        )) + " " + _analysis_price_assessment(row)
                    ).casefold()
                ]

            if not universal_rows:
                st.write("No analyses match the current filters.")
            else:
                log_frame = pd.DataFrame([{
                    "ID": str(row.get("id", ""))[:8],
                    "Date": str(row.get("event_date") or row.get("created_at", ""))[:10],
                    "Sport": row.get("sport", ""),
                    "Event": row.get("event_name", ""),
                    "Prediction": row.get("prediction", ""),
                    "Prediction Confidence": row.get("confidence"),
                    "Actual Line": _analysis_market_line(row),
                    "Fair Line": row.get("fair_line", ""),
                    "Price Assessment": _analysis_price_assessment(row),
                    "Verdict": _analysis_verdict(row),
                    "Prediction Result": _prediction_result(row),
                } for row in universal_rows])
                st.dataframe(
                    log_frame, use_container_width=True, hide_index=True,
                )

                selected_id = st.selectbox(
                    "Open analysis",
                    [row["id"] for row in universal_rows],
                    format_func=lambda analysis_id: next(
                        f"{row.get('event_date', '')} — {row.get('sport', '')}: {row.get('event_name', '')}"
                        for row in universal_rows if row["id"] == analysis_id
                    ),
                    key="universal_analysis_selected_id",
                )
                selected_row = next(row for row in universal_rows if row["id"] == selected_id)

                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("Prediction", selected_row.get("prediction") or "—")
                d2.metric("Prediction Confidence", f"{float(selected_row.get('confidence')):.0f}/100" if selected_row.get("confidence") is not None else "—")
                d3.metric("Actual Line", _analysis_market_line(selected_row))
                d4.metric("Fair Line", selected_row.get("fair_line") or "—")
                d5.metric("Price Assessment", _analysis_price_assessment(selected_row))
                d6.metric("Verdict", _analysis_verdict(selected_row))

                with st.expander("Why did Macabets give these labels?", expanded=False):
                    specific, assessment_definition, verdict_definition = _analysis_price_verdict_explanation(selected_row)
                    st.write(specific)
                    st.markdown(f"**{_analysis_price_assessment(selected_row)}:** {assessment_definition}")
                    st.markdown(f"**{_analysis_verdict(selected_row)}:** {verdict_definition}")

                action_cols = st.columns([1, 1])
                with action_cols[0]:
                    if str(selected_row.get("sport", "")) == "Tennis":
                        if st.button(
                            "Open in Tennis Analysis",
                            type="primary",
                            use_container_width=True,
                            key=f"reopen_tennis_{selected_id}",
                        ):
                            original_inputs = selected_row.get("input_snapshot") or {}
                            if not isinstance(original_inputs, dict):
                                original_inputs = {}

                            event_date_value = (
                                original_inputs.get("match_date")
                                or original_inputs.get("event_date")
                                or selected_row.get("event_date")
                                or date.today()
                            )
                            try:
                                event_date_value = pd.to_datetime(event_date_value).date()
                            except Exception:
                                event_date_value = date.today()

                            participant_a = str(
                                original_inputs.get("player_a")
                                or original_inputs.get("participant_a")
                                or selected_row.get("participant_a")
                                or ""
                            ).strip()
                            participant_b = str(
                                original_inputs.get("player_b")
                                or original_inputs.get("participant_b")
                                or selected_row.get("participant_b")
                                or ""
                            ).strip()
                            tournament_value = str(
                                original_inputs.get("tournament")
                                or selected_row.get("event_name")
                                or "Montreal"
                            ).strip()
                            surface_value = str(
                                original_inputs.get("surface")
                                or "Hard"
                            ).title()
                            if surface_value not in {"Hard", "Clay", "Grass", "Carpet"}:
                                surface_value = "Hard"
                            round_value = str(original_inputs.get("round") or original_inputs.get("round_label") or "R32")
                            category_value = str(original_inputs.get("tournament_category") or original_inputs.get("event_category") or "ATP 250")
                            if category_value not in {
                                "Grand Slam", "Masters 1000", "ATP 500", "ATP 250",
                                "Challenger", "Tour Finals", "Davis Cup"
                            }:
                                category_value = "ATP 250"
                            format_value = str(original_inputs.get("match_format") or "Best of 3")
                            if format_value not in {"Best of 3", "Best of 5"}:
                                format_value = "Best of 3"

                            odds_a = original_inputs.get("market_odds_a", selected_row.get("market_odds_a", -180))
                            odds_b = original_inputs.get("market_odds_b", selected_row.get("market_odds_b", 155))
                            st.session_state.pending_fair_line_prefill = {
                                "fle_date": event_date_value,
                                "fle_tournament": tournament_value,
                                "fle_surface": surface_value,
                                "fle_round": round_value,
                                "fle_favorite": participant_a,
                                "fle_opponent": participant_b,
                                "fle_market_a": safe_int(odds_a, -180),
                                "fle_market_b": safe_int(odds_b, 155),
                                "auto_match_date": event_date_value,
                                "auto_tournament": tournament_value,
                                "auto_surface": surface_value,
                                "auto_round": round_value,
                                "auto_tournament_category": category_value,
                                "auto_environment": str(original_inputs.get("environment") or "Outdoor"),
                                "auto_match_format": format_value,
                                "auto_player_a": participant_a,
                                "auto_player_b": participant_b,
                                "auto_considering_bet": "Just analyze",
                                "auto_market_a": safe_int(odds_a, -180),
                                "auto_market_b": safe_int(odds_b, 155),
                                "auto_simulations": safe_int(original_inputs.get("simulations", 20000), 20000),
                            }
                            # Reopening is for review only. Rerun the current model without
                            # creating a second Analysis Log entry for the same matchup.
                            st.session_state.suppress_next_tennis_log = True
                            st.session_state.run_analysis_from_daily_slate = True
                            st.session_state.open_analysis_engine_tab = True
                            st.session_state.reopened_analysis_notice = (
                                f"Reopened {participant_a} vs {participant_b} from the Analysis Log. "
                                "The matchup has been rerun using the current Macabets model."
                            )
                            st.rerun()
                    else:
                        st.button(
                            "Open in Analysis Engine",
                            use_container_width=True,
                            disabled=True,
                            key=f"reopen_disabled_{selected_id}",
                            help="Direct reopening is currently available for Tennis analyses.",
                        )
                with action_cols[1]:
                    with st.popover("View Frozen Snapshot", use_container_width=True):
                        st.caption("This is the original saved output and does not change when the model is updated.")
                        st.json(selected_row.get("analysis_snapshot", {}), expanded=False)
                        st.markdown("#### Original Inputs")
                        st.json(selected_row.get("input_snapshot", {}), expanded=False)

                edit1, edit2 = st.columns(2)
                result_options = ["Pending", "Correct", "Incorrect", "Push", "Void"]
                current_result = _prediction_result(selected_row)
                updated_result = edit1.selectbox(
                    "Was Macabets' prediction correct?", result_options,
                    index=result_options.index(current_result) if current_result in result_options else 0,
                    key=f"universal_status_{selected_id}",
                    help="Grade only the predicted winner. Correct is stored as Won and Incorrect as Lost for database compatibility.",
                )
                status_storage_map = {
                    "Pending": "Pending", "Correct": "Won", "Incorrect": "Lost",
                    "Push": "Push", "Void": "Void",
                }
                updated_status = status_storage_map[updated_result]
                updated_favorite = edit2.checkbox(
                    "Favorite / keep", value=bool(selected_row.get("favorite", False)),
                    key=f"universal_favorite_{selected_id}",
                )
                updated_notes = st.text_area(
                    "Notes", value=str(selected_row.get("notes", "")),
                    key=f"universal_notes_{selected_id}",
                )
                updated_review = st.text_area(
                    "Post-event review", value=str(selected_row.get("review", "")),
                    key=f"universal_review_{selected_id}",
                )
                updated_lesson = st.text_area(
                    "What should Macabets learn?", value=str(selected_row.get("lesson", "")),
                    key=f"universal_lesson_{selected_id}",
                )

                save_log_col, delete_log_col = st.columns(2)
                if save_log_col.button("Save Analysis Updates", type="primary", use_container_width=True, key=f"save_universal_{selected_id}"):
                    try:
                        db_update_analysis(selected_id, {
                            "status": updated_status, "favorite": updated_favorite,
                            "notes": updated_notes.strip(), "review": updated_review.strip(),
                            "lesson": updated_lesson.strip(),
                        })
                        st.success("Analysis updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not update analysis: {exc}")

                confirm_delete = st.checkbox(
                    "Confirm permanent deletion", key=f"confirm_delete_universal_{selected_id}"
                )
                if delete_log_col.button(
                    "Delete Analysis Permanently", use_container_width=True,
                    disabled=not confirm_delete, key=f"delete_universal_{selected_id}"
                ):
                    try:
                        db_delete_analysis(selected_id)
                        st.success("Analysis permanently deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not delete analysis: {exc}")

                export_frame = pd.DataFrame(universal_rows)
                st.download_button(
                    "Download Analysis Log CSV",
                    export_frame.to_csv(index=False).encode("utf-8"),
                    f"macabets_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv", use_container_width=True,
                )

    with archive_tabs[1]:
        st.subheader("Legacy Tennis Analysis Archive")
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

    with archive_tabs[2]:
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


with tabs[6]:
    st.subheader("How to Read Macabets Betting Grades")
    st.caption(
        "Macabets separates winner prediction accuracy from whether the offered moneyline is worth betting."
    )

    st.markdown("### The Three-Part Decision")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Winner Prediction")
        st.write(
            "Who Macabets believes is most likely to win. Every prediction can be graded for accuracy, "
            "even when the betting verdict is Pass."
        )
    with c2:
        st.markdown("#### 2. Matchup Quality")
        st.write(
            "How stable and repeatable the projected result appears after considering surface, style, "
            "serve-return interaction, fatigue, form and realistic upset paths."
        )
    with c3:
        st.markdown("#### 3. Price Decision")
        st.write(
            "Whether the current moneyline is acceptable after weighing Macabets' fair line, matchup "
            "quality, confidence and risk. A slightly expensive favorite can still be playable."
        )

    st.divider()
    st.markdown("### Betting Grades")
    grade_rows = [
        {
            "Grade": "Strong Bet",
            "Meaning": "One of the strongest opportunities on the board.",
            "What Macabets sees": "High conviction, strong matchup structure and an acceptable or favorable price.",
        },
        {
            "Grade": "Worth Betting",
            "Meaning": "A bet Macabets believes is justified.",
            "What Macabets sees": "The price and matchup are strong enough together to support a wager, even if the line is not perfect.",
        },
        {
            "Grade": "Lean",
            "Meaning": "A smaller or more cautious betting case.",
            "What Macabets sees": "A preferred side exists, but the edge, stability or price is not strong enough for a full recommendation.",
        },
        {
            "Grade": "Pass",
            "Meaning": "Likely winner, but not a good enough bet at the current number.",
            "What Macabets sees": "The player may still be favored to win, but price, volatility or matchup risk makes the wager unattractive.",
        },
        {
            "Grade": "Complete Pass",
            "Meaning": "Stay away from the wager.",
            "What Macabets sees": "The price is too poor, confidence is too low, uncertainty is too high, or the underdog has too many realistic upset paths.",
        },
    ]
    st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)

    st.info(
        "Important: Macabets does not automatically pass a bet just because the market line is worse "
        "than its fair line. It also evaluates whether this specific matchup is stable enough to justify "
        "paying a reasonable premium."
    )

    st.markdown("### Price Assessment Terms")
    price_rows = [
        {"Label": label, "Definition": definition}
        for label, definition in PRICE_ASSESSMENT_DEFINITIONS.items()
    ]
    st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)

    st.markdown("### Example: Same Price, Different Decision")
    left, right = st.columns(2)
    with left:
        st.success("Worth Betting at -340")
        st.write(
            "The favorite has a highly stable matchup, multiple paths to victory, limited fatigue or injury risk, "
            "and the underdog has very few realistic upset paths."
        )
    with right:
        st.warning("Pass at -340")
        st.write(
            "The favorite is still likely to win, but the matchup is volatile, the underdog has a dangerous weapon, "
            "or the price does not compensate for the added risk."
        )
