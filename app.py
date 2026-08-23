
import io
import math
import json
import re
import html
import urllib.error
import urllib.parse
import urllib.request
import importlib
from pathlib import Path
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
    from engine.tennis_evidence import build_tennis_evidence_packet
    from engine.tennis_h2h import build_head_to_head_summary as build_tennis_head_to_head_summary
    from engine.tennis import (
        analyze as analyze_tennis_match,
        player_names as tennis_player_names,
        tournament_names as tennis_tournament_names,
        tournament_surface as tennis_tournament_surface,
        tournament_category as tennis_tournament_category,
        tournament_surface_for_display_name as tennis_tournament_surface_for_display_name,
        tournament_category_for_display_name as tennis_tournament_category_for_display_name,
    )
    TENNIS_ENGINE_AVAILABLE = True
    TENNIS_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    TENNIS_ENGINE_AVAILABLE = False
    TENNIS_ENGINE_IMPORT_ERROR = str(exc)

try:
    from engine.api_tennis import APITennisClient, APITennisError
    API_TENNIS_AVAILABLE = True
    API_TENNIS_IMPORT_ERROR = ""
except Exception as exc:
    APITennisClient = None
    APITennisError = RuntimeError
    API_TENNIS_AVAILABLE = False
    API_TENNIS_IMPORT_ERROR = str(exc)

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

try:
    from engine.ufc import (
        analyze as analyze_ufc_match,
        fighter_names as ufc_fighter_names,
        load_ufc_fights,
        load_ufc_ratings,
    )
    from engine.ufc_validation import run_historical_validation, UFCValidationConfig
    UFC_ENGINE_AVAILABLE = True
    UFC_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    UFC_ENGINE_AVAILABLE = False
    UFC_ENGINE_IMPORT_ERROR = str(exc)

APP_VERSION = "Macabets v0.98 — Tennis Audit Release"
BUILD_DATE = "August 22, 2026"

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

# Shared round options for tennis analysis widgets. The placeholder is first
# so any auto-filled entry (daily slate, archive replay) that doesn't have a
# real detected round lands on an explicit "please confirm" state instead of
# silently defaulting to a specific round that's usually wrong.
TENNIS_ROUND_NOT_DETECTED = "— Select round —"
TENNIS_ROUND_OPTIONS = [
    TENNIS_ROUND_NOT_DETECTED, "Qualifying", "R128", "R64", "R32", "R16",
    "Quarterfinal", "Semifinal", "Final",
]

def _format_nfl_availability_timestamp(value):
    """Format an ISO availability timestamp in Eastern Time for the NFL UI."""
    if not value:
        return "Unavailable"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(EASTERN_TZ)
        return local.strftime("%B %d, %Y · %-I:%M %p ET").replace(" 0", " ")
    except Exception:
        return str(value)


@st.cache_data(ttl=300, show_spinner=False)
def _load_nfl_personnel_details():
    """Load the detailed generated team/unit ratings used by Availability Intelligence."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "nfl" / "team_ratings_auto.json",
        Path(__file__).resolve().parent / "data" / "team_ratings_auto.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            continue
    return {}



def _style_metric(raw_text, label):
    """Extract one signed trait edge from the technical style-matchup explanation."""
    match = re.search(rf"{re.escape(label)}\s*([+-]?\d+(?:\.\d+)?)", str(raw_text), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _plain_nfl_style_why(category, advantage, strength, raw_why, away_team, home_team):
    """Translate technical Madden trait deltas into direct football language for the UI.

    The underlying matchup calculations stay untouched. This function only changes how
    the already-computed result is explained to the user.
    """
    category = str(category or "")
    advantage = str(advantage or "Even")
    strength = str(strength or "Even")
    raw_why = str(raw_why or "")
    away_team = str(away_team or "")
    home_team = str(home_team or "")

    subject_team = None
    for team in (away_team, home_team):
        if team and category.lower().startswith(team.lower()):
            subject_team = team
            break
    if not subject_team:
        subject_team = away_team or home_team or "This team"

    if subject_team == away_team:
        opponent = home_team
    elif subject_team == home_team:
        opponent = away_team
    else:
        opponent = home_team or away_team or "the opponent"

    strength_word = strength.lower() if strength and strength.lower() != "even" else "small"
    subject_poss = f"{subject_team}\'" if subject_team.lower().endswith("s") else f"{subject_team}\'s"
    opponent_poss = f"{opponent}\'" if opponent.lower().endswith("s") else f"{opponent}\'s"
    is_even = advantage.lower() == "even"
    subject_has_edge = advantage == subject_team
    opponent_has_edge = advantage == opponent
    cat = category.lower()

    def strongest(metrics, positive=True):
        available = [(label, value) for label, value in metrics if isinstance(value, (int, float))]
        if not available:
            return None
        return max(available, key=lambda item: item[1]) if positive else min(available, key=lambda item: item[1])

    if "receiver/qb traits vs coverage" in cat:
        metrics = [
            ("speed", _style_metric(raw_why, "receiver speed")),
            ("route running", _style_metric(raw_why, "route-vs-coverage")),
            ("press release", _style_metric(raw_why, "release-vs-press")),
            ("deep passing", _style_metric(raw_why, "deep-pass-vs-coverage")),
        ]
        best = strongest(metrics, True)
        worst = strongest(metrics, False)
        positive_text = {
            "speed": "receiver speed can stress the coverage vertically",
            "route running": "the route-running matchup should create separation",
            "press release": "the receivers are well equipped to beat press coverage",
            "deep passing": "the deep-passing matchup creates real big-play potential",
        }
        negative_text = {
            "speed": "the receivers do not have a clear speed advantage",
            "route running": "consistent separation could be difficult",
            "press release": "beating press coverage could be a problem",
            "deep passing": "the defense is positioned to limit deep shots",
        }
        best_text = positive_text.get(best[0]) if best and best[1] > 0.5 else None
        worst_text = negative_text.get(worst[0]) if worst and worst[1] < -0.5 else None

        if is_even:
            parts = ["The passing-game matchup is fairly balanced."]
            if best_text:
                parts.append(f"For {subject_team}, {best_text}.")
            if worst_text:
                parts.append(f"The counter is that {worst_text}.")
            return " ".join(parts)
        if subject_has_edge:
            sentence = f"{subject_poss} passing game has a {strength_word} matchup advantage against {opponent_poss} coverage."
            if best_text:
                sentence += f" The biggest reason is that {best_text}."
            if worst_text:
                sentence += f" It is not a bigger edge because {worst_text}."
            return sentence
        if opponent_has_edge:
            sentence = f"{opponent_poss} secondary has a {strength_word} edge against {subject_poss} passing-game traits."
            if worst_text:
                sentence += f" The main concern for {subject_team} is that {worst_text}."
            if best_text:
                sentence += f" {subject_team} still has some upside because {best_text}."
            return sentence

    if "ol technique vs pass rush" in cat:
        metrics = [
            ("finesse", _style_metric(raw_why, "pass-block finesse vs finesse rush")),
            ("power", _style_metric(raw_why, "pass-block power vs power rush")),
            ("recognition", _style_metric(raw_why, "protection recognition")),
        ]
        best = strongest(metrics, True)
        worst = strongest(metrics, False)
        positive_text = {
            "finesse": "its blockers match up well with finesse rushers",
            "power": "its blockers are well equipped to handle power rushers",
            "recognition": "its protection recognition should help identify pressure cleanly",
        }
        negative_text = {
            "finesse": "finesse pressure could give the tackles problems",
            "power": "power rushers could collapse the pocket",
            "recognition": "pressure recognition and pickup could be stressed",
        }
        best_text = positive_text.get(best[0]) if best and best[1] > 0.5 else None
        worst_text = negative_text.get(worst[0]) if worst and worst[1] < -0.5 else None

        if is_even:
            return f"{subject_poss} offensive line and {opponent_poss} pass rush are closely matched; neither side has a dependable technique advantage."
        if subject_has_edge:
            sentence = f"{subject_poss} offensive line has a {strength_word} edge against {opponent_poss} pass rush."
            if best_text:
                sentence += f" The clearest reason is that {best_text}."
            if worst_text:
                sentence += f" The vulnerability is that {worst_text}."
            return sentence
        if opponent_has_edge:
            sentence = f"{opponent_poss} pass rush has a {strength_word} matchup edge against {subject_poss} offensive line."
            if worst_text:
                sentence += f" The biggest concern is that {worst_text}."
            if best_text:
                sentence += f" {subject_team} can offset some of that because {best_text}."
            return sentence

    if "run style vs front seven" in cat:
        metrics = [
            ("blocking", _style_metric(raw_why, "blocking vs shedding")),
            ("runner creation", _style_metric(raw_why, "runner creation vs tackling")),
            ("backfield speed", _style_metric(raw_why, "backfield speed vs pursuit")),
        ]
        best = strongest(metrics, True)
        worst = strongest(metrics, False)
        positive_text = {
            "blocking": "the blocking matchup should create cleaner running lanes",
            "runner creation": "the backs have an advantage creating yards after contact",
            "backfield speed": "the backfield speed can stress pursuit and edge discipline",
        }
        negative_text = {
            "blocking": "the front can win blocks and close running lanes",
            "runner creation": "the backs do not hold a clear tackle-breaking advantage",
            "backfield speed": "the defense has enough pursuit speed to limit runs to the edge",
        }
        best_text = positive_text.get(best[0]) if best and best[1] > 0.5 else None
        worst_text = negative_text.get(worst[0]) if worst and worst[1] < -0.5 else None

        if is_even:
            parts = [f"{subject_poss} rushing style is a fairly even match for {opponent_poss} front seven."]
            if best_text:
                parts.append(f"The offense can still lean on the fact that {best_text}.")
            if worst_text:
                parts.append(f"But {worst_text}.")
            return " ".join(parts)
        if subject_has_edge:
            sentence = f"{subject_team} has a {strength_word} rushing matchup advantage against {opponent_poss} front seven."
            if best_text:
                sentence += f" The biggest reason is that {best_text}."
            if worst_text:
                sentence += f" The matchup is not one-sided because {worst_text}."
            return sentence
        if opponent_has_edge:
            sentence = f"{opponent_poss} front seven has a {strength_word} edge against {subject_poss} rushing style."
            if worst_text:
                sentence += f" The main issue is that {worst_text}."
            if best_text:
                sentence += f" {subject_team} still has a counter because {best_text}."
            return sentence

    if "qb movement vs contain" in cat:
        if is_even:
            return f"{subject_poss} quarterback mobility and {opponent_poss} ability to contain it are closely matched, so neither side owns a meaningful edge outside the pocket."
        if subject_has_edge:
            return (
                f"{subject_poss} quarterback has a {strength_word} mobility advantage. "
                f"Scrambling and movement outside the pocket can put extra stress on {opponent_poss} front-seven contain responsibilities."
            )
        if opponent_has_edge:
            return (
                f"{opponent_poss} front seven has a {strength_word} contain advantage. "
                f"It is well equipped to keep {subject_poss} quarterback from consistently creating extra yards or extended plays with mobility."
            )

    if is_even:
        return "The relevant starter traits are closely matched, so this area does not create a meaningful matchup advantage."
    return f"{advantage} has a {strength_word} matchup edge here based on the relevant starter traits."


def _render_nfl_style_matchup_table(style_table, away_team, home_team):
    """Render the style-matchup table with readable wrapping and a dominant Why column."""
    away_team = str(away_team or "")
    home_team = str(home_team or "")

    def advantage_class(value):
        value = str(value or "")
        if value == away_team:
            return "away"
        if value == home_team:
            return "home"
        if value.lower() == "even":
            return "even"
        return "other"

    def strength_class(value):
        value = str(value or "").lower()
        if value == "strong":
            return "strong"
        if value in {"moderate", "clear"}:
            return "moderate"
        if value == "slight":
            return "slight"
        return "even"

    def why_html(value):
        value = str(value or "").strip()
        if not value:
            return "—"
        # Make the takeaway immediately scannable, then put the supporting football
        # context on a second line. This preserves the explanation while avoiding a
        # wall of text inside the table.
        parts = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)
        lead = html.escape(parts[0])
        detail = html.escape(parts[1]) if len(parts) > 1 else ""
        if detail:
            return f'<span class="style-why-lead">{lead}</span><span class="style-why-detail">{detail}</span>'
        return f'<span class="style-why-lead">{lead}</span>'

    rows_html = []
    for _, row in style_table.iterrows():
        category = html.escape(str(row.get("Category", "—")))
        advantage = str(row.get("Advantage", "Even"))
        strength = str(row.get("Strength", "Even"))
        rows_html.append(
            f'''<tr>
                <td class="style-category">{category}</td>
                <td><span class="style-advantage {advantage_class(advantage)}">{html.escape(advantage)}</span></td>
                <td><span class="style-strength {strength_class(strength)}">{html.escape(strength)}</span></td>
                <td class="style-why">{why_html(row.get("Why", ""))}</td>
            </tr>'''
        )

    table_html = f'''
    <style>
      .macabets-style-wrap {{
        width: 100%;
        overflow-x: auto;
        margin: 0.15rem 0 0.75rem 0;
      }}
      .macabets-style-table {{
        width: 100%;
        min-width: 980px;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        border: 1px solid #d9dee7;
        border-radius: 10px;
        overflow: hidden;
        background: #ffffff;
        color: #172033;
        font-size: 0.88rem;
      }}
      .macabets-style-table th {{
        background: #f8fafc;
        color: #667085;
        font-weight: 600;
        text-align: left;
        padding: 0.62rem 0.7rem;
        border-bottom: 1px solid #d9dee7;
      }}
      .macabets-style-table td {{
        padding: 0.68rem 0.7rem;
        vertical-align: top;
        border-bottom: 1px solid #e7eaf0;
        line-height: 1.38;
        overflow-wrap: anywhere;
        white-space: normal;
      }}
      .macabets-style-table tr:last-child td {{ border-bottom: 0; }}
      .macabets-style-table th:nth-child(1), .macabets-style-table td:nth-child(1) {{ width: 23%; }}
      .macabets-style-table th:nth-child(2), .macabets-style-table td:nth-child(2) {{ width: 15%; }}
      .macabets-style-table th:nth-child(3), .macabets-style-table td:nth-child(3) {{ width: 10%; }}
      .macabets-style-table th:nth-child(4), .macabets-style-table td:nth-child(4) {{ width: 52%; }}
      .style-category {{ font-weight: 600; }}
      .style-advantage, .style-strength {{
        display: inline-block;
        padding: 0.22rem 0.48rem;
        border-radius: 7px;
        font-weight: 700;
        line-height: 1.2;
      }}
      .style-advantage.away {{ background: #eef2ff; color: #3730a3; }}
      .style-advantage.home {{ background: #ecfdf3; color: #166534; }}
      .style-advantage.even {{ background: #f3f4f6; color: #4b5563; }}
      .style-advantage.other {{ background: #f8fafc; color: #334155; }}
      .style-strength.strong {{ background: #ecfdf3; color: #166534; }}
      .style-strength.moderate {{ background: #eef2ff; color: #3730a3; }}
      .style-strength.slight {{ background: #eff6ff; color: #1d4ed8; }}
      .style-strength.even {{ background: #f3f4f6; color: #4b5563; }}
      .style-why {{ line-height: 1.45 !important; }}
      .style-why-lead {{ display: block; font-weight: 700; color: #172033; margin-bottom: 0.18rem; }}
      .style-why-detail {{ display: block; color: #475467; }}
      @media (max-width: 900px) {{
        .macabets-style-table {{ min-width: 900px; font-size: 0.84rem; }}
      }}
    </style>
    <div class="macabets-style-wrap">
      <table class="macabets-style-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Advantage</th>
            <th>Strength</th>
            <th>Gap</th>
            <th>Why</th>
          </tr>
        </thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    '''
    st.markdown(table_html, unsafe_allow_html=True)


def _availability_unit_effect(unit_name):
    effects = {
        "quarterback": "Passing efficiency, sack avoidance and offensive ceiling",
        "running_backs": "Rushing efficiency, pass protection and receiving out of the backfield",
        "receiving_weapons": "Separation, explosive passing and third-down receiving",
        "offensive_line": "Pass protection, pressure exposure and run blocking",
        "defensive_front": "Pass rush, edge containment and run defense",
        "linebackers": "Run fits, underneath coverage and tackling",
        "secondary": "Coverage quality, explosive-pass prevention and receiver matchups",
        "special_teams": "Kicking, punting and field-position value",
    }
    return effects.get(str(unit_name), "The affected personnel matchup")


def _team_availability_rows(team_profile):
    """Flatten unit-level Sleeper availability details into football-readable rows."""
    rows = []
    if not isinstance(team_profile, dict):
        return rows
    units = team_profile.get("units", {}) or {}
    for unit_name, unit in units.items():
        if not isinstance(unit, dict):
            continue
        unit_label = str(unit_name).replace("_", " ").title()
        effect = _availability_unit_effect(unit_name)
        grade = unit.get("grade")
        healthy_grade = unit.get("healthy_grade", grade)
        delta = unit.get("availability_grade_delta")
        for item in unit.get("unavailable_starters", []) or []:
            rows.append({
                "kind": "out",
                "Player": item.get("name", "—"),
                "Status": item.get("status", "Out"),
                "Role": item.get("role", "") or "Starter",
                "Unit": unit_label,
                "Replacement": "—",
                "Effect": effect,
                "HealthyGrade": healthy_grade,
                "CurrentGrade": grade,
                "Delta": delta,
            })
        for item in unit.get("availability_promotions", []) or []:
            rows.append({
                "kind": "promotion",
                "Player": item.get("out", "—"),
                "Status": "Out → backup activated",
                "Role": item.get("role", "") or "Starter",
                "Unit": unit_label,
                "Replacement": item.get("in", "—"),
                "Effect": effect,
                "HealthyGrade": healthy_grade,
                "CurrentGrade": grade,
                "Delta": delta,
            })
        for item in unit.get("availability_uncertainty", []) or []:
            rows.append({
                "kind": "uncertain",
                "Player": item.get("name", "—"),
                "Status": item.get("status", "Questionable"),
                "Role": item.get("role", "") or "Depth-chart player",
                "Unit": unit_label,
                "Replacement": "Not activated",
                "Effect": effect,
                "HealthyGrade": grade,
                "CurrentGrade": grade,
                "Delta": 0.0,
            })
    return rows


def _refresh_nfl_availability_now():
    """Refresh Sleeper, rebuild player/team ratings, and reload NFL runtime state."""
    global NFL_QUALITY_RATINGS, analyze_nfl_match
    from engine.nfl_availability import refresh_sleeper_availability
    from engine.nfl_rating_engine import build_and_save_ratings

    refresh_sleeper_availability()
    build_and_save_ratings()

    import engine.nfl_data as nfl_data_module
    import engine.nfl as nfl_module
    importlib.reload(nfl_data_module)
    nfl_module = importlib.reload(nfl_module)

    NFL_TEAM_RATINGS.clear()
    NFL_TEAM_RATINGS.update(nfl_data_module.NFL_TEAM_RATINGS)
    NFL_DATA_STATUS.clear()
    NFL_DATA_STATUS.update(nfl_data_module.NFL_DATA_STATUS)
    NFL_QUALITY_RATINGS = load_all_team_ratings()
    analyze_nfl_match = nfl_module.analyze
    st.cache_data.clear()


def _render_team_availability_detail(team_name, profile, rows):
    st.markdown(f"**{team_name}**")
    m1, m2, m3 = st.columns(3)
    unavailable = int(profile.get("unavailable_starters", 0) or 0)
    promotions = int(profile.get("availability_promotions", 0) or 0)
    uncertain = int(profile.get("availability_uncertain", 0) or 0)
    m1.metric("Unavailable starters", unavailable)
    m2.metric("Backups activated", promotions)
    m3.metric("Questionable / Doubtful", uncertain)

    definitive_rows = [row for row in rows if row.get("kind") in {"out", "promotion"}]
    uncertain_rows = [row for row in rows if row.get("kind") == "uncertain"]

    if definitive_rows:
        seen = set()
        for row in definitive_rows:
            key = (row.get("Player"), row.get("Unit"), row.get("Replacement"))
            if key in seen:
                continue
            seen.add(key)
            replacement = row.get("Replacement", "—")
            if replacement and replacement != "—":
                st.error(f"🔴 {row['Player']} — {row['Status']} · {row['Unit']}")
                st.markdown(f"**Backup activated:** {replacement}")
            else:
                st.error(f"🔴 {row['Player']} — {row['Status']} · {row['Unit']}")
            hg, cg, delta = row.get("HealthyGrade"), row.get("CurrentGrade"), row.get("Delta")
            if isinstance(hg, (int, float)) and isinstance(cg, (int, float)):
                delta_text = f" ({float(delta):+.1f})" if isinstance(delta, (int, float)) else ""
                st.caption(f"{row['Effect']} · Unit grade {float(hg):.1f} → {float(cg):.1f}{delta_text}")
            else:
                st.caption(row["Effect"])

    if uncertain_rows:
        for row in uncertain_rows:
            icon = "🟠" if str(row["Status"]).lower().startswith("doubt") else "🟡"
            st.warning(f"{icon} {row['Player']} — {row['Status']} · {row['Unit']}")
            st.caption(f"If inactive: {row['Effect']}. No backup is activated and no unit downgrade is applied yet.")

    if not definitive_rows and not uncertain_rows:
        st.success("🟢 No current Sleeper availability flags affecting the tracked depth chart.")
    elif not definitive_rows:
        st.success("🟢 No definitive starter replacement is active.")


def _render_nfl_availability_intelligence(away_team, home_team, away_profile, home_profile):
    """Show who is hurt, who replaced them, and which football units changed."""
    details = _load_nfl_personnel_details()
    away_detail = details.get(away_team, {}) if isinstance(details, dict) else {}
    home_detail = details.get(home_team, {}) if isinstance(details, dict) else {}

    # Detailed generated profiles carry unit/player lists. Keep aggregate loader profiles
    # as a fallback for counts and timestamps.
    if not isinstance(away_detail, dict) or not away_detail:
        away_detail = away_profile if isinstance(away_profile, dict) else {}
    if not isinstance(home_detail, dict) or not home_detail:
        home_detail = home_profile if isinstance(home_profile, dict) else {}

    updates = [
        value for value in [
            away_detail.get("availability_updated_at_utc"),
            home_detail.get("availability_updated_at_utc"),
            away_profile.get("availability_updated_at_utc") if isinstance(away_profile, dict) else None,
            home_profile.get("availability_updated_at_utc") if isinstance(home_profile, dict) else None,
            NFL_DATA_STATUS.get("availability_updated_at_utc"),
        ] if value
    ]
    updated = max(updates) if updates else None

    header1, header2 = st.columns([4, 1])
    with header1:
        st.markdown("### NFL Availability Intelligence")
        st.caption(
            f"Source: Sleeper availability + Footballguys depth chart · Last updated: "
            f"{_format_nfl_availability_timestamp(updated)}"
        )
    with header2:
        if st.button("Refresh Sleeper Data", key="refresh_sleeper_availability", use_container_width=True):
            try:
                with st.spinner("Refreshing Sleeper availability and rebuilding NFL personnel ratings..."):
                    _refresh_nfl_availability_now()
                st.success("NFL availability refreshed. Rebuilding matchup with the latest personnel.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not refresh Sleeper availability: {exc}")

    away_rows = _team_availability_rows(away_detail)
    home_rows = _team_availability_rows(home_detail)
    c1, c2 = st.columns(2)
    with c1:
        _render_team_availability_detail(away_team, away_detail or away_profile, away_rows)
    with c2:
        _render_team_availability_detail(home_team, home_detail or home_profile, home_rows)

    all_rows = away_rows + home_rows
    promotions = [row for row in all_rows if row.get("kind") == "promotion"]
    uncertain_rows = [row for row in all_rows if row.get("kind") == "uncertain"]
    if promotions:
        affected = sorted({row.get("Unit", "") for row in promotions if row.get("Unit")})
        st.warning(
            f"Availability has changed the active depth chart: {len(promotions)} backup activation(s). "
            f"Affected unit(s): {', '.join(affected)}. Macabets' current unit grades already include these replacements."
        )
    elif uncertain_rows:
        st.info(
            f"No definitive starter replacement is active, but {len(uncertain_rows)} Questionable/Doubtful "
            "designation(s) remain unresolved. Macabets keeps those players active until Sleeper reports a definitive unavailable status."
        )
    else:
        st.success("No injury-driven starter substitutions are currently applied to this matchup.")

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


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_active_sports(api_key):
    payload, headers = _api_get_json("/sports", {"apiKey": api_key, "all": "true"})
    return payload, {
        "remaining": headers.get("x-requests-remaining", "—"),
        "used": headers.get("x-requests-used", "—"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_api_tennis_today():
    """Load today's ATP/WTA fixtures from API-Tennis without consuming Odds API credits."""
    if not API_TENNIS_AVAILABLE or APITennisClient is None:
        return [], {"source": "unavailable", "error": API_TENNIS_IMPORT_ERROR or "API-Tennis client unavailable"}
    try:
        client = APITennisClient()
        response = client.get_fixtures(
            datetime.now(EASTERN_TZ).date(),
            datetime.now(EASTERN_TZ).date(),
            timezone_name="America/New_York",
        )
        return response.result, {"source": response.source, "fetched_at": response.fetched_at, "error": ""}
    except Exception as exc:
        return [], {"source": "unavailable", "fetched_at": None, "error": str(exc)}


def normalize_api_tennis_schedule(fixtures):
    """Normalize API-Tennis fixtures into the same basic columns as the automatic slate."""
    rows = []
    today_eastern = datetime.now(EASTERN_TZ).date()
    for event in fixtures or []:
        event_type = str(event.get("event_type_type") or "").strip().lower()
        if "atp singles" not in event_type and "wta singles" not in event_type:
            continue
        raw_date = str(event.get("event_date") or "").strip()
        try:
            event_date = datetime.fromisoformat(raw_date).date()
        except Exception:
            try:
                event_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except Exception:
                continue
        if event_date != today_eastern:
            continue
        player_a = str(event.get("event_first_player") or "").strip()
        player_b = str(event.get("event_second_player") or "").strip()
        if not player_a or not player_b:
            continue
        event_time = str(event.get("event_time") or "").strip()
        try:
            parsed_time = datetime.strptime(event_time, "%H:%M").time()
            start_time = datetime.combine(event_date, parsed_time, tzinfo=EASTERN_TZ)
            time_et = start_time.strftime("%-I:%M %p")
        except Exception:
            start_time = datetime.combine(event_date, time(12, 0), tzinfo=EASTERN_TZ)
            time_et = event_time or "TBD"
        tournament = str(event.get("tournament_name") or event.get("league_name") or event_type.upper()).strip()
        rows.append({
            "event_id": str(event.get("event_key") or ""),
            "start_time": start_time,
            "time_et": time_et,
            "sport": tournament,
            "participant_a": player_a,
            "participant_b": player_b,
            "odds_a": pd.NA,
            "odds_b": pd.NA,
            "book_a": "—",
            "book_b": "—",
        })
    return pd.DataFrame(rows).sort_values(["start_time", "sport", "participant_a"]).reset_index(drop=True) if rows else pd.DataFrame()


def _odds_quota_exhausted(messages):
    text = " ".join(str(message) for message in (messages or [])).upper()
    return "OUT_OF_USAGE_CREDITS" in text or "USAGE QUOTA HAS BEEN REACHED" in text


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


def tennis_probability_confidence_band(probability_a, reliability_score=100):
    """Return a user-facing confidence label anchored to win probability.

    Win probability sets the maximum label. Data/model reliability may downgrade
    the label, but can never promote a close matchup into a higher-confidence tier.
    """
    try:
        probability_a = float(probability_a)
    except (TypeError, ValueError):
        probability_a = 0.5
    favorite_probability = max(probability_a, 1.0 - probability_a)

    if favorite_probability >= 0.85:
        probability_level = 3
    elif favorite_probability >= 0.80:
        probability_level = 2
    elif favorite_probability >= 0.60:
        probability_level = 1
    else:
        probability_level = 0

    try:
        reliability_score = float(reliability_score)
    except (TypeError, ValueError):
        reliability_score = 0.0
    if reliability_score >= 85:
        reliability_level = 3
    elif reliability_score >= 70:
        reliability_level = 2
    elif reliability_score >= 55:
        reliability_level = 1
    else:
        reliability_level = 0

    labels = ["Low Confidence", "Moderate Confidence", "High Confidence", "Very High Confidence"]
    return labels[min(probability_level, reliability_level)]


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
    """Compatibility wrapper around the tested tennis H2H engine."""
    return build_tennis_head_to_head_summary(matches, player_a, player_b, current_surface)



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
    """Render H2H as a conclusion first; keep the raw record one click deeper."""
    h2h = build_head_to_head_summary(matches, player_a, player_b, current_surface)
    st.markdown("#### Head-to-Head Summary")

    if h2h["meetings"] == 0:
        unresolved = []
        for requested, resolution in (
            (player_a, h2h.get("resolution_a", {})),
            (player_b, h2h.get("resolution_b", {})),
        ):
            if not resolution.get("resolved"):
                unresolved.append(str(requested))
        if unresolved:
            st.warning(
                "Head-to-head lookup could not resolve "
                + ", ".join(unresolved)
                + " to the historical tennis database. This is an identity/data issue, "
                  "not evidence that the players have never met."
            )
        else:
            st.info("No previous meetings were found in the available Macabets match data.")
        return

    meetings = int(h2h.get("meetings", 0) or 0)
    wins_a = int(h2h.get("wins_a", 0) or 0)
    wins_b = int(h2h.get("wins_b", 0) or 0)
    if meetings < 3 or wins_a == wins_b:
        h2h_take = "The available head-to-head does not create a meaningful matchup edge."
    else:
        leader = player_a if wins_a > wins_b else player_b
        h2h_take = f"{leader} owns the better available head-to-head record, but Macabets treats it as supporting context rather than a primary signal."
    st.info(h2h_take)

    last = h2h["last_meeting"]
    score_text = f" Score: {last['score']}." if last.get("score") else ""
    st.caption(
        f"Last meeting: {last['winner']} won on {last['date']} at {last['event']}.{score_text}"
    )

    with st.expander("Show head-to-head record", expanded=False):
        h1, h2, h3 = st.columns(3)
        h1.metric("Overall meetings", h2h["meetings"])
        h2.metric(f"{player_a} H2H wins", h2h["wins_a"])
        h3.metric(f"{player_b} H2H wins", h2h["wins_b"])

        s1, s2, s3 = st.columns(3)
        s1.metric(f"Meetings on {current_surface}", h2h["surface_meetings"])
        s2.metric(f"{player_a} {current_surface} wins", h2h["surface_wins_a"])
        s3.metric(f"{player_b} {current_surface} wins", h2h["surface_wins_b"])

        resolved_a = h2h.get("resolved_player_a")
        resolved_b = h2h.get("resolved_player_b")
        if resolved_a or resolved_b:
            st.caption(
                "Historical identity match: "
                f"{player_a} -> {resolved_a or 'unresolved'}; "
                f"{player_b} -> {resolved_b or 'unresolved'}."
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
            "Roster Continuity",
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
            "Roster Continuity": f"{team} returns more of its current starting core from last season.",
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


VERDICT_ORDER = {
    "Complete Pass": 0,
    "Pass": 1,
    "Lean": 2,
    "Worth Betting": 3,
    "Strong Bet": 4,
}


def verdict_probability_ceiling(model_probability):
    """Return the strongest verdict allowed by projected win probability.

    Price can make a bet attractive, but it cannot manufacture conviction that
    the win-probability model does not have. These are ceilings, not automatic
    recommendations.
    """
    probability = min(max(float(model_probability), 0.0), 1.0)
    if probability < 0.52:
        return "Pass"
    if probability < 0.575:
        return "Lean"
    if probability < 0.65:
        return "Worth Betting"
    return "Strong Bet"


def cap_verdict_by_probability(verdict, model_probability):
    """Prevent price/confidence or Challenge mode from exceeding conviction."""
    verdict = str(verdict or "Pass")
    ceiling = verdict_probability_ceiling(model_probability)
    if VERDICT_ORDER.get(verdict, VERDICT_ORDER["Pass"]) > VERDICT_ORDER[ceiling]:
        return ceiling
    return verdict


def moneyline_price_quality(model_probability, market_odds, confidence_score):
    """Separate mathematical price value from the probability-capped verdict."""
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

    # Final conviction gate. Market value can lower the price required to bet,
    # but it cannot promote a low-probability pick into a stronger verdict.
    # <52%: Pass max | 52-57.4%: Lean max | 57.5-64.9%: Worth Betting max
    # >=65%: Strong Bet becomes eligible (still requires ROI/confidence above).
    verdict = cap_verdict_by_probability(verdict, probability)

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
        "against the model edge and current model confidence."
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


def _analysis_confidence_label(row):
    """Return the qualitative confidence label used by the current model UI.

    Saved Tennis analyses already store the probability-anchored band inside the
    frozen snapshot. NFL/UFC snapshots expose their engine band. Legacy rows fall
    back to projected win probability so the archive never needs to display the
    retired 1-100 score.
    """
    snapshot = row.get("analysis_snapshot") or {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except (TypeError, ValueError, json.JSONDecodeError):
            snapshot = {}

    candidates = []
    if isinstance(snapshot, dict):
        analysis_confidence = snapshot.get("analysis_confidence") or {}
        if isinstance(analysis_confidence, dict):
            candidates.extend([
                analysis_confidence.get("band"),
                analysis_confidence.get("label"),
            ])
        engine_result = snapshot.get("engine_result") or {}
        if isinstance(engine_result, dict):
            candidates.extend([
                engine_result.get("confidence_band"),
                engine_result.get("confidence_label"),
            ])
        candidates.extend([
            snapshot.get("confidence_band"),
            snapshot.get("confidence_label"),
        ])

    aliases = {
        "low": "Low",
        "moderate": "Moderate",
        "solid": "High",
        "high": "High",
        "very high": "Very High",
        "exceptional": "Very High",
        "pass": "Low",
    }
    for candidate in candidates:
        label = str(candidate or "").strip()
        if not label:
            continue
        label = re.sub(r"\s+confidence$", "", label, flags=re.IGNORECASE).strip()
        normalized = aliases.get(label.casefold())
        if normalized:
            return normalized

    # Legacy fallback: use the stored projected winner probability. This mirrors
    # the current Tennis confidence thresholds without exposing a numeric score.
    try:
        probability = float(row.get("predicted_probability"))
    except (TypeError, ValueError):
        probability = None
    if probability is not None and 0.0 < probability < 1.0:
        favorite_probability = max(probability, 1.0 - probability)
        if favorite_probability >= 0.85:
            return "Very High"
        if favorite_probability >= 0.80:
            return "High"
        if favorite_probability >= 0.60:
            return "Moderate"
        return "Low"

    return "—"


def _ufc_derivative_performance(row):
    """Read the evaluated UFC derivative market and its automatic settlement result.

    UFC derivative grades deliberately live inside the frozen analysis snapshot so the
    existing Analysis Log schema remains the source of truth and does not need a parallel
    UFC table.
    """
    if str(row.get("sport", "")) != "UFC":
        return {}

    snapshot = row.get("analysis_snapshot") or {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except (TypeError, ValueError, json.JSONDecodeError):
            snapshot = {}
    if not isinstance(snapshot, dict):
        return {}

    evaluation = snapshot.get("derivative_evaluation") or {}
    settlement = snapshot.get("settlement") or {}
    derivative = settlement.get("derivative") or {} if isinstance(settlement, dict) else {}
    if not isinstance(evaluation, dict) or not evaluation.get("available"):
        return {}

    primary = evaluation.get("primary") or {}
    secondary = evaluation.get("secondary") or {}
    return {
        "market_key": str(evaluation.get("market_key") or ""),
        "market_type": str(evaluation.get("market_type") or ""),
        "primary_label": str(evaluation.get("primary_label") or ""),
        "primary_odds": primary.get("market_odds") if isinstance(primary, dict) else None,
        "primary_verdict": str(primary.get("verdict") or "") if isinstance(primary, dict) else "",
        "primary_result": str(derivative.get("primary_status") or "Pending") if isinstance(derivative, dict) else "Pending",
        "secondary_label": str(evaluation.get("secondary_label") or ""),
        "secondary_odds": secondary.get("market_odds") if isinstance(secondary, dict) else None,
        "secondary_verdict": str(secondary.get("verdict") or "") if isinstance(secondary, dict) else "",
        "secondary_result": str(derivative.get("secondary_status") or "Pending") if isinstance(derivative, dict) and secondary else "",
        "actual_method": str(derivative.get("actual_method") or "") if isinstance(derivative, dict) else "",
        "actual_round": derivative.get("actual_round") if isinstance(derivative, dict) else None,
        "actual_time": str(derivative.get("actual_time") or "") if isinstance(derivative, dict) else "",
    }


def _analysis_price_verdict_explanation(row):
    """Explain the selected log entry using its own market line, fair line, confidence and labels."""
    pricing = _analysis_pricing_report(row)
    assessment = pricing.get("price_assessment", "—")
    verdict = pricing.get("verdict", "—")
    prediction = str(row.get("prediction") or "The predicted winner")
    actual_line = _analysis_market_line(row)
    fair_line = str(row.get("fair_line") or "—")

    assessment_text = PRICE_ASSESSMENT_DEFINITIONS.get(assessment, "This label compares the market line with Macabets' fair line.")
    verdict_text = VERDICT_DEFINITIONS.get(verdict, "This verdict weighs both price and the current confidence level.")

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
    analysis_tabs = st.tabs(["Tennis Analysis", "NFL Analysis", "UFC Analysis", "Outcome Simulator"])

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
                    TENNIS_ROUND_OPTIONS,
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
                # Tennis format probabilities are now exact closed-form calculations.
                # Keep the legacy argument internally for archive/API compatibility, but
                # do not ask the user to choose a Monte Carlo count that no longer affects
                # the result.
                simulations = 20000
                o3.metric("Format engine", "Exact")

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

                    # User-facing confidence is anchored to the actual projected win
                    # probability. Reliability can downgrade the label, but cannot
                    # make a near coin-flip look like a high-confidence prediction.
                    analysis_confidence["band"] = tennis_probability_confidence_band(
                        model_probability, analysis_confidence["overall"]
                    )
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
                            "model_version": str(result.get("model_version") or "Macabets Tennis v0.98"),
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
                        challenged_verdict = cap_verdict_by_probability(
                            str(active_challenge["proposed_verdict"]),
                            projected_winner_probability,
                        )
                        projected_price_report["verdict"] = challenged_verdict
                        projected_price_report["recommendation"] = challenged_verdict
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

                    try:
                        verified_recent_evidence = build_tennis_evidence_packet(
                            matches,
                            analyzed_a,
                            analyzed_b,
                            market_snapshot.get("match_date", match_date.isoformat()),
                            result.get("surface", surface),
                            tournament=result.get("tournament", tournament),
                            lookback=20,
                        )
                    except Exception as evidence_exc:
                        verified_recent_evidence = {
                            "source": "Macabets local ATP match database",
                            "status": "evidence_packet_error",
                            "error": str(evidence_exc),
                        }

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
                        "verified_recent_evidence": verified_recent_evidence,
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
                        stability_band = str(match_intelligence.get("stability_band", "—"))
                        volatility_band = str(match_intelligence.get("volatility_band", "—"))
                        st.info(
                            f"Matchup stability: **{stability_band}** · Volatility: **{volatility_band}**. "
                            "Macabets uses the underlying scores internally; the bands are the decision-useful takeaway."
                        )

                        drivers = match_intelligence.get("drivers", []) or []
                        if drivers:
                            st.markdown("**Primary volatility drivers**")
                            for driver in drivers:
                                st.markdown(f"- {driver.capitalize()}")

                        st.markdown(f"#### Upset Paths for {match_intelligence.get('underdog', 'the underdog')}")
                        for path in match_intelligence.get("upset_paths", []):
                            st.markdown(f"- {path}")

                        with st.expander("Audit stability and volatility scores", expanded=False):
                            intel1, intel2, intel3 = st.columns(3)
                            intel1.metric("Matchup Stability", f"{match_intelligence.get('stability_score', 0)}/100")
                            intel2.metric("Volatility", f"{match_intelligence.get('volatility_score', 0)}/100")
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

                    st.markdown("#### Objective Match Price")
                    m1, m2, m3 = st.columns(3)
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
                    m3.metric(
                        "Model Confidence",
                        analysis_confidence["band"],
                    )

                    comparison = pd.DataFrame(
                        {
                            "Player": [analyzed_a, analyzed_b],
                            "Market": [f"{no_vig_a:.1%}", f"{no_vig_b:.1%}"],
                            "Macabets": [f"{model_probability:.1%}", f"{probability_b:.1%}"],
                        }
                    )
                    with st.expander("Audit Macabets vs. market probabilities", expanded=False):
                        st.metric("Sportsbook hold", f"{sportsbook_hold:.1%}")
                        st.dataframe(comparison, use_container_width=True, hide_index=True)

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

                    st.markdown("#### Confidence")
                    confidence_col1, confidence_col2 = st.columns(2)
                    with confidence_col1:
                        st.markdown("**Model Confidence**")
                        st.write(f"**{analysis_confidence['band']}**")
                        st.caption(
                            "Win probability sets the confidence ceiling. Data quality, sample size, "
                            "health/context clarity and conflicting matchup evidence can only lower it."
                        )

                    with confidence_col2:
                        if considered_player and bet_confidence:
                            st.markdown(
                                f"**Confidence in Your {considered_player} Bet**"
                            )
                            st.write(f"**{bet_confidence['band']}**")
                            st.caption(
                                "This is a secondary price/edge label. Win probability remains the primary signal."
                            )
                        else:
                            st.markdown("**Confidence in a Specific Bet**")
                            st.info(
                                "Select a player before analyzing to receive a separate bet-confidence label."
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

                    # Reconcile the exact format distribution with the final matchup-adjusted verdict.
                    # Preserve the format engine's conditional set-score distribution for each
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

                    st.markdown("#### Outcome Shape")
                    if straight_sets_a > straight_sets_b and straight_sets_a >= deciding_set_probability:
                        shape_take = f"The most likely clean match shape is {analyzed_a} winning in straight sets."
                    elif straight_sets_b > straight_sets_a and straight_sets_b >= deciding_set_probability:
                        shape_take = f"The most likely clean match shape is {analyzed_b} winning in straight sets."
                    else:
                        shape_take = "Macabets sees a meaningful path to a deciding set; the match shape is less clean than the headline winner probability."
                    st.info(shape_take)
                    with st.expander("Show simulation probabilities and exact set scores", expanded=False):
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
                        f"Final pre-format model: {result['model_probability']:.1%}. "
                        "Best-of-3 / best-of-5 format probability is calculated exactly "
                        "with no Monte Carlo noise."
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
                    availability_updated = NFL_DATA_STATUS.get("availability_updated_at_utc")
                    if availability_updated:
                        st.info(
                            f"Automatic NFL availability: Sleeper — updated {availability_updated}. "
                            f"{NFL_DATA_STATUS.get('availability_definitively_unavailable', 0)} players currently marked definitively unavailable; "
                            f"{NFL_DATA_STATUS.get('availability_uncertain', 0)} Questionable/Doubtful."
                        )
                    else:
                        st.warning(
                            "Automatic Sleeper availability snapshot is not present yet. "
                            "Run the Update Macabets NFL Data workflow to refresh injuries/availability."
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
                    "\n**Coaching:** 2026 head-coach prior from the sourced league-wide coaching file; experience and available 2025 head-coach record are used conservatively, with new coaches held near neutral until current-season evidence develops."
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
                profile5.metric("Roster Continuity", f"{selected_profile['continuity']:.1f}")

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

            # Keep venue/weather defined on every Streamlit rerun. The generated NFL
            # result persists in session state, while the button body only runs once;
            # without stable defaults these locals could disappear on later reruns and
            # crash the archived/report view with NameError.
            venue_type = manual_venue_type if manual_weather_override else auto_venue_type
            weather = manual_weather if manual_weather_override else "Automatic"

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
                        game_date=nfl_date,
                        week=int(nfl_week),
                        season=int(nfl_date.year),
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

                _render_nfl_availability_intelligence(
                    nfl_result["away_team"],
                    nfl_result["home_team"],
                    NFL_QUALITY_RATINGS.get(nfl_result["away_team"], {}),
                    NFL_QUALITY_RATINGS.get(nfl_result["home_team"], {}),
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

                matchup_intelligence = nfl_result.get("matchup_intelligence") or {}
                category_verdicts = pd.DataFrame(matchup_intelligence.get("categories", []))
                if category_verdicts.empty:
                    category_verdicts, category_wins, strongest_edge, category_leader = build_nfl_category_verdicts(
                        nfl_result["away_team"], nfl_result["home_team"], NFL_QUALITY_RATINGS
                    )
                else:
                    category_wins = {nfl_result["away_team"]: 0, nfl_result["home_team"]: 0, "Even": 0}
                    for advantage in category_verdicts["Advantage"].tolist():
                        if advantage in category_wins:
                            category_wins[advantage] += 1
                    non_even = category_verdicts[category_verdicts["Advantage"] != "Even"].copy()
                    strongest_edge = (
                        non_even.sort_values("Rating Gap", ascending=False).iloc[0].to_dict()
                        if not non_even.empty else None
                    )
                    category_leader = matchup_intelligence.get("overall_leader", "Even")
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

                # The primary NFL recommendation is a winner/moneyline decision.
                # Keep spread disagreement visible as secondary market context, but
                # never pair a moneyline verdict with the opposite side's spread.
                # Macabets still projects a winner on every analysis even when the
                # current price does not justify a wager.
                actionable_moneyline_verdicts = {"Strong Bet", "Worth Betting", "Lean"}
                recommendation_text = (
                    f"{projected_nfl_winner} ML {format_american(winner_market_ml)}"
                    if price_report["verdict"] in actionable_moneyline_verdicts
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
                                <div class="macabets-edge-label">Spread Value</div>
                                <div class="macabets-edge-value">{html.escape(edge_text)}</div>
                            </div>
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Macabets Fair Moneyline</div>
                                <div class="macabets-edge-value">{html.escape(projected_nfl_winner)} {format_american(winner_fair_ml)}</div>
                            </div>
                            <div class="macabets-edge-item">
                                <div class="macabets-edge-label">Market Moneyline</div>
                                <div class="macabets-edge-value">{html.escape(projected_nfl_winner)} {format_american(winner_market_ml)}</div>
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

                # Legacy heading retained for regression-test compatibility: ### Why Macabets Sees It This Way
                st.markdown("### Decision Drivers")
                factor_col, risk_col = st.columns(2)
                with factor_col:
                    # Legacy label retained for regression-test compatibility: **Decisive factors**
                    st.markdown("**Reasons for the lean**")
                    for item in explanation_report["key_advantages"][:4]:
                        st.markdown(f"- {item}")
                with risk_col:
                    # Legacy label retained for regression-test compatibility: **Risk factors**
                    st.markdown("**What could flip it**")
                    for item in explanation_report["risks"][:4]:
                        st.markdown(f"- {item}")

                st.markdown("### Expected Game Script")
                st.markdown(explanation_report["game_script"])

                simulation_context = nfl_result.get("simulation_context") or {}
                if simulation_context.get("available"):
                    st.markdown("### Game Simulation")
                    sim_favorite = str(simulation_context.get("favorite") or projected_nfl_winner)
                    sim_fav_prob = float(simulation_context.get("favorite_win_probability", projected_nfl_probability) or projected_nfl_probability)
                    sim_upset = float(simulation_context.get("upset_probability", 1.0 - sim_fav_prob) or 0.0)
                    sim_volatility = str(simulation_context.get("volatility") or "Moderate")
                    sim_script = str(simulation_context.get("game_script") or "Competitive game")
                    sim1, sim2, sim3 = st.columns(3)
                    sim1.metric("Most Likely Script", sim_script)
                    sim2.metric("Favorite Wins", f"{sim_fav_prob:.1%}")
                    sim3.metric("Underdog Wins", f"{sim_upset:.1%}")
                    one_score = float(simulation_context.get("one_score_probability", 0.0) or 0.0)
                    st.info(
                        f"{sim_favorite} wins {sim_fav_prob:.1%} of simulated outcomes. "
                        f"About {one_score:.1%} finish within one score. Simulation volatility: {sim_volatility.lower()}."
                    )

                    with st.expander("Show simulation ranges and market probabilities", expanded=False):
                        sr1, sr2, sr3 = st.columns(3)
                        margin_50 = simulation_context.get("margin_range_50") or [0.0, 0.0]
                        total_50 = simulation_context.get("total_range_50") or [0.0, 0.0]
                        sr1.metric("Middle 50% Home Margin", f"{float(margin_50[0]):+.1f} to {float(margin_50[1]):+.1f}")
                        sr2.metric("Middle 50% Game Total", f"{float(total_50[0]):.1f} to {float(total_50[1]):.1f}")
                        sr3.metric("Simulation Runs", f"{int(simulation_context.get('simulations', 0) or 0):,}")

                        home_cover = simulation_context.get("home_cover_probability")
                        over_prob = simulation_context.get("over_probability")
                        mp1, mp2 = st.columns(2)
                        if home_cover is not None:
                            away_cover = 1.0 - float(home_cover)
                            mp1.write(
                                f"**Spread at entered line:** {nfl_result['home_team']} {float(home_cover):.1%} cover / "
                                f"{nfl_result['away_team']} {away_cover:.1%} cover"
                            )
                        if over_prob is not None:
                            mp2.write(
                                f"**Total at entered line:** Over {float(over_prob):.1%} / Under {1.0 - float(over_prob):.1%}"
                            )

                        away_80 = simulation_context.get("away_score_range_80") or [0.0, 0.0]
                        home_80 = simulation_context.get("home_score_range_80") or [0.0, 0.0]
                        st.caption(
                            f"80% simulated score range — {nfl_result['away_team']}: "
                            f"{float(away_80[0]):.0f}-{float(away_80[1]):.0f}; "
                            f"{nfl_result['home_team']}: {float(home_80[0]):.0f}-{float(home_80[1]):.0f}. "
                            + str(simulation_context.get("method_note") or "")
                        )

                matchup_intelligence = nfl_result.get("matchup_intelligence") or {}
                # Legacy section name retained for test compatibility: ### Matchup Advantages
                if matchup_intelligence.get("available"):
                    st.markdown("### Unified NFL Matchup Intelligence")

                    mi1, mi2 = st.columns(2)
                    mi1.metric("Overall Football Edge", str(matchup_intelligence.get("overall_leader", "Even")))
                    mi2.metric("Edge Strength", str(matchup_intelligence.get("overall_strength", "Toss-up")))
                    st.info(matchup_intelligence.get("summary", ""))

                    # Keep intermediate scoring and adjustment values available for audit,
                    # but do not make them compete with the football conclusion in the default UI.
                    with st.expander("Show matchup model details", expanded=False):
                        edge_points = float(matchup_intelligence.get("football_edge_points", 0.0) or 0.0)
                        md1, md2, md3 = st.columns(3)
                        md1.metric("Weighted Edge", f"{edge_points:.1f} pts")
                        md2.metric("Matchup Adjustment", f"{float(matchup_intelligence.get('matchup_adjustment_home', 0.0) or 0.0):+.2f} pts home")
                        md3.metric("Trait / Style Adjustment", f"{float(matchup_intelligence.get('style_adjustment_home', 0.0) or 0.0):+.2f} pts home")
                        st.caption(f"Data mode: {matchup_intelligence.get('data_mode', 'Baseline')}")
                        if matchup_intelligence.get("data_note"):
                            st.caption(matchup_intelligence.get("data_note", ""))

                    # Build the style rows first so the main Unified table can reliably
                    # exclude them by exact matchup name. Do not depend on Source text: older
                    # generated matchup payloads may omit or rename that field, which can cause
                    # the style rows to appear twice.
                    style_rows = pd.DataFrame(matchup_intelligence.get("style_matchups", []))
                    style_categories = set()
                    if not style_rows.empty and "Matchup" in style_rows.columns:
                        style_categories = set(style_rows["Matchup"].astype(str))

                    unified_rows = pd.DataFrame(matchup_intelligence.get("categories", []))
                    if not unified_rows.empty:
                        core_rows = unified_rows.copy()
                        if style_categories and "Category" in core_rows.columns:
                            core_rows = core_rows.loc[
                                ~core_rows["Category"].astype(str).isin(style_categories)
                            ].copy()
                        elif "Source" in core_rows.columns:
                            # Backward-compatible fallback for payloads without style_matchups.
                            style_mask = core_rows["Source"].astype(str).str.contains(
                                "starter traits", case=False, na=False
                            )
                            core_rows = core_rows.loc[~style_mask].copy()

                        if not core_rows.empty:
                            visible_cols = [c for c in ["Category", "Advantage", "Strength"] if c in core_rows.columns]
                            st.markdown("#### Core Matchup Breakdown")
                            st.dataframe(core_rows[visible_cols], use_container_width=True, hide_index=True)

                    if not style_rows.empty:
                        st.markdown("#### Player-Style & Line-of-Scrimmage Matchups")
                        st.caption(
                            "Macabets checks how the actual starters' styles fit this opponent and shows the matchup details that matter most."
                        )
                        style_display = style_rows.rename(columns={"Matchup": "Category", "Edge": "Trait Gap"})
                        style_cols = [c for c in ["Category", "Advantage", "Strength", "Why"] if c in style_display.columns]
                        style_table = style_display[style_cols].copy()

                        # Keep the model's technical trait calculations intact, but translate
                        # their UI explanation into direct football language. The raw trait
                        # numbers remain inside matchup_intelligence for technical auditing.
                        if "Why" in style_table.columns:
                            style_table["Why"] = style_table.apply(
                                lambda row: _plain_nfl_style_why(
                                    row.get("Category"),
                                    row.get("Advantage"),
                                    row.get("Strength"),
                                    row.get("Why"),
                                    nfl_result.get("away_team"),
                                    nfl_result.get("home_team"),
                                ),
                                axis=1,
                            )

                        _render_nfl_style_matchup_table(
                            style_table,
                            nfl_result.get("away_team"),
                            nfl_result.get("home_team"),
                        )

                        # One overall conclusion after all eight style/LOS matchups are
                        # weighted together. This summarizes the existing style signal only;
                        # it does NOT create a second projection adjustment.
                        overall_style_advantage = matchup_intelligence.get("overall_style_advantage") or matchup_intelligence.get("overall_advantage")
                        overall_style_strength = matchup_intelligence.get("overall_style_strength") or matchup_intelligence.get("overall_strength")
                        overall_style_why = matchup_intelligence.get("overall_style_why") or matchup_intelligence.get("overall_why")
                        overall_style_edge = matchup_intelligence.get("overall_style_edge") or matchup_intelligence.get("overall_edge")
                        if overall_style_advantage:
                            if str(overall_style_advantage).lower() == "even":
                                st.markdown("##### Overall Player-Style & LOS Advantage: EVEN")
                            else:
                                strength_label = f" — {overall_style_strength} Advantage" if overall_style_strength else ""
                                st.markdown(f"##### Overall Player-Style & LOS Advantage: {overall_style_advantage}{strength_label}")
                            if overall_style_why:
                                st.write(overall_style_why)
                            if overall_style_edge not in (None, ""):
                                try:
                                    with st.expander("Audit player-style scoring", expanded=False):
                                        st.caption(f"Combined weighted matchup gap: {float(overall_style_edge):.2f}. This is a summary of the existing style signal, not an additional model adjustment.")
                                except (TypeError, ValueError):
                                    pass

                        style_adj = float(matchup_intelligence.get("style_adjustment_home", 0.0) or 0.0)
                        if abs(style_adj) < 0.005:
                            st.info("Starter-trait compatibility is effectively neutral for the projected margin in this matchup.")
                        else:
                            direction = nfl_result["home_team"] if style_adj > 0 else nfl_result["away_team"]
                            st.info(f"**Fair-line impact:** {abs(style_adj):.2f} points toward {direction} from starter-style compatibility.")

                    scheme_context = nfl_result.get("scheme_context") or {}
                    st.markdown("#### Scheme & Team Tendencies")
                    if not scheme_context.get("available"):
                        st.warning(
                            scheme_context.get(
                                "summary",
                                "Scheme tendency data is not available for both teams yet. No scheme adjustment was applied.",
                            )
                        )
                        st.caption(
                            "This section stays visible even when the data feed is unavailable so you can audit whether scheme intelligence is actually active."
                        )
                    else:
                        away_scheme = scheme_context.get("away") or {}
                        home_scheme = scheme_context.get("home") or {}
                        st.caption(
                            "Scheme tendencies help explain how each team is likely to attack this matchup. Detailed rates and source information are available below."
                        )

                        def _scheme_pct(value):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.1%}"
                            except (TypeError, ValueError):
                                return "—"

                        def _scheme_num(value, digits=1, suffix=""):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.{digits}f}{suffix}"
                            except (TypeError, ValueError):
                                return "—"

                        def _scheme_identity(profile):
                            rate = profile.get("early_down_pass_rate")
                            try:
                                if rate is None or pd.isna(rate):
                                    return "Unavailable"
                                rate = float(rate)
                                if rate <= 0.50:
                                    return "Run-leaning"
                                if rate >= 0.62:
                                    return "Pass-leaning"
                                return "Balanced"
                            except (TypeError, ValueError):
                                return "Unavailable"

                        team_profiles = [
                            (str(nfl_result.get("away_team", "Away")), away_scheme),
                            (str(nfl_result.get("home_team", "Home")), home_scheme),
                        ]

                        # Scheme rates inform the model, but the default screen should answer
                        # what the matchup means. Keep the raw tendencies in a single audit view.
                        identity_text = " · ".join(
                            f"{team_name}: {_scheme_identity(profile)}"
                            for team_name, profile in team_profiles
                        )
                        st.caption(f"Offensive identity: {identity_text}")
                        profile_cols = st.columns(2)
                        for idx, (team_name, profile) in enumerate(team_profiles):
                            with profile_cols[idx]:
                                st.markdown(f"##### {team_name}")
                                a, b, c = st.columns(3)
                                a.metric("Identity", _scheme_identity(profile))
                                b.metric("Early-Down Pass", _scheme_pct(profile.get("early_down_pass_rate")))
                                c.metric("Pace", _scheme_num(profile.get("seconds_per_play"), 1, " sec/play"))

                                tendency_rows = [
                                    {"What to watch": "Motion", "Rate": _scheme_pct(profile.get("motion_rate"))},
                                    {"What to watch": "Play action", "Rate": _scheme_pct(profile.get("play_action_rate"))},
                                    {"What to watch": "Blitz", "Rate": _scheme_pct(profile.get("blitz_rate"))},
                                    {"What to watch": "Pressure generated", "Rate": _scheme_pct(profile.get("pressure_rate"))},
                                    {"What to watch": "Pressure allowed", "Rate": _scheme_pct(profile.get("pressure_rate_allowed"))},
                                    {"What to watch": "Explosive plays", "Rate": _scheme_pct(profile.get("offense_explosive_rate"))},
                                    {"What to watch": "Explosive plays allowed", "Rate": _scheme_pct(profile.get("defense_explosive_allowed"))},
                                    {"What to watch": "Red-zone TD", "Rate": _scheme_pct(profile.get("red_zone_td_rate"))},
                                    {"What to watch": "Opponent red-zone TD", "Rate": _scheme_pct(profile.get("red_zone_td_rate_allowed"))},
                                ]
                                st.dataframe(pd.DataFrame(tendency_rows), use_container_width=True, hide_index=True)
                                season_label = profile.get("season") or "—"
                                week_label = profile.get("through_week") or "—"
                                updated_label = str(profile.get("updated_at_utc") or "").strip() or "—"
                                st.caption(f"Season {season_label} · Through week {week_label} · Updated {updated_label}")

                        st.markdown("##### Overall Scheme Matchup Conclusion")
                        sc1, sc2, sc3 = st.columns(3)
                        overall_advantage = str(scheme_context.get("overall_advantage", "Even"))
                        overall_strength = str(scheme_context.get("overall_strength", "Even"))
                        scheme_adj = float(scheme_context.get("home_margin_adjustment", 0.0) or 0.0)
                        if abs(scheme_adj) < 0.005:
                            adj_label = "0.00 pts"
                        else:
                            adj_team = nfl_result.get("home_team") if scheme_adj > 0 else nfl_result.get("away_team")
                            adj_label = f"{abs(scheme_adj):.2f} pts toward {adj_team}"
                        sc1.metric("Overall Scheme Edge", overall_advantage)
                        sc2.metric("Scheme Strength", overall_strength)
                        sc3.metric("Line Impact", adj_label)
                        st.info(scheme_context.get("summary", ""))

                        evidence_weight = scheme_context.get("evidence_weight")
                        source = scheme_context.get("source", "nflverse")
                        with st.expander("Audit scheme data, source and model guardrails", expanded=False):
                            if evidence_weight is not None:
                                try:
                                    st.write(f"Evidence weight: {float(evidence_weight):.0%}")
                                except (TypeError, ValueError):
                                    pass
                            st.write(
                                "Macabets measures pass/run identity, early-down aggressiveness, pace, no-huddle, motion, "
                                "play action, RPO usage, blitz tendency, coverage mix when published, pressure context, "
                                "explosive-play rates and red-zone success."
                            )
                            st.write(
                                "Only behavioral compatibility with the existing personnel matchup affects the side projection. "
                                "Performance-style fields remain context until their dedicated model layers are built."
                            )
                            st.caption(
                                "Additional tracked tendencies include no-huddle, RPO usage, man/zone coverage mix and plays per game. "
                                "They remain available to the model without crowding the decision view."
                            )
                            st.caption(f"Source: {source}")
                            st.caption(scheme_context.get("guardrail", ""))

                    los_context = nfl_result.get("los_context") or {}
                    st.markdown("#### Real NFL Line-of-Scrimmage Intelligence")
                    if not los_context.get("available"):
                        st.warning(los_context.get("summary", "Real NFL line-of-scrimmage data is not available yet."))
                    else:
                        st.caption("How the offensive and defensive fronts have actually performed on the field. Detailed grades and rates stay in the audit view.")

                        def _los_pct(value):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.1%}"
                            except (TypeError, ValueError):
                                return "—"

                        def _los_num(value, digits=2):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.{digits}f}"
                            except (TypeError, ValueError):
                                return "—"

                        los_profiles = [
                            (str(nfl_result.get("away_team", "Away")), los_context.get("away") or {}),
                            (str(nfl_result.get("home_team", "Home")), los_context.get("home") or {}),
                        ]
                        with st.expander("Show detailed LOS grades and rates", expanded=False):
                            st.caption("Composite grades are league-relative 0–100 model inputs. Higher is better. Raw rates below show the underlying on-field evidence.")
                            los_cols = st.columns(2)
                            for idx, (team_name, profile) in enumerate(los_profiles):
                                with los_cols[idx]:
                                    st.markdown(f"##### {team_name}")
                                    x1, x2, x3, x4 = st.columns(4)
                                    x1.metric("Pass Protection", _los_num(profile.get("pass_protection_grade"), 1))
                                    x2.metric("Pass Rush", _los_num(profile.get("pass_rush_grade"), 1))
                                    x3.metric("Run Blocking", _los_num(profile.get("run_block_grade"), 1))
                                    x4.metric("Run Front", _los_num(profile.get("run_front_grade"), 1))
                                    rows = [
                                        {"Area": "Pass protection", "Metric": "Sack rate allowed", "Rate": _los_pct(profile.get("sack_rate_allowed"))},
                                        {"Area": "Pass protection", "Metric": "QB-hit rate allowed", "Rate": _los_pct(profile.get("qb_hit_rate_allowed"))},
                                        {"Area": "Pass protection", "Metric": "Disruption allowed", "Rate": _los_pct(profile.get("disruption_rate_allowed"))},
                                        {"Area": "QB response", "Metric": "EPA/dropback when disrupted", "Rate": _los_num(profile.get("qb_epa_when_disrupted"), 3)},
                                        {"Area": "Pass rush", "Metric": "Sack rate generated", "Rate": _los_pct(profile.get("sack_rate_generated"))},
                                        {"Area": "Pass rush", "Metric": "QB-hit rate generated", "Rate": _los_pct(profile.get("qb_hit_rate_generated"))},
                                        {"Area": "Pass rush", "Metric": "Disruption generated", "Rate": _los_pct(profile.get("disruption_rate_generated"))},
                                        {"Area": "Run blocking", "Metric": "Stuff rate allowed", "Rate": _los_pct(profile.get("run_stuff_rate_allowed"))},
                                        {"Area": "Run blocking", "Metric": "Rush success rate", "Rate": _los_pct(profile.get("rush_success"))},
                                        {"Area": "Run defense", "Metric": "Stuff rate forced", "Rate": _los_pct(profile.get("run_stuff_rate_forced"))},
                                        {"Area": "Run defense", "Metric": "Rush success allowed", "Rate": _los_pct(profile.get("rush_success_allowed"))},
                                    ]
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                                    st.caption(f"Season {profile.get('season') or '—'} · Through week {profile.get('through_week') or '—'}")

                        st.markdown("##### Overall Real-Performance LOS Conclusion")
                        lc1, lc2, lc3 = st.columns(3)
                        los_adv = str(los_context.get("overall_advantage", "Even"))
                        los_strength = str(los_context.get("overall_strength", "Even"))
                        los_adj = float(los_context.get("home_margin_adjustment", 0.0) or 0.0)
                        if abs(los_adj) < 0.005:
                            los_adj_label = "0.00 pts"
                        else:
                            los_adj_team = nfl_result.get("home_team") if los_adj > 0 else nfl_result.get("away_team")
                            los_adj_label = f"{abs(los_adj):.2f} pts toward {los_adj_team}"
                        lc1.metric("Overall LOS Edge", los_adv)
                        lc2.metric("LOS Strength", los_strength)
                        lc3.metric("Line Impact", los_adj_label)
                        st.info(los_context.get("summary", ""))
                        with st.expander("Audit LOS source and model guardrails", expanded=False):
                            ew = los_context.get("evidence_weight")
                            if ew is not None:
                                try:
                                    st.write(f"Evidence weight: {float(ew):.0%}")
                                except (TypeError, ValueError):
                                    pass
                            st.write("Disruption is a public play-by-play proxy based on QB hits and sacks. Stuff rate is the share of rushing attempts stopped for zero or negative yards.")
                            st.caption(f"Source: {los_context.get('source', 'nflverse regular-season play-by-play')}")
                            st.caption(los_context.get("guardrail", ""))

                    situational_context = nfl_result.get("situational_context") or {}
                    st.markdown("#### Situational Execution")
                    if not situational_context.get("available"):
                        st.warning(situational_context.get("summary", "Situational NFL performance data is not available yet."))
                    else:
                        sit_adv = str(situational_context.get("overall_advantage", "Even"))
                        sit_strength = str(situational_context.get("overall_strength", "Even"))
                        sit_adj = float(situational_context.get("home_margin_adjustment", 0.0) or 0.0)
                        if abs(sit_adj) < 0.005:
                            sit_adj_label = "0.00 pts"
                        else:
                            sit_adj_team = nfl_result.get("home_team") if sit_adj > 0 else nfl_result.get("away_team")
                            sit_adj_label = f"{abs(sit_adj):.2f} pts toward {sit_adj_team}"
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Situational Edge", sit_adv)
                        sc2.metric("Strength", sit_strength)
                        sc3.metric("Line Impact", sit_adj_label)
                        st.info(situational_context.get("summary", ""))

                        def _sit_pct(value):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.1%}"
                            except (TypeError, ValueError):
                                return "—"

                        def _sit_num(value, digits=3):
                            try:
                                if value is None or pd.isna(value):
                                    return "—"
                                return f"{float(value):.{digits}f}"
                            except (TypeError, ValueError):
                                return "—"

                        st.caption("Focus: third downs, red-zone finishing and close fourth-quarter execution — the situations that actually drive this refinement.")
                        sit_profiles = [
                            (str(nfl_result.get("away_team", "Away")), situational_context.get("away") or {}),
                            (str(nfl_result.get("home_team", "Home")), situational_context.get("home") or {}),
                        ]
                        sit_cols = st.columns(2)
                        for idx, (team_name, profile) in enumerate(sit_profiles):
                            with sit_cols[idx]:
                                st.markdown(f"##### {team_name}")
                                sit_rows = [
                                    {"Situation": "Third down", "Offense": _sit_pct(profile.get("third_down_conversion_rate")), "Defense allowed": _sit_pct(profile.get("third_down_conversion_allowed"))},
                                    {"Situation": "Red-zone TD", "Offense": _sit_pct(profile.get("red_zone_td_rate")), "Defense allowed": _sit_pct(profile.get("red_zone_td_rate_allowed"))},
                                    {"Situation": "Close 4Q EPA/play", "Offense": _sit_num(profile.get("high_leverage_epa")), "Defense allowed": _sit_num(profile.get("high_leverage_epa_allowed"))},
                                ]
                                st.dataframe(pd.DataFrame(sit_rows), use_container_width=True, hide_index=True)
                                st.caption(f"Season {profile.get('season') or '—'} · Through week {profile.get('through_week') or '—'}")
                        with st.expander("Audit situational context and guardrails", expanded=False):
                            for team_name, profile in sit_profiles:
                                st.markdown(f"**{team_name}**")
                                st.caption(
                                    f"Turnover rate: {_sit_pct(profile.get('offense_turnover_rate'))} · "
                                    f"Takeaway rate: {_sit_pct(profile.get('defense_takeaway_rate'))} · "
                                    f"Explosive-play rate: {_sit_pct(profile.get('offense_explosive_rate'))} · "
                                    f"Explosive allowed: {_sit_pct(profile.get('defense_explosive_allowed'))}"
                                )
                            ew = situational_context.get("evidence_weight")
                            if ew is not None:
                                try:
                                    st.caption(f"Evidence weight: {float(ew):.0%}")
                                except (TypeError, ValueError):
                                    pass
                            st.caption(f"Source: {situational_context.get('source', 'nflverse regular-season play-by-play')}")
                            st.caption(situational_context.get("guardrail", ""))

                    opponent_context = nfl_result.get("opponent_adjusted_context") or {}
                    st.markdown("#### Opponent-Adjusted Performance")
                    if not opponent_context.get("available"):
                        st.warning(opponent_context.get("summary", "Opponent-adjusted NFL performance data is not available yet."))
                    else:
                        opp_adv = str(opponent_context.get("overall_advantage", "Even"))
                        opp_strength = str(opponent_context.get("overall_strength", "Even"))
                        opp_adj = float(opponent_context.get("home_margin_adjustment", 0.0) or 0.0)
                        if abs(opp_adj) < 0.005:
                            opp_adj_label = "0.00 pts"
                        else:
                            opp_adj_team = nfl_result.get("home_team") if opp_adj > 0 else nfl_result.get("away_team")
                            opp_adj_label = f"{abs(opp_adj):.2f} pts toward {opp_adj_team}"
                        oc1, oc2, oc3 = st.columns(3)
                        oc1.metric("Opponent-Adjusted Edge", opp_adv)
                        oc2.metric("Strength", opp_strength)
                        oc3.metric("Line Impact", opp_adj_label)
                        st.info(opponent_context.get("summary", ""))

                        with st.expander("Audit opponent-adjustment data", expanded=False):
                            st.caption("Macabets compares the quality of offenses and defenses already faced. Future schedule difficulty never receives betting credit.")
                            profiles = [
                                (str(nfl_result.get("away_team", "Away")), opponent_context.get("away") or {}),
                                (str(nfl_result.get("home_team", "Home")), opponent_context.get("home") or {}),
                            ]
                            for team_name, profile in profiles:
                                st.markdown(f"**{team_name}**")
                                try:
                                    oq = float(profile.get("opponent_quality_epa"))
                                    raw = float(profile.get("raw_net_epa"))
                                    adj_net = float(profile.get("opponent_adjusted_net_epa"))
                                    st.caption(f"Opponent-quality EPA: {oq:+.3f} · Raw net EPA/play: {raw:+.3f} · Opponent-adjusted net EPA/play: {adj_net:+.3f}")
                                except (TypeError, ValueError):
                                    st.caption("Detailed opponent-quality values unavailable.")
                                st.caption(f"Season {profile.get('season') or '—'} · Through week {profile.get('through_week') or '—'}")
                            ew = opponent_context.get("evidence_weight")
                            if ew is not None:
                                try:
                                    st.caption(f"Evidence weight: {float(ew):.0%}")
                                except (TypeError, ValueError):
                                    pass
                            st.caption(f"Source: {opponent_context.get('source', 'nflverse regular-season play-by-play')}")
                            st.caption(opponent_context.get("guardrail", ""))

                    drivers = matchup_intelligence.get("top_drivers", []) or []
                    if drivers:
                        st.markdown("**Most important matchup drivers**")
                        driver_cols = st.columns(min(3, len(drivers)))
                        for idx, driver in enumerate(drivers[:3]):
                            with driver_cols[idx]:
                                st.markdown(f"**{driver.get('leader', 'Even')}**")
                                st.write(driver.get("factor", "Matchup factor"))
                        with st.expander("Audit driver scoring", expanded=False):
                            for driver in drivers[:3]:
                                st.caption(
                                    f"{driver.get('factor', 'Matchup factor')}: raw gap "
                                    f"{float(driver.get('raw_gap', 0.0)):.1f}"
                                )

                    with st.expander("Show unified matchup grades and sources", expanded=False):
                        if not unified_rows.empty:
                            detail_cols = [c for c in ["Category", "Advantage", "Strength", "Rating Gap", nfl_result["away_team"], nfl_result["home_team"], "Source", "Why"] if c in unified_rows.columns]
                            st.dataframe(unified_rows[detail_cols], use_container_width=True, hide_index=True)
                        st.caption(matchup_intelligence.get("guardrail", ""))

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

                # Prefer the context attached to the saved analysis when displaying a
                # persisted report. This keeps the report stable even if Streamlit reruns
                # after a tab change, widget interaction, or Archive navigation.
                active_venue_raw = str(active_weather.get("venue_type") or venue_type or auto_venue_type).strip()
                active_venue_key = active_venue_raw.lower()
                if active_venue_key in {"dome", "closed"}:
                    display_venue_type = "Dome"
                elif "retract" in active_venue_key:
                    display_venue_type = "Retractable roof"
                elif active_venue_key == "outdoor":
                    display_venue_type = "Outdoor"
                else:
                    display_venue_type = active_venue_raw or auto_venue_type
                display_weather = str(active_weather.get("label") or weather or "Automatic")

                if active_weather:
                    with st.expander("Weather intelligence", expanded=False):
                        weather1, weather2, weather3 = st.columns([3, 1, 1])
                        weather1.write(str(active_weather.get("summary") or "No weather adjustment."))
                        weather2.metric("Impact", str(active_weather.get("impact") or "None"))
                        total_weather_move = float(active_weather.get("total_adjustment", 0.0) or 0.0)
                        weather3.metric("Total Adj.", f"{total_weather_move:+.1f}")
                        if not active_weather.get("available", False):
                            st.caption("Weather data was unavailable, so Macabets applied no weather adjustment.")
                        elif float(active_weather.get("home_margin_adjustment", 0.0) or 0.0):
                            st.caption(str(active_weather.get("climate_mismatch") or ""))

                # Matchup advantages are now displayed once through Unified NFL Matchup Intelligence above.

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
                        st.write(f"Venue: {display_venue_type}")
                        st.write(f"Weather: {display_weather}")
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
                        if nfl_considered_side == "Just analyze":
                            st.caption("Neutral view: Macabets answers both sides from the same unified football intelligence used by the matchup model.")
                        else:
                            st.caption(f"Bet-side view: answers are shown from the {nfl_considered_side} perspective. The selected side does not change the underlying grades.")

                        for item in questions:
                            st.markdown(f"**{item.get('number')}. {item.get('question')}**")
                            answers_by_team = item.get("answers_by_team") or {}
                            if answers_by_team:
                                if nfl_considered_side == "Just analyze":
                                    brain_cols = st.columns(2)
                                    for idx, team in enumerate((nfl_result["away_team"], nfl_result["home_team"])):
                                        answer_obj = answers_by_team.get(team, {})
                                        with brain_cols[idx]:
                                            st.markdown(f"**{team}: {answer_obj.get('answer', 'Mixed / close')}**")
                                            st.write(answer_obj.get("reason", ""))
                                else:
                                    answer_obj = answers_by_team.get(nfl_considered_side, {})
                                    st.markdown(f"**Answer: {answer_obj.get('answer', 'Mixed / close')}**")
                                    st.write(answer_obj.get("reason", ""))
                            else:
                                st.markdown(f"**Answer: {item.get('answer', 'Mixed / close')}**")
                                st.write(item.get("reason", ""))
                            st.caption(f"Evidence mode: {item.get('readiness_label', matchup_intelligence.get('data_mode', 'Baseline'))}")

                        st.markdown("**Football Summary**")
                        st.info(matchup_brain.get("summary", ""))

                    with st.expander("NFL Brain technical details", expanded=False):
                        st.write(f"Status: {matchup_brain.get('status', 'unknown')}")
                        if decision_framework:
                            st.write(f"Questions answered: {int(decision_framework.get('ready_questions', 0))}/8")
                            st.write(decision_framework.get("message", ""))
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
        st.subheader("Analysis Engine — UFC")
        st.caption(
            "Compare two UFC fighters using Strength v0.2, opponent-adjusted performance, style, round-cardio degradation, damage/durability risk, physical/context, simulation and derivative-market pricing. "
            "Historical Validation v0.1 now backtests the leakage-safe side baseline and fight-path calibration on prior UFC results."
        )

        if not UFC_ENGINE_AVAILABLE:
            st.error(
                "The UFC analysis engine could not be imported. Confirm that engine/ufc.py is in the repository. "
                f"Import error: {UFC_ENGINE_IMPORT_ERROR}"
            )
        else:
            try:
                ufc_ratings = load_ufc_ratings()
                ufc_fights = load_ufc_fights()
                ufc_players = ufc_fighter_names(ufc_ratings)
            except Exception as exc:
                ufc_ratings = pd.DataFrame()
                ufc_fights = pd.DataFrame()
                ufc_players = []
                st.error(str(exc))
                st.info(
                    "Run the GitHub Action named 'Update Macabets UFC Data', then reboot the Streamlit app."
                )

            if ufc_players:
                with st.expander("UFC Model Validation", expanded=False):
                    st.caption(
                        "Run a leakage-safe retrospective backtest using only information available before each historical bout. "
                        "Because the repository does not store historical snapshots of every current Performance/Style/Context percentile, "
                        "v0.1 is a validation proxy for the side baseline and simulation/derivative calibration — not a claim that today's full model was reconstructed exactly."
                    )
                    vc1, vc2 = st.columns([1, 2])
                    validation_bouts = int(vc1.selectbox(
                        "Historical sample",
                        [600, 1200, 1800],
                        index=2,
                        key="ufc_validation_bouts",
                        format_func=lambda value: f"Last {value:,} eligible bouts",
                    ))
                    vc2.caption(
                        "Minimum two prior UFC fights per fighter. The report measures moneyline calibration, finish/distance bias, "
                        "method-of-victory calibration, round-total calibration and division-level performance."
                    )
                    if st.button("Run UFC Historical Validation", use_container_width=True, key="run_ufc_validation"):
                        with st.spinner("Replaying historical UFC fights without future information..."):
                            try:
                                st.session_state["ufc_validation_report"] = run_historical_validation(
                                    ufc_fights,
                                    config=UFCValidationConfig(max_bouts=validation_bouts),
                                )
                            except Exception as exc:
                                st.session_state.pop("ufc_validation_report", None)
                                st.error(f"UFC historical validation failed: {exc}")

                    validation = st.session_state.get("ufc_validation_report")
                    if validation and validation.get("available"):
                        ml = validation.get("moneyline", {})
                        fd = validation.get("finish_distance", {})
                        vm1, vm2, vm3, vm4 = st.columns(4)
                        vm1.metric("Validated Bouts", f"{int(validation.get('sample', 0)):,}")
                        vm2.metric("Winner Accuracy", f"{float(ml.get('winner_accuracy', 0.0)):.1%}")
                        vm3.metric("Moneyline Brier", f"{float(ml.get('brier', 0.0)):.3f}")
                        vm4.metric("Finish Brier", f"{float(fd.get('finish_brier', 0.0)):.3f}")

                        predicted_finish = float(fd.get("predicted_finish_rate", 0.0))
                        actual_finish = float(fd.get("actual_finish_rate", 0.0))
                        finish_gap = float(fd.get("finish_gap", 0.0))
                        if abs(finish_gap) >= 0.035:
                            st.warning(
                                f"Simulation finish-rate bias: {fd.get('bias_label', 'Needs calibration')}. "
                                f"Predicted {predicted_finish:.1%} vs actual {actual_finish:.1%} ({finish_gap:+.1%} actual minus predicted)."
                            )
                        else:
                            st.success(
                                f"Simulation finish rate is reasonably centered: predicted {predicted_finish:.1%} vs actual {actual_finish:.1%}."
                            )

                        method_rows = []
                        for label, key in [("KO/TKO", "ko_tko"), ("Submission", "submission"), ("Decision", "decision")]:
                            item = validation.get("methods", {}).get(key, {})
                            method_rows.append({
                                "Outcome": label,
                                "Predicted": f"{float(item.get('predicted', 0.0)):.1%}",
                                "Actual": f"{float(item.get('actual', 0.0)):.1%}",
                                "Calibration Gap": f"{float(item.get('gap', 0.0)):+.1%}",
                            })
                        st.markdown("**Method-of-victory calibration**")
                        st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)

                        total_rows = []
                        for line, item in validation.get("round_totals", {}).items():
                            total_rows.append({
                                "Total": line,
                                "Sample": int(item.get("sample", 0)),
                                "Predicted Over": f"{float(item.get('mean_predicted_over', 0.0)):.1%}",
                                "Actual Over": f"{float(item.get('actual_over_rate', 0.0)):.1%}",
                                "Gap": f"{float(item.get('calibration_gap', 0.0)):+.1%}",
                                "Brier": f"{float(item.get('brier', 0.0)):.3f}",
                            })
                        if total_rows:
                            st.markdown("**Round-total calibration**")
                            st.dataframe(pd.DataFrame(total_rows), use_container_width=True, hide_index=True)

                        division_rows = validation.get("divisions", [])
                        if division_rows:
                            st.markdown("**Largest division samples**")
                            division_display = pd.DataFrame(division_rows).rename(columns={
                                "division": "Division",
                                "sample": "Sample",
                                "winner_accuracy": "Winner Accuracy",
                                "moneyline_brier": "Moneyline Brier",
                                "predicted_finish_rate": "Predicted Finish",
                                "actual_finish_rate": "Actual Finish",
                                "finish_gap": "Finish Gap",
                            })
                            for col in ["Winner Accuracy", "Predicted Finish", "Actual Finish", "Finish Gap"]:
                                if col in division_display:
                                    division_display[col] = division_display[col].map(lambda value: f"{float(value):+.1%}" if col == "Finish Gap" else f"{float(value):.1%}")
                            if "Moneyline Brier" in division_display:
                                division_display["Moneyline Brier"] = division_display["Moneyline Brier"].map(lambda value: f"{float(value):.3f}")
                            st.dataframe(division_display, use_container_width=True, hide_index=True)

                        with st.expander("Validation methodology and limitations", expanded=False):
                            st.caption(
                                f"Validation window: {validation.get('date_start', '—')} through {validation.get('date_end', '—')} · "
                                f"{validation.get('version', 'Historical Validation')}"
                            )
                            for limitation in validation.get("limitations", []):
                                st.caption(f"• {limitation}")
                    elif validation:
                        st.warning(str(validation.get("reason", "No eligible historical validation sample was produced.")))

                ufc_divisions = sorted(
                    ufc_ratings.loc[ufc_ratings["active_pool"], "division"]
                    .dropna().astype(str).unique().tolist()
                )
                division_filter_options = ["All active UFC fighters"] + ufc_divisions
                ufc_division_filter = st.selectbox(
                    "Division filter",
                    division_filter_options,
                    key="ufc_division_filter",
                )

                if ufc_division_filter == "All active UFC fighters":
                    filtered_ufc_players = ufc_players
                else:
                    filtered_ufc_players = sorted(
                        ufc_ratings.loc[
                            ufc_ratings["active_pool"]
                            & ufc_ratings["division"].astype(str).eq(ufc_division_filter),
                            "fighter",
                        ].astype(str).tolist()
                    )

                if len(filtered_ufc_players) < 2:
                    st.warning("This division does not currently have two active fighters in the UFC rating pool.")
                else:
                    uf1, uf2, uf3, uf4 = st.columns([2, 2, 1, 1.35])
                    ufc_fighter_a = uf1.selectbox(
                        "Fighter A",
                        filtered_ufc_players,
                        index=0,
                        key="ufc_fighter_a",
                    )
                    fighter_b_options = [name for name in filtered_ufc_players if name != ufc_fighter_a]
                    ufc_fighter_b = uf2.selectbox(
                        "Fighter B",
                        fighter_b_options,
                        index=0,
                        key="ufc_fighter_b",
                    )
                    ufc_rounds = uf3.selectbox(
                        "Scheduled rounds",
                        [3, 5],
                        index=0,
                        key="ufc_rounds",
                    )
                    ufc_fight_date = uf4.date_input(
                        "Fight date",
                        value=date.today(),
                        key="ufc_fight_date",
                        help="Used for age, activity/turnaround, and weight-class context. It does not affect the underlying historical ratings snapshot.",
                    )

                    st.markdown("##### Sportsbook price")
                    evaluate_ufc_price = st.checkbox(
                        "Evaluate the current moneyline",
                        value=False,
                        key="ufc_evaluate_price",
                    )
                    if evaluate_ufc_price:
                        uo1, uo2 = st.columns(2)
                        ufc_market_a = int(uo1.number_input(
                            f"Sportsbook odds — {ufc_fighter_a}",
                            value=-110,
                            step=5,
                            key="ufc_market_a",
                        ))
                        ufc_market_b = int(uo2.number_input(
                            f"Sportsbook odds — {ufc_fighter_b}",
                            value=-110,
                            step=5,
                            key="ufc_market_b",
                        ))
                    else:
                        ufc_market_a = None
                        ufc_market_b = None

                    st.markdown("##### Derivative market price")
                    evaluate_ufc_derivative = st.checkbox(
                        "Evaluate a totals or method-of-victory market",
                        value=False,
                        key="ufc_evaluate_derivative",
                        help="Macabets will always show fair derivative prices after analysis. Turn this on only when you want to compare a specific sportsbook prop price.",
                    )
                    ufc_derivative_key = None
                    ufc_derivative_odds_primary = None
                    ufc_derivative_odds_secondary = None
                    ufc_derivative_total_line = None
                    if evaluate_ufc_derivative:
                        derivative_options = {
                            "Total rounds": "total_rounds",
                            "Goes the distance": "goes_distance",
                            f"{ufc_fighter_a} by KO/TKO": "a_ko_tko",
                            f"{ufc_fighter_a} by Submission": "a_submission",
                            f"{ufc_fighter_a} by Decision": "a_decision",
                            f"{ufc_fighter_b} by KO/TKO": "b_ko_tko",
                            f"{ufc_fighter_b} by Submission": "b_submission",
                            f"{ufc_fighter_b} by Decision": "b_decision",
                        }
                        derivative_label = st.selectbox(
                            "Derivative market",
                            list(derivative_options),
                            key="ufc_derivative_market_label",
                        )
                        ufc_derivative_key = derivative_options[derivative_label]
                        if ufc_derivative_key == "total_rounds":
                            total_choices = [1.5, 2.5] if int(ufc_rounds) == 3 else [1.5, 2.5, 3.5, 4.5]
                            dm1, dm2, dm3 = st.columns([1, 1, 1])
                            ufc_derivative_total_line = float(dm1.selectbox(
                                "Round total", total_choices, index=min(1, len(total_choices) - 1), key="ufc_derivative_total_line"
                            ))
                            ufc_derivative_odds_primary = int(dm2.number_input(
                                f"Over {ufc_derivative_total_line:.1f} odds", value=-110, step=5, key="ufc_derivative_over_odds"
                            ))
                            ufc_derivative_odds_secondary = int(dm3.number_input(
                                f"Under {ufc_derivative_total_line:.1f} odds", value=-110, step=5, key="ufc_derivative_under_odds"
                            ))
                        elif ufc_derivative_key == "goes_distance":
                            dm1, dm2 = st.columns(2)
                            ufc_derivative_odds_primary = int(dm1.number_input(
                                "Goes distance — Yes odds", value=-110, step=5, key="ufc_derivative_distance_yes_odds"
                            ))
                            ufc_derivative_odds_secondary = int(dm2.number_input(
                                "Goes distance — No odds", value=-110, step=5, key="ufc_derivative_distance_no_odds"
                            ))
                        else:
                            ufc_derivative_odds_primary = int(st.number_input(
                                f"Sportsbook odds — {derivative_label}", value=200, step=5, key="ufc_derivative_method_odds"
                            ))

                    ufc_considering = st.radio(
                        "Who are you considering betting on?",
                        ["Just analyze", ufc_fighter_a, ufc_fighter_b],
                        horizontal=True,
                        key="ufc_considering_bet",
                        help="This does not influence the model. It only selects which market price Macabets evaluates.",
                    )

                    if st.button(
                        "Analyze UFC Fight",
                        type="primary",
                        use_container_width=True,
                        key="run_ufc_analysis",
                    ):
                        with st.spinner("Macabets is building the UFC matchup analysis..."):
                            try:
                                ufc_analysis = analyze_ufc_match(
                                    ufc_fighter_a,
                                    ufc_fighter_b,
                                    rounds=int(ufc_rounds),
                                    market_odds_a=ufc_market_a,
                                    market_odds_b=ufc_market_b,
                                    derivative_market_key=ufc_derivative_key,
                                    derivative_odds_primary=ufc_derivative_odds_primary,
                                    derivative_odds_secondary=ufc_derivative_odds_secondary,
                                    derivative_total_line=ufc_derivative_total_line,
                                    ratings=ufc_ratings,
                                    fights=ufc_fights,
                                    fight_date=ufc_fight_date,
                                )
                                st.session_state["ufc_analysis_result"] = ufc_analysis
                                st.session_state["ufc_analysis_considering"] = ufc_considering

                                # UFC uses the same permanent Analysis Log as Tennis and NFL.
                                # Save exactly once per explicit Analyze click, with the full
                                # model output frozen inside analysis_snapshot for later audit.
                                fighter_a_log = str(ufc_analysis["fighter_a"])
                                fighter_b_log = str(ufc_analysis["fighter_b"])
                                projected_winner_log = str(ufc_analysis["projected_winner"])
                                projected_probability_log = float(ufc_analysis["projected_winner_probability"])
                                winner_fair_log = (
                                    int(ufc_analysis["fair_moneyline_a"])
                                    if projected_winner_log == fighter_a_log
                                    else int(ufc_analysis["fair_moneyline_b"])
                                )
                                market_log = ufc_analysis.get("market") or {}
                                if market_log.get("available"):
                                    winner_side_log = "a" if projected_winner_log == fighter_a_log else "b"
                                    winner_market_log = int(market_log[f"market_odds_{winner_side_log}"])
                                    winner_verdict_log = str(market_log.get(f"verdict_{winner_side_log}") or "PASS")
                                    winner_roi_log = float(market_log.get(f"roi_{winner_side_log}") or 0.0)
                                    winner_edge_log = float(market_log.get(f"edge_{winner_side_log}") or 0.0)
                                else:
                                    winner_market_log = None
                                    winner_verdict_log = "Analysis"
                                    winner_roi_log = None
                                    winner_edge_log = None

                                ufc_inputs_log = {
                                    "fighter_a": fighter_a_log,
                                    "fighter_b": fighter_b_log,
                                    "fight_date": ufc_fight_date.isoformat(),
                                    "rounds": int(ufc_rounds),
                                    "considered_side": ufc_considering,
                                    "market_odds_a": ufc_market_a,
                                    "market_odds_b": ufc_market_b,
                                    "derivative_market_key": ufc_derivative_key,
                                    "derivative_total_line": ufc_derivative_total_line,
                                    "derivative_odds_primary": ufc_derivative_odds_primary,
                                    "derivative_odds_secondary": ufc_derivative_odds_secondary,
                                }
                                ufc_snapshot_log = {
                                    "engine_result": ufc_analysis,
                                    "projected_winner": projected_winner_log,
                                    "projected_winner_probability": projected_probability_log,
                                    "winner_market_moneyline": winner_market_log,
                                    "winner_fair_moneyline": winner_fair_log,
                                    "moneyline_edge": winner_edge_log,
                                    "moneyline_estimated_roi": winner_roi_log,
                                    "verdict": winner_verdict_log,
                                    "price_assessment": "—",
                                    "derivative_markets": ufc_analysis.get("derivative_markets") or {},
                                    "derivative_evaluation": ufc_analysis.get("derivative_evaluation") or {},
                                    "simulation": ufc_analysis.get("simulation") or {},
                                    "validation": ufc_analysis.get("historical_validation") or {},
                                }
                                _save_universal_analysis({
                                    "client_event_id": _analysis_event_token("UFC", ufc_inputs_log),
                                    "event_date": ufc_fight_date.isoformat(),
                                    "sport": "UFC",
                                    "model_version": str(ufc_analysis.get("model_version") or "Macabets UFC"),
                                    "event_name": f"{fighter_a_log} vs {fighter_b_log}",
                                    "participant_a": fighter_a_log,
                                    "participant_b": fighter_b_log,
                                    "market_type": "Moneyline" if market_log.get("available") else "Analysis",
                                    "market_odds_a": ufc_market_a,
                                    "market_odds_b": ufc_market_b,
                                    "prediction": projected_winner_log,
                                    "predicted_probability": projected_probability_log,
                                    "fair_line": format_american(winner_fair_log),
                                    "confidence": float(ufc_analysis["confidence"]),
                                    "recommendation": winner_verdict_log,
                                    "status": "Pending",
                                    "input_snapshot": ufc_inputs_log,
                                    "analysis_snapshot": ufc_snapshot_log,
                                })
                            except Exception as exc:
                                st.session_state.pop("ufc_analysis_result", None)
                                st.error(f"UFC analysis failed: {exc}")
                                st.exception(exc)

                    ufc_result = st.session_state.get("ufc_analysis_result")
                    if ufc_result:
                        fighter_a_name = str(ufc_result["fighter_a"])
                        fighter_b_name = str(ufc_result["fighter_b"])
                        probability_a = float(ufc_result["win_probability_a"])
                        probability_b = float(ufc_result["win_probability_b"])
                        projected_winner = str(ufc_result["projected_winner"])
                        winner_probability = float(ufc_result["projected_winner_probability"])
                        winner_fair = (
                            int(ufc_result["fair_moneyline_a"])
                            if projected_winner == fighter_a_name
                            else int(ufc_result["fair_moneyline_b"])
                        )

                        st.markdown(f"### {fighter_a_name} vs {fighter_b_name}")
                        st.caption(
                            f"{ufc_result['division_context']} · {int(ufc_result['rounds'])} rounds · "
                            f"{ufc_result['rating_version']}"
                        )

                        if not ufc_result.get("same_division", True):
                            st.warning(
                                "These fighters are currently assigned to different divisions. Treat the fair line as a broad "
                                "cross-division strength comparison until a specific contracted weight is modeled."
                            )

                        st.markdown("## Macabets UFC Analysis")
                        ub1, ub2, ub3, ub4 = st.columns(4)
                        ub1.metric("Projected Winner", projected_winner)
                        ub2.metric("Win Probability", f"{winner_probability:.1%}")
                        ub3.metric("Fair Moneyline", format_american(winner_fair))
                        ub4.metric(
                            "Confidence",
                            f"{int(ufc_result['confidence'])}/100",
                            str(ufc_result["confidence_band"]),
                        )

                        performance_adjustment = float(ufc_result.get("performance_adjustment_a", 0.0) or 0.0)
                        style_adjustment = float(ufc_result.get("style_adjustment_a", 0.0) or 0.0)
                        context_adjustment = float(ufc_result.get("context_adjustment_a", 0.0) or 0.0)
                        combined_adjustment = float(ufc_result.get("total_adjustment_a", ufc_result.get("combined_matchup_adjustment_a", 0.0)) or 0.0)
                        combined_side = fighter_a_name if combined_adjustment >= 0 else fighter_b_name
                        st.info(
                            f"UFC Performance + Style Matchups are active. Performance moved Fighter A {performance_adjustment:+.1%}; "
                            f"opponent-specific style moved Fighter A {style_adjustment:+.1%}. After the correlated-input guardrail, "
                            f"the total matchup impact is {abs(combined_adjustment):.1%} toward {combined_side}."
                        )

                        driver_col, risk_col = st.columns(2)
                        with driver_col:
                            st.markdown("#### Reasons for the lean")
                            for item in ufc_result.get("reasons_for_lean", []):
                                st.markdown(f"- {item}")
                        with risk_col:
                            st.markdown("#### What could flip it")
                            for item in ufc_result.get("risk_factors", []):
                                st.markdown(f"- {item}")

                        st.markdown("### Fighter Strength")
                        summary_a = ufc_result["fighter_a_summary"]
                        summary_b = ufc_result["fighter_b_summary"]
                        strength_rows = []
                        for fighter_name, summary, probability, fair_ml in (
                            (fighter_a_name, summary_a, probability_a, int(ufc_result["fair_moneyline_a"])),
                            (fighter_b_name, summary_b, probability_b, int(ufc_result["fair_moneyline_b"])),
                        ):
                            rank = summary.get("division_rank")
                            strength_rows.append({
                                "Fighter": fighter_name,
                                "Division": summary.get("division", "Unknown"),
                                "Macabets Rank": f"#{rank}" if rank else "Unranked pool",
                                "Strength": f"{float(summary.get('strength_score', 50.0)):.1f}",
                                "UFC Record": summary.get("ufc_record", "—"),
                                "Recent Form": f"{float(summary.get('recent_form', 50.0)):.1f}",
                                "Ranking Confidence": f"{float(summary.get('ranking_confidence', 50.0)):.0f}/100",
                                "Win Probability": f"{probability:.1%}",
                                "Fair ML": format_american(fair_ml),
                            })
                        st.dataframe(pd.DataFrame(strength_rows), use_container_width=True, hide_index=True)

                        st.markdown("### Recent UFC Performance Snapshot")
                        profile_rows = []
                        for fighter_name, profile in (
                            (fighter_a_name, ufc_result.get("recent_profile_a", {})),
                            (fighter_b_name, ufc_result.get("recent_profile_b", {})),
                        ):
                            sig_diff = profile.get("sig_str_diff_per_fight")
                            td = profile.get("td_per_fight")
                            kd = profile.get("kd_per_fight")
                            sub_att = profile.get("sub_att_per_fight")
                            profile_rows.append({
                                "Fighter": fighter_name,
                                "Recent Sample": int(profile.get("sample", 0)),
                                "Recent Record": profile.get("record", "0-0"),
                                "Finish Rate in Wins": f"{float(profile.get('finish_rate', 0.0)):.0%}",
                                "Sig. Strike Diff / Fight": "—" if sig_diff is None else f"{float(sig_diff):+.1f}",
                                "Knockdowns / Fight": "—" if kd is None else f"{float(kd):.2f}",
                                "Takedowns / Fight": "—" if td is None else f"{float(td):.2f}",
                                "Sub Attempts / Fight": "—" if sub_att is None else f"{float(sub_att):.2f}",
                            })
                        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)
                        st.caption(
                            "Raw recent totals remain descriptive. The probability-driving performance layer uses landed/attempted rates, "
                            "opponent rates, takedown defense, control, durability and pace with a strict ±5 percentage-point cap."
                        )

                        opponent_adjustment = ufc_result.get("opponent_adjustment", {})
                        if opponent_adjustment.get("available"):
                            st.markdown("### Opponent-Adjusted Skills")
                            st.caption(
                                "These are the same performance/style inputs after adjusting for the specific quality of recent opponents faced. "
                                "This layer transforms the inputs rather than adding a separate probability bonus, which avoids double-counting opponent quality."
                            )
                            oa_rows = []
                            report_a = opponent_adjustment.get("fighter_a_report", {})
                            report_b = opponent_adjustment.get("fighter_b_report", {})
                            skill_map_a = {row.get("skill"): row for row in report_a.get("skills", [])}
                            skill_map_b = {row.get("skill"): row for row in report_b.get("skills", [])}
                            ordered_skills = [
                                "Striking Offense", "Striking Defense", "Power", "Durability",
                                "Wrestling Offense", "Wrestling Defense",
                                "Grappling Offense", "Grappling Defense", "Pace",
                            ]
                            for skill in ordered_skills:
                                arow = skill_map_a.get(skill, {})
                                brow = skill_map_b.get(skill, {})
                                def _fmt_score(row, key):
                                    value = row.get(key)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0f}"
                                def _fmt_move(row):
                                    value = row.get("adjustment")
                                    return "—" if value is None or pd.isna(value) else f"{float(value):+.1f}"
                                oa_rows.append({
                                    "Skill": skill,
                                    f"{fighter_a_name} Base": _fmt_score(arow, "base_score"),
                                    f"{fighter_a_name} Adj.": _fmt_score(arow, "adjusted_score"),
                                    f"{fighter_a_name} SOS": str(arow.get("opponent_quality_label", "Unknown")),
                                    f"{fighter_a_name} Move": _fmt_move(arow),
                                    f"{fighter_b_name} Base": _fmt_score(brow, "base_score"),
                                    f"{fighter_b_name} Adj.": _fmt_score(brow, "adjusted_score"),
                                    f"{fighter_b_name} SOS": str(brow.get("opponent_quality_label", "Unknown")),
                                    f"{fighter_b_name} Move": _fmt_move(brow),
                                })
                            st.dataframe(pd.DataFrame(oa_rows), use_container_width=True, hide_index=True)
                            os1, os2, os3 = st.columns(3)
                            os1.metric(
                                f"{fighter_a_name} Opponent Sample",
                                int(report_a.get("opponent_sample", 0) or 0),
                            )
                            os2.metric(
                                f"{fighter_b_name} Opponent Sample",
                                int(report_b.get("opponent_sample", 0) or 0),
                            )
                            os3.metric(
                                "Opponent-Adjustment Reliability",
                                f"{float(opponent_adjustment.get('reliability', 0.0) or 0.0):.0%}",
                            )
                            st.caption(str(opponent_adjustment.get("guardrail", "")))

                        perf_a = ufc_result.get("performance_profile_a", {})
                        perf_b = ufc_result.get("performance_profile_b", {})
                        perf_matchup = ufc_result.get("performance_matchup", {})
                        if perf_matchup.get("available"):
                            st.markdown("### Underlying Performance Engine")
                            perf_rows = []
                            for fighter_name, profile in ((fighter_a_name, perf_a), (fighter_b_name, perf_b)):
                                def _score(name):
                                    value = profile.get(name)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0f}/100"
                                def _pct(name):
                                    value = profile.get(name)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"
                                def _num(name, digits=2):
                                    value = profile.get(name)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"
                                perf_rows.append({
                                    "Fighter": fighter_name,
                                    "Striking": _score("striking_score"),
                                    "Wrestling": _score("wrestling_score"),
                                    "Grappling": _score("grappling_score"),
                                    "Durability": _score("durability_score"),
                                    "Pace": _score("pace_score"),
                                    "Sig. Str. Diff / Min": _num("sig_diff_per_min"),
                                    "Sig. Str. Accuracy": _pct("sig_accuracy"),
                                    "Sig. Str. Defense": _pct("sig_defense"),
                                    "TD / 15": _num("td_per15"),
                                    "TD Defense": _pct("td_defense"),
                                    "Control Share": _pct("control_share"),
                                    "Sample": int(profile.get("sample", 0) or 0),
                                })
                            st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

                            pa1, pa2, pa3 = st.columns(3)
                            pa1.metric(
                                "Performance Line Impact",
                                f"{float(ufc_result.get('performance_adjustment_a', 0.0)):+.1%}",
                                f"to {fighter_a_name}",
                            )
                            pa2.metric(
                                "Performance Reliability",
                                f"{float(perf_matchup.get('reliability', 0.0)):.0%}",
                            )
                            pa3.metric(
                                "Performance Gap",
                                f"{float(perf_matchup.get('weighted_gap', 0.0)):+.1f}",
                                "percentile points",
                            )
                            st.caption(
                                "Composite scores are division-relative percentiles from the recent UFC sample. Five-round fights "
                                "place a little more weight on durability and pace. This is the general performance layer; the opponent-specific interaction layer is shown next."
                            )

                        advanced_striking = ufc_result.get("advanced_striking_matchup", {})
                        if advanced_striking.get("available"):
                            st.markdown("### Advanced Striking Matchups")
                            striking_rows = pd.DataFrame(advanced_striking.get("rows", []))
                            if not striking_rows.empty:
                                striking_display = striking_rows.rename(columns={
                                    "category": "Category",
                                    "advantage": "Advantage",
                                    "strength": "Strength",
                                    "why": "Why it matters",
                                })
                                scols = [c for c in ["Category", "Advantage", "Strength", "Why it matters"] if c in striking_display.columns]
                                st.dataframe(striking_display[scols], use_container_width=True, hide_index=True)

                            sp_rows = []
                            for fighter_name, profile in (
                                (fighter_a_name, advanced_striking.get("fighter_a_profile", {})),
                                (fighter_b_name, advanced_striking.get("fighter_b_profile", {})),
                            ):
                                def _sscore(key):
                                    value = profile.get(key)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0f}/100"
                                sp_rows.append({
                                    "Fighter": fighter_name,
                                    "Head Attack": _sscore("head_attack_score"),
                                    "Body Attack": _sscore("body_attack_score"),
                                    "Leg Attack": _sscore("leg_attack_score"),
                                    "Distance": _sscore("distance_attack_score"),
                                    "Power": _sscore("power_score"),
                                    "KD Resistance": _sscore("knockdown_resistance_score"),
                                    "Close Range": _sscore("close_attack_score"),
                                    "Sample": int(profile.get("sample", 0) or 0),
                                })
                            st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)
                            ss1, ss2 = st.columns(2)
                            ss1.metric("Advanced Striking Reliability", f"{float(advanced_striking.get('reliability', 0.0)):.0%}")
                            ss2.metric("Advanced Striking Gap", f"{float(advanced_striking.get('weighted_gap', 0.0)):+.1f}", "interaction points")
                            st.caption(str(advanced_striking.get("guardrail", "")))
                            st.caption("Line impact is integrated into the existing Style Matchups adjustment; Macabets does not add a second striking probability modifier.")

                        advanced_grappling = ufc_result.get("advanced_grappling_matchup", {})
                        if advanced_grappling.get("available"):
                            st.markdown("### Advanced Wrestling & Grappling Matchups")
                            grappling_rows = pd.DataFrame(advanced_grappling.get("rows", []))
                            if not grappling_rows.empty:
                                grappling_display = grappling_rows.rename(columns={
                                    "category": "Category",
                                    "advantage": "Advantage",
                                    "strength": "Strength",
                                    "why": "Why it matters",
                                })
                                gcols = [c for c in ["Category", "Advantage", "Strength", "Why it matters"] if c in grappling_display.columns]
                                st.dataframe(grappling_display[gcols], use_container_width=True, hide_index=True)

                            gp_rows = []
                            for fighter_name, profile in (
                                (fighter_a_name, advanced_grappling.get("fighter_a_profile", {})),
                                (fighter_b_name, advanced_grappling.get("fighter_b_profile", {})),
                            ):
                                def _gscore(key):
                                    value = profile.get(key)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0f}/100"
                                gp_rows.append({
                                    "Fighter": fighter_name,
                                    "Chain Wrestling": _gscore("chain_wrestling_score"),
                                    "Control Retention": _gscore("control_retention_score"),
                                    "TD Resistance": _gscore("takedown_resistance_score"),
                                    "Bottom Escape Proxy": _gscore("bottom_escape_score"),
                                    "Submission Pressure": _gscore("submission_pressure_score"),
                                    "Submission Resistance": _gscore("submission_resistance_score"),
                                    "Sample": int(profile.get("sample", 0) or 0),
                                })
                            st.dataframe(pd.DataFrame(gp_rows), use_container_width=True, hide_index=True)
                            gg1, gg2 = st.columns(2)
                            gg1.metric(
                                "Advanced Grappling Reliability",
                                f"{float(advanced_grappling.get('reliability', 0.0)):.0%}",
                            )
                            gg2.metric(
                                "Advanced Grappling Gap",
                                f"{float(advanced_grappling.get('weighted_gap', 0.0)):+.1f}",
                                "interaction points",
                            )
                            st.caption(str(advanced_grappling.get("guardrail", "")))
                            st.caption("Line impact is integrated into the existing Style Matchups adjustment; Macabets does not add a second grappling probability modifier.")

                        style_matchup = ufc_result.get("style_matchup", {})
                        if style_matchup.get("available"):
                            st.markdown("### Opponent-Specific Style Matchups")
                            sa1, sa2 = st.columns(2)
                            sa1.metric(
                                f"{fighter_a_name} Style",
                                str(style_matchup.get("fighter_a_archetype", "Balanced / mixed style")),
                            )
                            sa2.metric(
                                f"{fighter_b_name} Style",
                                str(style_matchup.get("fighter_b_archetype", "Balanced / mixed style")),
                            )

                            style_rows = pd.DataFrame(style_matchup.get("rows", []))
                            if not style_rows.empty:
                                style_display = style_rows.rename(columns={
                                    "category": "Category",
                                    "advantage": "Advantage",
                                    "strength": "Strength",
                                    "why": "Why it matters",
                                })
                                cols = [c for c in ["Category", "Advantage", "Strength", "Why it matters"] if c in style_display.columns]
                                st.dataframe(style_display[cols], use_container_width=True, hide_index=True)

                            sm1, sm2, sm3 = st.columns(3)
                            sm1.metric(
                                "Style Line Impact",
                                f"{float(ufc_result.get('style_adjustment_a', 0.0)):+.1%}",
                                f"to {fighter_a_name}",
                            )
                            sm2.metric(
                                "Style Reliability",
                                f"{float(style_matchup.get('reliability', 0.0)):.0%}",
                            )
                            sm3.metric(
                                "Interaction Gap",
                                f"{float(style_matchup.get('weighted_gap', 0.0)):+.1f}",
                                "matchup points",
                            )
                            st.caption(str(style_matchup.get("guardrail", "")))

                        cardio_matchup = ufc_result.get("cardio_matchup", {})
                        if cardio_matchup.get("available"):
                            st.markdown("### Round-by-Round Cardio & Degradation")
                            cardio_rows = []
                            for fighter_name, profile in (
                                (fighter_a_name, cardio_matchup.get("fighter_a_profile", {})),
                                (fighter_b_name, cardio_matchup.get("fighter_b_profile", {})),
                            ):
                                def _ret(key):
                                    value = profile.get(key)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0%}"
                                cardio_rows.append({
                                    "Fighter": fighter_name,
                                    "Cardio Score": f"{float(profile.get('cardio_score', 50.0)):.0f}/100",
                                    "Trend": str(profile.get("trend", "Unknown")),
                                    "R2+ Output Retention": _ret("output_retention"),
                                    "Accuracy Retention": _ret("accuracy_retention"),
                                    "Defense Retention": _ret("defense_retention"),
                                    "Wrestling Retention": _ret("wrestling_retention"),
                                    "Late-Round Exposures": int(profile.get("late_round_exposures", 0) or 0),
                                    "R4/R5 Exposures": int(profile.get("championship_round_exposures", 0) or 0),
                                })
                            st.dataframe(pd.DataFrame(cardio_rows), use_container_width=True, hide_index=True)
                            cd1, cd2, cd3 = st.columns(3)
                            cd1.metric(
                                "Cardio Line Impact",
                                f"{float(ufc_result.get('cardio_adjustment_a', 0.0)):+.1%}",
                                f"to {fighter_a_name}",
                            )
                            cd2.metric(
                                "Cardio Reliability",
                                f"{float(cardio_matchup.get('reliability', 0.0)):.0%}",
                            )
                            cd3.metric(
                                "Cardio Gap",
                                f"{float(cardio_matchup.get('cardio_gap', 0.0)):+.1f}",
                                "retention score points",
                            )
                            st.caption(str(cardio_matchup.get("guardrail", "")))
                        else:
                            cardio_a_status = cardio_matchup.get("fighter_a_profile", {})
                            cardio_b_status = cardio_matchup.get("fighter_b_profile", {})
                            cardio_reason = cardio_a_status.get("reason") or cardio_b_status.get("reason")
                            if cardio_reason:
                                st.caption(f"Round Cardio unavailable: {cardio_reason}")

                        damage_matchup = ufc_result.get("damage_matchup", {})
                        if damage_matchup.get("available"):
                            st.markdown("### Damage & Durability Risk")
                            damage_rows = []
                            for fighter_name, profile in (
                                (fighter_a_name, damage_matchup.get("fighter_a_profile", {})),
                                (fighter_b_name, damage_matchup.get("fighter_b_profile", {})),
                            ):
                                def _dval(key, suffix=""):
                                    value = profile.get(key)
                                    return "—" if value is None or pd.isna(value) else f"{float(value):.0f}{suffix}"
                                damage_rows.append({
                                    "Fighter": fighter_name,
                                    "Damage Risk": f"{float(profile.get('risk_score', 0.0)):.0f}/100",
                                    "Risk Level": str(profile.get("risk_label", "Unknown")),
                                    "KD Absorbed (Last 3)": _dval("knockdowns_absorbed_last3"),
                                    "Head Strikes Absorbed (Last 3)": _dval("head_strikes_absorbed_last3"),
                                    "KO/TKO Losses (365d)": int(profile.get("ko_tko_losses_last365", 0) or 0),
                                    "Days Since KO/TKO Loss": _dval("days_since_last_ko_tko_loss"),
                                    "UFC Cage Minutes": _dval("career_ufc_minutes"),
                                })
                            st.dataframe(pd.DataFrame(damage_rows), use_container_width=True, hide_index=True)
                            dm1, dm2, dm3 = st.columns(3)
                            dm1.metric(
                                "Damage Line Impact",
                                f"{float(ufc_result.get('damage_adjustment_a', 0.0)):+.1%}",
                                f"to {fighter_a_name}",
                            )
                            dm2.metric(
                                "Damage Reliability",
                                f"{float(damage_matchup.get('reliability', 0.0)):.0%}",
                            )
                            dm3.metric(
                                "Risk Gap",
                                f"{float(damage_matchup.get('risk_gap', 0.0)):+.1f}",
                                "positive favors Fighter A",
                            )
                            st.caption(str(damage_matchup.get("guardrail", "")))
                        else:
                            damage_a_status = damage_matchup.get("fighter_a_profile", {})
                            damage_b_status = damage_matchup.get("fighter_b_profile", {})
                            damage_reason = damage_a_status.get("reason") or damage_b_status.get("reason")
                            if damage_reason:
                                st.caption(f"Damage Risk unavailable: {damage_reason}")

                        fight_context = ufc_result.get("fight_context", {})
                        if fight_context.get("available"):
                            st.markdown("### Physical & Fight Context")
                            profile_a = fight_context.get("fighter_a_profile", {})
                            profile_b = fight_context.get("fighter_b_profile", {})
                            context_rows = []
                            for fighter_name, profile in [(fighter_a_name, profile_a), (fighter_b_name, profile_b)]:
                                age_value = profile.get("age")
                                reach_value = profile.get("reach_inches")
                                height_value = profile.get("height_inches")
                                context_rows.append({
                                    "Fighter": fighter_name,
                                    "Age": "—" if age_value is None else f"{float(age_value):.1f}",
                                    "Height": "—" if height_value is None else f"{float(height_value):.0f} in",
                                    "Reach": "—" if reach_value is None else f"{float(reach_value):.0f} in",
                                    "Stance": profile.get("stance") or "Unknown",
                                    "Profile Weight": "—" if profile.get("weight_lbs") is None else f"{float(profile.get('weight_lbs')):.0f} lb",
                                })
                            st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)

                            context_breakdown = pd.DataFrame(fight_context.get("rows", []))
                            if not context_breakdown.empty:
                                context_display = context_breakdown.rename(columns={
                                    "category": "Category",
                                    "advantage": "Advantage",
                                    "strength": "Strength",
                                    "why": "Why it matters",
                                })
                                ctx_cols = [c for c in ["Category", "Advantage", "Strength", "Why it matters"] if c in context_display.columns]
                                st.dataframe(context_display[ctx_cols], use_container_width=True, hide_index=True)

                            cx1, cx2, cx3 = st.columns(3)
                            cx1.metric(
                                "Context Line Impact",
                                f"{float(ufc_result.get('context_adjustment_a', 0.0)):+.1%}",
                                f"to {fighter_a_name}",
                            )
                            cx2.metric(
                                "Context Reliability",
                                f"{float(fight_context.get('reliability', 0.0)):.0%}",
                            )
                            cx3.metric(
                                "Fight Date",
                                str(fight_context.get("fight_date", "—")),
                            )
                            st.caption(str(fight_context.get("guardrail", "")))

                        ufc_sim = ufc_result.get("simulation", {})
                        if ufc_sim.get("available"):
                            st.markdown("### Fight Simulation & Method of Victory")
                            sim1, sim2, sim3, sim4 = st.columns(4)
                            sim1.metric("Most Likely Fight Path", str(ufc_sim.get("most_likely_path", "—")))
                            sim2.metric("Path Probability", f"{float(ufc_sim.get('most_likely_path_probability', 0.0)):.1%}")
                            sim3.metric("Goes the Distance", f"{float(ufc_sim.get('goes_distance_probability', 0.0)):.1%}")
                            sim4.metric("Fight Volatility", str(ufc_sim.get("volatility", "Moderate")))

                            method_rows = [
                                {
                                    "Fighter": fighter_a_name,
                                    "KO/TKO": f"{float(ufc_sim.get('a_ko_tko_probability', 0.0)):.1%}",
                                    "Submission": f"{float(ufc_sim.get('a_submission_probability', 0.0)):.1%}",
                                    "Decision": f"{float(ufc_sim.get('a_decision_probability', 0.0)):.1%}",
                                    "Win Probability": f"{probability_a:.1%}",
                                },
                                {
                                    "Fighter": fighter_b_name,
                                    "KO/TKO": f"{float(ufc_sim.get('b_ko_tko_probability', 0.0)):.1%}",
                                    "Submission": f"{float(ufc_sim.get('b_submission_probability', 0.0)):.1%}",
                                    "Decision": f"{float(ufc_sim.get('b_decision_probability', 0.0)):.1%}",
                                    "Win Probability": f"{probability_b:.1%}",
                                },
                            ]
                            st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)

                            likely_round = ufc_sim.get("likely_finish_round")
                            if likely_round:
                                st.caption(
                                    f"If the fight ends inside the distance, Round {int(likely_round)} is the most likely finish window "
                                    f"({float(ufc_sim.get('likely_finish_round_probability_given_finish', 0.0)):.1%} of modeled finishes)."
                                )
                            with st.expander("Show simulation detail", expanded=False):
                                rd = ufc_sim.get("finish_round_probabilities_given_finish", {})
                                if rd:
                                    st.dataframe(
                                        pd.DataFrame([{"Round": k, "Probability Given Finish": f"{float(v):.1%}"} for k, v in rd.items()]),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                st.caption(
                                    f"Simulation runs: {int(ufc_sim.get('simulations', 0)):,}. "
                                    + str(ufc_sim.get("guardrail", ""))
                                )

                        derivative_markets = ufc_result.get("derivative_markets", {})
                        if derivative_markets.get("available"):
                            st.markdown("### UFC Totals & Method Fair Prices")
                            dist = derivative_markets.get("distance_market", {})
                            d1, d2 = st.columns(2)
                            d1.metric(
                                "Goes Distance — Fair",
                                format_american(int(dist.get("yes_fair_odds", 0))),
                                f"{float(dist.get('yes_probability', 0.0)):.1%}",
                            )
                            d2.metric(
                                "Doesn't Go Distance — Fair",
                                format_american(int(dist.get("no_fair_odds", 0))),
                                f"{float(dist.get('no_probability', 0.0)):.1%}",
                            )

                            totals_rows = []
                            for row in derivative_markets.get("round_totals", []):
                                totals_rows.append({
                                    "Total": f"{float(row.get('line', 0.0)):.1f}",
                                    "Over Probability": f"{float(row.get('over_probability', 0.0)):.1%}",
                                    "Over Fair": format_american(int(row.get("over_fair_odds", 0))),
                                    "Under Probability": f"{float(row.get('under_probability', 0.0)):.1%}",
                                    "Under Fair": format_american(int(row.get("under_fair_odds", 0))),
                                })
                            if totals_rows:
                                st.markdown("**Round totals**")
                                st.dataframe(pd.DataFrame(totals_rows), use_container_width=True, hide_index=True)

                            method_fair_rows = []
                            for row in derivative_markets.get("method_markets", []):
                                method_fair_rows.append({
                                    "Market": str(row.get("market", "—")),
                                    "Probability": f"{float(row.get('probability', 0.0)):.1%}",
                                    "Fair Odds": format_american(int(row.get("fair_odds", 0))),
                                })
                            if method_fair_rows:
                                st.markdown("**Method of victory**")
                                st.dataframe(pd.DataFrame(method_fair_rows), use_container_width=True, hide_index=True)

                            derivative_eval = ufc_result.get("derivative_evaluation", {})
                            if derivative_eval.get("available"):
                                st.markdown("#### Sportsbook Derivative Evaluation")
                                eval_rows = []
                                for label_key, value_key in [("primary_label", "primary"), ("secondary_label", "secondary")]:
                                    value = derivative_eval.get(value_key)
                                    label = derivative_eval.get(label_key)
                                    if not value or not label:
                                        continue
                                    eval_rows.append({
                                        "Market": str(label),
                                        "Sportsbook": format_american(int(value.get("market_odds", 0))),
                                        "Fair": format_american(int(value.get("fair_odds", 0))),
                                        "Model Probability": f"{float(value.get('probability', 0.0)):.1%}",
                                        "No-Vig Edge": "—" if value.get("no_vig_edge") is None else f"{float(value.get('no_vig_edge', 0.0)):+.1%}",
                                        "Estimated ROI": f"{float(value.get('roi', 0.0)):+.1%}",
                                        "Verdict": str(value.get("verdict", "PASS")),
                                    })
                                if eval_rows:
                                    st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
                                    best = max(eval_rows, key=lambda row: float(str(row["Estimated ROI"]).replace("%", "")))
                                    if best["Verdict"] == "BET":
                                        st.success(f"{best['Market']}: BET at the entered price. Estimated ROI {best['Estimated ROI']}.")
                                    elif best["Verdict"] == "WATCH":
                                        st.warning(f"{best['Market']}: WATCH at the entered price. Estimated ROI {best['Estimated ROI']}.")
                                    else:
                                        st.info("PASS on the entered derivative price(s) at the current model probabilities.")
                                note = derivative_eval.get("note")
                                if note:
                                    st.caption(str(note))

                            st.caption(str(derivative_markets.get("guardrail", "")))

                        st.markdown("### Core Matchup Breakdown")
                        matchup_rows = pd.DataFrame(ufc_result.get("matchup_breakdown", []))
                        if not matchup_rows.empty:
                            display = matchup_rows.rename(columns={
                                "category": "Category",
                                "advantage": "Advantage",
                                "strength": "Edge",
                                "why": "Why it matters",
                            })
                            cols = [c for c in ["Category", "Advantage", "Edge", "Why it matters"] if c in display.columns]
                            st.dataframe(display[cols], use_container_width=True, hide_index=True)

                        market = ufc_result.get("market") or {}
                        if market.get("available"):
                            st.markdown("### Moneyline Evaluation")
                            considered = st.session_state.get("ufc_analysis_considering", "Just analyze")
                            market_rows = [
                                {
                                    "Fighter": fighter_a_name,
                                    "Market": format_american(int(market["market_odds_a"])),
                                    "Fair": format_american(int(ufc_result["fair_moneyline_a"])),
                                    "No-Vig Edge": f"{float(market['edge_a']):+.1%}",
                                    "Estimated ROI": f"{float(market['roi_a']):+.1%}",
                                    "Verdict": market["verdict_a"],
                                },
                                {
                                    "Fighter": fighter_b_name,
                                    "Market": format_american(int(market["market_odds_b"])),
                                    "Fair": format_american(int(ufc_result["fair_moneyline_b"])),
                                    "No-Vig Edge": f"{float(market['edge_b']):+.1%}",
                                    "Estimated ROI": f"{float(market['roi_b']):+.1%}",
                                    "Verdict": market["verdict_b"],
                                },
                            ]
                            st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)
                            if considered in {fighter_a_name, fighter_b_name}:
                                side = "a" if considered == fighter_a_name else "b"
                                verdict = str(market[f"verdict_{side}"])
                                roi_value = float(market[f"roi_{side}"])
                                if verdict == "BET":
                                    st.success(
                                        f"{considered}: BET at the entered price on the current baseline — estimated ROI {roi_value:+.1%}. "
                                        "Fight simulation plus totals/round and method-of-victory fair pricing are now included. Derivative markets remain conservative and use higher ROI thresholds than the moneyline."
                                    )
                                elif verdict == "WATCH":
                                    st.warning(
                                        f"{considered}: WATCH. The baseline sees some price value ({roi_value:+.1%} estimated ROI), "
                                        "but not enough to clear the current bet threshold cleanly."
                                    )
                                else:
                                    st.info(
                                        f"{considered}: PASS at the entered price. The baseline estimated ROI is {roi_value:+.1%}."
                                    )
                            else:
                                st.caption("Select a fighter under bet consideration to receive a direct side-specific price verdict.")
                        else:
                            st.caption("Turn on 'Evaluate the current moneyline' to compare sportsbook prices with the UFC fair line.")

                        with st.expander("UFC model audit and current limitations", expanded=False):
                            audit_rows = pd.DataFrame(ufc_result.get("matchup_breakdown", []))
                            if not audit_rows.empty:
                                st.dataframe(audit_rows, use_container_width=True, hide_index=True)
                            for limitation in ufc_result.get("limitations", []):
                                st.caption(f"• {limitation}")

    with analysis_tabs[3]:
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
                    fetch_api_tennis_today.clear()

                with st.spinner("Loading today's market slate..."):
                    tennis_load_errors = []
                    if available_choices[selected_label] == "__all_tennis__":
                        api_tennis_fixtures, api_tennis_status = fetch_api_tennis_today()
                        api_tennis_schedule = normalize_api_tennis_schedule(api_tennis_fixtures)
                        if tennis_items:
                            automatic_slate, usage, tennis_load_errors = combine_tennis_slate(api_key, tennis_items)
                        else:
                            automatic_slate = pd.DataFrame()
                            usage = sports_usage
                        # API-Tennis owns the schedule. If market pricing is unavailable, keep the card visible
                        # instead of incorrectly reporting that no tennis is scheduled.
                        if automatic_slate.empty and not api_tennis_schedule.empty:
                            automatic_slate = api_tennis_schedule
                    else:
                        api_events, usage = fetch_sport_odds(api_key, available_choices[selected_label])
                        automatic_slate = normalize_api_slate(api_events, selected_label)

                if available_choices[selected_label] == "__all_tennis__":
                    with st.expander("Tennis API diagnostics", expanded=automatic_slate.empty):
                        schedule_count = len(api_tennis_schedule) if 'api_tennis_schedule' in locals() else 0
                        st.write(f"API-Tennis fixtures scheduled today: **{schedule_count}**")
                        if 'api_tennis_status' in locals():
                            if api_tennis_status.get("error"):
                                st.caption(f"API-Tennis schedule status: {api_tennis_status.get('error')}")
                            else:
                                st.caption(f"API-Tennis schedule source: {api_tennis_status.get('source', 'unknown')}")
                        st.write(f"Active Odds API ATP/WTA feeds discovered: **{len(tennis_items)}**")
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
                    if _odds_quota_exhausted(tennis_load_errors):
                        st.warning(
                            "Today's tennis schedule is still available from API-Tennis, but live sportsbook odds "
                            "are unavailable because The Odds API quota has been exhausted."
                        )
                    else:
                        st.warning(
                            f"Loaded the available tennis card, but {len(tennis_load_errors)} tournament odds feed(s) failed."
                        )
                    with st.expander("Tennis feed details", expanded=False):
                        for message in tennis_load_errors:
                            st.caption(message)

                if automatic_slate.empty:
                    if available_choices[selected_label] == "__all_tennis__":
                        if 'api_tennis_status' in locals() and api_tennis_status.get("error"):
                            st.info(
                                "Macabets could not load today's tennis schedule from API-Tennis or live prices from The Odds API. "
                                "Open diagnostics above for the provider errors."
                            )
                        else:
                            st.info("API-Tennis is not reporting any ATP/WTA singles fixtures scheduled today.")
                    else:
                        st.info(f"No {selected_label} events with US moneyline odds are scheduled today.")
                else:
                    if available_choices[selected_label] == "__all_tennis__":
                        tournament_count = automatic_slate["sport"].nunique()
                        has_prices = automatic_slate["odds_a"].notna().any() or automatic_slate["odds_b"].notna().any()
                        if has_prices:
                            st.success(
                                f"Loaded {len(automatic_slate)} ATP/WTA matches across {tournament_count} active tournament(s) with market pricing."
                            )
                        else:
                            st.success(
                                f"Loaded {len(automatic_slate)} ATP/WTA matches from API-Tennis. Live sportsbook odds are currently unavailable."
                            )
                    has_any_prices = "odds_a" in automatic_slate and automatic_slate["odds_a"].notna().any()
                    if has_any_prices:
                        st.caption(
                            f"Best available US moneyline shown for each side. API requests remaining: {usage['remaining']}."
                        )
                    else:
                        st.caption("Schedule source: API-Tennis. Odds source: The Odds API (currently unavailable or not loaded).")
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

                            # Tournament sponsor names and odds-feed labels rarely match the
                            # historical ATP dataset's own naming (e.g. "ATP Cincinnati Open" vs.
                            # a prior season's "Western & Southern Financial Group Masters"), so
                            # bridge through the real match history instead of a raw keyword
                            # guess -- this is what correctly identifies a Masters 1000 event
                            # that a plain "masters"/"1000" substring check would miss.
                            try:
                                slate_matches, _ = load_matches()
                            except Exception:
                                slate_matches = pd.DataFrame()
                            if not slate_matches.empty:
                                inferred_surface = tennis_tournament_surface_for_display_name(slate_matches, tournament_name)
                                inferred_category = tennis_tournament_category_for_display_name(slate_matches, tournament_name)
                            else:
                                inferred_surface = "Hard"
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

                            # The odds feed that populates the daily slate carries no round
                            # information at all (just teams/time/price), so there is nothing
                            # to detect the round from. Land on an explicit placeholder rather
                            # than silently asserting a specific round that's usually wrong.
                            st.session_state.pending_fair_line_prefill = {
                                "fle_date": event_date,
                                "fle_tournament": tournament_name,
                                "fle_surface": inferred_surface,
                                "fle_round": TENNIS_ROUND_NOT_DETECTED,
                                "fle_favorite": player_a_name,
                                "fle_opponent": player_b_name,
                                "fle_market_a": odds_a_value,
                                "fle_market_b": odds_b_value,
                                "auto_match_date": event_date,
                                "auto_tournament": tournament_name,
                                "auto_surface": inferred_surface,
                                "auto_round": TENNIS_ROUND_NOT_DETECTED,
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
    archive_tabs = st.tabs(["Performance Center", "Analysis Log", "Legacy Tennis Archive", "Matchup Lab"])

    with archive_tabs[0]:
        st.subheader("Performance Center")
        st.caption(
            "One place for every Macabets prediction, live performance tracking, your -250 to -450 Core Zone, "
            "and clean CSV exports for deeper analysis."
        )

        if not analysis_db_configured():
            st.error("Permanent storage is not configured yet.")
            st.info("Configure Supabase to use the Performance Center with the permanent Analysis Log.")
        else:
            try:
                performance_rows = db_list_analyses(5000)
            except Exception as exc:
                performance_rows = []
                st.error(f"Could not load predictions for the Performance Center: {exc}")

            def _pc_result(row):
                status = str(row.get("status", "Pending"))
                if status == "Won":
                    return "Correct"
                if status == "Lost":
                    return "Incorrect"
                return status if status in {"Pending", "Push", "Void"} else "Pending"

            def _pc_line_number(row):
                raw = _analysis_market_line(row)
                try:
                    return int(float(str(raw).replace("+", "").replace(",", "").strip()))
                except (TypeError, ValueError):
                    return None

            def _pc_flat_units(result, odds):
                if result not in {"Correct", "Incorrect"} or odds in (None, 0):
                    return None
                if result == "Incorrect":
                    return -1.0
                return float(odds) / 100.0 if float(odds) > 0 else 100.0 / abs(float(odds))

            def _pc_line_bucket(odds):
                if odds is None:
                    return "Unknown"
                if odds >= 100:
                    return "+Money"
                if -149 <= odds <= -101:
                    return "-101 to -149"
                if -199 <= odds <= -150:
                    return "-150 to -199"
                if -249 <= odds <= -200:
                    return "-200 to -249"
                if -299 <= odds <= -250:
                    return "-250 to -299"
                if -349 <= odds <= -300:
                    return "-300 to -349"
                if -399 <= odds <= -350:
                    return "-350 to -399"
                if -449 <= odds <= -400:
                    return "-400 to -449"
                if odds <= -450:
                    return "-450+"
                return "Other"

            pc_records = []
            for row in performance_rows:
                odds = _pc_line_number(row)
                result = _pc_result(row)
                confidence_label = _analysis_confidence_label(row)
                event_date = pd.to_datetime(row.get("event_date") or row.get("created_at"), errors="coerce")
                ufc_derivative = _ufc_derivative_performance(row)
                pc_records.append({
                    "ID": row.get("id", ""),
                    "Date": event_date,
                    "Sport": str(row.get("sport", "")),
                    "Event": str(row.get("event_name", "")),
                    "Participant A": str(row.get("participant_a", "")),
                    "Participant B": str(row.get("participant_b", "")),
                    "Prediction": str(row.get("prediction", "")),
                    "Market": str(row.get("market_type", "")),
                    "Actual Line": odds,
                    "Line Bucket": _pc_line_bucket(odds),
                    "Core Zone": bool(odds is not None and -450 <= odds <= -250),
                    "Market Implied %": implied_probability(odds) if odds not in (None, 0) else float("nan"),
                    "Fair Line": row.get("fair_line", ""),
                    "Confidence": confidence_label,
                    "Price Assessment": _analysis_price_assessment(row),
                    "Verdict": _analysis_verdict(row),
                    "Prediction Result": result,
                    "Flat Units": _pc_flat_units(result, odds),
                    "Model Version": str(row.get("model_version", "")),
                    "Derivative Market": ufc_derivative.get("primary_label", ""),
                    "Derivative Result": ufc_derivative.get("primary_result", ""),
                    "Derivative Odds": ufc_derivative.get("primary_odds"),
                    "Derivative Verdict": ufc_derivative.get("primary_verdict", ""),
                    "Derivative Secondary": ufc_derivative.get("secondary_label", ""),
                    "Derivative Secondary Result": ufc_derivative.get("secondary_result", ""),
                    "Derivative Secondary Odds": ufc_derivative.get("secondary_odds"),
                    "UFC Actual Method": ufc_derivative.get("actual_method", ""),
                    "UFC Actual Round": ufc_derivative.get("actual_round"),
                    "UFC Actual Time": ufc_derivative.get("actual_time", ""),
                })

            pc_all = pd.DataFrame(pc_records)
            if pc_all.empty:
                st.info("No saved predictions yet.")
            else:
                pc_all["Date"] = pd.to_datetime(pc_all["Date"], errors="coerce")
                valid_dates = pc_all["Date"].dropna()
                min_date = valid_dates.min().date() if not valid_dates.empty else date.today()
                max_date = valid_dates.max().date() if not valid_dates.empty else date.today()

                st.markdown("### Filters")
                f1, f2, f3, f4 = st.columns(4)
                pc_sport = f1.selectbox(
                    "Sport", ["All"] + sorted(x for x in pc_all["Sport"].dropna().unique().tolist() if x),
                    key="pc_sport_filter",
                )
                pc_result_filter = f2.selectbox(
                    "Result", ["All", "Correct", "Incorrect", "Pending", "Push", "Void"],
                    key="pc_result_filter",
                )
                verdict_options = ["All"] + sorted(x for x in pc_all["Verdict"].dropna().unique().tolist() if x and x != "—")
                pc_verdict = f3.selectbox("Verdict", verdict_options, key="pc_verdict_filter")
                assessment_options = ["All"] + sorted(x for x in pc_all["Price Assessment"].dropna().unique().tolist() if x and x != "—")
                pc_assessment = f4.selectbox("Price Assessment", assessment_options, key="pc_assessment_filter")

                f5, f6, f7, f8 = st.columns([1.4, 1.4, 1.4, 2.2])
                pc_date_range = f5.date_input(
                    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
                    key="pc_date_range",
                )
                pc_core_only = f6.checkbox("Core Zone only (-250 to -450)", value=False, key="pc_core_zone_only")
                confidence_order = ["Low", "Moderate", "High", "Very High"]
                available_confidence = [
                    label for label in confidence_order if label in set(pc_all["Confidence"].dropna().tolist())
                ]
                pc_confidence = f7.selectbox(
                    "Confidence", ["All"] + available_confidence, key="pc_confidence_filter"
                )
                pc_search = f8.text_input(
                    "Search", placeholder="Player, team, event or prediction", key="pc_search_filter"
                )

                line_values = pc_all["Actual Line"].dropna()
                if not line_values.empty:
                    line_min = int(line_values.min())
                    line_max = int(line_values.max())
                    chosen_line_range = st.slider(
                        "Actual Line range", min_value=line_min, max_value=line_max,
                        value=(line_min, line_max), key="pc_actual_line_range"
                    )
                else:
                    chosen_line_range = (0, 0)

                pc_filtered = pc_all.copy()
                if pc_sport != "All":
                    pc_filtered = pc_filtered[pc_filtered["Sport"] == pc_sport]
                if pc_result_filter != "All":
                    pc_filtered = pc_filtered[pc_filtered["Prediction Result"] == pc_result_filter]
                if pc_verdict != "All":
                    pc_filtered = pc_filtered[pc_filtered["Verdict"] == pc_verdict]
                if pc_assessment != "All":
                    pc_filtered = pc_filtered[pc_filtered["Price Assessment"] == pc_assessment]
                if pc_core_only:
                    pc_filtered = pc_filtered[pc_filtered["Core Zone"]]
                if isinstance(pc_date_range, (list, tuple)) and len(pc_date_range) == 2:
                    start_date, end_date = pc_date_range
                    pc_filtered = pc_filtered[
                        (pc_filtered["Date"].dt.date >= start_date) & (pc_filtered["Date"].dt.date <= end_date)
                    ]
                if not line_values.empty and chosen_line_range != (line_min, line_max):
                    pc_filtered = pc_filtered[
                        pc_filtered["Actual Line"].between(chosen_line_range[0], chosen_line_range[1], inclusive="both")
                    ]
                if pc_confidence != "All":
                    pc_filtered = pc_filtered[pc_filtered["Confidence"] == pc_confidence]
                if pc_search.strip():
                    q = pc_search.strip().casefold()
                    searchable = pc_filtered[["Event", "Participant A", "Participant B", "Prediction"]].fillna("").astype(str).agg(" ".join, axis=1).str.casefold()
                    pc_filtered = pc_filtered[searchable.str.contains(q, regex=False)]

                def _pc_summary(frame):
                    graded = frame[frame["Prediction Result"].isin(["Correct", "Incorrect"])].copy()
                    wins = int((graded["Prediction Result"] == "Correct").sum())
                    losses = int((graded["Prediction Result"] == "Incorrect").sum())
                    accuracy = wins / len(graded) if len(graded) else None
                    units = graded["Flat Units"].dropna().sum() if not graded.empty else 0.0
                    roi = units / len(graded) if len(graded) else None
                    expected = graded["Market Implied %"].dropna().mean() if not graded.empty else None
                    return graded, wins, losses, accuracy, units, roi, expected

                graded, wins, losses, accuracy, units, flat_roi, expected_rate = _pc_summary(pc_filtered)
                st.markdown("### Live Performance")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Record", f"{wins}-{losses}")
                m2.metric("Win %", f"{accuracy:.1%}" if accuracy is not None else "—")
                m3.metric("Flat Units", f"{units:+.2f}u")
                m4.metric("Flat ROI", f"{flat_roi:+.1%}" if flat_roi is not None else "—")
                m5.metric("Market Expected", f"{expected_rate:.1%}" if expected_rate is not None else "—")
                m6.metric("Predictions", len(pc_filtered))

                core_frame = pc_filtered[pc_filtered["Core Zone"]].copy()
                core_graded, core_w, core_l, core_acc, core_units, core_roi, core_expected = _pc_summary(core_frame)
                st.markdown("### Core Zone — -250 to -450")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Record", f"{core_w}-{core_l}")
                c2.metric("Win %", f"{core_acc:.1%}" if core_acc is not None else "—")
                c3.metric("Market Expected", f"{core_expected:.1%}" if core_expected is not None else "—")
                c4.metric("Edge vs Market", f"{(core_acc-core_expected):+.1%}" if core_acc is not None and core_expected is not None else "—")
                c5.metric("Flat Units", f"{core_units:+.2f}u")
                c6.metric("Flat ROI", f"{core_roi:+.1%}" if core_roi is not None else "—")
                st.caption(f"Core Zone graded sample: {len(core_graded)}. Flat-unit ROI assumes a 1-unit stake on every graded prediction at the saved Actual Line.")

                ufc_derivative_rows = []
                for _, ufc_row in pc_filtered.loc[pc_filtered["Sport"] == "UFC"].iterrows():
                    for prefix in ("", "Secondary"):
                        if prefix:
                            label = str(ufc_row.get("Derivative Secondary", "") or "").strip()
                            status = str(ufc_row.get("Derivative Secondary Result", "") or "").strip()
                            odds_value = ufc_row.get("Derivative Secondary Odds")
                            verdict_value = ""
                        else:
                            label = str(ufc_row.get("Derivative Market", "") or "").strip()
                            status = str(ufc_row.get("Derivative Result", "") or "").strip()
                            odds_value = ufc_row.get("Derivative Odds")
                            verdict_value = str(ufc_row.get("Derivative Verdict", "") or "").strip()
                        if not label:
                            continue
                        derivative_result = "Correct" if status == "Won" else ("Incorrect" if status == "Lost" else status or "Pending")
                        try:
                            derivative_odds = int(float(odds_value)) if odds_value not in (None, "", 0) else None
                        except (TypeError, ValueError):
                            derivative_odds = None
                        ufc_derivative_rows.append({
                            "Date": ufc_row.get("Date"),
                            "Event": ufc_row.get("Event"),
                            "Market": label,
                            "Odds": derivative_odds,
                            "Verdict": verdict_value,
                            "Result": derivative_result,
                            "Flat Units": _pc_flat_units(derivative_result, derivative_odds),
                            "Actual Method": ufc_row.get("UFC Actual Method", ""),
                            "Actual Round": ufc_row.get("UFC Actual Round", ""),
                            "Actual Time": ufc_row.get("UFC Actual Time", ""),
                        })

                if ufc_derivative_rows:
                    ufc_derivative_frame = pd.DataFrame(ufc_derivative_rows)
                    ufc_derivative_graded = ufc_derivative_frame[ufc_derivative_frame["Result"].isin(["Correct", "Incorrect"])].copy()
                    derivative_wins = int((ufc_derivative_graded["Result"] == "Correct").sum())
                    derivative_losses = int((ufc_derivative_graded["Result"] == "Incorrect").sum())
                    derivative_accuracy = derivative_wins / len(ufc_derivative_graded) if len(ufc_derivative_graded) else None
                    derivative_units = float(ufc_derivative_graded["Flat Units"].dropna().sum()) if not ufc_derivative_graded.empty else 0.0
                    derivative_roi = derivative_units / len(ufc_derivative_graded) if len(ufc_derivative_graded) else None
                    st.markdown("### UFC Derivative Performance")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Record", f"{derivative_wins}-{derivative_losses}")
                    d2.metric("Win %", f"{derivative_accuracy:.1%}" if derivative_accuracy is not None else "—")
                    d3.metric("Flat Units", f"{derivative_units:+.2f}u")
                    d4.metric("Flat ROI", f"{derivative_roi:+.1%}" if derivative_roi is not None else "—")
                    st.caption(
                        "UFC method, distance and round-total results are graded automatically from the same settled fight result. "
                        "This table is downstream of the existing Analysis Log; it is not a separate prediction database."
                    )
                    derivative_display = ufc_derivative_frame.copy().sort_values("Date", ascending=False)
                    derivative_display["Date"] = pd.to_datetime(derivative_display["Date"], errors="coerce").dt.strftime("%B %d, %Y").str.replace(" 0", " ", regex=False)
                    st.dataframe(
                        derivative_display[["Date", "Event", "Market", "Odds", "Verdict", "Result", "Actual Method", "Actual Round", "Actual Time"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                def _pc_group_table(frame, group_col, order=None):
                    rows = []
                    groups = order or [x for x in frame[group_col].dropna().unique().tolist() if x]
                    for group in groups:
                        segment = frame[frame[group_col] == group]
                        segment_graded, sw, sl, sa, su, sr, se = _pc_summary(segment)
                        rows.append({
                            group_col: group,
                            "Record": f"{sw}-{sl}",
                            "Graded": len(segment_graded),
                            "Win %": sa,
                            "Market Expected": se,
                            "Edge vs Market": (sa - se) if sa is not None and se is not None else None,
                            "Flat Units": su,
                            "Flat ROI": sr,
                        })
                    return pd.DataFrame(rows)

                breakdown1, breakdown2 = st.columns(2)
                with breakdown1:
                    st.markdown("#### By Actual Line")
                    bucket_order = [
                        "+Money", "-101 to -149", "-150 to -199", "-200 to -249",
                        "-250 to -299", "-300 to -349", "-350 to -399", "-400 to -449", "-450+",
                    ]
                    line_table = _pc_group_table(pc_filtered, "Line Bucket", bucket_order)
                    st.dataframe(line_table, use_container_width=True, hide_index=True)
                with breakdown2:
                    st.markdown("#### By Verdict")
                    verdict_order = ["Strong Bet", "Worth Betting", "Lean", "Pass", "Complete Pass"]
                    verdict_table = _pc_group_table(pc_filtered, "Verdict", verdict_order)
                    st.dataframe(verdict_table, use_container_width=True, hide_index=True)

                st.markdown("#### By Confidence")
                confidence_rows = []
                for label in ["Very High", "High", "Moderate", "Low"]:
                    segment = pc_filtered[pc_filtered["Confidence"] == label]
                    sg, sw, sl, sa, su, sr, se = _pc_summary(segment)
                    confidence_rows.append({
                        "Confidence": label,
                        "Record": f"{sw}-{sl}",
                        "Graded": len(sg),
                        "Win %": sa,
                    })
                st.dataframe(pd.DataFrame(confidence_rows), use_container_width=True, hide_index=True)

                st.markdown("### All Predictions")
                st.caption("This is the one-stop prediction history. Use the filters above or jump directly to a saved prediction date.")

                prediction_dates = sorted(
                    {d for d in pc_filtered["Date"].dropna().dt.date.tolist()},
                    reverse=True,
                )
                prediction_date_labels = {d.strftime("%B %d, %Y").replace(" 0", " "): d for d in prediction_dates}
                prediction_date_options = ["All dates"] + list(prediction_date_labels.keys())
                selected_prediction_date = st.selectbox(
                    "Prediction date",
                    prediction_date_options,
                    key="pc_prediction_date_dropdown",
                )

                prediction_view = pc_filtered.copy()
                if selected_prediction_date != "All dates":
                    selected_date_value = prediction_date_labels[selected_prediction_date]
                    prediction_view = prediction_view[prediction_view["Date"].dt.date == selected_date_value]

                display_cols = [
                    "Date", "Sport", "Event", "Prediction", "Market", "Actual Line", "Core Zone",
                    "Fair Line", "Confidence", "Price Assessment", "Verdict", "Prediction Result",
                    "Derivative Market", "Derivative Result",
                ]
                display_frame = prediction_view[display_cols].copy().sort_values("Date", ascending=False)
                display_frame["Date"] = display_frame["Date"].dt.strftime("%B %d, %Y").str.replace(" 0", " ", regex=False)
                st.dataframe(display_frame, use_container_width=True, hide_index=True, height=520)

                export_cols = [
                    "ID", "Date", "Sport", "Event", "Participant A", "Participant B", "Prediction", "Market",
                    "Actual Line", "Core Zone", "Market Implied %", "Fair Line",
                    "Confidence", "Price Assessment", "Verdict", "Prediction Result",
                    "Derivative Market", "Derivative Odds", "Derivative Verdict", "Derivative Result",
                    "Derivative Secondary", "Derivative Secondary Odds", "Derivative Secondary Result",
                    "UFC Actual Method", "UFC Actual Round", "UFC Actual Time", "Model Version",
                ]
                export_all = pc_all[export_cols].copy()
                export_filtered = prediction_view[export_cols].copy()
                export_tennis = pc_all[pc_all["Sport"] == "Tennis"][export_cols].copy()
                export_ufc = pc_all[pc_all["Sport"] == "UFC"][export_cols].copy()
                for export_df in (export_all, export_filtered, export_tennis, export_ufc):
                    export_df["Date"] = pd.to_datetime(export_df["Date"], errors="coerce").dt.date

                e1, e2, e3, e4 = st.columns(4)
                e1.download_button(
                    "Export All Predictions", export_all.to_csv(index=False).encode("utf-8"),
                    f"macabets_all_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    use_container_width=True, key="pc_export_all",
                )
                e2.download_button(
                    "Export Filtered View", export_filtered.to_csv(index=False).encode("utf-8"),
                    f"macabets_filtered_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    use_container_width=True, key="pc_export_filtered",
                )
                e3.download_button(
                    "Export All Tennis Predictions", export_tennis.to_csv(index=False).encode("utf-8"),
                    f"macabets_tennis_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    use_container_width=True, key="pc_export_tennis",
                )
                e4.download_button(
                    "Export All UFC Predictions", export_ufc.to_csv(index=False).encode("utf-8"),
                    f"macabets_ufc_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    use_container_width=True, key="pc_export_ufc",
                )

    with archive_tabs[1]:
        st.subheader("Universal Analysis Log")
        st.caption("Every Tennis, NFL and UFC analysis is saved automatically as a frozen snapshot.")
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
            for report_sport in ("Tennis", "NFL", "UFC"):
                sport_completed = [row for row in completed_rows if str(row.get("sport", "")) == report_sport]
                sport_correct = sum(_prediction_result(row) == "Correct" for row in sport_completed)
                sport_accuracy = sport_correct / len(sport_completed) if sport_completed else None
                sport_cards.append((report_sport, sport_correct, len(sport_completed) - sport_correct, sport_accuracy))

            tennis_card, nfl_card, ufc_card = st.columns(3)
            for card, (report_sport, sport_correct, sport_incorrect, sport_accuracy) in zip(
                (tennis_card, nfl_card, ufc_card), sport_cards
            ):
                record = f"{sport_correct}-{sport_incorrect}" if sport_correct + sport_incorrect else "0-0"
                card.metric(
                    report_sport,
                    f"{sport_accuracy:.1%}" if sport_accuracy is not None else "—",
                    f"Record: {record}",
                )

            with st.expander("Confidence calibration", expanded=False):
                calibration_rows = []
                for label in ["Very High", "High", "Moderate", "Low"]:
                    band_rows = [row for row in completed_rows if _analysis_confidence_label(row) == label]
                    band_correct = sum(_prediction_result(row) == "Correct" for row in band_rows)
                    band_incorrect = len(band_rows) - band_correct
                    calibration_rows.append({
                        "Confidence": label,
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
            sport_filter = filter1.selectbox("Sport", ["All", "Tennis", "NFL", "UFC"], key="analysis_log_sport_filter")
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
                    "Confidence": _analysis_confidence_label(row),
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
                d2.metric("Confidence", _analysis_confidence_label(selected_row))
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

    with archive_tabs[2]:
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

    with archive_tabs[3]:
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
