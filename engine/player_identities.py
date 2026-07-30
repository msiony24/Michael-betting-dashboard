"""Narrative ATP player identities for Macabets.

This module is explanation-only. Nothing returned here should be used to alter
win probabilities, Elo ratings, fair lines, or betting verdicts.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


# Keep these descriptions concise and tactical. They are presentation context,
# not numerical inputs to the prediction model.
PLAYER_IDENTITIES = {
    "carlos alcaraz": {
        "identity": "Explosive All-Court Attacker",
        "strengths": ["Elite movement and court coverage", "Creates offense from defensive positions", "Comfortable changing pace and moving forward"],
        "watch_for": ["Can press for low-percentage winners when rushed", "Level can fluctuate within sets"],
        "preferred_pattern": "Use athleticism and variety to take control of rallies, then finish from inside the court.",
    },
    "jannik sinner": {
        "identity": "First-Strike Power Baseliner",
        "strengths": ["Clean, early ball striking", "Elite backhand stability", "Takes time away with compact offense"],
        "watch_for": ["Can be pulled out of rhythm by heavy variation", "Less comfortable when forced to defend repeatedly at full stretch"],
        "preferred_pattern": "Control the baseline early, redirect pace, and keep points on his terms.",
    },
    "novak djokovic": {
        "identity": "Elite Counterpunching Controller",
        "strengths": ["Exceptional return positioning and anticipation", "Redirects pace with precision", "Outstanding point construction under pressure"],
        "watch_for": ["Can become more passive when timing is disrupted", "Physical durability is more important in extended matches"],
        "preferred_pattern": "Neutralize the first strike, establish depth, and gradually take control through precision.",
    },
    "alexander zverev": {
        "identity": "Powerful Two-Way Baseliner",
        "strengths": ["High-end first serve", "Backhand controls exchanges", "Strong court coverage for his size"],
        "watch_for": ["Second serve can invite pressure", "Forehand can become less assertive in tense moments"],
        "preferred_pattern": "Win free points, settle behind the backhand, and dictate without overextending.",
    },
    "daniil medvedev": {
        "identity": "Deep-Court Counterpuncher",
        "strengths": ["Exceptional defensive reach", "Absorbs and redirects pace", "Makes opponents play extra balls"],
        "watch_for": ["Can concede too much court against confident attackers", "Less natural when required to finish frequently at net"],
        "preferred_pattern": "Extend exchanges, disrupt rhythm, and turn the match into a consistency test.",
    },
    "casper ruud": {
        "identity": "Heavy-Forehand Clay-Court Builder",
        "strengths": ["Heavy forehand creates court position", "Disciplined rally construction", "Reliable movement on clay"],
        "watch_for": ["Backhand can be targeted by elite pace", "Can struggle to shorten points on faster courts"],
        "preferred_pattern": "Use shape and depth to create forehand opportunities and control the center of the court.",
    },
    "lorenzo musetti": {
        "identity": "Creative All-Court Shotmaker",
        "strengths": ["Excellent variety and touch", "Comfortable changing height and pace", "Natural movement on clay"],
        "watch_for": ["Can lose court position against relentless pace", "Shot selection may become ambitious under pressure"],
        "preferred_pattern": "Break rhythm with variety, create angles, and open the court before attacking.",
    },
    "andrey rublev": {
        "identity": "Forehand-Led Power Baseliner",
        "strengths": ["Heavy first-strike forehand", "Sustained baseline aggression", "Punishes short balls quickly"],
        "watch_for": ["Can become predictable when Plan A stalls", "Emotional frustration can accelerate errors"],
        "preferred_pattern": "Take the ball early, dominate with the forehand, and keep points direct.",
    },
    "holger rune": {
        "identity": "Aggressive All-Court Disruptor",
        "strengths": ["Takes the ball early", "Can attack from both wings", "Comfortable changing direction and moving forward"],
        "watch_for": ["Decision-making can become inconsistent", "Energy and concentration can fluctuate"],
        "preferred_pattern": "Apply early pressure, change direction quickly, and prevent the opponent from settling.",
    },
    "taylor fritz": {
        "identity": "Serve-Plus-One Power Player",
        "strengths": ["Reliable first serve", "Flat forehand through the court", "Strong first-ball offense"],
        "watch_for": ["Movement can be tested in extended defensive exchanges", "Return position may concede initiative"],
        "preferred_pattern": "Use the serve to earn a short reply and finish with direct baseline power.",
    },
    "ben shelton": {
        "identity": "Explosive Left-Handed Attacker",
        "strengths": ["High-impact left-handed serve", "Athletic court coverage", "Comfortable finishing at net"],
        "watch_for": ["Return consistency can vary", "Aggression can produce streaky error runs"],
        "preferred_pattern": "Create immediate pressure with the serve and attack before rallies become neutral.",
    },
    "tommy paul": {
        "identity": "Athletic All-Court Counterpuncher",
        "strengths": ["Excellent movement", "Redirects pace well", "Comfortable transitioning forward"],
        "watch_for": ["Can lack a single overwhelming finishing weapon", "May be forced into defensive patterns by elite power"],
        "preferred_pattern": "Use speed and clean redirection to turn defense into controlled offense.",
    },
    "frances tiafoe": {
        "identity": "Athletic First-Strike Shotmaker",
        "strengths": ["Explosive forehand", "Creative offense and net instincts", "Thrives in fast, emotional match environments"],
        "watch_for": ["Baseline tolerance can fluctuate", "Shot selection can become loose"],
        "preferred_pattern": "Keep points dynamic, use athletic offense, and avoid repetitive neutral exchanges.",
    },
    "stefanos tsitsipas": {
        "identity": "Forehand-Dominant All-Court Attacker",
        "strengths": ["Heavy forehand", "Strong serve-plus-one patterns", "Natural transition and net play"],
        "watch_for": ["Backhand return can be pressured", "Deep, high-quality backhand exchanges can pin him down"],
        "preferred_pattern": "Build around the serve and forehand, then finish forward before the backhand is isolated.",
    },
    "hubert hurkacz": {
        "identity": "Serve-Led All-Court Player",
        "strengths": ["Elite first serve", "Good movement for his size", "Comfortable closing at net"],
        "watch_for": ["Return games can be low-margin", "Baseline aggression may fade when the first serve is neutralized"],
        "preferred_pattern": "Protect service games, create short points, and pressure opponents through scoreboard control.",
    },
    "alex de minaur": {
        "identity": "Speed-Based Counterpuncher",
        "strengths": ["Elite speed and recovery", "Takes time away with early contact", "Competes relentlessly in neutral rallies"],
        "watch_for": ["Can be overpowered by sustained heavy hitting", "Free-point production is not always enough against elite servers"],
        "preferred_pattern": "Rush the opponent through speed, depth, and early redirection rather than raw power.",
    },
    "grigor dimitrov": {
        "identity": "Fluid All-Court Shotmaker",
        "strengths": ["Broad shot variety", "Strong transition game", "Can redirect pace from both wings"],
        "watch_for": ["Backhand can be pinned under sustained pressure", "Execution level can vary across long matches"],
        "preferred_pattern": "Use variety and court craft to avoid static power exchanges and create attacking openings.",
    },
    "jack draper": {
        "identity": "Left-Handed Power Baseliner",
        "strengths": ["Heavy left-handed serve and forehand", "Takes the ball early", "Strong physical presence in baseline exchanges"],
        "watch_for": ["Extended physical matches can test durability", "Can overpress when trying to end points quickly"],
        "preferred_pattern": "Use left-handed patterns to open the court and control rallies with first-strike power.",
    },
    "arthur fils": {
        "identity": "Explosive Power Athlete",
        "strengths": ["Heavy forehand acceleration", "Strong athletic defense", "Can overwhelm opponents with pace"],
        "watch_for": ["Point construction can become impatient", "Error rate rises when forced to create repeatedly from neutral positions"],
        "preferred_pattern": "Use physical power to seize court position and finish before rallies become overly complex.",
    },
    "sebastian korda": {
        "identity": "Clean-Ball-Striking All-Court Player",
        "strengths": ["Effortless timing", "Takes the ball early", "Balanced offense from both wings"],
        "watch_for": ["Physical availability has sometimes interrupted continuity", "Can be pushed back by heavier, more physical opponents"],
        "preferred_pattern": "Control tempo through early contact and smooth directional changes.",
    },
}


def _statistical_identity(profile: dict, style: dict, surface: str) -> dict:
    """Create an evidence-limited fallback from verified model inputs."""
    label = str(style.get("label") or "Balanced Baseline Player")
    if label == "Style data unavailable":
        label = "Statistical Baseline Profile"

    strengths: list[str] = []
    watch_for: list[str] = []
    serve = profile.get("serve_points_won")
    ret = profile.get("return_points_won")
    recent = profile.get("recent_win")
    surface_win = profile.get("surface_win")

    if pd.notna(serve):
        if float(serve) >= 0.655:
            strengths.append("Serve results are a meaningful source of control")
        elif float(serve) < 0.625:
            watch_for.append("Limited free-point production in the available sample")
    if pd.notna(ret):
        if float(ret) >= 0.400:
            strengths.append("Return results show consistent pressure")
        elif float(ret) < 0.380:
            watch_for.append("Return results leave less margin in close service sets")
    if pd.notna(recent):
        if float(recent) >= 0.65:
            strengths.append("Strong recent win rate")
        elif float(recent) <= 0.40:
            watch_for.append("Recent results have been below a winning level")
    if pd.notna(surface_win):
        if float(surface_win) >= 0.62:
            strengths.append(f"Strong recent results on {surface}")
        elif float(surface_win) <= 0.42:
            watch_for.append(f"Limited recent success on {surface}")

    if not strengths:
        strengths.append("No single verified statistical trait clearly dominates")
    if not watch_for:
        watch_for.append("Tactical detail is limited to the available match statistics")

    preferred = {
        "Big Server": "Use service games to control the scoreboard and keep return sets short.",
        "Elite Returner": "Apply return pressure and extend enough rallies to expose weaker service patterns.",
        "Aggressive All-Court": "Take control early and look to finish before the opponent establishes rhythm.",
        "Counterpuncher": "Extend exchanges, absorb pace, and force the opponent to create repeatedly.",
        "Balanced Baseliner": "Win through steady baseline execution without relying on one dominant pattern.",
    }.get(str(style.get("label")), "Rely on the strongest verified statistical pattern available in this matchup.")

    return {
        "identity": label,
        "strengths": strengths[:3],
        "watch_for": watch_for[:2],
        "preferred_pattern": preferred,
        "source": "statistical",
    }


def get_player_identity(player_name: str, profile: dict, style: dict, surface: str) -> dict:
    """Return a curated identity when available, otherwise a verified-data fallback."""
    curated = PLAYER_IDENTITIES.get(_key(player_name))
    if curated:
        return {**curated, "source": "curated"}
    return _statistical_identity(profile, style, surface)
