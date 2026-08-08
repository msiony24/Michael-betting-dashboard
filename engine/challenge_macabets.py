"""Match-specific conversational challenge layer for Macabets.

The core prediction engines remain deterministic. This module lets a user debate a
single matchup with an OpenAI model, which can defend the original analysis or
propose a temporary matchup-only revision. Nothing here changes global player
ratings, model weights, or future matchups.
"""
from __future__ import annotations

import json
import re
from typing import Any


class ChallengeMacabetsError(RuntimeError):
    """Raised when the conversational challenge layer cannot return a response."""


VERDICTS = ["Strong Bet", "Worth Betting", "Lean", "Pass", "Complete Pass"]
STRONG_BET_MIN_WIN_PROBABILITY = 0.65


def _sanitize_reply_text(value: Any) -> str:
    """Keep only the human-facing Challenge reply and drop accidental model/tool debris."""
    text = str(value or "").strip()
    if not text:
        return ""

    # Hosted model/tool traces should never be visible in the Streamlit chat.
    hard_markers = [
        "</macabets_challenge_response>",
        "<macabets_challenge_response>",
        "debug_info_remaining_points",
        "production_proof",
        "sensitive_c",
    ]
    for marker in hard_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].rstrip()

    # The interface renders these structured fields separately. If the model echoes
    # them into reply, truncate the duplicate/debug tail rather than showing it raw.
    structured_tail = re.search(
        r"(?im)^\s*(agree_points|pushback_points|adjustment_reason|adjustment_category|"
        r"proposed_probability_a|proposed_confidence|proposed_verdict|revision_summary|"
        r"should_offer_apply|uses_unverified_user_claim)\s*:",
        text,
    )
    if structured_tail:
        text = text[: structured_tail.start()].rstrip()

    return text[:5000].strip()

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "message_intent": {
            "type": "string",
            "enum": ["research_question", "evidence_claim", "challenge", "command"],
        },
        "adjustment_reason": {"type": "string"},
        "verified_new_evidence_used": {"type": "boolean"},
        "agree_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "pushback_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "question_to_user": {"type": "string"},
        "stance": {
            "type": "string",
            "enum": ["defend", "partially_agree", "revise"],
        },
        "adjustment_category": {
            "type": "string",
            "enum": [
                "none",
                "reliability",
                "matchup_style",
                "surface",
                "recent_form",
                "fatigue",
                "injury",
                "motivation",
                "market_price",
                "other",
            ],
        },
        "proposed_probability_a": {
            "type": "number",
            "minimum": 0.05,
            "maximum": 0.95,
        },
        "proposed_confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "proposed_verdict": {
            "type": "string",
            "enum": VERDICTS,
        },
        "revision_summary": {"type": "string"},
        "should_offer_apply": {"type": "boolean"},
        "uses_unverified_user_claim": {"type": "boolean"},
    },
    "required": [
        "reply",
        "message_intent",
        "adjustment_reason",
        "verified_new_evidence_used",
        "agree_points",
        "pushback_points",
        "question_to_user",
        "stance",
        "adjustment_category",
        "proposed_probability_a",
        "proposed_confidence",
        "proposed_verdict",
        "revision_summary",
        "should_offer_apply",
        "uses_unverified_user_claim",
    ],
    "additionalProperties": False,
}

_SYSTEM_INSTRUCTIONS = """
You are the Macabets Challenge Analyst. You are debating one specific sports
matchup with the user. The deterministic Macabets model has already produced an
opinion. Your job is to pressure-test that opinion, not to flatter the user and
not to blindly defend the model.

Rules:
0. FIRST classify the user's new message into exactly one message_intent:
   - research_question: the user is asking for facts, history, stats, explanation, or clarification.
   - evidence_claim: the user is supplying a factual claim or new evidence for Macabets to verify.
   - challenge: the user is arguing that the prediction, probability, confidence, price assessment, or verdict is wrong or should change.
   - command: the user is asking to show/reset/finalize/format something, rather than debating the matchup.
   Answer the user's actual request first. Do not reinterpret a factual question as a betting challenge.
1. Treat verified_recent_evidence in the supplied matchup context as verified Macabets
   match history. Use it actively: check recent results, opponent quality, surface,
   tournament, round, score, and rankings before judging the user's argument.
2. The evidence packet includes latest_match_date_in_database. If a user's claim is
   dated after that cutoff, do NOT call it false merely because it is absent locally.
   Use web search when available to verify recent factual claims. If web search verifies
   the claim, treat it as verified and set uses_unverified_user_claim=false. If neither
   local evidence nor web search verifies it, reason conditionally and set the flag true.
3. Do not invent injuries, statistics, news, or facts. Prefer the structured Macabets
   evidence first, then current web verification only for missing/recent claims.
3. Distinguish "this player should be favored" from "this player deserves a
   Strong Bet." Reliability, volatility, matchup fragility, fatigue, and price
   can justify lowering confidence or the verdict even when the same winner is
   still preferred.
4. Push back when the user's argument is weak, vague, or contradicted by the
   supplied data. Ask for the missing reasoning inside your reply when useful.
5. If the argument materially changes your view, propose a matchup-only revision.
   Do not imply that player ratings or global model weights have changed.
6. Probability and confidence changes should normally be modest. Large changes
   require strong matchup-specific reasoning in the supplied context.
7. Sound like Macabets, not a generic chatbot. Speak like a sharp betting analyst:
   direct, specific, skeptical, and willing to say either the model or the user
   has the stronger case. Avoid filler such as "you're right to push back."
8. Separate two questions: (a) who is more likely to win and (b) whether that
   player deserves the current betting verdict. A player can remain the projected
   winner while a Strong Bet is downgraded because the matchup is fragile.
9. Use agree_points for the strongest parts of the user's case and pushback_points
   for the strongest reasons Macabets still resists it. Do not manufacture either.
10. question_to_user should contain one short, pointed follow-up question when a
   missing piece of reasoning would materially help the debate. Otherwise return
   an empty string.
11. Keep reply concise. It should summarize the debate position, not repeat every
   field that the interface will display separately.
12. proposed_probability_a is always Player A's win probability, regardless of
   which player the user discusses.
13. proposed_verdict is the proposed headline verdict for the current projected
   winner at the current market price.
14. Set should_offer_apply=true only when you actually recommend changing at
   least one of probability, confidence, or verdict.
15. The supplied current_opinion may already reflect earlier turns in this same
   debate even if the user has not finalized the revision. Treat it as the live
   debate position. The original_opinion remains the untouched baseline.
16. Never treat abbreviated provider names as separate people from obvious full-name
   references. For example, "Djokovic N." and "Novak Djokovic" are the same opponent.
17. When verified_recent_evidence directly confirms a user's factual point, say so
   plainly (for example, "Confirmed in Macabets match history") and incorporate it
   into the debate rather than asking the user to prove it again.

18. CRITICAL STATE RULE: research_question and command messages are informational only. They MUST NOT change probability, confidence, or verdict. For those intents, return the current values exactly, set should_offer_apply=false, adjustment_category="none", adjustment_reason="", and verified_new_evidence_used=false.
19. For evidence_claim messages, verify the claim using the supplied evidence first and web search only when needed. A numerical/model change is allowed only if the claim is verified, materially relevant to this matchup, and adjustment_reason names the exact verified evidence and why it changes the model. If the claim is unverified or immaterial, keep all model values unchanged.
20. For challenge messages, a model change is allowed only when adjustment_reason gives a concrete matchup-specific reason. Never move numbers merely because the user asked a question, expressed uncertainty, or continued the conversation.
21. Every numerical change must be traceable. If proposed_probability_a, proposed_confidence, or proposed_verdict differs from current_opinion, adjustment_reason must be non-empty and specific. Never make unexplained "small nudges" or confidence reductions.
22. When the user asks about recent matches, answer with the actual recent-match list from verified_recent_evidence, newest first, for both players when requested. Do not substitute a general betting interpretation for the requested facts.
23. For tennis, recent_resume_comparison is deterministic model evidence. Use it before web search when comparing recent form or strength of schedule. Never claim one player's recent wins/resume are stronger unless the supplied resume metrics support that conclusion; if the metrics are mixed, say they are mixed.
24. Treat fatigue as a risk signal, not proof of deterioration. If fatigue_resilience_a/b is positive, explicitly account for the fact that high-level recent performance has already softened the workload penalty. Do not repeatedly cite raw match counts as if the resilience adjustment did not exist.
25. Keep winner projection and betting verdict conceptually separate. If the user asks only "who wins?", answer from win probability and do not use market value as a reason the player is more likely to win. Strong Bet/Worth Betting describe price + confidence, not certainty of winning.
26. HARD VERDICT RULE: "Strong Bet" is unavailable unless the current projected winner has at least a 65% win probability. At 58.6%, for example, the maximum headline verdict is "Worth Betting" regardless of market edge.
27. The reply field is human-facing prose only. Never echo schema field names, JSON, XML/tool tags, hidden annotations, debug tokens, or internal metadata in reply. Keep reply concise; the UI renders agree_points, pushback_points, adjustment_reason, probability, confidence, and verdict separately.
26. Do not use web search merely to defend the current opinion when deterministic matchup evidence already answers the question. Web search is a freshness fallback for facts missing from or newer than the supplied evidence.
""".strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_response(
    payload: dict[str, Any],
    current_probability_a: float,
    current_confidence: int,
    current_verdict: str,
) -> dict[str, Any]:
    """Apply conservative hard caps so a conversation cannot wildly move a model."""
    normalized = dict(payload)
    try:
        proposed_probability = float(normalized.get("proposed_probability_a"))
    except (TypeError, ValueError):
        proposed_probability = current_probability_a
    try:
        proposed_confidence = int(normalized.get("proposed_confidence"))
    except (TypeError, ValueError):
        proposed_confidence = current_confidence

    # Per-turn caps. A longer back-and-forth can move further, but only gradually.
    normalized["proposed_probability_a"] = round(
        _clamp(
            proposed_probability,
            current_probability_a - 0.08,
            current_probability_a + 0.08,
        ),
        4,
    )
    normalized["proposed_confidence"] = int(
        round(
            _clamp(
                proposed_confidence,
                current_confidence - 10,
                current_confidence + 10,
            )
        )
    )
    if normalized.get("proposed_verdict") not in VERDICTS:
        normalized["proposed_verdict"] = current_verdict

    normalized["reply"] = _sanitize_reply_text(normalized.get("reply"))

    # Deterministic guardrail: Challenge Macabets cannot label a sub-65% projected
    # winner a Strong Bet, even if the language model tries to preserve that label.
    projected_winner_probability = max(
        float(normalized["proposed_probability_a"]),
        1.0 - float(normalized["proposed_probability_a"]),
    )
    if (
        str(normalized.get("proposed_verdict")) == "Strong Bet"
        and projected_winner_probability < STRONG_BET_MIN_WIN_PROBABILITY
    ):
        normalized["proposed_verdict"] = "Worth Betting"

    intent_contract_present = "message_intent" in payload
    intent = str(normalized.get("message_intent") or "challenge")
    if intent not in {"research_question", "evidence_claim", "challenge", "command"}:
        intent = "challenge"
    normalized["message_intent"] = intent
    normalized["adjustment_reason"] = str(normalized.get("adjustment_reason") or "").strip()
    normalized["verified_new_evidence_used"] = bool(normalized.get("verified_new_evidence_used"))

    # Research/questions and UI-style commands are read-only. Asking Macabets for
    # information must never silently move the betting model.
    if intent in {"research_question", "command"}:
        normalized["proposed_probability_a"] = round(float(current_probability_a), 4)
        normalized["proposed_confidence"] = int(current_confidence)
        normalized["proposed_verdict"] = current_verdict
        normalized["should_offer_apply"] = False
        normalized["adjustment_category"] = "none"
        normalized["adjustment_reason"] = ""
        normalized["verified_new_evidence_used"] = False
        normalized["agree_points"] = []
        normalized["pushback_points"] = []
        normalized["revision_summary"] = ""

    # Evidence claims may move the model only when the model says it actually
    # verified and used new relevant evidence. Challenges need an explicit reason.
    wants_change = (
        abs(float(normalized["proposed_probability_a"]) - current_probability_a) >= 0.002
        or int(normalized["proposed_confidence"]) != current_confidence
        or str(normalized["proposed_verdict"]) != current_verdict
    )
    if intent_contract_present and wants_change and intent == "evidence_claim" and (
        not normalized["verified_new_evidence_used"] or not normalized["adjustment_reason"]
    ):
        normalized["proposed_probability_a"] = round(float(current_probability_a), 4)
        normalized["proposed_confidence"] = int(current_confidence)
        normalized["proposed_verdict"] = current_verdict
        normalized["should_offer_apply"] = False
    elif intent_contract_present and wants_change and intent == "challenge" and not normalized["adjustment_reason"]:
        normalized["proposed_probability_a"] = round(float(current_probability_a), 4)
        normalized["proposed_confidence"] = int(current_confidence)
        normalized["proposed_verdict"] = current_verdict
        normalized["should_offer_apply"] = False

    # Re-apply the hard Strong Bet floor after intent-specific state restoration.
    final_winner_probability = max(
        float(normalized["proposed_probability_a"]),
        1.0 - float(normalized["proposed_probability_a"]),
    )
    if (
        str(normalized.get("proposed_verdict")) == "Strong Bet"
        and final_winner_probability < STRONG_BET_MIN_WIN_PROBABILITY
    ):
        normalized["proposed_verdict"] = "Worth Betting"

    changed = (
        abs(normalized["proposed_probability_a"] - current_probability_a) >= 0.002
        or normalized["proposed_confidence"] != current_confidence
        or normalized["proposed_verdict"] != current_verdict
    )
    normalized["should_offer_apply"] = bool(
        normalized.get("should_offer_apply") and changed
    )
    return normalized


def challenge_macabets(
    *,
    api_key: str,
    matchup_context: dict[str, Any],
    conversation: list[dict[str, str]],
    user_message: str,
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    """Return Macabets' response and a possible temporary matchup-only revision."""
    if not api_key:
        raise ChallengeMacabetsError("OPENAI_API_KEY is not configured.")
    if not str(user_message or "").strip():
        raise ChallengeMacabetsError("Enter a reason for challenging the analysis.")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - deployment dependency guard
        raise ChallengeMacabetsError(
            "The OpenAI Python package is not installed. Redeploy after updating requirements.txt."
        ) from exc

    current = matchup_context.get("current_opinion", {})
    current_probability = float(current.get("probability_a", 0.5))
    current_confidence = int(current.get("confidence", 50))
    current_verdict = str(current.get("verdict") or "Pass")
    if current_verdict not in VERDICTS:
        current_verdict = "Pass"

    compact_history = []
    for item in conversation[-10:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            compact_history.append({"role": role, "content": content[:3000]})

    request_payload = {
        "matchup": matchup_context,
        "conversation_so_far": compact_history,
        "new_user_challenge": str(user_message).strip(),
    }

    try:
        client = OpenAI(api_key=api_key)
        response_kwargs = {
            "model": model,
            "instructions": _SYSTEM_INSTRUCTIONS,
            "input": json.dumps(request_payload, ensure_ascii=False, default=str),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "macabets_challenge_response",
                    "strict": True,
                    "schema": _RESPONSE_SCHEMA,
                }
            },
            # The local ATP packet is the primary source. Web search is a fallback for
            # very recent claims that may be newer than the scheduled tennis-data feed.
            "tools": [{"type": "web_search"}],
        }
        try:
            response = client.responses.create(**response_kwargs)
        except Exception as web_exc:
            # Keep Challenge Macabets usable if a deployed model/account temporarily
            # lacks hosted web-search access. The verified local evidence still works.
            response_kwargs.pop("tools", None)
            response = client.responses.create(**response_kwargs)
        raw_text = str(response.output_text or "").strip()
        payload = json.loads(raw_text)
    except ChallengeMacabetsError:
        raise
    except Exception as exc:
        raise ChallengeMacabetsError(f"OpenAI challenge request failed: {exc}") from exc

    return _normalize_response(
        payload,
        current_probability_a=current_probability,
        current_confidence=current_confidence,
        current_verdict=current_verdict,
    )
