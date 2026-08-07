"""Match-specific conversational challenge layer for Macabets.

The core prediction engines remain deterministic. This module lets a user debate a
single matchup with an OpenAI model, which can defend the original analysis or
propose a temporary matchup-only revision. Nothing here changes global player
ratings, model weights, or future matchups.
"""
from __future__ import annotations

import json
from typing import Any


class ChallengeMacabetsError(RuntimeError):
    """Raised when the conversational challenge layer cannot return a response."""


VERDICTS = ["Strong Bet", "Worth Betting", "Lean", "Pass", "Complete Pass"]

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
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
1. Use only the supplied matchup context plus the user's reasoning. Do not invent
   injuries, statistics, news, or facts that are not in the context.
2. If the user introduces a factual claim not present in the supplied context,
   you may reason about it conditionally, but explicitly say it is unverified and
   set uses_unverified_user_claim=true.
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
        response = client.responses.create(
            model=model,
            instructions=_SYSTEM_INSTRUCTIONS,
            input=json.dumps(request_payload, ensure_ascii=False, default=str),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "macabets_challenge_response",
                    "strict": True,
                    "schema": _RESPONSE_SCHEMA,
                }
            },
        )
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
