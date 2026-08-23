"""Shared confidence helpers for Macabets models."""


def confidence_band(score: float) -> str:
    score = float(score)
    if score >= 90:
        return "Exceptional"
    if score >= 80:
        return "High"
    if score >= 70:
        return "Solid"
    if score >= 60:
        return "Moderate"
    if score >= 50:
        return "Low"
    return "Pass"


def recommendation_from_edge(edge_points: float, confidence: float) -> str:
    """Convert a *signed* model-vs-market edge into a restrained bet label.

    Positive edge means the model assigns the recommended side a higher win
    probability than the no-vig market. A zero or negative edge is never a bet,
    regardless of how large the disagreement is.
    """
    edge = float(edge_points)
    confidence = float(confidence)
    if confidence < 50 or edge < 0.75:
        return "Pass"
    if edge >= 3.0 and confidence >= 80:
        return "Strong Bet"
    if edge >= 2.0 and confidence >= 72:
        return "Good Bet"
    return "Lean"
