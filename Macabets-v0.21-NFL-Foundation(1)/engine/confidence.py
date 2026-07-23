"""Shared confidence helpers for Macabets sport engines."""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def confidence_score(data_quality: float, agreement: float, uncertainty: float) -> int:
    """Return a transparent 1-10 confidence score.

    Inputs are on 0-10 scales. Uncertainty is subtracted rather than hidden.
    """
    score = 0.50 * data_quality + 0.35 * agreement + 0.15 * (10 - uncertainty)
    return int(round(clamp(score, 1, 10)))
