"""Turn a CriticVerdict into feedback text, and merge it into a running history.

Used by build_graph.py, pipeline.py, and scripts/draft_and_critique_demo.py so the
three don't each keep their own copy of the same two rules.
"""
from __future__ import annotations

from career_copilot.schemas.critic import CriticVerdict


def verdict_to_feedback_items(verdict: CriticVerdict) -> list[str]:
    return verdict.issues + [
        f'Claim "{c.claim_text}" is not well-grounded: {c.reason}'
        for c in verdict.ungrounded_claims
    ]


def accumulate_feedback(history: list[str] | None, new_items: list[str]) -> list[str]:
    """Append new_items to history, skipping anything already in there."""
    merged = list(history or [])
    for item in new_items:
        if item not in merged:
            merged.append(item)
    return merged
