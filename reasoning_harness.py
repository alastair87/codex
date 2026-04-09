"""A lightweight, local reasoning harness.

This module provides a small framework for running multi-step reasoning with
trace capture, guardrails, and confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Sequence


@dataclass
class ReasoningStep:
    """A single step in a reasoning trace."""

    name: str
    content: str
    score: float = 0.0


@dataclass
class ReasoningResult:
    """Final output emitted by the harness."""

    question: str
    answer: str
    confidence: float
    trace: List[ReasoningStep] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


Reasoner = Callable[[str, Sequence[ReasoningStep]], ReasoningStep]


class ReasoningHarness:
    """Runs a fixed sequence of reasoning stages over an input question.

    The harness intentionally keeps each stage explicit so it can be tested and
    inspected without requiring a network LLM dependency.
    """

    def __init__(self, stages: Iterable[tuple[str, Reasoner]] | None = None):
        self.stages = list(stages or self._default_stages())

    def run(self, question: str) -> ReasoningResult:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        trace: list[ReasoningStep] = []
        for name, fn in self.stages:
            step = fn(question, trace)
            if step.name != name:
                step.name = name
            trace.append(step)

        answer_step = trace[-1]
        confidence = self._aggregate_confidence(trace)
        return ReasoningResult(
            question=question,
            answer=answer_step.content,
            confidence=confidence,
            trace=trace,
        )

    @staticmethod
    def _aggregate_confidence(trace: Sequence[ReasoningStep]) -> float:
        if not trace:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for idx, step in enumerate(trace, start=1):
            weight = idx
            weighted_sum += max(0.0, min(step.score, 1.0)) * weight
            weight_total += weight
        return round(weighted_sum / weight_total, 3)

    @staticmethod
    def _default_stages() -> list[tuple[str, Reasoner]]:
        return [
            ("interpret", _interpret_stage),
            ("hypothesize", _hypothesize_stage),
            ("challenge", _challenge_stage),
            ("answer", _answer_stage),
        ]


def _interpret_stage(question: str, _: Sequence[ReasoningStep]) -> ReasoningStep:
    normalized = " ".join(question.split())
    return ReasoningStep(
        name="interpret",
        content=f"Question interpreted as: {normalized}",
        score=0.75,
    )


def _hypothesize_stage(question: str, trace: Sequence[ReasoningStep]) -> ReasoningStep:
    del trace
    tokens = [t.strip(".,?!") for t in question.lower().split()]
    content_words = [t for t in tokens if len(t) > 3]
    topics = ", ".join(content_words[:5]) or "general domain"
    return ReasoningStep(
        name="hypothesize",
        content=f"Potential approach: break the problem into subparts around {topics}.",
        score=0.65,
    )


def _challenge_stage(_: str, trace: Sequence[ReasoningStep]) -> ReasoningStep:
    hypothesis = next((s for s in trace if s.name == "hypothesize"), None)
    content = "Risk check: validate assumptions and request missing constraints."
    if hypothesis and "general domain" in hypothesis.content:
        content += " Scope is broad, so confidence is reduced."
        score = 0.45
    else:
        score = 0.7
    return ReasoningStep(name="challenge", content=content, score=score)


def _answer_stage(question: str, trace: Sequence[ReasoningStep]) -> ReasoningStep:
    interpret = next((s for s in trace if s.name == "interpret"), None)
    challenge = next((s for s in trace if s.name == "challenge"), None)

    answer_lines = [
        "Proposed answer:",
        f"- Goal: respond to '{question.strip()}'",
    ]
    if interpret:
        answer_lines.append(f"- Context: {interpret.content}")
    if challenge:
        answer_lines.append(f"- Safety check: {challenge.content}")
    answer_lines.append("- Next step: execute and verify with a concrete test.")

    score = 0.8 if challenge and "reduced" not in challenge.content else 0.6
    return ReasoningStep(name="answer", content="\n".join(answer_lines), score=score)
