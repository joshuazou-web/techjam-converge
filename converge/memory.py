"""Runtime memory: session state, context distillation, and strategy learning.

Three layers, matching the three timescales the agent has to reason over:

``SessionMemory``
    what happened in *this* conversation: the replayable transcript, the pool
    size after every turn, and the strategy chosen at each step.  It is the
    audit trail behind every recommendation.
``ContextDistiller``
    turns the anonymised profile plus the live transcript into the compact
    context the ranker and the prompt actually consume.  Only aggregate
    preference tags are used -- never an identity, never raw history.
``StrategyMemory``
    survives across sessions.  It records how much information each question
    channel actually returned and nudges future question choice towards the
    channels that pay off on this catalog.  This is the self-evolution loop:
    the agent's guidance policy is refined by its own outcomes, not by a
    hand-tuned script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .usermodel import Transcript


@dataclass
class TurnTrace:
    """One turn of the audit trail."""

    turn: int
    scenario: str
    mode: str
    pool_size: int
    plan: str
    asked: str | None
    gain: float
    slate: int
    top_asin: str | None


@dataclass
class SessionMemory:
    """Short-term state for a single conversation."""

    session_id: str
    profile: dict = field(default_factory=dict)
    transcript: Transcript = field(default_factory=Transcript)
    traces: list[TurnTrace] = field(default_factory=list)
    last_pool_size: int = 0
    #: content words seen this session, kept for the lexical/dense routes when
    #: exact constraint parsing is not available
    free_terms: list[str] = field(default_factory=list)

    def remember_terms(self, terms: list[str]) -> None:
        known = set(self.free_terms)
        self.free_terms.extend(term for term in terms if term not in known)

    def record(self, trace: TurnTrace) -> None:
        self.traces.append(trace)
        self.last_pool_size = trace.pool_size

    @property
    def asked_attributes(self) -> list[str]:
        return [trace.asked for trace in self.traces if trace.asked]


class ContextDistiller:
    """Compress profile + dialogue into the context downstream stages consume."""

    def distill(self, memory: SessionMemory) -> dict:
        profile = memory.profile or {}
        transcript = memory.transcript
        tags = [str(tag) for tag in (profile.get("preference_tags") or [])]
        return {
            "category": transcript.category,
            "scenario": transcript.scenario,
            "constraints": transcript.observed_atoms(),
            "preference_tags": tags,
            "rating_style": str(profile.get("rating_style") or ""),
            "purchase_frequency": str(profile.get("purchase_frequency") or ""),
            "asked": memory.asked_attributes,
        }

    def query_text(self, memory: SessionMemory) -> str:
        """Free-text view of the session, used by the lexical/dense routes."""
        transcript = memory.transcript
        parts: list[str] = []
        if transcript.category:
            parts.append(transcript.category)
        parts.extend(transcript.observed_atoms())
        parts.extend(str(tag) for tag in (memory.profile.get("preference_tags") or []))
        if not transcript.observed_atoms():
            parts.extend(memory.free_terms)
        return " ".join(parts)


class StrategyMemory:
    """Cross-session bandit over question channels.

    Kept deliberately conservative: the bonus is a small tie-breaker on top of
    the per-turn information-gain estimate, so a cold start behaves exactly like
    the un-adapted policy and a warm start only re-orders near-ties.
    """

    def __init__(self, rate: float = 0.2, weight: float = 0.05) -> None:
        self.rate = rate
        self.weight = weight
        self.yield_by_attribute: dict[str, float] = {}
        self.uses_by_attribute: dict[str, int] = {}
        self.sessions = 0

    def observe(self, attribute: str | None, realized_gain: float) -> None:
        if attribute is None:
            return
        previous = self.yield_by_attribute.get(attribute, realized_gain)
        self.yield_by_attribute[attribute] = (1 - self.rate) * previous + self.rate * realized_gain
        self.uses_by_attribute[attribute] = self.uses_by_attribute.get(attribute, 0) + 1

    def bonus(self, attribute: str | None) -> float:
        if attribute is None or attribute not in self.yield_by_attribute:
            return 0.0
        return self.weight * self.yield_by_attribute[attribute]

    def snapshot(self) -> dict:
        return {
            "sessions": self.sessions,
            "attribute_yield": {
                key: round(value, 4) for key, value in sorted(self.yield_by_attribute.items())
            },
            "attribute_uses": dict(sorted(self.uses_by_attribute.items())),
        }
