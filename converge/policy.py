"""Question-value estimation.

A clarification question is only worth a turn if it splits the candidate pool.
Because the customer is a deterministic function of the hidden product, we can
answer "how much would this question tell me?" *before* asking it: run the
forward user model over every surviving candidate, group candidates by the
answer they would produce, and measure the expected entropy left behind.

That turns clarification into an explicit expected-information-gain decision
instead of a hand-written question script, and it is what keeps mean turns to
conversion low: we never spend a turn on a question whose answer we can already
predict.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .cards import ALLOWED_ATTRIBUTES
from .catalog import Catalog
from .usermodel import Transcript, expected_reply

#: Concrete channels read more naturally than the open-ended "other" channel.
_CONCRETE = tuple(attribute for attribute in ALLOWED_ATTRIBUTES if attribute != "other")

#: Relative gain a concrete question may give up to stay conversational.
_NATURALNESS_MARGIN = 0.02


@dataclass
class Question:
    """A candidate clarification, with the evidence for asking it."""

    attribute: str | None
    gain: float
    residual_entropy: float
    partitions: int

    @property
    def worth_asking(self) -> bool:
        return self.attribute is not None and self.gain > 1e-9


def _entropy(weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        p = weight / total
        entropy -= p * math.log2(p)
    return entropy


def _candidate_weights(catalog: Catalog, pool: Sequence[int], prior_mix: float) -> list[float]:
    """Blend a uniform posterior with the purchase-volume prior."""
    if not pool:
        return []
    uniform = 1.0 / len(pool)
    weights: list[float] = []
    for idx in pool:
        weights.append((1.0 - prior_mix) * uniform + prior_mix * (catalog.popularity(idx) + 1e-6))
    return weights


def choose_question(
    catalog: Catalog,
    transcript: Transcript,
    pool: Sequence[int],
    prior_mix: float = 0.5,
    sample_cap: int = 1500,
    natural: bool = True,
) -> Question:
    """Pick the attribute with the highest expected information gain."""
    if len(pool) <= 1:
        return Question(None, 0.0, 0.0, 1)

    working = list(pool)
    if len(working) > sample_cap:
        # Deterministic sub-sample: the most probable candidates dominate the
        # expectation anyway, and this keeps per-turn latency flat.
        working.sort(key=catalog.popularity, reverse=True)
        working = working[:sample_cap]

    disclosed = transcript.disclosed_atoms()
    weights = _candidate_weights(catalog, working, prior_mix)
    total = sum(weights)
    base_entropy = _entropy(weights)

    best = Question(None, 0.0, base_entropy, 1)
    concrete_best: Question | None = None

    for attribute in ALLOWED_ATTRIBUTES:
        groups: dict[tuple[str, ...], float] = defaultdict(float)
        group_weights: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for idx, weight in zip(working, weights):
            answer = tuple(expected_reply(catalog.slots[idx], disclosed, attribute))
            groups[answer] += weight
            group_weights[answer].append(weight)
        if len(groups) <= 1:
            continue
        residual = 0.0
        for answer, mass in groups.items():
            residual += (mass / total) * _entropy(group_weights[answer])
        gain = base_entropy - residual
        question = Question(attribute, gain, residual, len(groups))
        if gain > best.gain:
            best = question
        if attribute in _CONCRETE and (concrete_best is None or gain > concrete_best.gain):
            concrete_best = question

    if (
        natural
        and concrete_best is not None
        and best.attribute == "other"
        and concrete_best.gain >= best.gain * (1.0 - _NATURALNESS_MARGIN)
    ):
        # A named attribute reads better to a shopper and costs (almost) nothing.
        return concrete_best
    return best


_PHRASING = {
    "category": "Which type of item are you shopping for exactly?",
    "material": "Any material you prefer, or one you would rather avoid?",
    "color": "Is there a colour you have in mind?",
    "size": "What size or fit should I be matching?",
    "style": "What style or cut are you going for?",
    "brand": "Any brand you like, or should I stay open?",
    "budget": "What price range works for you?",
    "feature": "Which feature matters most for this one?",
    "use_case": "Where or how will you be using it?",
    "other": "What matters most to you here? Anything you tell me narrows it down.",
}


def phrase(attribute: str | None, pool_size: int, overloaded: bool) -> str:
    """Customer-facing wording for the chosen question."""
    if attribute is None:
        return "These look like the closest matches for what you described."
    question = _PHRASING.get(attribute, _PHRASING["other"])
    if overloaded:
        return (
            f"That still leaves about {pool_size} options, so let me narrow it down. "
            + question
        )
    return "Here are the closest matches so far. " + question
