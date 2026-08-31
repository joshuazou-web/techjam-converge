"""Runtime workflow re-orchestration.

The agent does not run one fixed pipeline. Each turn it inspects the state of
the search -- how the candidate pool was produced, how wide it still is, how
many turns remain, and whether an intent override is still pending -- and
selects a plan that changes both the retrieval mix and the conversational move.

Keeping this in one small, declarative place is deliberate: the routing policy
is the part a product team would tune, so it must be readable and testable
rather than smeared across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Above this many survivors the slate is a lottery ticket; clarify instead.
OVERLOAD = 60


@dataclass
class Plan:
    """The pipeline configuration for a single turn."""

    name: str
    weights: dict[str, float] = field(default_factory=dict)
    diversity: float = 0.0
    ask: bool = True
    use_lexical: bool = False
    overloaded: bool = False
    rerank_with_model: bool = False


_COMMIT = {"prior": 3.00, "rating": 0.10, "profile": 0.30, "semantic": 0.45, "fusion": 0.6}
_FOCUS = {"prior": 3.00, "rating": 0.15, "profile": 0.30, "semantic": 0.40, "fusion": 0.8}
_EXPLORE = {"prior": 3.00, "rating": 0.25, "profile": 0.45, "semantic": 0.25, "fusion": 0.8}
_RECOVER = {"prior": 1.50, "rating": 0.10, "profile": 0.25, "semantic": 0.60, "fusion": 1.4, "soft": 2.5}


def plan_for(
    mode: str,
    pool_size: int,
    turn: int,
    max_turns: int,
    scenario: str,
    override_pending: bool,
    model_rerank: bool = False,
) -> Plan:
    """Choose this turn's retrieval mix and conversational move."""
    turns_left = max_turns - turn

    if mode == "soft":
        # The exact posterior collapsed: widen with lexical recall and keep talking.
        return Plan("recover", dict(_RECOVER), diversity=0.2, ask=True, use_lexical=True,
                    overloaded=pool_size > OVERLOAD, rerank_with_model=model_rerank)

    if mode == "exact" and pool_size <= 10:
        # Everything that survives fits on one slate. We still keep asking while
        # a question can split the pool: converting at rank 1 next turn is worth
        # far more than converting at rank 6 now (see converge.slate).
        return Plan("commit", dict(_COMMIT), diversity=0.0, ask=turns_left > 0,
                    rerank_with_model=model_rerank and pool_size > 1)

    if mode == "exact":
        return Plan("focus", dict(_FOCUS), diversity=0.0, ask=turns_left > 0,
                    overloaded=pool_size > OVERLOAD, rerank_with_model=model_rerank)

    # No constraints yet: an open-ended Browsing slate should span the bucket
    # rather than stack ten variants of the same listing.
    return Plan(
        "explore",
        dict(_EXPLORE),
        diversity=0.35 if scenario != "buying" else 0.15,
        ask=turns_left > 0,
        use_lexical=pool_size == 0,
        overloaded=pool_size > OVERLOAD,
    )
