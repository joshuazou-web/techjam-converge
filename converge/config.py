"""Feature switches.

Every component of the pipeline can be turned off independently so that
``tools/ablation.py`` can measure what each one is actually worth on the public
set, rather than asserting it in prose. The defaults are the submitted
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Toggles for the retrieval, ranking, and dialogue stages."""

    #: exact posterior from the inverse user model
    structural: bool = True
    #: purchase-volume prior over the catalog
    prior: bool = True
    #: safe personalisation from the anonymised preference tags
    profile: bool = True
    #: character-n-gram semantic smoothing
    semantic: bool = True
    #: MMR spread on wide, open-ended slates
    diversity: bool = True
    #: expected-information-gain question selection (vs. a fixed script)
    information_gain: bool = True
    #: hold back the long tail of the slate while a question can still split
    #: the candidate pool
    confidence_gating: bool = True
    #: cross-session strategy adaptation
    strategy_memory: bool = True
    #: optional LLM tie-breaker (off unless credentials are configured)
    model_rerank: bool | None = None

    def apply(self, weights: dict[str, float]) -> dict[str, float]:
        tuned = dict(weights)
        if not self.prior:
            tuned["prior"] = 0.0
            tuned["rating"] = 0.0
        if not self.profile:
            tuned["profile"] = 0.0
        if not self.semantic:
            tuned["semantic"] = 0.0
        return tuned


DEFAULT = Config()

#: Fixed question script used when information-gain selection is disabled.
SCRIPTED_QUESTIONS = ("category", "material", "color", "style", "feature", "size", "brand", "budget")
