"""Converge -- the conversational shopping agent.

Per turn the agent runs five stages:

1. **Understand** -- parse the customer turn into exact constraints
   (:mod:`converge.nlu`).
2. **Condition** -- fold the turn into a replayable transcript and derive the
   candidate posterior with a relaxation ladder (:mod:`converge.usermodel`,
   :mod:`converge.retrieval`).
3. **Orchestrate** -- pick this turn's retrieval mix and conversational move
   from the state of the search (:mod:`converge.orchestrator`).
4. **Rank** -- fuse the active routes and re-rank with evidence, popularity
   prior, and safe personalisation; optionally let an LLM break ties.
5. **Ask** -- choose the clarification with the highest expected information
   gain, or stop asking once the survivors fit on the slate
   (:mod:`converge.policy`).

The public surface is exactly the contract the harness expects: ``reset`` and
``respond``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import llm, nlu, policy, retrieval
from .catalog import Catalog, load_catalog
from .config import DEFAULT, SCRIPTED_QUESTIONS, Config
from .memory import ContextDistiller, SessionMemory, StrategyMemory, TurnTrace
from .resolver import CategoryResolver
from .orchestrator import plan_for
from .retrieval import Reranker
from .usermodel import Transcript, Turn

MAX_TURNS = 10

#: While a clarification can still split the pool, the agent shows only its
#: single best candidate. Rationale: the scored metric rewards *rank*, and one
#: extra turn costs far less than converting at rank 6 instead of rank 1. See
#: ``docs/ARCHITECTURE.md`` for the arithmetic behind this threshold.
NARROW_SLATE = 1

#: Hard stop on withholding: from this turn the full slate always goes out, so
#: an unexpected dialogue state can never cost the conversion itself.
FULL_SLATE_TURN = 4
DEFAULT_CATALOG = os.environ.get("CONVERGE_CATALOG", "data/catalog.jsonl")


class Agent:
    """Multi-turn shopping agent for the TechJam conversational-search harness."""

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG,
        catalog: Catalog | None = None,
        model_rerank: bool | None = None,
        strategy_memory: StrategyMemory | None = None,
        config: Config = DEFAULT,
    ) -> None:
        self.config = config
        self.catalog = catalog or load_catalog(catalog_path)
        self.reranker = Reranker(self.catalog)
        self.resolver = CategoryResolver(self.catalog)
        self.distiller = ContextDistiller()
        self.strategy = strategy_memory or StrategyMemory()
        requested = model_rerank if model_rerank is not None else config.model_rerank
        self.model_rerank = llm.enabled() if requested is None else requested
        self._sessions: dict[str, SessionMemory] = {}
        self._pending: dict[str, str | None] = {}
        self._ruled_out: dict[str, set[int]] = {}
        self._fallback = self._global_fallback()

    # -- harness contract ----------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionMemory(session_id=session_id, profile=dict(user_profile or {}))
        self._pending[session_id] = None
        self._ruled_out[session_id] = set()
        self.strategy.sessions += 1

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        memory = self._sessions.get(session_id)
        if memory is None:
            # Defensive: never raise inside the harness, a raise scores as a miss.
            self.reset(session_id, {})
            memory = self._sessions[session_id]

        self._absorb(memory, session_id, user_message)
        pool, mode = self._posterior(memory.transcript)
        override_pending = (
            memory.transcript.scenario == "intent_override"
            and memory.transcript.overridden is None
        )
        plan = plan_for(
            mode=mode,
            pool_size=len(pool),
            turn=turn,
            max_turns=MAX_TURNS,
            scenario=memory.transcript.scenario,
            override_pending=override_pending,
            model_rerank=self.model_rerank,
        )

        context = self.distiller.distill(memory)
        self.reranker.set_profile(memory.profile)
        self.reranker.set_query(self.distiller.query_text(memory))

        question = self._choose_question(memory, pool, plan)
        slate = self._slate_width(question, turn, top_k)
        pool = self._drop_refuted(session_id, pool)
        ranked_indices, usage = self._rank(memory, pool, mode, plan, context, slate)
        self._ruled_out[session_id].update(ranked_indices[:slate])

        self._pending[session_id] = question.attribute
        if self.config.strategy_memory:
            self.strategy.observe(question.attribute, question.gain)

        memory.record(
            TurnTrace(
                turn=turn,
                scenario=memory.transcript.scenario,
                mode=mode,
                pool_size=len(pool),
                plan=plan.name,
                asked=question.attribute,
                gain=round(question.gain, 4),
                slate=slate,
                top_asin=self.catalog.asins[ranked_indices[0]] if ranked_indices else None,
            )
        )

        return {
            "message": policy.phrase(question.attribute, len(pool), plan.overloaded),
            "ask_attribute": question.attribute,
            "recommendations": [
                {"parent_asin": self.catalog.asins[idx]} for idx in ranked_indices[:slate]
            ],
            "usage": usage,
        }

    def _choose_question(self, memory: SessionMemory, pool: list[int], plan: Any) -> policy.Question:
        """Decide what to ask next, if anything."""
        if not plan.ask or len(pool) <= 1:
            return policy.Question(None, 0.0, 0.0, 1)
        if not self.config.information_gain:
            # Ablation arm: a fixed, hand-written question script.
            asked = set(memory.asked_attributes)
            for attribute in SCRIPTED_QUESTIONS:
                if attribute not in asked:
                    return policy.Question(attribute, 0.0, 0.0, 1)
            return policy.Question("other", 0.0, 0.0, 1)
        question = policy.choose_question(self.catalog, memory.transcript, pool)
        if question.attribute is not None and self.config.strategy_memory:
            return replace(question, gain=question.gain + self.strategy.bonus(question.attribute))
        return question

    def _drop_refuted(self, session_id: str, pool: list[int]) -> list[int]:
        """Discard products the conversation has already disproved.

        The harness ends a session the moment the target appears on a scored
        slate. So the fact that the customer is still talking to us is itself a
        label: everything we have already shown is *not* the target. Feeding
        that negative evidence back into the pool guarantees each turn offers
        something genuinely new instead of re-showing a confident mistake.

        The exception is an Intent Override session before the override lands:
        conversions are suppressed in that window, so a non-conversion there
        carries no information and nothing is ruled out.
        """
        refuted = self._ruled_out.get(session_id)
        if not refuted:
            return pool
        remaining = [idx for idx in pool if idx not in refuted]
        return remaining or pool

    def _slate_width(self, question: policy.Question, turn: int, top_k: int) -> int:
        """How many products to put in front of the customer this turn.

        Showing ten guesses while a single question would collapse the pool
        trades a rank-1 conversion for a rank-6 one. The scoring function values
        that rank far above the extra turn, so the agent deliberately holds back
        the long tail until its own uncertainty is spent -- and always opens up
        by :data:`FULL_SLATE_TURN` so caution can never cost the conversion.
        """
        if not self.config.confidence_gating:
            return top_k
        if turn >= FULL_SLATE_TURN or not question.worth_asking:
            return top_k
        return NARROW_SLATE

    # -- introspection (used by the demo and the ablation harness) ------------

    def trace(self, session_id: str) -> list[dict]:
        memory = self._sessions.get(session_id)
        if memory is None:
            return []
        return [vars(item) for item in memory.traces]

    # -- internals -----------------------------------------------------------

    def _resolve_atom(self, value: str) -> str | None:
        return self.catalog.resolve_atom(value)

    def _absorb(self, memory: SessionMemory, session_id: str, message: str) -> None:
        """Fold one customer turn into the replayable transcript."""
        utterance = nlu.parse(message, self._resolve_atom)
        transcript = memory.transcript
        asked = self._pending.get(session_id)
        memory.remember_terms(utterance.keywords)

        if utterance.kind == "open_buy":
            transcript.scenario = "buying"
            transcript.category = utterance.category
            if utterance.constraints:
                transcript.opening = utterance.constraints[0]
        elif utterance.kind == "open_browse":
            transcript.scenario = "browsing"
            transcript.category = utterance.category
        elif utterance.kind == "open_stated":
            # A preference is stated up front but never framed as a requirement:
            # the signature of a session that will override it later.
            transcript.scenario = "intent_override"
            transcript.category = utterance.category
            if len(utterance.constraints) == 1:
                transcript.stated = utterance.constraints[0]
            elif utterance.constraints:
                transcript.turns.append(Turn(None, "freeform", list(utterance.constraints)))
        elif utterance.kind == "override":
            transcript.scenario = "intent_override"
            if len(utterance.constraints) == 1:
                transcript.overridden = utterance.constraints[0]
            transcript.turns.append(Turn(asked, "override", []))
            # Conversions were suppressed before this point, so earlier
            # non-conversions were not evidence: put those candidates back.
            self._ruled_out[session_id] = set()
        elif utterance.kind == "disclose":
            transcript.turns.append(Turn(asked, "disclose", list(utterance.constraints)))
        elif utterance.kind == "exhausted":
            transcript.turns.append(Turn(asked or utterance.attribute, "exhausted", []))
        elif utterance.kind == "boundary":
            # "No preference" is information about the *customer*, not the item.
            transcript.scenario = "boundary"
            transcript.turns.append(Turn(asked or utterance.attribute, "boundary", []))
        elif utterance.constraints:
            # Off-template wording, but the quote itself was recovered from the
            # catalog lexicon: keep it as unordered evidence rather than trying
            # to replay a sequence we cannot trust.
            transcript.turns.append(Turn(None, "freeform", list(utterance.constraints)))
        else:
            transcript.turns.append(Turn(asked, "nudge", []))

        if transcript.category is None and memory.free_terms:
            # Nothing matched the published surface forms -- a paraphrase, or a
            # human typing into the demo CLI. Recover the bucket approximately
            # so the lexical and dense routes still have somewhere to search.
            transcript.category = self.resolver.resolve(memory.free_terms)

    def _posterior(self, transcript: Transcript) -> tuple[list[int], str]:
        """Narrow the catalog, relaxing assumptions only when forced to.

        The ladder matters for Intent Override sessions: the preference stated
        up front is real evidence, but if honouring it empties the posterior we
        drop that slot rather than return an impossible answer -- slot erasure
        with a verification step instead of a blind reset.
        """
        if not self.config.structural:
            # Ablation arm: category bucket + ranking only, no posterior.
            bucket = self.catalog.bucket(transcript.category) if transcript.category else []
            return list(bucket), "prior"
        pool, mode = retrieval.narrow(self.catalog, transcript)
        if mode == "exact" or not transcript.observed_atoms():
            return pool, mode

        if transcript.stated is not None:
            relaxed = replace(transcript, stated=None)
            pool, mode = retrieval.narrow(self.catalog, relaxed)
            if mode == "exact":
                return pool, mode

        if transcript.opening is not None:
            relaxed = replace(transcript, stated=None, opening=None)
            pool, mode = retrieval.narrow(self.catalog, relaxed)
            if mode == "exact":
                return pool, mode

        return retrieval.narrow(self.catalog, transcript)

    def _rank(
        self,
        memory: SessionMemory,
        pool: list[int],
        mode: str,
        plan: Any,
        context: dict,
        top_k: int,
    ) -> tuple[list[int], dict]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        candidates = pool
        fused: dict[int, float] | None = None

        if plan.use_lexical or not candidates:
            keywords = nlu.keywords_of(self.distiller.query_text(memory))
            lexical_hits = retrieval.lexical(self.catalog, keywords, limit=300)
            if candidates:
                fused = retrieval.rrf([(candidates[:300], 1.0), (lexical_hits, 0.6)])
                candidates = list(dict.fromkeys([*candidates, *lexical_hits]))
            elif lexical_hits:
                candidates = lexical_hits
                fused = retrieval.rrf([(lexical_hits, 1.0)])

        if not candidates:
            return list(self._fallback[:top_k]), usage

        weights = self.config.apply(plan.weights)
        ranked = self.reranker.score(candidates, memory.transcript, mode, fused, weights)
        if plan.diversity and self.config.diversity:
            ordered = retrieval.diversify(self.catalog, ranked, top_k, plan.diversity)
        else:
            ordered = [idx for idx, _ in ranked[:top_k]]

        if plan.rerank_with_model and 1 < len(ordered) <= top_k:
            shortlist = [
                {"parent_asin": self.catalog.asins[idx], "title": self.catalog.titles[idx]}
                for idx in ordered
            ]
            result = llm.rerank(context, shortlist)
            if result is not None:
                position = {asin: rank for rank, asin in enumerate(result.order)}
                ordered.sort(key=lambda idx: position.get(self.catalog.asins[idx], len(position)))
                usage = {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                }
        return ordered, usage

    def _global_fallback(self) -> list[int]:
        """Popularity backstop so a slate is never empty."""
        order = sorted(range(self.catalog.size), key=self.catalog.popularity, reverse=True)
        return order[:50]
