"""Multi-route retrieval and semantic re-ranking.

Three routes, fused per turn by the orchestrator:

``structural``
    exact posterior from the inverse user model -- high precision, the route
    that drives Buying and post-clarification turns.
``lexical``
    SQLite FTS5 BM25 over the full product text -- the recall net for
    paraphrased or unparseable turns.
``dense``
    character-3-gram cosine computed in-memory over the active shortlist --
    cheap semantic smoothing that needs no model and no vector service.

Fusion is weighted Reciprocal Rank Fusion, so routes stay comparable without
score calibration. The final ordering comes from a re-ranker that mixes replay
evidence, a purchase-volume prior, and safe personalisation from the anonymised
profile.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .catalog import Catalog
from .usermodel import Transcript, is_consistent, soft_agreement

_NGRAM = 3


def _ngrams(text: str) -> Counter:
    padded = " " + text.lower().strip() + " "
    if len(padded) <= _NGRAM:
        return Counter([padded])
    return Counter(padded[i : i + _NGRAM] for i in range(len(padded) - _NGRAM + 1))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(count * large.get(gram, 0) for gram, count in small.items())
    if dot == 0:
        return 0.0
    norm_l = math.sqrt(sum(value * value for value in left.values()))
    norm_r = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_l * norm_r)


def narrow(catalog: Catalog, transcript: Transcript) -> tuple[list[int], str]:
    """Return the candidate posterior and the mode that produced it.

    ``exact``  survivors of the replay: the product is *possible*, not merely
    similar.
    ``soft``   the replay eliminated everything, so fall back to graded
    evidence rather than returning nothing.
    ``prior``  nothing to condition on yet, e.g. the first Browsing turn.
    """
    base: list[int] | None = None
    if transcript.category:
        bucket = catalog.bucket(transcript.category)
        if bucket:
            base = bucket

    atoms = transcript.observed_atoms()
    if atoms:
        postings = [catalog.atom_postings.get(atom, []) for atom in atoms]
        postings = [posting for posting in postings if posting]
        if postings:
            postings.sort(key=len)
            pool = set(postings[0])
            for posting in postings[1:]:
                pool &= set(posting)
                if not pool:
                    break
            if base is not None:
                pool &= set(base)
            survivors = [idx for idx in pool if is_consistent(catalog.slots[idx], transcript)]
            if survivors:
                survivors.sort()
                return survivors, "exact"

    if base is None:
        return [], "prior"
    if not atoms:
        return base, "prior"

    survivors = [idx for idx in base if is_consistent(catalog.slots[idx], transcript)]
    if survivors:
        return survivors, "exact"
    return base, "soft"


def lexical(catalog: Catalog, keywords: Sequence[str], limit: int = 300) -> list[int]:
    """BM25 recall net over the whole catalog."""
    return catalog.search_text(list(keywords), limit=limit)


def rrf(routes: Iterable[tuple[Sequence[int], float]], k: int = 60) -> dict[int, float]:
    """Weighted Reciprocal Rank Fusion across heterogeneous routes."""
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in routes:
        if weight <= 0:
            continue
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += weight / (k + rank)
    return fused


class Reranker:
    """Evidence-weighted final ordering over a shortlist."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._profile_terms: list[str] = []
        self._query_vector: Counter = Counter()

    def set_profile(self, profile: dict) -> None:
        tags = profile.get("preference_tags") or []
        self._profile_terms = [str(tag).lower() for tag in tags]

    def set_query(self, text: str) -> None:
        self._query_vector = _ngrams(text) if text else Counter()

    def _affinity(self, idx: int) -> float:
        """Safe personalisation: aggregate preference tags only, never identity."""
        if not self._profile_terms:
            return 0.0
        haystack = (self.catalog.titles[idx] + " " + " ".join(self.catalog.slots[idx])).lower()
        hits = sum(1 for term in self._profile_terms if term in haystack)
        return hits / len(self._profile_terms)

    def score(
        self,
        indices: Sequence[int],
        transcript: Transcript,
        mode: str,
        fused: dict[int, float] | None = None,
        weights: dict[str, float] | None = None,
        semantic_budget: int = 400,
    ) -> list[tuple[int, float]]:
        weights = weights or {}
        w_prior = weights.get("prior", 1.0)
        w_fusion = weights.get("fusion", 1.0)
        w_profile = weights.get("profile", 0.25)
        w_rating = weights.get("rating", 0.15)
        w_soft = weights.get("soft", 2.0)
        w_semantic = weights.get("semantic", 0.35)

        atoms = transcript.observed_atoms()
        fused = fused or {}
        # Stage 1: cheap ordering, so the expensive signals only touch a shortlist.
        coarse = sorted(
            indices,
            key=lambda idx: (
                (soft_agreement(self.catalog.slots[idx], atoms) if mode == "soft" else 0.0),
                fused.get(idx, 0.0),
                self.catalog.popularity(idx),
            ),
            reverse=True,
        )[:semantic_budget]

        scored: list[tuple[int, float]] = []
        for idx in coarse:
            value = w_prior * self.catalog.popularity(idx)
            value += w_rating * (self.catalog.ratings[idx] / 5.0)
            value += w_profile * self._affinity(idx)
            if fused:
                value += w_fusion * fused.get(idx, 0.0)
            if mode == "soft":
                value += w_soft * soft_agreement(self.catalog.slots[idx], atoms)
            if w_semantic and self._query_vector:
                value += w_semantic * _cosine(self._query_vector, _ngrams(self.catalog.titles[idx]))
            scored.append((idx, value))
        scored.sort(key=lambda item: (-item[1], self.catalog.asins[item[0]]))
        return scored


def diversify(
    catalog: Catalog,
    ranked: Sequence[tuple[int, float]],
    top_k: int,
    strength: float = 0.35,
) -> list[int]:
    """Maximal-marginal-relevance selection for open-ended Browsing turns.

    With a wide candidate pool, ten near-duplicate listings waste the slate; a
    spread across stores and titles covers more of the latent intent.
    """
    if strength <= 0:
        return [idx for idx, _ in ranked[:top_k]]
    selected: list[int] = []
    chosen: list[Counter] = []
    seen_stores: set[str] = set()
    for idx, _score in ranked[: top_k * 12]:
        if len(selected) >= top_k:
            break
        vector = _ngrams(catalog.titles[idx])
        redundancy = max((_cosine(vector, other) for other in chosen), default=0.0)
        store = catalog.stores[idx].lower()
        if store and store in seen_stores:
            redundancy = max(redundancy, 0.6)
        if redundancy > 1.0 - strength and len(ranked) > top_k * 2:
            continue
        selected.append(idx)
        chosen.append(vector)
        if store:
            seen_stores.add(store)
    for idx, _score in ranked:
        if len(selected) >= top_k:
            break
        if idx not in selected:
            selected.append(idx)
    return selected[:top_k]
