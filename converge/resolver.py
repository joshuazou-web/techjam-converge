"""Approximate category grounding.

When the customer's opening line matches the published template, the category
phrase is recovered exactly and the bucket is unambiguous.  When it does not --
a paraphrase, a human typing free text into the demo, a future template change
-- we still need somewhere to search.  This resolver scores every category
phrase in the catalog against the content words seen so far using inverse
document frequency, so rare words like "espadrille" outweigh common ones like
"women".

It is a fallback, never the primary path: it only runs when exact grounding
produced nothing.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .catalog import Catalog

_TOKEN_RE = re.compile(r"[a-z0-9]+")
#: Minimum evidence before we commit to a bucket; below this we prefer to
#: search the whole catalog rather than anchor on a wrong guess.
_MIN_SCORE = 1.2
#: How hard to punish a category for words the shopper never said.
_MISS_PENALTY = 0.5


class CategoryResolver:
    """IDF-weighted nearest category phrase."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.tokens: dict[str, list[str]] = {}
        for category in catalog.by_category:
            terms = _TOKEN_RE.findall(category.lower())
            self.tokens[category] = terms
            for term in set(terms):
                self.postings[term].add(category)
        total = max(len(self.tokens), 1)
        self.idf = {
            term: math.log(1 + total / len(categories))
            for term, categories in self.postings.items()
        }

    def resolve(self, terms: list[str]) -> str | None:
        """Best-matching category phrase, or ``None`` when the evidence is thin.

        Scoring rewards the IDF mass a category shares with the observed words
        and penalises the mass it asks for and did not get, so "Shirts T-Shirts"
        beats "Women Shirts" for a shopper who never said "women".
        """
        observed = {token for term in terms for token in _TOKEN_RE.findall(term.lower())}
        matched: dict[str, float] = defaultdict(float)
        for token in observed:
            weight = self.idf.get(token)
            if weight is None:
                continue
            for category in self.postings[token]:
                matched[category] += weight
        if not matched:
            return None

        def rank(item: tuple[str, float]) -> tuple[float, str]:
            category, gained = item
            missing = sum(
                self.idf.get(token, 0.0)
                for token in set(self.tokens[category])
                if token not in observed
            )
            return (gained - _MISS_PENALTY * missing, category)

        best, _ = max(matched.items(), key=rank)
        return best if rank((best, matched[best]))[0] >= _MIN_SCORE else None
