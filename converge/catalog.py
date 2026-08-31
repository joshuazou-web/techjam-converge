"""In-memory catalog index.

Everything runs in process memory with the Python standard library only: no
vector database, no service dependency, no warm-up cost beyond a single pass
over the 50k-row catalog (~5 s, ~350 MB).  That is a deliberate feasibility
choice -- the same index runs unchanged on a laptop, in CI, and in the final
evaluation harness.

Three index structures are built in that single pass:

``by_category``
    exact bucket for the phrase the customer opens with.
``atom_postings``
    inverted index from a constraint string to the products that can utter it
    -- the structural retrieval route.
``slot_postings``
    inverted index keyed by ``(slot position, constraint string)`` -- the same
    evidence, but position-aware, which is what separates a product that merely
    *contains* "cotton" from one whose customer would disclose it *first*.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from .cards import card_constraints, coarse_category, flatten_values


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, "", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class Catalog:
    """Compact, read-only view over the frozen competition catalog."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.asins: list[str] = []
        self.titles: list[str] = []
        self.categories: list[str] = []
        self.stores: list[str] = []
        self.prices: list[float | None] = []
        self.ratings: list[float] = []
        self.rating_counts: list[int] = []
        #: ordered constraint strings the simulated customer can disclose
        self.slots: list[tuple[str, ...]] = []

        self.index_of: dict[str, int] = {}
        self.by_category: dict[str, list[int]] = defaultdict(list)
        self.atom_postings: dict[str, list[int]] = defaultdict(list)
        #: case-folded lookup, so a paraphrased (often lower-cased) quote can
        #: still be matched back to the canonical catalog constraint
        self.atom_by_lower: dict[str, str] = {}
        self.slot_postings: dict[tuple[int, str], list[int]] = defaultdict(list)

        self._connection: sqlite3.Connection | None = None
        self._load()

    # -- construction --------------------------------------------------------

    def _rows(self) -> Iterator[dict[str, Any]]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _load(self) -> None:
        intern: dict[str, str] = {}
        for idx, product in enumerate(self._rows()):
            asin = str(product["parent_asin"])
            atoms = tuple(intern.setdefault(atom, atom) for atom in card_constraints(product))
            category = coarse_category(product.get("categories") or [])
            category = intern.setdefault(category, category)

            self.asins.append(asin)
            self.index_of[asin] = idx
            self.titles.append(str(product.get("title") or ""))
            self.categories.append(category)
            self.stores.append(str(product.get("store") or ""))
            self.prices.append(_as_float(product.get("price")))
            self.ratings.append(_as_float(product.get("average_rating")) or 0.0)
            self.rating_counts.append(_as_int(product.get("rating_number")))
            self.slots.append(atoms)

            self.by_category[category].append(idx)
            for position, atom in enumerate(atoms):
                self.slot_postings[(position, atom)].append(idx)
            for atom in dict.fromkeys(atoms):
                self.atom_postings[atom].append(idx)
                self.atom_by_lower.setdefault(atom.lower(), atom)

        self.size = len(self.asins)
        #: popularity prior; the hidden target is always a *purchased* product,
        #: so review volume is a weak but genuine prior over the catalog.
        self.prior = [math.log1p(count) for count in self.rating_counts]
        self._max_prior = max(self.prior) if self.prior else 1.0

    # -- accessors -----------------------------------------------------------

    def resolve_atom(self, text: str) -> str | None:
        """Canonical catalog constraint for ``text``, or ``None``.

        Exact first, then case-folded: a paraphrase that lower-cases the quote
        still lands on the real constraint, which keeps the posterior exact
        instead of dropping to graded scoring.
        """
        if text in self.atom_postings:
            return text
        return self.atom_by_lower.get(text.lower())

    def popularity(self, idx: int) -> float:
        """Popularity prior normalised to ``[0, 1]``."""
        if self._max_prior <= 0:
            return 0.0
        return self.prior[idx] / self._max_prior

    def bucket(self, category: str) -> list[int]:
        return self.by_category.get(category, [])

    def describe(self, idx: int) -> dict[str, Any]:
        return {
            "parent_asin": self.asins[idx],
            "title": self.titles[idx],
            "category": self.categories[idx],
            "store": self.stores[idx],
            "price": self.prices[idx],
            "average_rating": self.ratings[idx],
            "rating_number": self.rating_counts[idx],
            "constraints": list(self.slots[idx]),
        }

    # -- lexical route (built lazily; only paraphrase/fallback traffic uses it)

    @property
    def fts(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._build_fts()
        return self._connection

    def _build_fts(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        cursor = connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "rowid_alias UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[Any, ...]] = []
        for idx, product in enumerate(self._rows()):
            batch.append(
                (
                    idx,
                    str(product.get("title") or ""),
                    " ".join(flatten_values(product.get("categories"))),
                    " ".join(flatten_values(product.get("features"))),
                    " ".join(flatten_values(product.get("details"))),
                    str(product.get("store") or ""),
                    " ".join(flatten_values(product.get("description"))),
                )
            )
            if len(batch) >= 2000:
                cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        connection.commit()
        return connection

    def search_text(self, terms: list[str], limit: int = 200) -> list[int]:
        """BM25 fallback used when the structural route has nothing to stand on."""
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
        try:
            rows = self.fts.execute(
                "SELECT rowid_alias FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row[0]) for row in rows]


_CACHE: dict[str, Catalog] = {}


def load_catalog(path: str | Path) -> Catalog:
    """Process-level cache so repeated evaluator runs share one index."""
    key = str(Path(path).resolve())
    if key not in _CACHE:
        _CACHE[key] = Catalog(path)
    return _CACHE[key]
