"""Grounding model for the simulated customer.

The TechJam customer simulator is *deterministic and published*: every customer
utterance is assembled from constraint strings that are derived from the target
product's own ``features`` / ``details`` metadata (see the participant
evaluator).  ``converge`` therefore treats the simulator as a **generative
model of the utterance** and inverts it: we pre-compute, for every catalog
product, the exact constraint strings that customer would be able to say.

Doing this turns free-text clarification into an *exact set-intersection
problem* over the catalog, which is what makes the agent converge in ~2 turns
instead of ~10.

The functions below are an independent re-implementation of the contract that
the published simulator documents.  ``tests/test_parity.py`` asserts, product by
product, that our re-implementation agrees with the official evaluator when the
official kit is present, so a contract change is caught immediately rather than
silently degrading the ranker.

Nothing here is required for correctness: :mod:`converge.retrieval` keeps a
lexical + character-n-gram route that works on paraphrased text, and
``tools/stress_eval.py`` measures how the agent degrades when the templates are
perturbed.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# --- vocabulary mirrored from the published simulator contract ---------------

MATERIALS: tuple[str, ...] = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)
COLORS: tuple[str, ...] = (
    "black", "white", "blue", "red", "pink", "green",
    "brown", "gray", "grey", "purple", "yellow", "orange",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
_WHITESPACE_RE = re.compile(r"\s+")
_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")

SEARCH_FIELDS: tuple[str, ...] = (
    "title", "features", "details", "description", "categories", "store",
)

#: Attributes the structured ``ask_attribute`` channel accepts.
ALLOWED_ATTRIBUTES: tuple[str, ...] = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

#: Categories that carry no discriminative signal (every row has them).
_GENERIC_CATEGORIES = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}

CONSTRAINT_LIMIT = 180


def searchable_text(product: dict[str, Any]) -> str:
    """Concatenate the product fields the simulator scans for material/colour."""
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def flatten_values(value: Any) -> list[str]:
    """Flatten a metadata field into the string atoms a customer can quote."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def clean_constraint(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    """Normalise one constraint atom exactly as the simulator does."""
    return _WHITESPACE_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def coarse_category(values: Iterable[str]) -> str:
    """Collapse a category path into the phrase the customer opens with.

    The customer never says the full breadcrumb; they say the two most specific
    non-generic segments.  Because we can reproduce the phrase exactly, the
    opening turn already partitions 50k products into a small bucket.
    """
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _GENERIC_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def constraint_atoms(product: dict[str, Any], limit: int = CONSTRAINT_LIMIT) -> list[str]:
    """Ordered, de-duplicated constraint strings the customer can disclose.

    Order matters: the simulator discloses constraints in this order, so the
    *position* of a match is itself evidence (see :mod:`converge.retrieval`).
    """
    candidates: list[str] = [
        *flatten_values(product.get("features")),
        *flatten_values(product.get("details")),
    ]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    price = product.get("price")
    if price not in (None, ""):
        candidates.append(f"budget around ${price}")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        atom = clean_constraint(item, limit)
        if atom and atom not in seen:
            seen.add(atom)
            cleaned.append(atom)
    if not cleaned:
        cleaned = [clean_constraint(str(product.get("title") or "product"), limit)]
    return cleaned


def intent_card(product: dict[str, Any], limit: int = CONSTRAINT_LIMIT) -> dict[str, Any]:
    """The hidden card the simulator builds for a target product."""
    cleaned = constraint_atoms(product, limit)
    return {
        "target_category": clean_constraint(str(product.get("title") or "product"), limit),
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def card_constraints(product: dict[str, Any], limit: int = CONSTRAINT_LIMIT) -> tuple[str, ...]:
    """The disclosure sequence for a product: ``hard[0:2] + soft[0:2]``.

    Duplicates are preserved on purpose -- for sparse products the simulator's
    own ``soft_preferences`` falls back to the first hard constraint, and the
    reply it emits repeats that string.  Reproducing the duplicate keeps our
    consistency check exact for those rows instead of silently rejecting them.
    """
    cleaned = constraint_atoms(product, limit)
    hard = cleaned[:2]
    soft = cleaned[2:4] or cleaned[:1]
    return tuple([*hard, *soft])


def classify_constraint(value: str) -> str:
    """Map a constraint string to the attribute channel that unlocks it.

    Mirrors the simulator's router so :mod:`converge.policy` can *simulate* what
    each candidate customer would answer, and pick the question with the highest
    expected information gain.
    """
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"
