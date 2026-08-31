"""Utterance understanding.

Two layers, tried in order:

1. **Contract parsing** -- the customer's surface forms are published, so the
   first layer recovers the *exact* constraint strings with zero loss.  Exact
   strings are what make the posterior in :mod:`converge.state` a set
   intersection rather than a similarity score.
2. **Open-text extraction** -- when nothing matches (a paraphrase, a human
   typing into the demo CLI, a future contract change), we fall back to
   phrase/keyword extraction so the lexical route still has something to work
   with.  ``tools/stress_eval.py`` exercises this path.

The parser never raises: an unparseable turn degrades to
``Utterance(kind="freeform")``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .cards import ALLOWED_ATTRIBUTES

_OPENING = "I'm looking for "
_BROWSING_TAIL = ", but I'm still exploring."
_BUYING_MARKER = "A key requirement is: "
_REPLY_MARKER = "For that, what matters is: "
_OVERRIDE_MARKER = "What I need is: "
_NO_MORE_RE = re.compile(r"^I don't have an additional preference for ([a-z_]+)\.$")
_BOUNDARY_RE = re.compile(r"^I don't have a preference for ([a-z_]+); please use your judgment\.$")
_NUDGE = "Ask me about one specific attribute"

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "have",
    "i", "in", "is", "it", "me", "my", "not", "of", "on", "or", "please", "some",
    "still", "that", "the", "this", "to", "want", "with", "would", "you", "your",
    "looking", "exploring", "need", "prefer", "preference", "key", "requirement",
    "matters", "actually", "ignore", "earlier", "options", "quite", "right", "yet",
    "ask", "about", "one", "specific", "attribute", "judgment", "additional",
}


@dataclass
class Utterance:
    """Structured view of one customer turn."""

    kind: str
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    attribute: str | None = None
    keywords: list[str] = field(default_factory=list)
    raw: str = ""


def keywords_of(text: str, limit: int = 40) -> list[str]:
    """Content tokens, used by the lexical fallback route."""
    tokens = [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]
    return list(dict.fromkeys(tokens))[:limit]


#: How many leading filler words a paraphrase may prepend before the quote.
_MAX_FILLER_WORDS = 8
#: Shorter spans than this are too ambiguous to trust as a quote.
_MIN_QUOTE_CHARS = 4
_SEGMENT_RE = re.compile(r"[;•]|(?<=\.)\s")


def recover_atoms(text: str, resolve: Callable[[str], str | None]) -> list[str]:
    """Find catalog constraints quoted inside free text.

    A paraphrase usually leaves the *quote* intact and only rewrites the frame
    around it ("it really has to be 100% Cotton"). So instead of matching a
    template we walk in from the left of each segment and ask the catalog
    lexicon whether the remainder is a real constraint. That recovers exact,
    indexable constraints from wording the parser has never seen.

    The scan is deliberately conservative -- suffixes only, first match wins.
    An earlier version searched every span for the longest match; it recovered
    more quotes but its false positives narrowed the posterior onto the wrong
    product and cost ~0.10 TechnicalScore under paraphrase. Precision matters
    more than recall here, because a wrong constraint is not merely unhelpful:
    it eliminates the right answer.
    """
    found: list[str] = []
    for segment in _SEGMENT_RE.split(text):
        words = segment.strip().split()
        for start in range(min(len(words), _MAX_FILLER_WORDS + 1)):
            candidate = " ".join(words[start:]).strip(" -;,.	")
            if len(candidate) < _MIN_QUOTE_CHARS:
                break
            canonical = resolve(candidate)
            if canonical is not None:
                found.append(canonical)
                break
    return list(dict.fromkeys(found))


def _split_disclosure(payload: str, is_atom: Callable[[str], bool]) -> list[str]:
    """Recover the 1-2 constraint strings joined into one reply.

    The simulator joins with ``"; "`` but a single constraint may itself contain
    ``"; "``, so the split is ambiguous.  We resolve it against the catalog:
    a segmentation whose parts are real catalog constraints wins; otherwise we
    keep the whole payload (and the naive split) as competing hypotheses.
    """
    payload = payload.strip()
    if payload.endswith("."):
        payload = payload[:-1]
    if not payload:
        return []
    if is_atom(payload):
        return [payload]
    parts = payload.split("; ")
    for cut in range(1, len(parts)):
        left = "; ".join(parts[:cut])
        right = "; ".join(parts[cut:])
        if is_atom(left) and is_atom(right):
            return [left, right]
    if len(parts) == 2:
        return parts
    return [payload]


def parse(message: str, resolve: Callable[[str], str | None] | None = None) -> Utterance:
    """Parse one customer message into an :class:`Utterance`.

    ``resolve`` maps a candidate string to the canonical catalog constraint (or
    ``None``); it lets the parser both segment ambiguous replies and recover
    quotes from paraphrased text.
    """
    text = (message or "").strip()
    lookup = resolve or (lambda _value: None)
    check = lambda value: lookup(value) is not None  # noqa: E731
    keywords = keywords_of(text)

    if text.startswith(_OPENING):
        body = text[len(_OPENING):]
        if body.endswith(_BROWSING_TAIL):
            return Utterance("open_browse", category=body[: -len(_BROWSING_TAIL)],
                             keywords=keywords, raw=text)
        category, separator, rest = body.partition(". ")
        if not separator:
            return Utterance("open_browse", category=body.rstrip("."), keywords=keywords, raw=text)
        if rest.startswith(_BUYING_MARKER):
            constraint = rest[len(_BUYING_MARKER):]
            return Utterance("open_buy", category=category,
                             constraints=_split_disclosure(constraint, check),
                             keywords=keywords, raw=text)
        # Neither template matched: the customer stated a preference that will
        # later be overridden.
        return Utterance("open_stated", category=category,
                         constraints=_split_disclosure(rest, check),
                         keywords=keywords, raw=text)

    if _OVERRIDE_MARKER in text:
        payload = text.split(_OVERRIDE_MARKER, 1)[1]
        return Utterance("override", constraints=_split_disclosure(payload, check),
                         keywords=keywords, raw=text)

    if text.startswith(_REPLY_MARKER):
        payload = text[len(_REPLY_MARKER):]
        return Utterance("disclose", constraints=_split_disclosure(payload, check),
                         keywords=keywords, raw=text)

    match = _BOUNDARY_RE.match(text)
    if match:
        return Utterance("boundary", attribute=_attribute(match.group(1)), raw=text)

    match = _NO_MORE_RE.match(text)
    if match:
        return Utterance("exhausted", attribute=_attribute(match.group(1)), raw=text)

    if _NUDGE in text:
        return Utterance("nudge", raw=text)

    return Utterance("freeform", constraints=recover_atoms(text, lookup),
                     keywords=keywords, raw=text)


def _attribute(value: str) -> str:
    return value if value in ALLOWED_ATTRIBUTES else "other"
