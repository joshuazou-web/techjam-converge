"""Inverse user model: replay a transcript against a candidate product.

The customer is a deterministic program over the target product's constraint
card.  So instead of *scoring* similarity between a message and a product, we
ask a sharper question:

    if this product were the hidden target, would the customer have said
    exactly what we heard?

Products that fail the replay are impossible, not merely unlikely.  The set of
survivors is an exact posterior, and ranking only has to break ties inside it.
The same replay, run forward, tells us what each candidate customer *would*
answer next -- which is how :mod:`converge.policy` estimates the information
gain of a question before asking it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import classify_constraint

# ``classify_constraint`` is called for every (candidate, attribute) pair during
# information-gain search; the constraint vocabulary is small and highly
# repetitive, so memoising it removes the dominant cost of the search.
_CLASS_CACHE: dict[str, str] = {}


def attribute_of(constraint: str) -> str:
    channel = _CLASS_CACHE.get(constraint)
    if channel is None:
        channel = classify_constraint(constraint)
        _CLASS_CACHE[constraint] = channel
    return channel


@dataclass
class Turn:
    """One question we asked and the answer we heard."""

    attribute: str | None
    reply_kind: str
    constraints: list[str] = field(default_factory=list)


@dataclass
class Transcript:
    """Everything observed in a session, in a replayable form."""

    scenario: str = "unknown"
    category: str | None = None
    #: constraint disclosed by the opening message of a Buying session
    opening: str | None = None
    #: preference stated (and later overridden) in an Intent Override session
    stated: str | None = None
    #: constraint asserted by the override message
    overridden: str | None = None
    turns: list[Turn] = field(default_factory=list)

    def observed_atoms(self) -> list[str]:
        atoms: list[str] = []
        for value in (self.opening, self.stated, self.overridden):
            if value:
                atoms.append(value)
        for turn in self.turns:
            atoms.extend(turn.constraints)
        return list(dict.fromkeys(atoms))

    def disclosed_atoms(self) -> set[str]:
        """Constraints the simulator has already marked as spent."""
        spent: set[str] = set()
        if self.opening:
            spent.add(self.opening)
        if self.overridden:
            spent.add(self.overridden)
        for turn in self.turns:
            spent.update(turn.constraints)
        return spent


def expected_reply(constraints: tuple[str, ...], disclosed: set[str], attribute: str) -> list[str]:
    """What the customer discloses for ``attribute`` given what is already spent."""
    matches: list[str] = []
    for value in constraints:
        if value in disclosed:
            continue
        if attribute != "other" and attribute_of(value) != attribute:
            continue
        matches.append(value)
        if len(matches) == 2:
            break
    return matches


def is_consistent(constraints: tuple[str, ...], transcript: Transcript) -> bool:
    """Replay ``transcript`` assuming ``constraints`` belongs to the target."""
    if not constraints:
        return False
    disclosed: set[str] = set()

    if transcript.opening is not None:
        if constraints[0] != transcript.opening:
            return False
        disclosed.add(transcript.opening)
    if transcript.stated is not None:
        soft = constraints[2:] or constraints[:1]
        if soft[-1] != transcript.stated:
            return False
    if transcript.overridden is not None and constraints[0] != transcript.overridden:
        return False

    boundary_used = False
    for turn in transcript.turns:
        if turn.reply_kind == "override":
            # The override message pre-empts the answer to that question.
            disclosed.add(constraints[0])
            continue
        if turn.reply_kind == "boundary":
            if boundary_used:
                return False
            boundary_used = True
            continue
        if turn.attribute is None or turn.reply_kind in {"nudge", "freeform"}:
            continue
        expected = expected_reply(constraints, disclosed, turn.attribute)
        if turn.reply_kind == "exhausted":
            if expected:
                return False
            continue
        if turn.reply_kind == "disclose":
            if expected != turn.constraints:
                return False
            disclosed.update(expected)
            continue
    return True


def soft_agreement(constraints: tuple[str, ...], atoms: list[str]) -> float:
    """Fraction of observed constraints a product can account for.

    Used only when the exact posterior collapses to nothing -- a paraphrase, a
    mis-segmented reply, or a contract drift.  Degrading to a graded score keeps
    the agent useful instead of returning an empty list.
    """
    if not atoms:
        return 0.0
    present = set(constraints)
    return sum(1.0 for atom in atoms if atom in present) / len(atoms)
