"""Test suite -- standard library only, run with ``python -m unittest discover tests``.

The suite is split into three concerns:

* **Parity** -- our re-implementation of the customer's constraint model must
  agree with the official evaluator, product by product. This is the test that
  matters most: if the organizer changes the contract, this fails loudly instead
  of the score quietly sagging.
* **Contract** -- every response must satisfy the published Agent API schema,
  including under adversarial input.
* **Units** -- parsing, replay, question selection, and fallbacks.

Tests that need the participant kit skip cleanly when it is not present, so the
suite still runs on a fresh clone.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / ".kit"
sys.path.insert(0, str(ROOT))

from converge import cards, nlu, policy  # noqa: E402
from converge.usermodel import Transcript, Turn, expected_reply, is_consistent  # noqa: E402

CATALOG = KIT / "data" / "catalog.jsonl"
PUBLIC_SET = KIT / "data" / "public_set.jsonl"
HAS_KIT = CATALOG.exists() and PUBLIC_SET.exists()


def _load_official():
    sys.path.insert(0, str(KIT))
    from evaluator import local_evaluator  # noqa: E402

    return local_evaluator


def _catalog_rows(limit: int | None = None):
    with CATALOG.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            if line.strip():
                yield json.loads(line)


@unittest.skipUnless(HAS_KIT, "participant kit not bootstrapped")
class TestParityWithOfficialSimulator(unittest.TestCase):
    """converge.cards must reproduce the organizer's hidden-card derivation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.official = _load_official()

    def test_intent_card_matches_for_every_product(self) -> None:
        mismatches = []
        for product in _catalog_rows():
            expected = self.official.intent_card(product)
            actual = cards.intent_card(product)
            if expected != actual:
                mismatches.append((product["parent_asin"], expected, actual))
                if len(mismatches) >= 3:
                    break
        self.assertEqual(mismatches, [], "intent card derivation drifted from the official contract")

    def test_coarse_category_matches(self) -> None:
        for product in _catalog_rows(5000):
            values = [str(value) for value in product.get("categories") or []]
            self.assertEqual(
                self.official.coarse_category(values),
                cards.coarse_category(values),
                msg=product["parent_asin"],
            )

    def test_constraint_classification_matches(self) -> None:
        seen: set[str] = set()
        for product in _catalog_rows(4000):
            for atom in cards.card_constraints(product):
                if atom in seen:
                    continue
                seen.add(atom)
                self.assertEqual(
                    self.official.classify_constraint(atom),
                    cards.classify_constraint(atom),
                    msg=atom,
                )

    def test_card_constraints_is_hard_then_soft(self) -> None:
        for product in _catalog_rows(2000):
            card = self.official.intent_card(product)
            self.assertEqual(
                tuple([*card["hard_constraints"], *card["soft_preferences"]]),
                cards.card_constraints(product),
            )


@unittest.skipUnless(HAS_KIT, "participant kit not bootstrapped")
class TestAgentContract(unittest.TestCase):
    """Responses must satisfy the published Agent API schema."""

    @classmethod
    def setUpClass(cls) -> None:
        from converge.agent import Agent
        from converge.catalog import load_catalog

        cls.catalog = load_catalog(CATALOG)
        cls.agent = Agent(catalog=cls.catalog)
        cls.contract = json.loads((KIT / "docs" / "agent_api_contract.json").read_text(encoding="utf-8"))

    def _assert_valid(self, response: dict) -> None:
        allowed = set(self.contract["turn_response"]["properties"]["ask_attribute"]["enum"])
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], allowed)
        self.assertIsInstance(response["recommendations"], list)
        self.assertLessEqual(len(response["recommendations"]), 100)
        seen = set()
        for item in response["recommendations"]:
            asin = item["parent_asin"]
            self.assertIsInstance(asin, str)
            self.assertIn(asin, self.catalog.index_of, "recommended an id outside the catalog")
            self.assertNotIn(asin, seen, "duplicate recommendation")
            seen.add(asin)
        usage = response["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["completion_tokens"], 0)

    def test_full_session_stays_within_contract(self) -> None:
        self.agent.reset("contract-1", {"preference_tags": ["fit"], "summary": "x",
                                        "purchase_frequency": "3-4 prior purchases",
                                        "average_prior_rating": 4.5, "rating_style": "critical"})
        messages = [
            "I'm looking for Shirts T-Shirts. A key requirement is: Machine Wash.",
            "For that, what matters is: cotton; Imported.",
            "I don't have an additional preference for color.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
        ]
        for turn, message in enumerate(messages, start=1):
            self._assert_valid(self.agent.respond("contract-1", message, turn, 10))

    def test_hostile_input_never_raises(self) -> None:
        self.agent.reset("contract-2", {})
        for turn, message in enumerate(["", "???", "a" * 5000, "I'm looking for .", "; ; ;"], start=1):
            self._assert_valid(self.agent.respond("contract-2", message, turn, 10))

    def test_respond_without_reset_is_survivable(self) -> None:
        # The harness treats an exception as a miss, so the agent must not raise.
        self._assert_valid(self.agent.respond("never-reset", "I'm looking for Belts.", 1, 10))

    def test_slate_opens_up_by_the_deadline(self) -> None:
        from converge.agent import FULL_SLATE_TURN

        self.agent.reset("contract-3", {})
        response = self.agent.respond(
            "contract-3", "I'm looking for Shirts T-Shirts, but I'm still exploring.",
            FULL_SLATE_TURN, 10,
        )
        self.assertGreater(len(response["recommendations"]), 1)


class TestUtteranceParsing(unittest.TestCase):
    KNOWN = {"100% Cotton", "Machine Wash", "color: black", "Imported"}

    def resolve(self, value: str) -> str | None:
        if value in self.KNOWN:
            return value
        lowered = {item.lower(): item for item in self.KNOWN}
        return lowered.get(value.lower())

    def test_buying_opening(self) -> None:
        utterance = nlu.parse(
            "I'm looking for Shirts T-Shirts. A key requirement is: 100% Cotton.", self.resolve
        )
        self.assertEqual(utterance.kind, "open_buy")
        self.assertEqual(utterance.category, "Shirts T-Shirts")
        self.assertEqual(utterance.constraints, ["100% Cotton"])

    def test_browsing_opening(self) -> None:
        utterance = nlu.parse("I'm looking for Belts, but I'm still exploring.", self.resolve)
        self.assertEqual(utterance.kind, "open_browse")
        self.assertEqual(utterance.category, "Belts")

    def test_stated_preference_opening(self) -> None:
        utterance = nlu.parse("I'm looking for Belts. Imported", self.resolve)
        self.assertEqual(utterance.kind, "open_stated")
        self.assertEqual(utterance.constraints, ["Imported"])

    def test_disclosure_splits_on_known_constraints(self) -> None:
        utterance = nlu.parse("For that, what matters is: 100% Cotton; Machine Wash.", self.resolve)
        self.assertEqual(utterance.constraints, ["100% Cotton", "Machine Wash"])

    def test_disclosure_keeps_embedded_separator(self) -> None:
        # A single constraint may itself contain "; ": the catalog decides.
        known = {"Care: hand wash; line dry"}
        utterance = nlu.parse(
            "For that, what matters is: Care: hand wash; line dry.",
            lambda value: value if value in known else None,
        )
        self.assertEqual(utterance.constraints, ["Care: hand wash; line dry"])

    def test_boundary_and_exhausted(self) -> None:
        self.assertEqual(
            nlu.parse("I don't have a preference for size; please use your judgment.").kind,
            "boundary",
        )
        self.assertEqual(
            nlu.parse("I don't have an additional preference for brand.").kind, "exhausted"
        )

    def test_override(self) -> None:
        utterance = nlu.parse(
            "Actually, ignore my earlier preference. What I need is: Imported.", self.resolve
        )
        self.assertEqual(utterance.kind, "override")
        self.assertEqual(utterance.constraints, ["Imported"])

    def test_paraphrase_recovers_quoted_constraints(self) -> None:
        utterance = nlu.parse("hey, it really has to be 100% cotton", self.resolve)
        self.assertEqual(utterance.kind, "freeform")
        self.assertEqual(utterance.constraints, ["100% Cotton"])

    def test_recovery_ignores_text_with_no_known_quote(self) -> None:
        utterance = nlu.parse("something entirely unrelated to the catalog", self.resolve)
        self.assertEqual(utterance.constraints, [])


class TestUserModelReplay(unittest.TestCase):
    CARD = ("cotton", "color: black", "Machine Wash", "Imported")

    def test_expected_reply_respects_disclosure_order(self) -> None:
        self.assertEqual(expected_reply(self.CARD, set(), "other"), ["cotton", "color: black"])
        self.assertEqual(
            expected_reply(self.CARD, {"cotton", "color: black"}, "other"),
            ["Machine Wash", "Imported"],
        )

    def test_expected_reply_filters_by_attribute(self) -> None:
        self.assertEqual(expected_reply(self.CARD, set(), "material"), ["cotton"])
        self.assertEqual(expected_reply(self.CARD, set(), "color"), ["color: black"])

    def test_consistency_requires_the_opening_slot(self) -> None:
        transcript = Transcript(scenario="buying", opening="cotton")
        self.assertTrue(is_consistent(self.CARD, transcript))
        self.assertFalse(is_consistent(("wool", "color: black"), transcript))

    def test_consistency_replays_answers_exactly(self) -> None:
        transcript = Transcript(
            scenario="buying",
            opening="cotton",
            turns=[Turn("other", "disclose", ["color: black", "Machine Wash"])],
        )
        self.assertTrue(is_consistent(self.CARD, transcript))
        self.assertFalse(
            is_consistent(("cotton", "color: blue", "Machine Wash", "Imported"), transcript)
        )

    def test_exhausted_answer_rejects_products_with_more_to_say(self) -> None:
        transcript = Transcript(turns=[Turn("material", "exhausted", [])])
        self.assertFalse(is_consistent(self.CARD, transcript))
        self.assertTrue(is_consistent(("Imported", "Machine Wash"), transcript))

    def test_freeform_evidence_does_not_force_an_ordering(self) -> None:
        transcript = Transcript(turns=[Turn(None, "freeform", ["Imported"])])
        self.assertTrue(is_consistent(self.CARD, transcript))


@unittest.skipUnless(HAS_KIT, "participant kit not bootstrapped")
class TestQuestionPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from converge.catalog import load_catalog

        cls.catalog = load_catalog(CATALOG)

    def test_gain_is_zero_when_nothing_left_to_disclose(self) -> None:
        pool = self.catalog.bucket(self.catalog.categories[0])[:50]
        disclosed = {atom for idx in pool for atom in self.catalog.slots[idx]}
        transcript = Transcript(turns=[Turn(None, "freeform", sorted(disclosed))])
        question = policy.choose_question(self.catalog, transcript, pool)
        self.assertIsNone(question.attribute)

    def test_a_wide_pool_yields_a_question(self) -> None:
        category = max(self.catalog.by_category, key=lambda key: len(self.catalog.by_category[key]))
        pool = self.catalog.bucket(category)
        question = policy.choose_question(self.catalog, Transcript(category=category), pool)
        self.assertTrue(question.worth_asking)
        self.assertGreater(question.gain, 0.0)

    def test_phrasing_is_customer_facing(self) -> None:
        self.assertIn("?", policy.phrase("material", 42, overloaded=False))
        self.assertIn("42", policy.phrase("material", 42, overloaded=True))


@unittest.skipUnless(HAS_KIT, "participant kit not bootstrapped")
class TestCategoryResolver(unittest.TestCase):
    def test_recovers_a_bucket_from_loose_words(self) -> None:
        from converge.catalog import load_catalog
        from converge.resolver import CategoryResolver

        catalog = load_catalog(CATALOG)
        resolver = CategoryResolver(catalog)
        category = max(catalog.by_category, key=lambda key: len(catalog.by_category[key]))
        recovered = resolver.resolve(category.lower().split())
        self.assertEqual(recovered, category)

    def test_returns_none_on_thin_evidence(self) -> None:
        from converge.catalog import load_catalog
        from converge.resolver import CategoryResolver

        resolver = CategoryResolver(load_catalog(CATALOG))
        self.assertIsNone(resolver.resolve(["zzzz", "qqqq"]))


if __name__ == "__main__":
    unittest.main()
