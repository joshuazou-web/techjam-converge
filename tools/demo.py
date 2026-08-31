"""Watch Converge work: a narrated end-to-end session.

Runs a real session through the official customer simulator and prints both
sides of the conversation *plus* the agent's internal state each turn -- which
retrieval mode produced the pool, how many candidates survive, which plan the
orchestrator picked, what it decided to ask and why, and how wide a slate it was
willing to show.

Usage::

    python tools/demo.py                    # one session per scenario type
    python tools/demo.py --scenario buying --count 3
    python tools/demo.py --chat             # type your own messages
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / ".kit"

MAX_TURNS = 10
TOP_K = 10

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def run_session(agent, sample, catalog, catalog_ids, categories, products) -> None:
    from evaluator.local_evaluator import (
        coarse_category,
        customer_reply,
        initial_message,
        materialize_hidden_fields,
        normalize_recommendations,
    )

    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    print(_c(f"\n=== {sample['sample_id']}  scenario={sample['scenario_type']} "
             f"difficulty={sample.get('difficulty_bucket', '?')} ===", BOLD))
    print(_c(f"hidden target : {target}  {catalog.titles[catalog.index_of[target]][:70]}", DIM))
    print(_c(f"profile       : {sample['user_profile']['summary']}", DIM))

    agent.reset(session_id, sample["user_profile"])
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(_c(f"\ncustomer  > {message}", CYAN))
        response = agent.respond(session_id, message, turn, TOP_K)
        trace = agent.trace(session_id)[-1]
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

        print(f"agent     > {response['message']}")
        print(_c(f"            [ask={response['ask_attribute']} "
                 f"pool={trace['pool_size']} mode={trace['mode']} plan={trace['plan']} "
                 f"gain={trace['gain']} slate={trace['slate']}]", DIM))
        for rank, asin in enumerate(ranked, start=1):
            mark = _c("  <-- target", GREEN) if asin == target else ""
            title = catalog.titles[catalog.index_of[asin]][:64]
            print(f"            {rank:>2}. {asin}  {title}{mark}")

        if override_applied and target in ranked:
            print(_c(f"\nCONVERTED on turn {turn} at rank {ranked.index(target) + 1}", GREEN + BOLD))
            return
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
            print(_c("            [intent override incoming: earlier slate is no longer refuted]", YELLOW))
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
    print(_c("\nNOT CONVERTED within 10 turns", YELLOW + BOLD))


def chat(agent, catalog) -> None:
    """Free-text session: no simulator, no templates, just typing."""
    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, {"preference_tags": [], "summary": "", "purchase_frequency": "",
                             "average_prior_rating": None, "rating_style": ""})
    print(_c("Type what you are shopping for. Ctrl-C to leave.\n", BOLD))
    turn = 1
    while turn <= MAX_TURNS:
        try:
            message = input(_c("you       > ", CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        response = agent.respond(session_id, message, turn, TOP_K)
        trace = agent.trace(session_id)[-1]
        print(f"agent     > {response['message']}")
        print(_c(f"            [ask={response['ask_attribute']} pool={trace['pool_size']} "
                 f"mode={trace['mode']} plan={trace['plan']} category={trace['scenario']}]", DIM))
        for rank, item in enumerate(response["recommendations"], start=1):
            idx = catalog.index_of[item["parent_asin"]]
            print(f"            {rank:>2}. {item['parent_asin']}  {catalog.titles[idx][:64]}")
        turn += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None,
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    if not KIT.exists():
        raise SystemExit("participant kit missing -- run: python tools/bootstrap.py")
    sys.path.insert(0, str(KIT))
    sys.path.insert(0, str(ROOT))
    os.chdir(KIT)

    from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402

    from converge.agent import Agent  # noqa: E402
    from converge.catalog import load_catalog  # noqa: E402

    catalog = load_catalog(args.catalog)
    agent = Agent(catalog=catalog)

    if args.chat:
        chat(agent, catalog)
        return 0

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    scenarios = [args.scenario] if args.scenario else ["buying", "browsing", "intent_override", "boundary"]
    for scenario in scenarios:
        chosen = [item for item in samples if item["scenario_type"] == scenario][: args.count]
        for sample in chosen:
            run_session(agent, sample, catalog, catalog_ids, categories, products)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
