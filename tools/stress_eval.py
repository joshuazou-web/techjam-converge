"""Robustness harness: how much does Converge rely on the published wording?

The official evaluator states that final-evaluation messages follow the same
templates, so the submitted score is produced against those templates. That is
also the obvious criticism of a system that parses them exactly -- so we measure
the failure mode instead of arguing about it.

This harness keeps the customer *policy* byte-for-byte identical (it calls the
official ``intent_card``, ``behavior_for`` and ``customer_reply``) and perturbs
only the **surface form** of what the customer says: casual openings, contracted
requirements, filler, lower-casing. Nothing here is used to produce the reported
score; it exists to show where the agent degrades and what catches it.

Usage::

    python tools/stress_eval.py                    # all perturbation levels
    python tools/stress_eval.py --levels none heavy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / ".kit"

MAX_TURNS = 10
TOP_K = 10


def paraphrase(message: str, level: str, rng: random.Random) -> str:
    """Rewrite a customer line without changing what it discloses."""
    if level == "none":
        return message
    text = message
    if text.startswith("I'm looking for "):
        opener = rng.choice([
            "hi! i want ", "hey, i need ", "so i'm after ", "im shopping for ",
        ])
        text = opener + text[len("I'm looking for "):]
    text = text.replace(", but I'm still exploring.", " - just browsing for now")
    text = text.replace("A key requirement is: ", rng.choice([
        "it really has to be ", "must-have: ", "the important bit is ",
    ]))
    text = text.replace("For that, what matters is: ", rng.choice([
        "well, ", "i'd say ", "for me it's ",
    ]))
    text = text.replace("Actually, ignore my earlier preference. What I need is: ", rng.choice([
        "actually scratch that, i want ", "hmm, change of plan - i need ",
    ]))
    text = text.replace("I don't have an additional preference for ", "no strong feelings on ")
    text = text.replace("; please use your judgment.", " - you pick")
    if level == "heavy":
        text = text.lower()
        text = text.replace(";", ",").replace(":", "")
        text = rng.choice(["", "um, ", "ok so ", "right, "]) + text
    return text


def run_level(level: str, samples, catalog_ids, categories, products, agent_factory) -> dict:
    from evaluator.local_evaluator import (  # noqa: E402
        coarse_category,
        customer_reply,
        initial_message,
        materialize_hidden_fields,
        normalize_recommendations,
    )

    agent = agent_factory()
    hits: list[int] = []
    ranks: list[float] = []
    turns: list[int] = []

    for sample in samples:
        rng = random.Random(f"{level}\0{sample['sample_id']}")
        session_id = f"stress_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, paraphrase(message, level, rng), turn, TOP_K)
            except Exception:  # a crash scores as a miss, exactly as in the official harness
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        hits.append(int(hit_turn is not None))
        ranks.append(0.0 if best_rank is None else 1.0 / best_rank)
        turns.append(hit_turn if hit_turn is not None else MAX_TURNS + 1)

    hit_rate = statistics.fmean(hits)
    mrr = statistics.fmean(ranks)
    mttc = statistics.fmean(turns)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "level": level,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="*", default=["none", "light", "heavy"])
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

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    catalog = load_catalog(args.catalog)

    rows = [
        run_level(level, samples, catalog_ids, categories, products, lambda: Agent(catalog=catalog))
        for level in args.levels
    ]
    for row in rows:
        print(f"{row['level']:8s} score={row['technical_score']:.4f} hit={row['hit_rate_at_10']:.3f} "
              f"mrr={row['mrr']:.4f} mttc={row['mttc']:.3f}")

    lines = [
        "# Paraphrase robustness",
        "",
        "The customer *policy* is unchanged (the official `customer_reply` decides what is",
        "disclosed); only the wording is perturbed. `none` reproduces the reported score.",
        "",
        "| perturbation | Hit@10 | MRR | MTTC | TechnicalScore |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['level']}` | {row['hit_rate_at_10']:.3f} | {row['mrr']:.4f} | "
            f"{row['mttc']:.2f} | **{row['technical_score']:.4f}** |"
        )
    lines.append("")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "robustness.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "robustness.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {ROOT / 'results' / 'robustness.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
