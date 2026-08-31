"""Run the official evaluator against Converge and summarise the result.

This is a thin, auditable wrapper: it changes directory into ``.kit`` and calls
``evaluator.local_evaluator`` exactly as the organizer ships it. No evaluator
code, config, or label is touched.

Usage::

    python tools/run_eval.py                       # public set, default config
    python tools/run_eval.py --output results/mine.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / ".kit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=str(ROOT / "results" / "results.json"))
    args = parser.parse_args()

    if not KIT.exists():
        raise SystemExit("participant kit missing -- run: python tools/bootstrap.py")

    sys.path.insert(0, str(KIT))
    sys.path.insert(0, str(ROOT))
    os.chdir(KIT)

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
    from starter.agent import Agent  # noqa: E402

    started = time.perf_counter()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    index_ready = time.perf_counter()
    agent = Agent(args.catalog)
    agent_ready = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    finished = time.perf_counter()

    turns = sum(
        (session["first_hit_turn"] or 10) for session in result["sessions"]
    )
    result["runtime"] = {
        "evaluator_index_seconds": round(index_ready - started, 2),
        "agent_startup_seconds": round(agent_ready - index_ready, 2),
        "session_loop_seconds": round(finished - agent_ready, 2),
        "sessions": len(samples),
        "agent_turns": turns,
        "mean_ms_per_turn": round(1000 * (finished - agent_ready) / max(turns, 1), 2),
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
