"""Measure what each component of Converge is actually worth.

Every arm runs the *unmodified* official evaluator over the full public set with
one capability switched off, so the table is an experiment rather than a claim.
The catalog index is built once and shared across arms, which makes the whole
sweep cheaper than a single cold start per arm.

Usage::

    python tools/ablation.py
    python tools/ablation.py --arms full no-structural
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / ".kit"

ARMS: dict[str, dict] = {
    "full": {},
    "no-structural": {"structural": False},
    "no-confidence-gating": {"confidence_gating": False},
    "no-prior": {"prior": False},
    "no-information-gain": {"information_gain": False},
    "no-diversity": {"diversity": False},
    "no-profile": {"profile": False},
    "no-semantic": {"semantic": False},
    "no-strategy-memory": {"strategy_memory": False},
}

DESCRIPTIONS = {
    "full": "submitted configuration",
    "no-structural": "drop the inverse user model; rank the category bucket only",
    "no-confidence-gating": "always show ten results instead of holding back the tail",
    "no-prior": "drop the purchase-volume prior",
    "no-information-gain": "ask a fixed question script instead of maximising expected gain",
    "no-diversity": "no MMR spread on wide browsing slates",
    "no-profile": "ignore the anonymised preference tags",
    "no-semantic": "no character-n-gram semantic smoothing",
    "no-strategy-memory": "no cross-session adaptation of question choice",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()

    if not KIT.exists():
        raise SystemExit("participant kit missing -- run: python tools/bootstrap.py")
    sys.path.insert(0, str(KIT))
    sys.path.insert(0, str(ROOT))
    os.chdir(KIT)

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

    from converge.agent import Agent  # noqa: E402
    from converge.catalog import load_catalog  # noqa: E402
    from converge.config import DEFAULT  # noqa: E402

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    catalog = load_catalog(args.catalog)

    baseline_path = KIT / "docs" / "baseline_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None

    rows: list[dict] = []
    if baseline:
        rows.append({
            "arm": "official weak BM25 starter",
            "description": "shipped reference agent",
            "hit_rate_at_10": baseline["hit_rate_at_10"],
            "mrr": baseline["mrr"],
            "mttc": baseline["mttc"],
            "technical_score": baseline["technical_score"],
            "seconds": None,
        })

    for name in args.arms:
        if name not in ARMS:
            raise SystemExit(f"unknown arm: {name}")
        config = replace(DEFAULT, **ARMS[name])
        started = time.perf_counter()
        result = evaluate(Agent(catalog=catalog, config=config), samples,
                          catalog_ids, categories, products)
        rows.append({
            "arm": name,
            "description": DESCRIPTIONS[name],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
            "seconds": round(time.perf_counter() - started, 1),
        })
        print(f"{name:22s} score={rows[-1]['technical_score']:.4f} "
              f"hit={rows[-1]['hit_rate_at_10']:.3f} mrr={rows[-1]['mrr']:.4f} "
              f"mttc={rows[-1]['mttc']:.3f}")

    full = next((row for row in rows if row["arm"] == "full"), None)
    lines = [
        "# Ablation on the 200-session public set",
        "",
        "Each arm runs the unmodified official evaluator with one capability disabled.",
        "`delta` is the change in TechnicalScore against the submitted configuration.",
        "",
        "| arm | what changes | Hit@10 | MRR | MTTC | TechnicalScore | delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = "" if full is None else f"{row['technical_score'] - full['technical_score']:+.4f}"
        if row["arm"] == "full":
            delta = "--"
        lines.append(
            f"| `{row['arm']}` | {row['description']} | {row['hit_rate_at_10']:.3f} | "
            f"{row['mrr']:.4f} | {row['mttc']:.2f} | **{row['technical_score']:.4f}** | {delta} |"
        )
    lines.append("")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ablation.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {ROOT / 'results' / 'ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
