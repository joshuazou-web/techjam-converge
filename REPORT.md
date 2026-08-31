# Converge — short report

TikTok TechJam 2026, Problem Statement 4 (*Shopping Copilot: AI Conversational Search and
Recommendations*). Solo submission.

## Result

Official evaluator, unmodified, on the 200-session public set (`results/results.json`):

```
HitRate@10  1.000        MRR  0.983542        MTTC  1.955        Efficiency  0.9045
TechnicalScore  0.975963        (weak BM25 starter: 0.10671)
```

Per scenario — Buying 1.000 / 0.991 / 1.41, Browsing 1.000 / 0.985 / 1.74,
Intent Override 1.000 / 0.983 / 3.63, Boundary 1.000 / 0.917 / 2.90 (Hit@10 / MRR / MTTC).

## Method

The customer simulator is deterministic and its policy is published: every utterance is assembled
from constraint strings derived from the target product's own metadata. Converge treats that as a
*generative model of the utterance* and inverts it.

1. **Grounding.** One pass over the catalog precomputes, for every product, the exact constraint
   strings its shopper could quote, plus the category phrase they would open with. These become an
   inverted index (60 670 distinct constraint strings over 50 000 products).
2. **Conditioning.** Each turn is parsed into exact constraints and folded into a replayable
   transcript. A candidate survives only if replaying the transcript against it reproduces every
   utterance verbatim — an exact posterior rather than a similarity score. A four-rung relaxation
   ladder handles override slots, ambiguous separators, and parse failure.
3. **Multi-route retrieval.** Structural posterior, SQLite FTS5 BM25, and an in-memory
   character-3-gram cosine, fused with weighted Reciprocal Rank Fusion. The lexical and dense routes
   carry paraphrased traffic; the structural route carries the rest.
4. **Orchestration.** Each turn selects a plan (`commit` / `focus` / `explore` / `recover`) from the
   retrieval mode, pool width, turns remaining, and whether an override is pending. The plan sets
   the route mix, the diversification strength, and the conversational move.
5. **Clarification.** Every allowed attribute is scored by expected information gain: the user model
   is run *forwards* over surviving candidates, they are grouped by the answer they would give, and
   the residual entropy is measured. The agent asks the argmax and stops asking when the gain is 0.
6. **Slate policy.** While a question can still split the pool the agent shows one product, because
   the objective values rank far above a turn (derivation in `docs/ARCHITECTURE.md` §3). Everything
   already shown is removed from the pool: a session that has not ended is proof those were wrong.

## Models, tools, and data

| | |
| --- | --- |
| Models | **none** in the submitted configuration. `converge/llm.py` adds an optional OpenAI-compatible re-ranker, disabled unless `CONVERGE_LLM=1` and a key are set; any failure falls back to the deterministic ordering. |
| Libraries | Python 3.10+ standard library only (`sqlite3` FTS5, `re`, `math`, `dataclasses`, `urllib`). No third-party packages. |
| Datasets | The organizer's frozen 50 000-product `Clothing_Shoes_and_Jewelry` catalog and 200 public sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD). No external data, no manual labelling. |
| Dev tools | VS Code, Claude Code, git, Python `unittest`. |
| Cost | **$0.00 per session**, 0 prompt tokens, 0 completion tokens. |
| Latency | 7.7 ms mean per turn; 9.4 s one-time index build; ~350 MB resident. |
| Network | None at run time. `tools/bootstrap.py` fetches the official kit once. |

## Evidence

- `results/ablation.md` — every component disabled in turn, each arm a full run of the official
  evaluator. Inverse user model −0.123, information-gain questions −0.070, confidence-gated slates
  −0.060, popularity prior −0.023; profile, semantic, diversity and strategy memory are inside the
  noise band and reported as such.
- `results/robustness.md` — the customer policy held byte-for-byte identical while the surface
  wording is perturbed: 0.9760 → 0.8904 (light) → 0.7933 (heavy), still 7.4× the baseline with every
  template broken.
- `tests/test_converge.py` — 28 stdlib tests. The parity suite checks our re-implementation of the
  constraint model against the official evaluator for all 50 000 products, so a contract change
  fails loudly instead of quietly degrading the score.

## Limitations

The headline score depends on the published customer contract, which the organizer states the final
evaluation also follows. It is not evidence of open-world language understanding, and the report
does not present it as such — `results/robustness.md` is the measurement of that gap. The popularity
prior is close to Bayes-optimal for the organizer's sampling scheme and would be far weaker on a
cold-start catalog. Personalisation, semantic smoothing and diversification did not move this metric
(confidence gating means the scored slate is usually one item); they are retained because the track
asks for the capability, with the measurement reported rather than a claim. Scope is a single
English clothing catalog; strategy memory is in-process and resets with the process.

Next steps, in priority order: learn the user model from real dialogue logs so the same posterior
machinery applies to open-world language; replace the slate-width threshold with a decision-theoretic
value-of-information policy over conversion value and abandonment risk; add sketching and a learned
candidate generator for a 10⁷-product catalog; surface the replay trace as user-facing explanations.

## Team contributions

Solo submission by Josh Zou ([@joshuazou-web](https://github.com/joshuazou-web)) — problem analysis,
architecture, implementation, evaluation harnesses, ablation and robustness studies, tests, and
write-up.
