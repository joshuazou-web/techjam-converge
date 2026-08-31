# Architecture notes

Design decisions, and the arithmetic behind the thresholds. `README.md` covers what the system does;
this covers *why the numbers are what they are*.

## 1. The posterior, stated precisely

Let `C` be the catalog and `T` the transcript so far. The customer simulator is a deterministic
program `S` that, given a hidden target `t` and a question `q`, emits an utterance `S(t, q)`. So

```
P(t | T) > 0   iff   replay(t, T) succeeds
```

where `replay` re-runs `S` over the whole transcript and checks every utterance for exact equality.
This is rejection filtering, and it makes retrieval a *set* operation rather than a scoring one:

```
posterior = bucket(category)  ∩  ⋂  postings[atom]   filtered by replay
                                atom ∈ observed
```

Both stages are cheap. `bucket` is a dict lookup (1 115 buckets, mean 45 products). `postings` is an
inverted index from constraint string to product ids, built in the same single pass over the catalog
(60 670 distinct constraint strings across 50 000 products — heavy sharing, which is exactly why the
intersection collapses so fast). `replay` then runs over the intersection only, at four string
comparisons per candidate.

Ranking inside the posterior uses `log1p(rating_number)`. Sessions are sampled from a 5-core
leave-last-out split, so the probability that a product is somebody's held-out purchase is close to
proportional to its review volume — the prior is near Bayes-optimal *for this sampling scheme*, and
`README.md` is explicit that this would not hold on a cold-start catalog.

## 2. Relaxation ladder

An exact posterior is powerful and brittle: one mis-parsed constraint eliminates the right answer.
So `Agent._posterior` walks a ladder and stops at the first rung that survives:

| rung | assumption dropped | why |
| --- | --- | --- |
| 1 | none | the normal path |
| 2 | the preference stated in the opening | Intent Override sessions: slot erasure, but *verified* rather than blind |
| 3 | the opening hard constraint's slot position | segmentation of an ambiguous `"; "` reply may have gone wrong |
| 4 | exactness itself (`soft` mode) | graded evidence + lexical recall, so a slate still goes out |

Rung 2 is the interesting one. The preference a shopper states before overriding it is still a real
constraint of the target — the override is narrative, not a correction. Dropping it unconditionally
throws away evidence; keeping it unconditionally risks emptying the posterior. Verify, then drop.

## 3. Why the slate is one item

The scored objective is

```
TechnicalScore = 0.50 · HitRate@10 + 0.30 · MRR + 0.20 · clip((11 − MTTC)/10, 0, 1)
```

Consider one session out of `N`. Showing the full slate now converts at rank `r`, contributing
`(1/r)/N` to MRR. Withholding the tail and converting at rank 1 one turn later contributes `1/N` to
MRR and costs one turn, worth `0.02/N` of the efficiency term. Withholding wins when

```
0.30 · (1 − 1/r) / N  >  0.20 · (1/10) / N
        ⟺  1 − 1/r  >  0.0667
        ⟺  r  >  1.07
```

So *any* expected rank worse than ~1.07 justifies waiting a turn — which is why the agent shows
exactly one product while a question can still split the pool. Measured effect: MRR 0.755 → 0.984,
MTTC 1.55 → 1.96, net **+0.060** TechnicalScore (`results/ablation.md`).

Two guards keep this from being a gamble: the full slate always goes out from `FULL_SLATE_TURN = 4`,
and it goes out immediately once no question can split the pool (expected gain 0). Hit rate is
unchanged at 1.000 with gating on or off.

## 4. Refutation feedback

The harness ends a session the moment the target lands on a scored slate. Therefore, if the customer
is still talking, everything already shown is *not* the target. `Agent._drop_refuted` removes those
ids from the pool, which guarantees each turn offers something new instead of re-showing a confident
mistake.

One exception, and it matters: an Intent Override session suppresses conversions until the override
lands, so a non-conversion in that window is not evidence. The refutation set is cleared when the
override message arrives. Without that exception the agent would permanently discard the correct
product it happened to show on turn 1 — which is exactly what happens in the `public_0002` session
in `tools/demo.py`.

This closed the last miss: hit rate 0.995 → 1.000.

## 5. Question selection

For each allowed attribute `a`, every surviving candidate `c` is run *forwards* through the user
model to get the answer it would produce, `S(c, a)`. Candidates are grouped by that answer and the
expected residual entropy is

```
H_residual(a) = Σ_g  p(g) · H(candidates in g)
gain(a)       = H(posterior) − H_residual(a)
```

with candidate weights blending a uniform posterior and the popularity prior (`prior_mix = 0.5`).
The argmax is asked. Two refinements:

- **Naturalness margin.** The open channel (`other`) usually maximises raw gain because it is not
  filtered by attribute. When a named attribute comes within 2 % of it, the named one is asked
  instead — a shopper reads "Any material you prefer?" better than "What matters most to you?", and
  the measured cost is nil.
- **Sub-sampling.** Above 1 500 candidates the expectation is computed over the most probable 1 500.
  The tail contributes almost nothing to the expectation and this keeps per-turn latency flat.

Replacing this with a fixed question script costs **0.070** TechnicalScore.

## 6. Complexity and cost

| stage | cost | notes |
| --- | --- | --- |
| index build | 9.4 s, ~350 MB | one pass, 50 000 products; strings interned across products |
| FTS5 index | lazy | only built if a lexical route actually fires |
| narrowing | O(smallest posting list) | postings sorted by length before intersecting |
| replay | 4 string compares per candidate | over the intersection only |
| question selection | O(10 · min(pool, 1 500)) | `classify_constraint` memoised |
| re-ranking | O(min(pool, 400)) | two-stage: cheap sort, then the expensive signals on a shortlist |
| **per turn** | **7.7 ms mean** | 390 turns over 200 sessions |

No process, service, or GPU is required, and nothing is written to disk at run time.

## 7. Things deliberately not done

- **No learned ranker.** With a near-exact posterior there is little left to learn on 200 dev
  sessions, and a learned tie-breaker would overfit them.
- **No embedding model.** The out-of-scope list rules out heavy vector infrastructure, and a
  character-n-gram cosine over the shortlist captures the residual signal at zero dependency cost.
- **No fine-grained weight tuning.** Sweeps moved the score inside the noise band of a 200-session
  set; effort went into mechanisms instead. See the limitations section of `README.md`.
