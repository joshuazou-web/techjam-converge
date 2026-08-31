# Ablation on the 200-session public set

Each arm runs the unmodified official evaluator with one capability disabled.
`delta` is the change in TechnicalScore against the submitted configuration.

| arm | what changes | Hit@10 | MRR | MTTC | TechnicalScore | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `official weak BM25 starter` | shipped reference agent | 0.125 | 0.0680 | 9.81 | **0.1067** | -0.8693 |
| `full` | submitted configuration | 1.000 | 0.9835 | 1.96 | **0.9760** | -- |
| `no-structural` | drop the inverse user model; rank the category bucket only | 0.970 | 0.7202 | 3.38 | **0.8534** | -0.1226 |
| `no-confidence-gating` | always show ten results instead of holding back the tail | 1.000 | 0.7553 | 1.55 | **0.9156** | -0.0604 |
| `no-prior` | drop the purchase-volume prior | 0.995 | 0.9493 | 2.46 | **0.9532** | -0.0228 |
| `no-information-gain` | ask a fixed question script instead of maximising expected gain | 1.000 | 0.7270 | 1.62 | **0.9056** | -0.0704 |
| `no-diversity` | no MMR spread on wide browsing slates | 1.000 | 0.9835 | 1.96 | **0.9760** | +0.0000 |
| `no-profile` | ignore the anonymised preference tags | 1.000 | 0.9854 | 1.97 | **0.9763** | +0.0004 |
| `no-semantic` | no character-n-gram semantic smoothing | 1.000 | 0.9823 | 1.96 | **0.9755** | -0.0005 |
| `no-strategy-memory` | no cross-session adaptation of question choice | 1.000 | 0.9835 | 1.96 | **0.9760** | +0.0000 |
