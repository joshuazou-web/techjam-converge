# Paraphrase robustness

The customer *policy* is unchanged (the official `customer_reply` decides what is
disclosed); only the wording is perturbed. `none` reproduces the reported score.

| perturbation | Hit@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| `none` | 1.000 | 0.9835 | 1.96 | **0.9760** |
| `light` | 0.925 | 0.8853 | 2.88 | **0.8904** |
| `heavy` | 0.875 | 0.7099 | 3.86 | **0.7933** |
