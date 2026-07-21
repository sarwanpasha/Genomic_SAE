# Results summary (headline numbers)

## SAE reconstruction (top-k, k=32, 16x dictionary)
- NT best layer 14: 73.9% variance explained, 92.7% features alive
- DNABERT-2 best layer 6: 75.7% variance explained, 100% alive
- Middle layers most interpretable in both models

## Causal validation matrix (causal / 15 tested; random controls 0/15)
| TF            | NT (layer 14) | DNABERT-2 (layer 6) |
|---------------|---------------|---------------------|
| CTCF          | 7/15          | 10/15               |
| GATA1         | 9/15          | 13/15               |
| REST          | 7/15          | 14/15               |
| SCRAMBLE      | 0 features    | 0 features          |
| GATA1-SCRAM   | 0 features    | 0 features          |

## Exemplar single-feature causal result (NT CTCF feature 8087)
- KL_bound = 1.25e-5, KL_unbound = 6.88e-6, ratio 1.81, AUC 0.626, P = 1.7e-18
- Random-feature control: AUC 0.511 (null)
- Motif-only control: large absolute KL but AUC 0.514 (no binding specificity)

## Confound-control chain
Alu/repeat features -> GC-in-PWM -> GC-in-binding-sets -> scrambled-label controls
