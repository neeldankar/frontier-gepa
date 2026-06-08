# IFBench active-selection validation: results

Outcome = proposed instruction (reflect on a single cc3 example, gpt-4.1-mini) strictly improves the seed's frozen A=20 mean. Regress outcome ~ base_score + addressability (logistic, numpy IRLS; LR test for the addressability term).

- reflected examples scored: 45 (errors/skipped: 0)
- seed A=20 'before' mean: 0.40166666666666656
- outcome base rate (improved): 42/45 = 0.933

## Logistic regression (regressors standardized)

| model | terms | coefs | logLik | McFadden R2 |
| --- | --- | --- | --- | --- |
| reduced | 1, base_score | [2.817, -0.672] | -10.527 | 0.0449 |
| full | 1, base_score, addressability | [2.833, -0.668, -0.184] | -10.478 | 0.0493 |

- addressability coef (standardized): -0.1837 (Wald se 0.5932, z -0.310, p 0.7568)
- **Likelihood-ratio test (full vs reduced, 1 dof): LR=0.0979, p=0.7543**

## Finding
addressability does NOT add predictive power over base_score (LR p=0.7543 >= 0.05); the scalar base score is sufficient.