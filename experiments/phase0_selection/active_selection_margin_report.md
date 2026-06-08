# Active-selection re-analysis: continuous margin outcome

Outcome = margin = (proposed instruction's A=20 mean) - 0.4017 (frozen seed baseline). OLS, regressors standardized (ddof=0), no new LLM calls.

- n = 45
- margin: mean=0.0589 sd=0.0369 min=-0.0267 max=0.1283 (n<=0: 3)

## OLS: margin ~ base_score + addressability (standardized)

| term | coef | se | t | p (two-sided) |
| --- | --- | --- | --- | --- |
| intercept | 0.05889 | 0.00561 | 10.496 | 0.0000 |
| base_score | 0.00209 | 0.00561 | 0.372 | 0.7120 |
| **addressability** | **0.00249** | **0.00561** | **0.444** | **0.6593** |

- model df (residual): 42
- **R^2 reduced (base only): 0.0036**
- **R^2 full (base + addressability): 0.0082**
- incremental R^2 from addressability: 0.0047

## Finding
addressability does NOT predict margin size (p=0.6593 >= 0.05). Pre-registered prediction was the null (addressability does not predict margin).