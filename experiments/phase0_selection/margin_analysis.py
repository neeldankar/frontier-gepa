"""Re-analysis of the cached active-selection results with the CONTINUOUS margin
as the outcome (no new LLM calls).

margin = after - 0.4017 (the frozen seed A=20 baseline).
OLS: margin ~ base_score + addressability, both standardized (ddof=0, matching the
logistic setup). Reports the addressability coefficient, SE, t-stat, two-sided p, and
R^2 for the reduced (base only) and full (base + addressability) models.

t-distribution p-values use the regularized incomplete beta (Numerical Recipes betai),
since scipy is not in the venv.

Run: .venv/bin/python phase0/margin_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PHASE0 = Path(__file__).resolve().parent
RESULTS = PHASE0 / "active_selection_results.jsonl"
REPORT = PHASE0 / "active_selection_margin_report.md"
BASELINE = 0.4017  # frozen seed A=20 mean (per spec)


# ---- t / beta helpers (no scipy) ----

def _gammln(x: float) -> float:
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(_gammln(a + b) - _gammln(a) - _gammln(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided p-value for Student-t."""
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, k = X.shape
    df = n - k
    sigma2 = ss_res / df
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, df, r2, ss_res


def main() -> int:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("after") is not None]
    n = len(rows)
    after = np.array([r["after"] for r in rows])
    base = np.array([r["base_score"] for r in rows])
    addr = np.array([r["addressability"] for r in rows])
    margin = after - BASELINE

    # standardize regressors (ddof=0, same as the logistic setup)
    zbase = (base - base.mean()) / base.std()
    zaddr = (addr - addr.mean()) / addr.std()

    X_red = np.column_stack([np.ones(n), zbase])
    X_full = np.column_stack([np.ones(n), zbase, zaddr])

    b_red, se_red, df_red, r2_red, _ = _ols(X_red, margin)
    b_full, se_full, df_full, r2_full, _ = _ols(X_full, margin)

    # addressability term is index 2 in the full model
    addr_coef = float(b_full[2])
    addr_se = float(se_full[2])
    addr_t = addr_coef / addr_se
    addr_p = _t_sf_two_sided(addr_t, df_full)
    base_coef_full, base_t_full = float(b_full[1]), float(b_full[1] / se_full[1])
    base_p_full = _t_sf_two_sided(base_t_full, df_full)

    lines = []
    lines.append("# Active-selection re-analysis: continuous margin outcome\n")
    lines.append("Outcome = margin = (proposed instruction's A=20 mean) - 0.4017 (frozen "
                 "seed baseline). OLS, regressors standardized (ddof=0), no new LLM calls.\n")
    lines.append(f"- n = {n}")
    lines.append(f"- margin: mean={margin.mean():.4f} sd={margin.std(ddof=1):.4f} "
                 f"min={margin.min():.4f} max={margin.max():.4f} "
                 f"(n<=0: {int((margin<=0).sum())})\n")
    lines.append("## OLS: margin ~ base_score + addressability (standardized)\n")
    lines.append("| term | coef | se | t | p (two-sided) |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| intercept | {b_full[0]:.5f} | {se_full[0]:.5f} | "
                 f"{b_full[0]/se_full[0]:.3f} | {_t_sf_two_sided(b_full[0]/se_full[0], df_full):.4f} |")
    lines.append(f"| base_score | {base_coef_full:.5f} | {se_full[1]:.5f} | {base_t_full:.3f} | {base_p_full:.4f} |")
    lines.append(f"| **addressability** | **{addr_coef:.5f}** | **{addr_se:.5f}** | "
                 f"**{addr_t:.3f}** | **{addr_p:.4f}** |")
    lines.append("")
    lines.append(f"- model df (residual): {df_full}")
    lines.append(f"- **R^2 reduced (base only): {r2_red:.4f}**")
    lines.append(f"- **R^2 full (base + addressability): {r2_full:.4f}**")
    lines.append(f"- incremental R^2 from addressability: {r2_full - r2_red:.4f}")
    lines.append("")
    finding = ("addressability DOES predict margin "
               f"(p={addr_p:.4f} < 0.05)" if addr_p < 0.05 else
               "addressability does NOT predict margin size "
               f"(p={addr_p:.4f} >= 0.05)")
    lines.append(f"## Finding\n{finding}. Pre-registered prediction was the null "
                 "(addressability does not predict margin).")
    REPORT.write_text("\n".join(lines))

    print(f"n={n}  margin mean={margin.mean():.4f} sd={margin.std(ddof=1):.4f} "
          f"min={margin.min():.4f} max={margin.max():.4f}")
    print(f"addressability: coef={addr_coef:.5f} se={addr_se:.5f} t={addr_t:.3f} "
          f"p={addr_p:.4f} (df={df_full})")
    print(f"base_score:     coef={base_coef_full:.5f} se={se_full[1]:.5f} "
          f"t={base_t_full:.3f} p={base_p_full:.4f}")
    print(f"R2 reduced (base only)         = {r2_red:.4f}")
    print(f"R2 full (base + addressability)= {r2_full:.4f}")
    print(f"incremental R2 (addressability)= {r2_full - r2_red:.4f}")
    print(f"\nFinding: {finding}")
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
