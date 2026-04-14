"""
metrics.py — Value at Risk, Expected Shortfall, and summary statistics.

────────────────────────────────────────────────────────────────────────────
VALUE AT RISK (VaR)
────────────────────────────────────────────────────────────────────────────
  VaR_c = -Quantile(R, 1 - c)

  Interpretation: "With probability c, the portfolio loss will NOT exceed
  VaR over the horizon."

  Example: 95% VaR = 5.2% → there is only a 5% chance of losing more
  than 5.2% over the specified number of trading days.

  VaR tells you the THRESHOLD. It says nothing about what lies beyond it.

────────────────────────────────────────────────────────────────────────────
EXPECTED SHORTFALL (ES)  — also called CVaR, Conditional VaR
────────────────────────────────────────────────────────────────────────────
  ES_c = -E[ R | R ≤ -VaR_c ]

  Interpretation: "Given that we ARE in the worst (1-c)% of outcomes,
  what is the average loss?"

  ES ≥ VaR always. The ratio ES/VaR quantifies tail heaviness.

  Why ES > VaR in practice:
    Two portfolios can have identical VaR but one has a far worse tail.
    Basel III (2016) replaced VaR with ES as the regulatory standard
    for exactly this reason — ES captures tail severity, not just the
    threshold.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    var_pct:         float   # VaR expressed as % loss (positive = loss)
    es_pct:          float   # ES expressed as % loss  (always ≥ var_pct)
    mean_return_pct: float   # Mean simulated portfolio return (%)
    std_pct:         float   # Std deviation of returns (%)
    prob_loss_pct:   float   # % of simulations ending in a loss
    skewness:        float   # Negative = more downside than upside
    kurtosis:        float   # Excess kurtosis; >0 = fatter tails than Normal
    worst_pct:       float   # Single worst simulation (%)
    best_pct:        float   # Single best simulation (%)
    confidence:      float   # Confidence level used
    n_simulations:   int


def compute_metrics(returns: np.ndarray, confidence: float = 0.95) -> RiskMetrics:
    """
    Compute all risk metrics from an array of simulated portfolio log-returns.

    Parameters
    ----------
    returns    : 1-D array of simulated log-returns (one value per simulation)
    confidence : e.g. 0.95 for 95% VaR / ES

    Returns
    -------
    RiskMetrics dataclass
    """
    # ── VaR ────────────────────────────────────────────────────────────────
    # The (1-c) percentile of the distribution.
    # We negate so VaR is expressed as a positive loss number.
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    var = -var_threshold

    # ── ES ─────────────────────────────────────────────────────────────────
    # Mean of all returns that fall AT or BELOW the VaR threshold.
    tail = returns[returns <= var_threshold]
    es = -np.mean(tail) if len(tail) > 0 else var

    # ── Higher moments ──────────────────────────────────────────────────────
    mu  = returns.mean()
    sig = returns.std()
    skew = float(np.mean(((returns - mu) / sig) ** 3)) if sig > 0 else 0.0
    kurt = float(np.mean(((returns - mu) / sig) ** 4) - 3) if sig > 0 else 0.0

    return RiskMetrics(
        var_pct         = round(float(var)  * 100, 4),
        es_pct          = round(float(es)   * 100, 4),
        mean_return_pct = round(float(mu)   * 100, 4),
        std_pct         = round(float(sig)  * 100, 4),
        prob_loss_pct   = round(float(np.mean(returns < 0)) * 100, 2),
        skewness        = round(skew, 4),
        kurtosis        = round(kurt, 4),
        worst_pct       = round(float(returns.min()) * 100, 4),
        best_pct        = round(float(returns.max()) * 100, 4),
        confidence      = confidence,
        n_simulations   = len(returns),
    )


def build_histogram(returns: np.ndarray, bins: int = 120) -> dict:
    """
    Compute histogram data for the distribution chart.

    Returns bin centers and counts — ready for Chart.js bar chart.
    """
    counts, edges = np.histogram(returns * 100, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {
        "centers": centers,
        "counts":  counts.tolist(),
        "edges":   edges.tolist(),
    }