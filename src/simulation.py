"""
simulation.py — Monte Carlo simulation engine.

────────────────────────────────────────────────────────────────────────────
THE CORE PROBLEM: correlated multi-asset returns
────────────────────────────────────────────────────────────────────────────

Simulating each asset independently is WRONG. Assets are correlated:
SPY and AAPL tend to move together; GLD moves differently from equities.
Ignoring this misstates the portfolio's true risk.

SOLUTION — Cholesky Decomposition:
  Given covariance matrix Σ (n×n), decompose it as:
      Σ = L @ L.T          (L is a lower-triangular matrix)

  For independent standard normal draws Z ~ N(0, I):
      X = L @ Z            →  X ~ N(0, Σ)

  X has exactly the correlation structure of the real data.

────────────────────────────────────────────────────────────────────────────
TWO DISTRIBUTIONS SUPPORTED
────────────────────────────────────────────────────────────────────────────

  Normal (Gaussian):
    Standard assumption. Underestimates tail risk in practice.
    Z ~ N(0, 1)

  Student-t (fat tails), ν = 5:
    Larger probability of extreme moves — more realistic.
    Standardised: divide raw t draws by √(ν/(ν-2)) so Var = 1,
    then apply the same Cholesky step.
    At ν → ∞, Student-t converges to Normal.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from typing import Literal

DistributionType = Literal["normal", "student_t"]
DOF = 5  # fixed degrees of freedom for Student-t


def _cholesky(cov: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition of a covariance matrix.
    Adds a small regularisation term if the matrix is near-singular
    (can happen with highly correlated assets or few observations).
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Nugget regularisation: shift eigenvalues just above zero
        nugget = 1e-8 * np.eye(len(cov))
        return np.linalg.cholesky(cov + nugget)


def run_monte_carlo(
    weights:      np.ndarray,
    mu_daily:     np.ndarray,
    cov_daily:    np.ndarray,
    n_simulations: int,
    horizon_days:  int,
    distribution:  DistributionType = "normal",
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate portfolio log-returns over `horizon_days` trading days.

    Steps per simulation path:
      1. Draw correlated random shocks for each day using Cholesky.
      2. Add the daily drift (μ_daily) to each shock.
      3. Sum daily asset returns over T days (log returns are additive).
      4. Apply portfolio weights → scalar portfolio return.

    Parameters
    ----------
    weights        : shape (n_assets,), must sum to 1
    mu_daily       : shape (n_assets,), daily mean log-returns
    cov_daily      : shape (n_assets, n_assets), daily covariance
    n_simulations  : number of Monte Carlo paths
    horizon_days   : forward-looking window in trading days
    distribution   : "normal" or "student_t"

    Returns
    -------
    portfolio_returns : shape (n_simulations,)
                        Simulated total log-return over the horizon.
    """
    np.random.seed(seed)
    n = len(weights)

    cov_arr = cov_daily.values if hasattr(cov_daily, "values") else cov_daily
    mu_arr  = mu_daily.values  if hasattr(mu_daily,  "values") else mu_daily

    # ── Step 1: Cholesky factorisation ─────────────────────────────────────
    L = _cholesky(cov_arr)

    # ── Step 2: Draw independent shocks ────────────────────────────────────
    # Shape: (n_simulations, horizon_days, n_assets)
    if distribution == "normal":
        Z = np.random.standard_normal((n_simulations, horizon_days, n))

    else:  # student_t
        # Standardise so that Var(Z) = 1, matching the Normal baseline
        scale = np.sqrt(DOF / (DOF - 2))
        Z = np.random.standard_t(df=DOF, size=(n_simulations, horizon_days, n)) / scale

    # ── Step 3: Introduce asset correlation via Cholesky ───────────────────
    # Z_corr[i, t, :] = L @ Z[i, t, :]   for all i, t
    Z_corr = Z @ L.T   # shape: (n_sims, horizon_days, n_assets)

    # ── Step 4: Add drift and sum over time ────────────────────────────────
    # daily_returns[i, t, j] = mu_j + Z_corr[i, t, j]
    daily_returns = mu_arr + Z_corr                    # (n_sims, T, n)
    total_returns = daily_returns.sum(axis=1)          # (n_sims, n) — sum over T days

    # ── Step 5: Apply portfolio weights ────────────────────────────────────
    portfolio_returns = total_returns @ weights        # (n_sims,)

    return portfolio_returns


def simulate_paths(
    weights:     np.ndarray,
    mu_daily:    np.ndarray,
    cov_daily:   np.ndarray,
    n_paths:     int = 80,
    horizon_days: int = 10,
    distribution: DistributionType = "normal",
    seed: int = 99,
) -> np.ndarray:
    """
    Simulate individual daily portfolio value paths for the fan chart.

    Returns
    -------
    paths : shape (n_paths, horizon_days + 1)
            paths[:, 0] = 1.0  (normalised starting value)
            paths[:, t] = portfolio value at end of day t
    """
    np.random.seed(seed)
    n = len(weights)

    cov_arr = cov_daily.values if hasattr(cov_daily, "values") else cov_daily
    mu_arr  = mu_daily.values  if hasattr(mu_daily,  "values") else mu_daily

    L = _cholesky(cov_arr)

    if distribution == "normal":
        Z = np.random.standard_normal((n_paths, horizon_days, n))
    else:
        scale = np.sqrt(DOF / (DOF - 2))
        Z = np.random.standard_t(df=DOF, size=(n_paths, horizon_days, n)) / scale

    Z_corr = Z @ L.T
    daily_portfolio = (mu_arr + Z_corr) @ weights     # (n_paths, horizon_days)

    # Cumulative portfolio value: exp(sum of log returns up to day t)
    cum_values = np.exp(np.cumsum(daily_portfolio, axis=1))  # (n_paths, T)
    paths = np.hstack([np.ones((n_paths, 1)), cum_values])    # prepend t=0 = 1.0

    return paths