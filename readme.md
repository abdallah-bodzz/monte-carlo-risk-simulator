# Monte Carlo Portfolio Risk Simulator

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009485?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![yfinance](https://img.shields.io/badge/yfinance-4B8BBE?style=flat-square&logo=yahoo&logoColor=white)](https://pypi.org/project/yfinance/)

[![Monte Carlo](https://img.shields.io/badge/Monte_Carlo-8B0000?style=flat-square&logo=python&logoColor=white)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
[![Cholesky](https://img.shields.io/badge/Cholesky_Decomposition-4B0082?style=flat-square)](https://en.wikipedia.org/wiki/Cholesky_decomposition)
[![VaR & ES](https://img.shields.io/badge/VaR_%26_Expected_Shortfall-006400?style=flat-square)](https://en.wikipedia.org/wiki/Value_at_risk)
[![Quantitative Finance](https://img.shields.io/badge/Quantitative_Finance-1E3A8A?style=flat-square&logo=chart-dot-js&logoColor=white)](https://en.wikipedia.org/wiki/Quantitative_finance)

[![License: MIT](https://img.shields.io/badge/License-MIT-2E8B57?style=flat-square)](LICENSE)

A quantitative risk engine that simulates portfolio return distributions using correlated multi-asset Monte Carlo methods, and computes institutional-grade risk metrics — VaR and Expected Shortfall — with support for both Gaussian and fat-tailed Student-t distributions.

> **Development note:** Core simulation logic, mathematical framework, and application architecture were designed and implemented from scratch. The frontend UI in `index.html` was generated with AI assistance (Claude by Anthropic) based on a detailed design spec.

---

## What It Does

Given a portfolio of assets with user-defined weights, the simulator asks one question:

> **How bad can this portfolio realistically get over the next N trading days?**

It answers by running thousands of correlated Monte Carlo simulations — each one a full day-by-day path through simulated market returns — and then extracting two risk numbers that matter: **VaR** (the loss threshold) and **ES** (how bad it gets *beyond* that threshold).

---

## Why This Project Exists

Risk estimation is a core problem in quantitative finance. The naive approach — treating each asset independently and assuming normally distributed returns — is mathematically wrong on two fronts:

1. **Assets are correlated.** SPY and AAPL move together; GLD doesn't. Ignoring this misstates the actual diversification of the portfolio.
2. **Real markets have fat tails.** Extreme moves happen far more often than a normal distribution predicts. The 2008 crash, March 2020, etc. — all "statistically impossible" under Gaussian assumptions.

This simulator addresses both problems explicitly.

---

## Features

- Monte Carlo simulation with **10,000+ paths** (configurable up to 50,000)
- **Cholesky decomposition** to correctly model cross-asset correlation
- **Normal and Student-t (ν=5)** return distributions
- **Value at Risk (VaR)** and **Expected Shortfall (ES)** at configurable confidence levels (80–99%)
- Live market data via `yfinance` with synthetic GBM fallback (demo mode)
- Fan chart of simulated portfolio paths + full return distribution histogram
- Portfolio weight presets: Conservative, Balanced, Aggressive
- Clean FastAPI backend + single-page vanilla JS frontend

---

## Mathematical Foundation

### 1. Log Returns

The simulator uses **log returns** throughout, not simple percentage returns.

```
r_t = ln(P_t / P_{t-1})
```

The critical property is **additivity over time**:

```
r_total = r_1 + r_2 + ... + r_T
```

Simple returns compound multiplicatively: `(1+r_total) = (1+r_1)(1+r_2)···(1+r_T)`. Summing them directly across T days — as Monte Carlo does — would be mathematically incorrect. Log returns are additive, making them the natural unit for simulation. They also distribute more symmetrically, which improves the fit of both Normal and Student-t distributions.

---

### 2. Cholesky Decomposition — Correlated Asset Simulation

This is the mathematical core of the engine.

Assets are correlated. Simulating each with independent random draws ignores this and overstates the benefit of diversification (or understates it for correlated bets).

**Process:**

1. Estimate the historical **covariance matrix Σ** from 2 years of daily log returns.
2. Decompose: `Σ = L · Lᵀ` (Cholesky factorisation, L is lower-triangular).
3. Draw independent standard-normal vectors: `Z ~ N(0, I)`.
4. Transform: `X = L · Z` → gives `X ~ N(0, Σ)`.

```
X = L · Z   ⟹   X ~ N(0, Σ)
```

The resulting draws carry **exactly** the cross-asset correlation structure observed in historical data. Every simulation path is internally consistent across assets.

---

### 3. Return Distributions

**Normal (Gaussian):**

Standard assumption. Underestimates tail risk. Good baseline.

```
Z ~ N(0, 1)
```

**Student-t (fat tails), ν = 5:**

Student-t with 5 degrees of freedom has roughly twice the tail mass of Normal. To maintain consistent variance before applying the Cholesky step, draws are standardised:

```
Z_raw ~ t(ν=5)
Z = Z_raw / √(ν/(ν−2))   →   Var(Z) = 1
```

The Cholesky step is then applied identically. As ν → ∞, Student-t converges to Normal. Basel III and most modern quant risk systems use fat-tailed distributions precisely because Gaussian models systematically underestimate extreme losses.

---

### 4. Simulation Loop

For each of N simulation paths:

1. Draw correlated daily shocks for each asset via Cholesky: `Z_corr = Z @ L.T`
2. Add daily drift: `r_daily[t] = μ + Z_corr[t]` (per asset, per day)
3. Sum log returns over T days (additivity): `r_total = Σ r_daily`
4. Apply portfolio weights: `r_portfolio = w · r_total`

```python
Z_corr = Z @ L.T                        # shape: (n_sims, T, n_assets)
daily_returns = mu_daily + Z_corr       # add drift
total_returns = daily_returns.sum(axis=1)  # sum over T days
portfolio_returns = total_returns @ weights  # scalar per sim
```

No `√T` shortcuts. Daily parameters are used directly and summed — this is exact, not approximated.

---

### 5. Value at Risk (VaR)

```
VaR_c = −Quantile(R, 1 − c)
```

With 95% confidence: only 5% of the 10,000 simulated paths produce a worse result than VaR.

VaR is a **threshold**, not a description of the tail. It answers "where does the bad zone start?" — not "how bad does it get?"

---

### 6. Expected Shortfall (ES)

```
ES_c = −E[ R | R ≤ −VaR_c ]
```

ES is the **average loss given that you are already in the worst (1−c)% of scenarios**. It always satisfies ES ≥ VaR.

The ratio **ES/VaR** quantifies tail severity:
- Ratio ~1.1–1.2 → light tail, losses beyond VaR are contained
- Ratio ~1.5+ → heavy tail, losses beyond VaR can be substantially worse

**Basel III (2016) replaced VaR with ES as the regulatory standard** for market risk precisely because ES captures tail severity rather than just the threshold. VaR tells you the door to the bad zone; ES tells you how far in you go.

---

### 7. Annualisation

```
μ_annual = μ_daily × 252
Σ_annual = Σ_daily × 252
σ_annual = σ_daily × √252
```

Variance scales linearly with time; standard deviation scales with √T. These are used for display purposes only. The simulation itself operates on raw daily parameters and sums T days directly.

---

## Project Structure

```
.
├── main.py          # FastAPI application — routes, request validation, orchestration
├── config.py        # All constants: tickers, simulation defaults, portfolio presets
├── data.py          # Price loading (yfinance / synthetic GBM fallback), log return computation
├── simulation.py    # Monte Carlo engine — Cholesky decomposition, correlated path generation
├── metrics.py       # VaR, ES, skewness, kurtosis, histogram builder
└── index.html       # Single-page frontend — Chart.js visualisations, slider controls
```

**Data flow:**

```
startup → data.py (load prices, compute Σ, μ)
              ↓
POST /api/simulate → simulation.py (Cholesky, draw paths)
              ↓
           metrics.py (VaR, ES, stats)
              ↓
           JSON response → index.html (render charts + metrics)
```

---

## API

### `GET /api/assets`

Returns available tickers, annualised return/volatility per asset, date range, data source, and portfolio presets.

### `POST /api/simulate`

**Request:**
```json
{
  "weights":       { "SPY": 0.40, "AAPL": 0.30, "BTC-USD": 0.15, "GLD": 0.15 },
  "n_simulations": 10000,
  "horizon_days":  10,
  "confidence":    0.95,
  "distribution":  "student_t"
}
```

**Response:**
```json
{
  "metrics": {
    "var_pct": 4.82,
    "es_pct":  7.31,
    "mean_return_pct": 0.24,
    "std_pct": 2.91,
    "prob_loss_pct": 46.2,
    "skewness": -0.31,
    "kurtosis":  1.84,
    "worst_pct": -22.4,
    "best_pct":  19.7
  },
  "histogram": { "centers": [...], "counts": [...], "var_line": -4.82, "es_line": -7.31 },
  "paths":     { "values": [[...], ...], "days": [0, 1, ..., 10] }
}
```

---

## Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Numerical | NumPy, Pandas |
| Market data | yfinance (live) / synthetic GBM (fallback) |
| Frontend | Vanilla JS, Chart.js 4 |
| Validation | Pydantic v2 |

No ML frameworks. No ORMs. No unnecessary abstractions. The simulation is pure NumPy.

---

## Setup

```bash
pip install fastapi uvicorn numpy pandas yfinance pydantic
python main.py
# → http://localhost:8000
```

---

## Key Design Decisions

**Why log returns, not simple returns?**
Log returns are additive over time, which is exactly what Monte Carlo needs — daily returns summed across T days. Simple returns compound; summing them introduces systematic error.

**Why Cholesky and not just correlated random numbers directly?**
Cholesky is the standard numerical method for sampling from a multivariate normal with a given covariance structure. It guarantees the output distribution has exactly the target Σ, and it's numerically stable for the near-singular covariance matrices that appear with correlated assets.

**Why Student-t with ν=5 specifically?**
ν=5 is the standard choice in quantitative finance literature (see McNeil, Frey & Embrechts, *Quantitative Risk Management*, 2005). It produces meaningful fat tails without extreme instability — at ν=2 the variance doesn't even exist; at ν=10+ you're essentially back to Normal.

**Why ES over VaR for the primary metric?**
VaR tells you the threshold. ES tells you the expected severity once you're beyond it. A portfolio with 5% VaR of 3% and another with 5% VaR of 3% but ES of 12% are completely different risk profiles. ES is the more informative number — which is why regulators agree.

---

## Limitations (honest)

- Uses historical covariance, which is backward-looking. Covariance is not stable across market regimes.
- 2-year lookback window smooths over crisis periods.
- No options, no leverage, no liquidity adjustments — this is a pure return-distribution model.
- Not production-grade: no authentication, no persistent storage, no rate limiting.
