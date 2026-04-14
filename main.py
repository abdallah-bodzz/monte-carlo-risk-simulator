"""
main.py — FastAPI application.

Two endpoints only:
  GET  /              → serve index.html
  GET  /api/assets    → available tickers + asset metadata
  POST /api/simulate  → run Monte Carlo, return metrics + chart data
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Literal

import config
from src.data       import load_prices, compute_log_returns, compute_statistics
from src.simulation import run_monte_carlo, simulate_paths
from src.metrics    import compute_metrics, build_histogram

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Portfolio Risk Simulator", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load data once at startup ─────────────────────────────────────────────
print("[main] Loading market data...")
_prices, _data_source = load_prices(config.DEFAULT_TICKERS, config.YFINANCE_PERIOD)
_log_returns          = compute_log_returns(_prices)
_stats                = compute_statistics(_log_returns)
print(f"[main] Ready. Tickers: {_stats['tickers']} | Source: {_data_source} | Obs: {_stats['n_obs']}")


# ── Request schema ────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    weights:       dict[str, float]
    n_simulations: int   = config.N_SIMULATIONS
    horizon_days:  int   = config.HORIZON_DAYS
    confidence:    float = config.CONFIDENCE_LEVEL
    distribution:  Literal["normal", "student_t"] = "normal"

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Weights must sum to 1. Got {total:.4f}")
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v):
        if not (0.80 <= v < 1.0):
            raise ValueError("confidence must be between 0.80 and 0.99")
        return v

    @field_validator("n_simulations")
    @classmethod
    def valid_sims(cls, v):
        if not (1000 <= v <= 100_000):
            raise ValueError("n_simulations must be between 1,000 and 100,000")
        return v


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def serve_index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/api/assets")
def get_assets():
    """Return asset metadata, presets, and data source info."""
    asset_stats = []
    for t in _stats["tickers"]:
        asset_stats.append({
            "ticker":        t,
            "annual_ret_pct": round(float(_stats["annual_mu"][t]) * 100, 2),
            "annual_vol_pct": round(float(_stats["annual_vol"][t]) * 100, 2),
        })
    return {
        "tickers":     _stats["tickers"],
        "n_obs":       _stats["n_obs"],
        "date_from":   str(_prices.index[0].date()),
        "date_to":     str(_prices.index[-1].date()),
        "data_source": _data_source,   # "yfinance" or "synthetic"
        "asset_stats": asset_stats,
        "presets":     config.PRESETS,
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """
    Run Monte Carlo simulation.
    Returns metrics and all chart data needed by the frontend.
    """
    # ── Align weights to available tickers ─────────────────────────────
    available = _stats["tickers"]
    tickers   = [t for t in available if t in req.weights]
    if not tickers:
        raise HTTPException(400, "None of the requested tickers are available.")

    raw_w  = np.array([req.weights[t] for t in tickers])
    weights = raw_w / raw_w.sum()   # normalise to handle floating-point

    mu_d  = _stats["daily_mu"][tickers]
    cov_d = _stats["daily_cov"].loc[tickers, tickers]

    # ── Simulate ────────────────────────────────────────────────────────
    sim_returns = run_monte_carlo(
        weights       = weights,
        mu_daily      = mu_d,
        cov_daily     = cov_d,
        n_simulations = req.n_simulations,
        horizon_days  = req.horizon_days,
        distribution  = req.distribution,
    )

    # ── Metrics ─────────────────────────────────────────────────────────
    m = compute_metrics(sim_returns, req.confidence)

    # ── Histogram ───────────────────────────────────────────────────────
    hist = build_histogram(sim_returns)

    # ── Paths (fan chart) ────────────────────────────────────────────────
    paths = simulate_paths(
        weights      = weights,
        mu_daily     = mu_d,
        cov_daily    = cov_d,
        n_paths      = 80,
        horizon_days = req.horizon_days,
        distribution = req.distribution,
    )

    return {
        "metrics": {
            "var_pct":          m.var_pct,
            "es_pct":           m.es_pct,
            "mean_return_pct":  m.mean_return_pct,
            "std_pct":          m.std_pct,
            "prob_loss_pct":    m.prob_loss_pct,
            "skewness":         m.skewness,
            "kurtosis":         m.kurtosis,
            "worst_pct":        m.worst_pct,
            "best_pct":         m.best_pct,
            "confidence":       req.confidence,
            "horizon_days":     req.horizon_days,
            "n_simulations":    req.n_simulations,
            "distribution":     req.distribution,
        },
        "histogram": {
            "centers":       hist["centers"],
            "counts":        hist["counts"],
            "var_line":      -m.var_pct,       # in % space, negative = loss
            "es_line":       -m.es_pct,
        },
        "paths": {
            "values": paths[::2].tolist(),    # every other path to reduce payload
            "days":   list(range(req.horizon_days + 1)),
        },
        "portfolio": {
            "tickers": tickers,
            "weights": weights.tolist(),
        },
        "data_source": _data_source,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)