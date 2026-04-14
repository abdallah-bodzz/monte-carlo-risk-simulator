"""
data.py — Price loading and return computation.

We use LOG returns throughout: r_t = ln(P_t / P_{t-1})

Why log returns (not simple returns)?
  • Additive over time: r_total = r_1 + r_2 + ... + r_T
  • Simple returns are multiplicative — multi-period simulation requires
    compounding, which introduces error when scaled with √T.
  • Log returns are better approximated by a normal distribution.
"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Literal

warnings.filterwarnings("ignore")

DataSource = Literal["yfinance", "synthetic"]


def fetch_yfinance(tickers: list[str], period: str = "2y") -> pd.DataFrame | None:
    """Fetch daily adjusted close prices via yfinance."""
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return None
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        return prices.ffill().dropna(how="all")
    except Exception as e:
        print(f"[data] yfinance failed: {e}")
        return None


def generate_synthetic(tickers: list[str], n_days: int = 504) -> pd.DataFrame:
    """
    Synthetic prices via Geometric Brownian Motion.
    Uses realistic annualised μ and σ per asset class.
    Shown in the UI with a clear 'DEMO' label — not real data.
    """
    np.random.seed(42)
    params = {
        "SPY":     {"mu": 0.10, "sigma": 0.18},
        "AAPL":    {"mu": 0.20, "sigma": 0.30},
        "BTC-USD": {"mu": 0.50, "sigma": 0.80},
        "GLD":     {"mu": 0.05, "sigma": 0.14},
    }
    default = {"mu": 0.08, "sigma": 0.20}
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")
    prices = {}
    for t in tickers:
        p = params.get(t, default)
        dt = 1 / 252
        r = np.exp(
            (p["mu"] - 0.5 * p["sigma"] ** 2) * dt
            + p["sigma"] * np.sqrt(dt) * np.random.randn(n_days)
        )
        prices[t] = 100 * np.cumprod(r)
    return pd.DataFrame(prices, index=dates)


def load_prices(tickers: list[str], yfinance_period: str = "2y") -> tuple[pd.DataFrame, DataSource]:
    """
    Load prices: yfinance first, synthetic fallback.
    Returns (prices_df, source_label).
    """
    prices = fetch_yfinance(tickers, yfinance_period)
    if prices is not None and not prices.empty:
        # Keep only requested tickers that came back
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
        print(f"[data] yfinance OK | {prices.shape[0]} days | {list(prices.columns)}")
        return prices, "yfinance"

    print("[data] Using synthetic GBM prices (demo mode)")
    return generate_synthetic(tickers), "synthetic"


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}). First row is NaN — dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_statistics(log_returns: pd.DataFrame) -> dict:
    """
    Compute daily and annualised statistics.

    Annualisation:
      μ_annual = μ_daily × 252          (mean scales linearly with time)
      Σ_annual = Σ_daily × 252          (covariance scales linearly with time)
      σ_annual = σ_daily × √252         (std dev scales with √T)
    """
    T = 252
    return {
        "tickers":   list(log_returns.columns),
        "n_obs":     len(log_returns),
        "daily_mu":  log_returns.mean(),
        "daily_cov": log_returns.cov(),
        "annual_mu":  log_returns.mean() * T,
        "annual_vol": log_returns.std() * np.sqrt(T),
        "corr":      log_returns.corr(),
    }