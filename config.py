# ── Simulation Defaults ────────────────────────────────────────────────────
N_SIMULATIONS     = 10_000
HORIZON_DAYS      = 10
CONFIDENCE_LEVEL  = 0.95
TRADING_DAYS_YEAR = 252

# ── Student-t degrees of freedom (fixed, not exposed in UI) ───────────────
# ν = 5 is a standard choice in quantitative finance.
# Lower ν → fatter tails. At ν → ∞, Student-t → Normal.
STUDENT_T_DOF = 5

# ── Assets ─────────────────────────────────────────────────────────────────
DEFAULT_TICKERS  = ["SPY", "AAPL", "BTC-USD", "GLD"]
YFINANCE_PERIOD  = "2y"

# ── Portfolio Presets (weights must sum to 1.0) ────────────────────────────
PRESETS = {
    "Conservative": {"SPY": 0.50, "AAPL": 0.10, "BTC-USD": 0.05, "GLD": 0.35},
    "Balanced":     {"SPY": 0.40, "AAPL": 0.30, "BTC-USD": 0.15, "GLD": 0.15},
    "Aggressive":   {"SPY": 0.20, "AAPL": 0.30, "BTC-USD": 0.45, "GLD": 0.05},
}