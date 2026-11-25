"""Price data: a simulator with a *known* ground truth, and a CSV loader for real data."""
import numpy as np
import pandas as pd


def simulate_prices(n_assets=50, n_days=3000, momentum_strength=0.0, seed=0,
                    market_vol=0.010, idio_vol=0.015, persistence=0.995):
    """Simulate daily prices from a one-factor model.

    Each asset's expected daily return ``mu`` is a slowly-moving AR(1) process whose
    stationary standard deviation is ``momentum_strength`` (per day). With 0 the market is a
    pure random walk, so any profitable strategy found on it is overfitting by construction.
    With a positive value, past returns carry real information about future returns.
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros((n_days, n_assets))
    if momentum_strength > 0:
        innov_sd = momentum_strength * np.sqrt(1 - persistence ** 2)
        mu[0] = rng.normal(0, momentum_strength, n_assets)
        for t in range(1, n_days):
            mu[t] = persistence * mu[t - 1] + rng.normal(0, innov_sd, n_assets)
    market = rng.normal(0, market_vol, (n_days, 1))
    beta = rng.uniform(0.6, 1.4, (1, n_assets))
    rets = mu + beta * market + rng.normal(0, idio_vol, (n_days, n_assets))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])


def load_prices_csv(path, date_col="Date"):
    """Load a wide CSV (one column per ticker, adjusted close) into a price DataFrame."""
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col).sort_index()
    return df.dropna(how="all").ffill()
