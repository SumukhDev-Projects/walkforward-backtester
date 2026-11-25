"""Vectorised long/short backtester with trading costs and a look-ahead guard."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    gross: pd.Series      # daily returns before costs
    net: pd.Series        # daily returns after costs
    weights: pd.DataFrame # weights actually held each day
    turnover: pd.Series   # one-way-sum |dw| per day


def make_weights(scores: pd.DataFrame, quantile: float = 0.2, long_only: bool = False,
                 rebalance_every: int = 5) -> pd.DataFrame:
    """Rank assets each rebalance date; long the top quantile, short the bottom.

    Long/short books are dollar neutral (+1 long, -1 short). Weights are held between
    rebalances. Returned weights are *targets decided at the close of that row's date*.
    """
    n = scores.shape[1]
    k = max(1, int(round(n * quantile)))
    ranks = scores.rank(axis=1, method="first")
    valid = scores.notna().all(axis=1)
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    top = ranks.gt(n - k)
    bottom = ranks.le(k)
    w[top] = 1.0 / k
    if not long_only:
        w[bottom] = -1.0 / k
    w[~valid] = np.nan
    rebal = np.zeros(len(w), dtype=bool)
    rebal[::rebalance_every] = True
    w[~rebal] = np.nan
    return w.ffill().fillna(0.0)


def backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, cost_bps: float = 5.0,
             slippage_bps: float = 5.0) -> BacktestResult:
    """Apply weights decided at close t to the return from t to t+1, and charge costs.

    ``cost_bps`` and ``slippage_bps`` are each charged per unit of traded notional.
    """
    rets = prices.pct_change().fillna(0.0)
    held = target_weights.shift(1).fillna(0.0)          # the one-day delay that prevents look-ahead
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * (cost_bps + slippage_bps) / 1e4
    return BacktestResult(gross=gross, net=net, weights=held, turnover=turnover)


def assert_no_lookahead(signal_fn, prices: pd.DataFrame, n_checks: int = 5, seed: int = 0) -> None:
    """Raise if ``signal_fn`` at date t changes when data after t is removed.

    A causal signal gives identical values whether or not the future exists.
    """
    full = signal_fn(prices)
    rng = np.random.default_rng(seed)
    cuts = rng.integers(int(len(prices) * 0.3), len(prices) - 1, n_checks)
    for c in cuts:
        trunc = signal_fn(prices.iloc[: c + 1])
        a, b = full.iloc[: c + 1], trunc
        both = a.notna() & b.notna()
        if not np.allclose(a.where(both).to_numpy(), b.where(both).to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True):
            raise AssertionError(f"signal uses future data (differs when truncated at row {c})")
        if (a.notna() != b.notna()).to_numpy().any():
            raise AssertionError(f"signal availability depends on future data (row {c})")
