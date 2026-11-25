import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    return float("nan") if len(r) < 2 or sd == 0 else float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + pd.Series(returns).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def bootstrap_sharpe_ci(returns, n_boot=2000, block=20, seed=0, level=0.95):
    """Moving-block bootstrap confidence interval for the annualised Sharpe ratio."""
    r = np.asarray(pd.Series(returns).dropna())
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(r) / block))
    starts_max = len(r) - block
    stats = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([r[i:i + block] for i in s])[:len(r)]
        stats[b] = sample.mean() / sample.std(ddof=1) * np.sqrt(TRADING_DAYS)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarize(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = pd.Series(returns).dropna()
    out = {
        "ann_return": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "days": int(len(r)),
    }
    if turnover is not None:
        out["ann_turnover"] = float(pd.Series(turnover).mean() * TRADING_DAYS)
    return out
