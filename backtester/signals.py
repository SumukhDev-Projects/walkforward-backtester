"""Signals. Every signal must use only data available at the close of the row's own date."""
import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 120, skip: int = 5) -> pd.DataFrame:
    """Trailing return from ``lookback`` days ago to ``skip`` days ago (skips short-term reversal)."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0
