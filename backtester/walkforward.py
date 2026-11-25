"""Walk-forward evaluation: choose parameters on past data, score them on unseen data."""
import itertools
import pandas as pd
from .engine import make_weights, backtest
from .metrics import sharpe


def _run(prices, signal_fn, params, weight_kwargs, cost_bps, slippage_bps):
    scores = signal_fn(prices, **params)
    w = make_weights(scores, **weight_kwargs)
    return backtest(prices, w, cost_bps, slippage_bps)


def walk_forward(prices, signal_fn, param_grid: dict, train_days=750, test_days=250,
                 weight_kwargs=None, cost_bps=5.0, slippage_bps=5.0):
    """Roll a train/test split through time.

    For each fold, the parameter set with the best *net* Sharpe on the train window is used,
    unchanged, on the following test window. Only test-window returns are stitched together,
    so the result is a genuine out-of-sample track record.

    Returns (oos_net_returns, in_sample_best_sharpes, fold_table).
    """
    weight_kwargs = weight_kwargs or {}
    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    # Signals are causal, so computing once on the full history and slicing is equivalent.
    runs = {tuple(c.items()): _run(prices, signal_fn, c, weight_kwargs, cost_bps, slippage_bps) for c in combos}
    oos, folds, ins = [], [], []
    start = 0
    while start + train_days + test_days <= len(prices):
        tr = slice(start, start + train_days)
        te = slice(start + train_days, start + train_days + test_days)
        best, best_s = None, -1e9
        for c in combos:
            s = sharpe(runs[tuple(c.items())].net.iloc[tr])
            if s > best_s:
                best, best_s = c, s
        test_ret = runs[tuple(best.items())].net.iloc[te]
        oos.append(test_ret)
        ins.append(best_s)
        folds.append({"train_start": prices.index[tr.start], "test_start": prices.index[te.start],
                      "test_end": prices.index[te.stop - 1], "params": best,
                      "in_sample_sharpe": best_s, "oos_sharpe": sharpe(test_ret)})
        start += test_days
    return pd.concat(oos), ins, pd.DataFrame(folds)
