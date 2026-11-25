import numpy as np
import pandas as pd
import pytest
from backtester import (simulate_prices, momentum, backtest, make_weights, assert_no_lookahead,
                        walk_forward, sharpe, max_drawdown)


def toy_prices():
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame({"A": [100, 110, 121, 121, 121, 121.0], "B": [100, 100, 100, 100, 100, 100.0]}, index=idx)


def test_weights_are_delayed_one_day():
    p = toy_prices()
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=p.index)
    r = backtest(p, w, cost_bps=0, slippage_bps=0)
    # a weight decided on day 0 earns day 1's return (+10%), not day 0's
    assert r.gross.iloc[0] == 0 and r.gross.iloc[1] == pytest.approx(0.10)
    assert r.gross.iloc[2] == pytest.approx(0.10)


def test_costs_charged_on_turnover():
    p = toy_prices()
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=p.index)
    r = backtest(p, w, cost_bps=10, slippage_bps=0)
    # buying 100% once costs 10 bps on the day the position is first held (day 1)
    assert (r.gross - r.net).sum() == pytest.approx(10 / 1e4)


def test_zero_trading_means_zero_cost():
    p = toy_prices()
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    r = backtest(p, w, cost_bps=50, slippage_bps=50)
    assert r.net.abs().sum() == 0


def test_long_short_weights_are_dollar_neutral():
    p = simulate_prices(20, 300, seed=1)
    w = make_weights(momentum(p, 60), quantile=0.25, rebalance_every=5)
    live = w.iloc[100:]
    assert np.allclose(live.sum(axis=1), 0, atol=1e-9)
    assert np.allclose(live.abs().sum(axis=1), 2, atol=1e-9)


def test_lookahead_guard_accepts_causal_signal():
    assert_no_lookahead(lambda p: momentum(p, 60), simulate_prices(10, 400, seed=2))


def test_lookahead_guard_catches_leaky_signal():
    leaky = lambda p: p.shift(-5) / p - 1.0      # uses the price 5 days in the future
    with pytest.raises(AssertionError):
        assert_no_lookahead(leaky, simulate_prices(10, 400, seed=3))


def test_leaky_signal_looks_amazing_but_causal_does_not():
    p = simulate_prices(30, 1500, seed=4)  # no edge exists
    leaky = make_weights(p.shift(-5) / p - 1.0)
    honest = make_weights(momentum(p, 60))
    assert sharpe(backtest(p, leaky).net) > 3
    assert abs(sharpe(backtest(p, honest).net)) < 1.5


def test_walk_forward_returns_only_out_of_sample_days():
    p = simulate_prices(20, 2000, seed=5)
    oos, ins, folds = walk_forward(p, momentum, {"lookback": [20, 60]}, train_days=500, test_days=250)
    assert oos.index.min() >= p.index[500]
    assert oos.index.is_unique and len(oos) == 250 * len(folds)
    assert (folds["test_start"].iloc[1:].to_numpy() > folds["test_end"].iloc[:-1].to_numpy()).all()


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_momentum_found_when_planted_and_not_when_absent():
    sig = simulate_prices(50, 3000, momentum_strength=0.0008, seed=6)
    noise = simulate_prices(50, 3000, momentum_strength=0.0, seed=6)
    g = lambda p: sharpe(backtest(p, make_weights(momentum(p, 120)), 0, 0).net)
    assert g(sig) > 1.0 and abs(g(noise)) < 1.0
