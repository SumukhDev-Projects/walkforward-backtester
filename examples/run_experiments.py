"""Reproduces the README tables. Run: python examples/run_experiments.py"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtester import (simulate_prices, momentum, make_weights, backtest, walk_forward,
                        sharpe)

OUT = pathlib.Path(__file__).parent.parent / "results"; OUT.mkdir(exist_ok=True)
GRID = {"lookback": [20, 40, 60, 120, 250]}
N_SEEDS = 20

def full_sample_best(p, cost):
    """What a careless researcher does: try every lookback on all the data, report the best."""
    best = max(GRID["lookback"], key=lambda lb: sharpe(backtest(p, make_weights(momentum(p, lb)), cost, cost).net))
    return sharpe(backtest(p, make_weights(momentum(p, best)), cost, cost).net)

rows = {}
for name, strength in [("no edge (random walk)", 0.0), ("planted momentum", 0.0006)]:
    careless, careless_gross, oos_sh, ins_sh = [], [], [], []
    for seed in range(N_SEEDS):
        p = simulate_prices(50, 3000, momentum_strength=strength, seed=seed)
        careless.append(full_sample_best(p, 5.0)); careless_gross.append(full_sample_best(p, 0.0))
        oos, ins, folds = walk_forward(p, momentum, GRID, train_days=750, test_days=250, cost_bps=5, slippage_bps=5)
        oos_sh.append(sharpe(oos)); ins_sh.append(float(np.mean(ins)))
    rows[name] = {"full_sample_best_sharpe_gross": float(np.mean(careless_gross)),
                  "full_sample_best_sharpe": float(np.mean(careless)),
                  "walk_forward_in_sample_sharpe": float(np.mean(ins_sh)),
                  "walk_forward_out_of_sample_sharpe": float(np.mean(oos_sh)),
                  "oos_sharpe_std_across_seeds": float(np.std(oos_sh)),
                  "seeds": N_SEEDS}

# Cost sensitivity on the planted-momentum market (single lookback = 120, seeds averaged)
cost_rows = []
for bps in [0, 5, 10, 20, 40]:
    sh, to = [], []
    for seed in range(N_SEEDS):
        p = simulate_prices(50, 3000, momentum_strength=0.0006, seed=seed)
        r = backtest(p, make_weights(momentum(p, 120), rebalance_every=5), bps / 2, bps / 2)
        sh.append(sharpe(r.net)); to.append(r.turnover.mean() * 252)
    cost_rows.append({"round_trip_style_cost_bps": bps, "mean_net_sharpe": float(np.mean(sh)), "ann_turnover": float(np.mean(to))})

# Equity curves for every seed (no cherry-picking), planted-momentum market
fig, ax = plt.subplots(figsize=(7, 3.4))
finals, sharpes = [], []
for seed in range(N_SEEDS):
    p = simulate_prices(50, 3000, momentum_strength=0.0006, seed=seed)
    oos, _, _ = walk_forward(p, momentum, GRID)
    eq = (1 + oos).cumprod(); ax.plot(eq.values, color="#4c78a8", alpha=0.35, lw=1)
    finals.append(float(eq.iloc[-1])); sharpes.append(sharpe(oos))
ax.axhline(1, color="grey", lw=0.8)
ax.set_title("Out-of-sample equity, all 20 seeds (net of costs)"); ax.set_xlabel("test day")
fig.tight_layout(); fig.savefig(OUT / "equity.png", dpi=140)
example = {"seeds_with_positive_oos_sharpe": int(sum(x > 0 for x in sharpes)), "of": N_SEEDS,
           "median_oos_sharpe": float(np.median(sharpes)), "min": float(min(sharpes)), "max": float(max(sharpes))}

res = {"overfitting_test": rows, "cost_sensitivity": cost_rows, "example_run": example}
(OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
