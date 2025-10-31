#!/usr/bin/env python3

"""
Runs a fast, numba-accelerated backtest for a random trading strategy
on historical crypto price data from a CSV file.

This script requires the following libraries:
- pandas
- numpy
- matplotlib
- numba

You can install them using pip:
pip install pandas numpy matplotlib numba
"""

import pandas as pd
import numpy as np
import time
from src.simulation import run_all_simulations_parallel
from src.plotting import plot_backtest_results, plot_histories, plot_pnl_distribution


def main():
    """
    Main function to load data, run simulations, and plot results.
    """

    # --- Parameters ---
    CSV_FILE = (
        "data/crypto_prices_3min_ccxt.csv"  # The file saved by the previous script
    )
    PNL_CSV_FILE = (
        "data/historical_pnl_pct.csv"  # The file saved by the previous script
    )
    INITIAL_CAPITAL = 10_000
    TRADING_FEE_RATE = 0.001  # 0.1% (common fee on Binance)
    MAX_BUY_PERC_POWER = 0.5  # Max 10% of available *buying power* on a single 100
    NUM_SIMULATIONS = 10_000  # Number of backtests to run

    # --- New Strategy Parameters ---
    LEVERAGE = 15.0  # e.g., 3x leverage
    AVOID_SELLING_WINNERS = True  # If True, will not sell positions that are in profit

    # --- Strategy Probabilities (must sum to 1.0) ---
    PROB_HOLD = 0.8  # 85% chance to hold
    PROB_BUY = 0.10  # 7.5% chance to buy
    PROB_SELL = 0.10  # 7.5% chance to sell
    # --- End Parameters ---

    # Check probabilities
    if not np.isclose(PROB_HOLD + PROB_BUY + PROB_SELL, 1.0):
        print(
            f"Error: Probabilities must sum to 1.0. (Currently sum to {PROB_HOLD + PROB_BUY + PROB_SELL})"
        )
        return

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"], index_col="timestamp")
        pnl_df = pd.read_csv(
            PNL_CSV_FILE, parse_dates=["timestamp"], index_col="timestamp"
        )
    except FileNotFoundError:
        print(f"Error: File not found: '{CSV_FILE}'")
        print("Please run the 'fetch_crypto_prices_ccxt.py' script first,")
        print("or make sure the CSV_FILE variable matches your filename.")
        return

    print(f"Loaded {len(df)} timesteps for {len(df.columns)} coins.")

    current_pnl_pct = pnl_df.groupby("model_id").cum_pnl_pct.last()
    last_timestamp = pnl_df.index[-1]

    # Convert DataFrame to a NumPy array for Numba.
    # Fill NaNs with a 0 to avoid issues in calculation,
    # the JIT function will check for 0 or NaN prices.
    prices_array = df.fillna(0).values

    # 2. Run Simulations
    print(f"Running {NUM_SIMULATIONS} parallel simulations...")
    start_time = time.time()

    base_seed = int(start_time * 1000)

    all_histories, all_final_pnls = run_all_simulations_parallel(
        prices_array,
        NUM_SIMULATIONS,
        INITIAL_CAPITAL,
        TRADING_FEE_RATE,
        MAX_BUY_PERC_POWER,
        base_seed,
        PROB_HOLD,
        PROB_BUY,
        AVOID_SELLING_WINNERS,
        LEVERAGE,
    )

    end_time = time.time()
    print(f"Simulations complete. Total time: {end_time - start_time:.2f} seconds")

    # 3. Plot Results
    print("Generating plots...")

    current_pnl_dollars = {
        m: INITIAL_CAPITAL * pnl_pct * 0.01
        for m, pnl_pct in current_pnl_pct.sort_values().items()
    }

    plot_backtest_results(
        df.index,
        all_histories,
        all_final_pnls,
        INITIAL_CAPITAL,
        current_pnl_dollars,
        save_name=f"data/backtest_{last_timestamp}.pdf",
    )

    # plot_histories(
    #     df.index,
    #     all_histories,
    #     INITIAL_CAPITAL,
    #     num_to_plot=100,
    #     save_name=f"data/backtest_histories_{last_timestamp}.pdf",
    # )
    # plot_pnl_distribution(
    #     all_final_pnls,
    #     INITIAL_CAPITAL,
    #     save_name=f"data/backtest_disribution_{last_timestamp}.pdf",
    # )

    pnl_p_vals = {
        m: np.mean((pnl) < all_final_pnls) for m, pnl in current_pnl_dollars.items()
    }

    # for model, pnl in current_pnl_dollars.items():
    #     print("\n", model)
    #     print(f"pnl {pnl:,.2f} p = {np.mean((pnl) < all_final_pnls):.3f}")
    print(pd.Series(pnl_p_vals).round(2).rename("p_value").to_markdown())


if __name__ == "__main__":
    main()
