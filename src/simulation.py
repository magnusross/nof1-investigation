"""
Simulation module for running backtests.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def run_simulation_jit(
    prices,
    initial_capital,
    fee_rate,
    max_buy_perc_power,
    prob_hold,
    prob_buy,
    avoid_selling_winners,
    leverage,
):
    """
    Runs a single, JIT-compiled simulation of the random trading strategy.
    """
    # ...existing code from run_simulation_jit...
    num_timesteps, num_coins = prices.shape

    cash = initial_capital
    positions = np.zeros(num_coins)
    positions_avg_cost = np.zeros(num_coins)
    portfolio_value_history = np.zeros(num_timesteps)

    prob_hold_buy = prob_hold + prob_buy
    max_exposure = initial_capital * leverage

    for t in range(num_timesteps):
        current_prices = prices[t]
        mtm_value = cash
        current_position_value = 0.0
        for i_pos in range(num_coins):
            pos_val = positions[i_pos] * current_prices[i_pos]
            mtm_value += pos_val
            current_position_value += pos_val

        portfolio_value_history[t] = mtm_value

        if np.isnan(mtm_value):
            if t > 0:
                portfolio_value_history[t] = portfolio_value_history[t - 1]
            else:
                portfolio_value_history[t] = initial_capital
            continue

        if mtm_value < 0:
            portfolio_value_history[t + 1 :] = mtm_value

            return portfolio_value_history, -initial_capital

        available_to_deploy = max_exposure - current_position_value

        for i in range(num_coins):
            r = np.random.rand()
            action = 0
            if r > prob_hold:
                if r < prob_hold_buy:
                    action = 1
                else:
                    action = 2

            current_price = current_prices[i]

            if np.isnan(current_price) or current_price <= 0:
                continue

            if action == 1 and available_to_deploy > 1.0:
                buy_perc = np.random.uniform(0.01, max_buy_perc_power)
                amount_to_spend = available_to_deploy * buy_perc

                if amount_to_spend > available_to_deploy:
                    amount_to_spend = available_to_deploy

                if amount_to_spend > 1.0:
                    fee = amount_to_spend * fee_rate
                    net_spend = amount_to_spend - fee
                    quantity_bought = net_spend / current_price

                    total_cost_of_old_position = positions[i] * positions_avg_cost[i]
                    total_cost_of_new_purchase = quantity_bought * current_price
                    total_quantity = positions[i] + quantity_bought

                    new_avg_cost = (
                        total_cost_of_old_position + total_cost_of_new_purchase
                    ) / total_quantity
                    positions_avg_cost[i] = new_avg_cost

                    cash -= amount_to_spend
                    positions[i] += quantity_bought

                    available_to_deploy -= amount_to_spend

            elif action == 2 and positions[i] > 1e-8:
                sell_this_position = True

                if avoid_selling_winners:
                    avg_cost = positions_avg_cost[i]
                    if current_price > avg_cost:
                        sell_this_position = False

                if sell_this_position:
                    quantity_to_sell = positions[i]

                    sale_value = quantity_to_sell * current_price
                    fee = sale_value * fee_rate
                    cash_received = sale_value - fee

                    cash += cash_received
                    positions[i] = 0.0
                    positions_avg_cost[i] = 0.0

    last_prices = prices[-1]
    for i in range(num_coins):
        if positions[i] > 1e-8 and not np.isnan(last_prices[i]) and last_prices[i] > 0:
            quantity_to_sell = positions[i]
            sale_value = quantity_to_sell * last_prices[i]
            fee = sale_value * fee_rate
            cash_received = sale_value - fee

            cash += cash_received
            positions[i] = 0.0
            positions_avg_cost[i] = 0.0

    final_pnl = cash - initial_capital
    return portfolio_value_history, final_pnl


@njit(parallel=True, cache=True)
def run_all_simulations_parallel(
    prices,
    num_simulations,
    initial_capital,
    fee_rate,
    max_buy_perc_power,
    base_seed,
    prob_hold,
    prob_buy,
    avoid_selling_winners,
    leverage,
):
    """
    Runs multiple simulations in parallel using Numba.
    """
    num_timesteps = prices.shape[0]

    all_histories = np.zeros((num_simulations, num_timesteps))
    all_final_pnls = np.zeros(num_simulations)

    for i in prange(num_simulations):
        np.random.seed(base_seed + i)

        hist, pnl = run_simulation_jit(
            prices,
            initial_capital,
            fee_rate,
            max_buy_perc_power,
            prob_hold,
            prob_buy,
            avoid_selling_winners,
            leverage,
        )
        all_histories[i] = hist
        all_final_pnls[i] = pnl

    return all_histories, all_final_pnls
