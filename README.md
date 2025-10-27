The strategy is as follows:
- You start with a fixed amount of capital, say 10,000
- at each timestep, for each coin you can either buy, sell, or hold
- You can only sell if you already own some of that coin, and you can only sell all of it at once
- If you buy, buy a random amount, as a percentage of cash in the bank
- If you hold, you do nothing for that coin
- At the end of the dataset, sell everything
- Set the fees for each trade to be a reasonable standard rate, typically of the market 

You need to focus on making the simulation fast and efficient. You should run multiple random simulations and plot the results. The plot should be of the mark to market value of the portfolio over time, including fees. I would also like a plot of the disribution of the final PnL. Make it easy to control the parameters of the strategy.


TODO:
- [x] Incorporate margin call. 
- [ ] 


INITIAL_CAPITAL = 10_000
TRADING_FEE_RATE = 0.001  # 0.1% (common fee on Binance)
MAX_BUY_PERC_CASH = 0.2  # Max 10% of available cash on a single buy
NUM_SIMULATIONS = 10_000  # Number of backtests to run

# --- Strategy Probabilities (must sum to 1.0) ---
PROB_HOLD = 0.90  # 85% chance to hold
PROB_BUY = 0.09  # 7.5% chance to buy
PROB_SELL = 0.01  # 7.5% chance to sell