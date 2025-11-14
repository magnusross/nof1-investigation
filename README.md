# N of 1 Investigation

This is the accompanying analysis code for [this blog post](https://magnusross.github.io/posts/nof1-analysis/) about the [Alpha Arena benchmark](https://nof1.ai).

### 1. Fetch Data
```bash
python fetch_data.py
```
Downloads historical cryptocurrency prices (BTC, ETH, SOL, etc.) and model P&L data.

**Options:**
- `--symbols`: Fetch only crypto price data
- `--model-pnls`: Fetch only model P&L data
- No flags: Fetch both (default)

### 2. Run Backtest
```bash
python run_backtest.py
```
Runs 10,000 parallel Monte Carlo simulations of a random trading strategy on historical crypto data. Uses Numba for fast execution.

The strategy is as follows:
- You start with a fixed amount of capital, say 10,000
- at each timestep, for each coin you can either buy, sell, or hold
- You can only sell if you already own some of that coin, and you can only sell all of it at once
- If you buy, buy a random amount, as a percentage of cash in the bank
- If you hold, you do nothing for that coin
- At the end of the dataset, sell everything
- Set the fees for each trade to be a reasonable standard rate, typically of the market


**Parameters** (edit in script):
- `INITIAL_CAPITAL`: Starting capital (default: $10,000)
- `NUM_SIMULATIONS`: Number of simulations (default: 10,000)
- `TRADING_FEE_RATE`: Trading fees (default: 0.1%)
- `LEVERAGE`: Trading leverage (default: 15x)
- `PROB_HOLD/BUY/SELL`: Trading probabilities

### 3. Compute Win Probabilities
```bash
python compute_model_win_probs.py
```
Uses exponential smoothing to forecast model performance and calculates win probabilities. Generates forecast plots and analysis.

**Outputs:**
- Win/loss proportions for each model
- Forecast plots with confidence intervals
- Probability analysis of reaching profit thresholds

