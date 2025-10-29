"""
Plotting module for visualizing backtest results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_histories(df_index, all_histories, initial_capital, num_to_plot=100):
    """
    Plots the portfolio value over time for all simulations.
    """
    plt.figure(figsize=(14, 7))

    plot_subset = all_histories[:: max(1, len(all_histories) // num_to_plot)]

    for history in plot_subset:
        plt.plot(df_index, history, color="blue", alpha=0.1)

    average_history = np.mean(all_histories, axis=0)
    plt.plot(
        df_index,
        average_history,
        color="red",
        linewidth=2,
        label="Average Portfolio Value",
    )

    plt.axhline(
        initial_capital,
        color="black",
        linestyle="--",
        label=f"Initial Capital (${initial_capital:,.0f})",
    )

    plt.title(
        f"Random Strategy Performance ({len(all_histories)} Simulations)", fontsize=16
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_pnl_distribution(all_final_pnls, initial_capital):
    """
    Plots a histogram of the final Profit and Loss (PnL) distribution.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(all_final_pnls, bins=50, edgecolor="black", alpha=0.7)

    plt.axvline(
        0, color="red", linestyle="--", linewidth=2, label="Break-even (PnL = 0)"
    )

    mean_pnl = np.mean(all_final_pnls)
    plt.axvline(
        mean_pnl,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean PnL (${mean_pnl:,.2f})",
    )

    mean_return = (np.mean(all_final_pnls) / initial_capital) * 100

    plt.title(
        f"Distribution of Final PnL ({len(all_final_pnls)} Simulations)", fontsize=16
    )
    plt.xlabel("Final PnL ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    stats_text = (
        f"Simulations: {len(all_final_pnls)}\n"
        f"Mean PnL: ${np.mean(all_final_pnls):,.2f}\n"
        f"Mean Return: {mean_return:.2f}%\n"
        f"Median PnL: ${np.median(all_final_pnls):,.2f}\n"
        f"Std. Dev: ${np.std(all_final_pnls):,.2f}\n"
        f"Win Rate: {np.mean(all_final_pnls > 0) * 100:.2f}%"
    )
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_forecasts(historical_df, forecast_series_dict):
    colors = plt.cm.get_cmap("tab10", len(historical_df.columns))

    plt.figure(figsize=(10, 6))

    for i, column in enumerate(historical_df.columns):
        color = colors(i)

        # 1. Plot historical data
        plt.plot(
            historical_df.index,
            historical_df[column],
            label=f"{column}",
            color=color,
            linewidth=2,
        )

        # 2. Get the forecast samples TimeSeries
        forecast_samples = forecast_series_dict[column]

        # 3. Get median (50th percentile) and 90% confidence interval
        # (5th and 95th percentiles) using the .quantile() method
        median_forecast = forecast_samples.quantile(0.50)
        low_forecast = forecast_samples.quantile(0.05)
        high_forecast = forecast_samples.quantile(0.95)

        # 4. Plot the median forecast
        # Use .time_index for x-axis and .values().squeeze() for y-axis
        plt.plot(
            median_forecast.time_index,
            median_forecast.values().squeeze(),  # <-- FIX IS HERE
            # label=f'Forecast {column}',
            color=color,
            linestyle="--",
        )

        # 5. Plot the 90% confidence interval
        plt.fill_between(
            low_forecast.time_index,  # Use the index from one of the quantiles
            low_forecast.values().squeeze(),  # <-- FIX IS HERE
            high_forecast.values().squeeze(),  # <-- FIX IS HERE
            color=color,
            alpha=0.1,
            # label=f'90% CI {column}'
        )

    plt.title("Historical Data and Probabilistic Forecasts (Median & 90% CI)")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")

    # Place legend outside the plot
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.grid(True)
    # plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    plt.show()
