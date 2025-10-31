"""
Plotting module for visualizing backtest results.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Fira Code"})


def plot_histories(
    df_index, all_histories, initial_capital, num_to_plot=100, save_name=None
):
    """
    Plots the portfolio value over time for all simulations.
    """
    plt.figure(figsize=(7, 4))
    colors = plt.cm.get_cmap("tab20", 3)

    plot_subset = all_histories[:: max(1, len(all_histories) // num_to_plot)]

    for history in plot_subset:
        plt.plot(df_index, history, color=colors(0), alpha=0.1)

    average_history = np.mean(all_histories, axis=0)
    plt.plot(
        df_index,
        average_history,
        color=colors(1),
        linewidth=2,
        label="Average Portfolio Value",
    )

    plt.axhline(
        initial_capital,
        color="black",
        # color=colors(2),
        linestyle="--",
        label=f"Initial Capital (${initial_capital:,.0f})",
    )

    # plt.title(
    #     f"Random Strategy Performance ({len(all_histories)} Simulations)", fontsize=16
    # )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.tick_params(axis="x", rotation=20)

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_pnl_distribution(all_final_pnls, initial_capital, save_name=None):
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
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_forecasts(historical_df, forecast_series_dict, save_name=None):
    colors = plt.cm.get_cmap("tab20", len(historical_df.columns))

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

        forecast_samples = forecast_series_dict[column]

        median_forecast = forecast_samples.quantile(0.50)
        low_forecast = forecast_samples.quantile(0.05)
        high_forecast = forecast_samples.quantile(0.95)

        plt.plot(
            median_forecast.time_index,
            median_forecast.values().squeeze(),
            color=color,
            linestyle="--",
        )

        plt.fill_between(
            low_forecast.time_index,
            low_forecast.values().squeeze(),
            high_forecast.values().squeeze(),
            color=color,
            alpha=0.1,
        )

    plt.title(
        "Forecasted PnLs (Median & 90% CI)",
        fontname="Fira Code",
    )
    plt.xlabel("Timestamp", fontname="Fira Code")
    plt.ylabel("Value (%)", fontname="Fira Code")
    plt.xticks()
    plt.tick_params(axis="x", rotation=45)

    # Place legend outside the plot
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontname="Fira Code")
    leg = plt.legend(
        framealpha=0,
    )
    for text in leg.get_texts():
        plt.setp(text, color="black", fontname="Fira Code")
    plt.grid(True)

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_backtest_results(
    df_index,
    all_histories,
    all_final_pnls,
    initial_capital,
    current_pnl_dollars,
    num_to_plot=100,
    save_name=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.get_cmap("tab20", 3)
    model_colors = plt.cm.get_cmap("tab20", len(current_pnl_dollars))

    plot_subset = all_histories[:: max(1, len(all_histories) // num_to_plot)]

    for i, history in enumerate(plot_subset):
        ax1.plot(
            df_index,
            history,
            color=colors(0),
            alpha=0.1,
            label="" if i != 0 else "Sample portfolio",
        )

    ax1.axhline(
        initial_capital,
        color="red",
        # color=colors(2),
        linestyle="--",
        label=f"Initial Capital (${initial_capital:,.0f})",
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.tick_params(axis="x", rotation=20)

    ax2.hist(all_final_pnls, bins=50, edgecolor="black", color=colors(0), alpha=0.7)

    ax2.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
    )

    for i, (m, pnl) in enumerate(current_pnl_dollars.items()):
        ax2.axvline(
            pnl,
            color=model_colors(i),
            label=m,
        )

    ax2.set_xlabel("Final PnL ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()
