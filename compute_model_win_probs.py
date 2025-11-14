from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode

import pandas as pd

from src.plotting import plot_forecasts


def analyze_final_pnls(forecast_series_dict):
    final_values_dict = {}

    for column, forecast_samples in forecast_series_dict.items():
        final_values_all_samples = forecast_samples.all_values()[-1, 0, :]
        final_values_dict[column] = final_values_all_samples

    final_pnls_df = pd.DataFrame(final_values_dict)

    winners_series = final_pnls_df.idxmax(axis=1)
    losers_series = final_pnls_df.idxmin(axis=1)

    win_proportions = winners_series.value_counts(normalize=True).sort_values(
        ascending=False
    )
    loss_proportions = losers_series.value_counts(normalize=True).sort_values(
        ascending=False
    )
    print("\n")
    print((win_proportions.rename("wins_proportion")).round(2).head(3).to_markdown())

    print("\n")
    print((loss_proportions).rename("loss_proportion").round(2).head(3).to_markdown())
    print("\n")
    print("any > 50k?")
    print(f"{(final_pnls_df > 400).any(axis=1).mean():.2f}")


def main():
    CSV_FILE = "data/historical_pnl_pct.csv"

    try:
        df = pd.read_csv(
            CSV_FILE,
            index_col="timestamp",
            parse_dates=["timestamp"],
        )
        print(df)
    except FileNotFoundError:
        print(f"Error: File not found: '{CSV_FILE}'")
        print("Please run the 'fetch_data.py' script first,")
        print("or make sure the CSV_FILE variable matches your filename.")
        return

    pivot_pnl_pct = df.pivot(columns="model_id", values="cum_pnl_pct")
    hourly_pnl_pct = pivot_pnl_pct.resample("h").last().ffill().dropna()

    model_names = hourly_pnl_pct.columns

    last_timestamp = pivot_pnl_pct.index[-1]
    forecast_start_time = last_timestamp + pd.DateOffset(hours=1)
    forecast_end_time = pd.to_datetime("2025-11-03 21:00:00")

    # Added post comp finish to make final plots
    pnl_pre_end = hourly_pnl_pct.index < forecast_end_time
    hourly_pnl_pct = hourly_pnl_pct[pnl_pre_end]

    comp_is_over = (~pnl_pre_end).any()

    # Calculate the number of steps (hours) to forecast
    n_steps = len(
        pd.date_range(start=forecast_start_time, end=forecast_end_time, freq="h")
    )

    print(f"Last data point: {last_timestamp}")
    print(
        f"Forecasting {n_steps} steps from {forecast_start_time} to {forecast_end_time}."
    )

    all_forecast_samples_dfs = {}

    # Loop through each column, create a univariate series, and forecast it
    for llm_name in model_names:
        ts = TimeSeries.from_dataframe(hourly_pnl_pct[[llm_name]], freq="h")

        model = ExponentialSmoothing(
            trend=None,
            seasonal=None,
        )

        print(f"Fitting model for {llm_name}...")

        model.fit(ts)
        forecast_samples = model.predict(n=n_steps, num_samples=10_000)
        # Convert this column's forecast samples to a DataFrame and store
        all_forecast_samples_dfs[llm_name] = forecast_samples

    plot_forecasts(
        hourly_pnl_pct,
        all_forecast_samples_dfs,
        save_name=f"data/forecasts_last_point{last_timestamp}.pdf",
    )

    if not comp_is_over:
        analyze_final_pnls(all_forecast_samples_dfs)


if __name__ == "__main__":
    main()
