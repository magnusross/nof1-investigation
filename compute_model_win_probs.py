from darts import TimeSeries
from darts.models import ExponentialSmoothing

import pandas as pd

from src.plotting import plot_forecasts


def analyze_final_pnls(forecast_series_dict):
    final_values_dict = {}

    # 1. Extract the final value from all 500 samples for each model
    for column, forecast_samples in forecast_series_dict.items():
        final_values_all_samples = forecast_samples.all_values()[-1, 0, :]
        final_values_dict[column] = final_values_all_samples

    final_pnls_df = pd.DataFrame(final_values_dict)

    # 3. Find the winning model (column name with max value) for each row (sample)
    winners_series = final_pnls_df.idxmax(axis=1)
    losers_series = final_pnls_df.idxmin(axis=1)

    # 4. Calculate the proportion of wins for each model
    # normalize=True automatically divides counts by the total (500)
    win_proportions = winners_series.value_counts(normalize=True).sort_values(
        ascending=False
    )
    loss_proportions = losers_series.value_counts(normalize=True).sort_values(
        ascending=False
    )
    # Multiply by 100 and format for a nice percentage printout
    print("win prob")
    print((win_proportions * 100).to_string(float_format="%.2f%%"))

    print("loss prob")
    print((loss_proportions * 100).to_string(float_format="%.2f%%"))

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
        print("Please run the 'fetch_crypto_prices_ccxt.py' script first,")
        print("or make sure the CSV_FILE variable matches your filename.")
        return

    pivot_pnl_pct = df.pivot(columns="model_id", values="cum_pnl_pct")
    hourly_pnl_pct = pivot_pnl_pct.resample("h").last().ffill().dropna()

    model_names = hourly_pnl_pct.columns

    print(hourly_pnl_pct)

    last_timestamp = pivot_pnl_pct.index[-1]
    forecast_start_time = last_timestamp + pd.DateOffset(hours=1)
    forecast_end_time = pd.to_datetime("2025-11-03 21:00:00")

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
            trend=None,  # Assumes a linear trend
            seasonal=None,  # Explicitly disables seasonality
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
    analyze_final_pnls(all_forecast_samples_dfs)


if __name__ == "__main__":
    main()
