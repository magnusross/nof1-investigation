#!/usr/bin/env python3

"""
Fetches historical cryptocurrency prices from Binance using the ccxt library
at 3-minute intervals and compiles them into a single pandas DataFrame.

This script requires the following libraries:
- pandas
- ccxt

You can install them using pip:
pip install pandas ccxt
"""

import ccxt
import pandas as pd
from datetime import datetime
import time


def fetch_all_ohlcv(exchange, pair, timeframe, since, end_timestamp):
    """
    Fetches all OHLCV data for a symbol from a start date to an end date.
    Handles exchange pagination automatically.
    """
    all_klines = []

    # ccxt-specific: get timeframe duration in milliseconds
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    current_since = since

    while current_since < end_timestamp:
        try:
            print(
                f"  Fetching 1000 candles for {pair} from {exchange.iso8601(current_since)}..."
            )
            # Fetch klines
            klines = exchange.fetch_ohlcv(
                pair,
                timeframe,
                since=current_since,
                limit=1000,  # Fetch 1000 candles per request (common max)
            )

            if not klines:
                # No more data available
                break

            all_klines.extend(klines)

            # Get timestamp of the last candle
            last_timestamp = klines[-1][0]

            if last_timestamp >= end_timestamp:
                # We've fetched past our end date
                break

            # Set the 'since' for the next request to be 1 millisecond after the last candle
            current_since = last_timestamp + timeframe_ms

        except ccxt.NetworkError as e:
            print(f"  Network error: {e}. Retrying in 5s...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"  Exchange error: {e}. Stopping fetch for {pair}.")
            break
        except Exception as e:
            print(f"  An error occurred: {e}.")
            break

    return all_klines


def main():
    """
    Main function to define parameters and run the fetcher.
    """

    # --- Parameters to set ---
    symbols = ["BTC", "BNB", "ETH", "SOL", "DOGE", "XRP"]
    quote_currency = "USDT"
    timeframe = "3m"
    start_date_str = "2024-10-17 00:00:00"
    end_date_str = "2024-10-27 00:00:00"
    # --- End Parameters ---

    # Initialize the exchange (Binance)
    # 'enableRateLimit': True is important to avoid API bans
    exchange = ccxt.binance({"enableRateLimit": True})

    # Convert human-readable dates to millisecond timestamps
    try:
        since_timestamp = exchange.parse8601(start_date_str)
        end_timestamp = exchange.parse8601(end_date_str)
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return

    all_series = []

    print(f"Fetching 3m data from {start_date_str} to {end_date_str}...")

    for symbol in symbols:
        pair = f"{symbol}/{quote_currency}"
        print(f"Processing {pair}...")

        klines = fetch_all_ohlcv(
            exchange, pair, timeframe, since_timestamp, end_timestamp
        )

        if klines:
            # Convert list of lists to DataFrame
            df = pd.DataFrame(
                klines, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Remove potential duplicates from API pagination overlap
            df = df[~df.index.duplicated(keep="first")]

            # Filter to the exact date range (pagination might overshoot)
            df = df.loc[start_date_str:end_date_str]

            # Keep only the 'close' price and rename it to the symbol
            series = df["open"].astype(float).rename(symbol)
            all_series.append(series)
        else:
            print(f"No data returned for {pair}.")

    if all_series:
        # Combine all individual symbol Series into one DataFrame
        # 'axis=1' joins them as columns
        final_df = pd.concat(all_series, axis=1)

        # --- Optional: Save to CSV ---
        output_filename = "crypto_prices_3min_ccxt.csv"
        final_df.to_csv(output_filename)
        print(f"\nData successfully saved to {output_filename}")
    else:
        print("No data was fetched for any symbol.")


if __name__ == "__main__":
    main()
