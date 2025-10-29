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

import os
import ccxt
import pandas as pd
from datetime import datetime
import time
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import argparse

BASE_DIR = "data"


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


def fetch_all_symbols():
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
        output_filename = f"{BASE_DIR}/crypto_prices_3min_ccxt.csv"
        final_df.to_csv(output_filename)
        print(f"\nData successfully saved to {output_filename}")
    else:
        print("No data was fetched for any symbol.")


def fetch_historical_model_pnls_json():
    url = "https://nof1.ai/api/account-totals"
    try:
        req = Request(url, headers={"User-Agent": "python-urllib/3"})
        with urlopen(req, timeout=15) as resp:
            raw = resp.read()
            try:
                charset = resp.headers.get_content_charset() or "utf-8"
            except Exception:
                charset = "utf-8"
            data = json.loads(raw.decode(charset))
        return data

    except HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}")
    except URLError as e:
        print(f"URL error: {e.reason}")
    except Exception as e:
        print(f"Failed to fetch/parse JSON: {e}")

    return None


def fetch_historical_model_pnls():
    data = fetch_historical_model_pnls_json()

    pnl_df = data.get("accountTotals", [])
    pnl_df = pd.DataFrame(pnl_df)
    pnl_df = pnl_df[["timestamp", "realized_pnl", "model_id", "cum_pnl_pct"]]
    pnl_df["realized_pnl"] = pnl_df["realized_pnl"].astype(float)
    pnl_df["cum_pnl_pct"] = pnl_df["cum_pnl_pct"].astype(float)
    pnl_df["timestamp"] = pd.to_datetime(pnl_df.timestamp.astype(int), unit="s")
    pnl_df = pnl_df.set_index("timestamp")

    pnl_df.to_csv(f"{BASE_DIR}/historical_pnl_pct.csv")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Fetch historical crypto OHLCV or model P&Ls. If no flags are provided both will be fetched."
    )
    parser.add_argument(
        "--symbols",
        action="store_true",
        help="Fetch OHLCV time series for configured symbols (only).",
    )
    parser.add_argument(
        "--model-pnls",
        action="store_true",
        help="Fetch historical model P&Ls (only).",
    )

    args = parser.parse_args()

    # Behavior: if neither flag provided, run both. If one or both flags provided, run the requested actions.
    if not args.symbols and not args.model_pnls:
        # Default behavior: preserve previous behavior and run both
        fetch_all_symbols()
        fetch_historical_model_pnls()
    else:
        if args.symbols:
            fetch_all_symbols()
        if args.model_pnls:
            fetch_historical_model_pnls()
