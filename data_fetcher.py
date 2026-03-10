"""
data_fetcher.py
───────────────
Fetches historical stock data and company info from Yahoo Finance
using the yfinance library.
"""

import yfinance as yf
import pandas as pd


class StockDataFetcher:
    def __init__(self, ticker: str, period: str = "1y"):
        self.ticker = ticker
        self.period = period

    def fetch(self):
        """
        Returns:
            df   : pd.DataFrame with OHLCV data (DatetimeIndex)
            info : dict with company metadata
        """
        try:
            stock = yf.Ticker(self.ticker)
            df    = stock.history(period=self.period)

            if df is None or df.empty:
                return None, {}

            # Clean up
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)   # remove timezone for Plotly
            df.dropna(inplace=True)

            # Basic info (graceful fallback)
            try:
                info = stock.info or {}
            except Exception:
                info = {}

            return df, info

        except Exception as e:
            print(f"[DataFetcher] Error: {e}")
            return None, {}
