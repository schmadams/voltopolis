"""A thin wrapper around entsoe-py that handles the boring parts once:
the API key, splitting long date ranges into chunks, and returning clean UTC data.

Usage:
    from entsoe_client import EntsoeClient

    client = EntsoeClient()  # reads ENTSOE_API_KEY from .env or environment
    prices = client.da_prices("DE_LU", date(2026, 6, 1), date(2026, 7, 1))
"""

import os
from datetime import date, timedelta

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import Area
from helpers.check_env import check_env

check_env()

CHUNK_DAYS = 360  # stay under ENTSO-E per-request limits


class EntsoeClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise ValueError("ENTSOE_API_KEY is not set (environment variable or .env file)")
        self._client = EntsoePandasClient(api_key=api_key)

    def da_prices(self, zone: str, start: date, end: date) -> pd.Series:
        """Day-ahead auction prices for a bidding zone, as a UTC-indexed Series in EUR/MWh."""
        return self._query_chunked(self._client.query_day_ahead_prices, zone, start, end)

    def _query_chunked(self, query_function, zone: str, start: date, end: date) -> pd.Series:
        """Run an entsoe-py query in chunks and stitch the results into one UTC series."""
        timezone = Area[zone].tz
        pieces = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
            start_ts = pd.Timestamp(chunk_start, tz=timezone)
            end_ts = pd.Timestamp(chunk_end, tz=timezone)
            piece = query_function(zone, start=start_ts, end=end_ts)
            print(f"{zone} {chunk_start} -> {chunk_end}: {len(piece)} intervals")
            if len(piece) > 0:
                pieces.append(piece)
            chunk_start = chunk_end
        if not pieces:
            return pd.Series(dtype=float)
        combined = pd.concat(pieces)
        combined.index = combined.index.tz_convert("UTC")
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        return combined

if __name__ == "__main__":
    client = EntsoeClient()
    prices = client.da_prices("DE_LU", date(2026, 6, 1), date(2026, 6, 8))
    print(prices.head(10))
    print(f"\n{len(prices)} intervals, {prices.index.min()} -> {prices.index.max()}")