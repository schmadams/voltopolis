import pandas as pd
from data_loading.get_da_prices import get_day_ahead_prices
from data_loading.get_gb_da_prices import get_gb_day_ahead_prices
from data_loading.get_ic_da_auction_results import get_ic_da_prices



def run_backfill():
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
    end = pd.Timestamp.now(tz="UTC")

    get_day_ahead_prices(start, end, regions=["FR", "DE_LU", "BE", "NL", "DK_1", "DK_2"])
    get_gb_day_ahead_prices(start, end)
    get_ic_da_prices(start, end)


if __name__ == "__main__":
    run_backfill()