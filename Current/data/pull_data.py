import os
from datetime import datetime

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

SYMBOL = "SPY"
START = datetime(2017, 1, 1)
END = datetime.now()
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), f"{SYMBOL}_1min.csv")


def load_keys():
    load_dotenv()
    return os.getenv("public_key"), os.getenv("secret_key")


def pull_data(symbol=SYMBOL, start=START, end=END, output_csv=OUTPUT_CSV):
    api_key, secret_key = load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed="iex",  # switch to "sip" if you have a paid market data subscription
    )

    bars = client.get_stock_bars(request).df
    bars.to_csv(output_csv)
    print(f"Saved {len(bars)} rows to {output_csv}")


if __name__ == "__main__":
    pull_data()
