from abc import ABC

import yfinance as yf


class Backtester(ABC):
    def download_data(
        self, ticker: str, start_date: str = "2020-01-01", end_date: str = "2024-01-01"
    ):
        return yf.download(ticker, start=start_date, end=end_date)
