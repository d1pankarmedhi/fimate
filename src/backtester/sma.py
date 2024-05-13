import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brute

from backtester.base import Backtester


class SMABacktester(Backtester):
    """class for backtesting with SMA strategy"""

    def __init__(self, ticker, SMA1: int, SMA2: int, start, end):
        self.ticker = ticker
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.data = None

    def prepare_data(self, data: pd.DataFrame):
        """Retrieves and prepares the data."""
        data["return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["SMA1"] = data["Close"].rolling(self.SMA1).mean()
        data["SMA2"] = data["Close"].rolling(self.SMA2).mean()
        self.data = data

    def set_parameters(self, SMA1: int = None, SMA2: int = None):
        """Updates SMA parameters and resp. time series."""
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data["SMA1"] = self.data["Close"].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data["SMA2"] = self.data["Close"].rolling(self.SMA2).mean()

    def run_strategy(self):
        """Backtests the trading strategy."""
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA1"] > data["SMA2"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["return"]
        data.dropna(inplace=True)
        data["creturns"] = data["return"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data["creturns"].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_returns(
        self,
    ):
        # Calculate cumulative returns
        self.results["creturns"] = self.results["Close"].pct_change().cumsum()
        self.results["cstrategy"] = self.results["creturns"] * (
            (self.results["SMA1"] > self.results["SMA2"]).astype(int) * 2 - 1
        )

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting cumulative returns
        title = f"{self.ticker} | SMA1={self.SMA1}, SMA2={self.SMA2}"
        self.results[["creturns", "cstrategy"]].plot(ax=ax, title=title)

        # Adding labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")

        return fig

    def plot_sma(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting Closing Prices
        sns.lineplot(
            data=self.data[["Close", "SMA1", "SMA2"]],
            palette="tab10",
            linewidth=2.5,
            ax=ax,
        )

        # Adding labels and title
        ax.set_title("Stock Closing Prices with Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.tick_params(axis="x", rotation=45)

        return fig

    def update_and_run(self, SMA):
        """Updates SMA parameters and returns negative absolute performance
        (for minimazation algorithm).
        Parameters
        ==========
        SMA: tuple
        SMA parameter tuple
        """
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        """Finds global maximum given the SMA parameter ranges.
        Parameters
        ==========
        SMA1_range, SMA2_range: tuple
        tuples of the form (start, end, step size)
        """

        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)
