import pandas as pd
import numpy as np
import yfinance as yf

class DataManager:

    def __init__(self, tickers, benchmark_ticker):

        try:
            # We get the information about each stock in the portfolio
            assets_info = {}
            for curr_ticker_name in tickers:
                curr_ticker = yf.Ticker(curr_ticker_name)
                if not curr_ticker.info:
                    raise ValueError(f'No data found for ticker {curr_ticker}.')
                assets_info[curr_ticker_name] = {"region": curr_ticker.info["region"], "sector": curr_ticker.info["sector"]}
            assets_info = pd.DataFrame(assets_info)

            # We get the market data for all the tickers
            market_data = yf.download(tickers, period="3y", group_by='ticker')
            market_data = market_data.xs('Close', level=1, axis=1)

            # Get benchmark prices
            benchmark_data = yf.download(benchmark_ticker, period="3y")["Close"][benchmark_ticker]

            # We only keep the intersection of dates
            dates_intersection = market_data.index.intersection(benchmark_data.index)
            market_data = market_data.loc[dates_intersection, :]
            benchmark_data = benchmark_data.loc[dates_intersection]

            assert((market_data.index == benchmark_data.index).all())

            # We store the data
            self.assets_info = assets_info
            self.market_data = market_data
            self.benchmark_data = benchmark_data

        except Exception as e:
            print(e)
            raise Exception

    ##################################################################
    #                        Simple Getters                          #
    ##################################################################

    def get_current_price(self):
        return self.market_data.iloc[-1, :]
    
    def get_assets_info(self, ticker):
        return self.assets_info[ticker]["sector"], self.assets_info[ticker]["region"]
    
    def get_market_data(self, window_size):
        return self.market_data.iloc[int(-252 * window_size):, :]
    
    def get_benchmark_data(self, window_size):
        return self.benchmark_data.iloc[int(-252 * window_size):, :]
    
    ##################################################################
    #                       Retrieve more data                       #
    ##################################################################

    def update_benchmark(self, new_benchmark_ticker):
        benchmark_data = yf.download(new_benchmark_ticker, period="3y")["Close"][new_benchmark_ticker]
        # We only keep the intersection of dates
        dates_intersection = self.market_data.index.intersection(benchmark_data.index)
        benchmark_data = benchmark_data.loc[dates_intersection]
        self.benchmark_data = benchmark_data

    def retrieve_new_data(self, ticker):
        curr_ticker = yf.Ticker(ticker)
        if not curr_ticker.info:
            raise ValueError(f'No data found for ticker {curr_ticker}.')
        self.assets_info[ticker] = {"region": curr_ticker.info["region"], "sector": curr_ticker.info["sector"]}
        self.market_data[ticker] = yf.download(ticker, period="1y")["Close"]

    def remove_not_usefull_data(self, ticker):
        
        self.assets_info.drop(labels=ticker, axis=1, inplace=True)
        self.market_data.drop(labels=ticker, axis=1, inplace=True)

    ##################################################################
    #                          Computations                          #
    ##################################################################

    def get_benchmark_returns(self, window_size, type="logret"):
        reduced_benchmark_returns = self.benchmark_data.iloc[int(-252 * window_size):]
        if type=="logret":
            return (np.log(reduced_benchmark_returns/reduced_benchmark_returns.shift(1)).iloc[1:])
        elif type=="ret":
            return reduced_benchmark_returns.pct_change().iloc[1:]
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        
    def get_market_data_returns(self, window_size, type="logret"):
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :])
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :]
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")

    def get_mean_returns(self, window_size, type="logret"):
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).mean()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].mean()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        
    def get_covariance_matrix(self, window_size, type="logret"):
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).cov()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].cov()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
    
    def get_correlation_matrix(self, window_size, type="logret"):
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).corr()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].corr()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        