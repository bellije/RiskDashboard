"""DataManager: retrieve and serve market and benchmark data for portfolio risk calculations."""

import pandas as pd
import numpy as np
import yfinance as yf

class DataManager:
    """Manage assets and benchmark price history for a portfolio using yfinance."""

    def __init__(self, tickers, benchmark_ticker):
        """Download and align historical data for tickers and benchmark.

        Parameters:
            tickers (list[str] | pd.Index): List of ticker symbols.
            benchmark_ticker (str): Benchmark ticker symbol.

        Raises:
            ValueError: If ticker metadata or prices cannot be retrieved.
        """

        try:
            # We get the information about each stock in the portfolio
            assets_info = {}
            other_currencies = {}
            for curr_ticker_name in tickers:
                curr_ticker = yf.Ticker(curr_ticker_name)
            
                # We check if the retrieved data is valid
                if (not curr_ticker.info) or (curr_ticker.info["quoteType"] != "EQUITY"):
                    raise ValueError(f'No data found for ticker {curr_ticker}.')
                
                # We retrieve the ticker characteristics
                assets_info[curr_ticker_name] = {"name": curr_ticker.info["longName"], "country": curr_ticker.info["country"], "sector": curr_ticker.info["sector"]}
            
                # We deal with currency problems
                if curr_ticker.info["currency"] != "USD":
                    # We store directly the FX rate name
                    other_currencies[curr_ticker_name] = curr_ticker.info["currency"] + "USD=X"
            
            assets_info = pd.DataFrame(assets_info)

            # We get the market data for all the tickers
            market_data = yf.download(tickers, period="3y", group_by='ticker')
            market_data = market_data.xs('Close', level=1, axis=1)

            # Get benchmark prices
            benchmark_data = yf.download(benchmark_ticker, period="3y")["Close"][benchmark_ticker]

            # We only keep the intersection of dates
            dates_intersection = market_data.index.intersection(benchmark_data.index)
            dates_intersection = dates_intersection.intersection()
            market_data = market_data.loc[dates_intersection, :]
            benchmark_data = benchmark_data.loc[dates_intersection]

            # We retrieve the needed fx_rates if needed
            if len(other_currencies):
                fx_rates = yf.download(list(set(other_currencies.values())), period="3y", group_by='ticker').xs("Close", level=1, axis=1)
                
                # We only keep the intersection of dates
                dates_intersection = dates_intersection.intersection(fx_rates.index)
                market_data = market_data.loc[dates_intersection, :]
                benchmark_data = benchmark_data.loc[dates_intersection]
                fx_rates = fx_rates.loc[dates_intersection]

                # we convert the market data
                for curr_ticker_name in other_currencies:
                    market_data[curr_ticker_name] = market_data[curr_ticker_name] * fx_rates[curr_ticker_name]
                
                # We store the FX rates
                self.fx_rates = fx_rates

            # We store the data
            self.assets_info = assets_info
            self.market_data = market_data
            self.benchmark_data = benchmark_data

        except KeyError as e:
            print("On a une Key error")
            print(curr_ticker.info)
            raise Exception

        except Exception as e:
            print(e)
            raise Exception

    ##################################################################
    #                        Simple Getters                          #
    ##################################################################

    def get_current_price(self):
        """Return current price for each asset in the portfolio.

        Returns:
            pd.Series: Latest close price by ticker.
        """
        return self.market_data.iloc[-1, :]
    
    def get_assets_info(self, ticker):
        """Return metadata (sector, country) for a given ticker.

        Parameters:
            ticker (str): Asset ticker.

        Returns:
            tuple[str, str]: Sector and country.
        """
        return self.assets_info[ticker]["sector"], self.assets_info[ticker]["country"]
    
    def get_market_data(self, window_size):
        """Return historical market data sliced to a lookback window.

        Parameters:
            window_size (float): Years of historical data to return (e.g., 1=1y).

        Returns:
            pd.DataFrame: Prices for each asset within the window.
        """
        return self.market_data.iloc[int(-252 * window_size):, :]
    
    def get_benchmark_data(self, window_size):
        """Return historical benchmark data sliced to a lookback window.

        Parameters:
            window_size (float): Years of historical data to return.

        Returns:
            pd.Series: Benchmark prices within the window.
        """
        return self.benchmark_data.iloc[int(-252 * window_size):, :]
    
    ##################################################################
    #                       Retrieve more data                       #
    ##################################################################

    def update_benchmark(self, new_benchmark_ticker):
        """Replace benchmark series and align its date index with the existing market data.

        Parameters:
            new_benchmark_ticker (str): New benchmark ticker symbol.
        """
        benchmark_data = yf.download(new_benchmark_ticker, period="3y")["Close"][new_benchmark_ticker]
        
        # We only keep the intersection of dates
        dates_intersection = self.market_data.index.intersection(benchmark_data.index)
        benchmark_data = benchmark_data.loc[dates_intersection]
        self.benchmark_data = benchmark_data

    def retrieve_new_data(self, ticker):
        """Add market and metadata for a new asset ticker.

        Parameters:
            ticker (str): Asset ticker symbol.

        Raises:
            ValueError: If ticker data cannot be found.
        """
        curr_ticker = yf.Ticker(ticker)
        
        # We check if the retrieved data is valid
        if (not curr_ticker.info) or (curr_ticker.info["quoteType"] != "EQUITY"):
            raise ValueError(f'No data found for ticker {curr_ticker}.')
        
        # We retrieve the ticker characteristics        
        self.assets_info[ticker] = {"name": curr_ticker.info["longName"], "country": curr_ticker.info["country"], "sector": curr_ticker.info["sector"]}
        
        # We retrieve the market data for the ticker
        self.market_data[ticker] = yf.download(ticker, period="3y")["Close"]

        # We deal with the currency issue if needed
        if curr_ticker.info["currency"] != "USD":
            fx_conversion_name = curr_ticker.info["currency"] + "USD=X"

            # We check if we already have the fx rate
            if fx_conversion_name in self.fx_rates.columns:
                self.market_data[ticker] = self.market_data[ticker] * self.fx_rates[fx_conversion_name]
            else:
                self.fx_rates[fx_conversion_name] = yf.download(fx_conversion_name, period="3y")["Close"][fx_conversion_name]
                # TODO: passer ça au dessus + finir les conversions + vérifier que toujours les mêmes indexes

    def remove_not_usefull_data(self, ticker):
        """Remove an asset’s data from assets_info and market_data.

        Parameters:
            ticker (str): Asset ticker to drop.
        """
        self.assets_info.drop(labels=ticker, axis=1, inplace=True)
        self.market_data.drop(labels=ticker, axis=1, inplace=True)

    ##################################################################
    #                          Computations                          #
    ##################################################################

    def get_benchmark_returns(self, window_size, type="logret"):
        """Compute benchmark returns for a specified lookback window.

        Parameters:
            window_size (float): Years of history used.
            type (str): 'logret' or 'ret'.

        Returns:
            pd.Series: Benchmark returns.
        """
        reduced_benchmark_returns = self.benchmark_data.iloc[int(-252 * window_size):]
        if type=="logret":
            return (np.log(reduced_benchmark_returns/reduced_benchmark_returns.shift(1)).iloc[1:])
        elif type=="ret":
            return reduced_benchmark_returns.pct_change().iloc[1:]
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        
    def get_market_data_returns(self, window_size, type="logret"):
        """Compute asset returns for a specified lookback window.

        Parameters:
            window_size (float): Years of history used.
            type (str): 'logret' or 'ret'.

        Returns:
            pd.DataFrame: Returns for each asset.
        """
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :])
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :]
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")

    def get_mean_returns(self, window_size, type="logret"):
        """Compute mean returns by asset for a specified lookback window.

        Parameters:
            window_size (float): Years of history used.
            type (str): 'logret' or 'ret'.

        Returns:
            pd.Series: Mean asset returns.
        """
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).mean()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].mean()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        
    def get_covariance_matrix(self, window_size, type="logret"):
        """Compute the covariance matrix of asset returns.

        Parameters:
            window_size (float): Years of history used.
            type (str): 'logret' or 'ret'.

        Returns:
            pd.DataFrame: Covariance matrix.
        """
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).cov()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].cov()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
    
    def get_correlation_matrix(self, window_size, type="logret"):
        """Compute the correlation matrix of asset returns.

        Parameters:
            window_size (float): Years of history used.
            type (str): 'logret' or 'ret'.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        reduced_asset_returns = self.market_data.iloc[int(-252 * window_size):, :]
        if type=="logret":
            return (np.log(reduced_asset_returns/reduced_asset_returns.shift(1)).iloc[1:, :]).corr()
        elif type=="ret":
            return reduced_asset_returns.pct_change().iloc[1:, :].corr()
        else:
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")
        