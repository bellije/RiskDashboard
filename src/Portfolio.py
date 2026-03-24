import pandas as pd
import numpy as np

class Portfolio:

    def __init__(self, portfolio):
        self.positions = portfolio

    # Modify portfolio
    
    def add_position(self, isin, quantity):
        self.positions[isin] = quantity

    def remove_position(self, isin):
        self.positions.drop(labels=[isin], inplace=True)

    def update_position(self, index, isin, quantity):
        curr_positions = self.positions.reset_index()
        curr_positions.loc[index, "Ticker"] = isin
        curr_positions.loc[index, "Quantity"] = quantity
        self.positions = curr_positions.set_index("Ticker")["Quantity"]

    # Util functions

    def get_tickers(self):
        return self.positions.index.to_list()
    
    def get_weights(self, prices):
        return (self.positions * prices)/(self.positions * prices).sum()
    
    def get_positions(self):
        return self.positions
    
    def get_portfolio_value(self, prices):
        return (self.positions * prices).sum()
    
    def get_portfolio_values(self, market_data):
        return (self.positions * market_data).sum(axis=1)
    
    def get_portfolio_returns(self, market_data, type="logret"):
        prices = (self.positions * market_data).sum(axis=1)
        if type=="logret":
            return np.log(prices/prices.shift(1)).iloc[1:]
        elif type=="ret":
            return prices.pct_change().iloc[1:]
        else: 
            raise Exception("Type ", type, " does not exist, must be in ['logret', 'ret'].")