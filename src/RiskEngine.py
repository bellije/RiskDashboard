import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import norm

from .Portfolio import Portfolio
from .DataManager import DataManager
from .MonteCarlo import monte_carlo_returns_simulation
from .Utils import frequency_horizon_converter

class RiskEngine:

    num_simu_monte_carlo = 100000

    def __init__(self, curr_positions, benchmark_ticker):

        self.portfolio = Portfolio(curr_positions)
        
        # TODO: Manage risk free rate
        self.risk_free_rate = 0.01

        # We retrieve the data for the tickers
        self.data_manager = DataManager(self.portfolio.get_tickers(), benchmark_ticker)
        

    def update_positions(self, new_positions):
        # New positions format: {
        #   'edited_rows': {index: {"Ticker or Quantity": new_value}}, 
        #   'added_rows': [{'Ticker': 'new ticker', 'Quantity': "new_quantity"}], 
        #   'deleted_rows': [indexes]
        # }
        
        former_positions = self.portfolio.get_positions()
        
        # If ticker in former position, we modify the quantity
        for row_index, row_value in new_positions["edited_rows"].items():
            new_values = {
                "Ticker": former_positions.index[row_index],
                "Quantity": former_positions.iloc[row_index]
            }
            for sub_row_index, sub_row_value in row_value.items():
                if sub_row_index == "Ticker":
                    self.data_manager.retrieve_new_data(sub_row_value)
                    self.data_manager.remove_not_usefull_data(new_values["Ticker"])
                new_values[sub_row_index] = sub_row_value
            self.portfolio.update_position(row_index, new_values["Ticker"], new_values["Quantity"])    
        
        # We remove the tickers that are not in the new positions
        for row in new_positions["deleted_rows"]:
            self.data_manager.remove_not_usefull_data(former_positions.index[row])
            self.portfolio.remove_position(former_positions.index[row])

        # If ticker not in position, we add it
        for row in new_positions["added_rows"]:
            self.data_manager.retrieve_new_data(row["Ticker"])
            self.portfolio.add_position(row["Ticker"], row["Quantity"])
                

    ##################################################################
    #                           Exposition                           #
    ##################################################################
    
    def get_exposition(self):
        """ This method is aimed to describe the exposition of the portfolio. This involves three type of exposure:
        - The weights of the portoflio, to visually see the proportion invested in each asset,
        - The sectorial aggregated weights, to see if the portfolio is heavily exposed to a specific sector,
        - The geographically aggregated weights, to see if the portfolio is heavily exposed to a specific region.
        This method returns three Series containing each exposure statistics.
        """
        # Weights retrieval
        weights = self.portfolio.get_weights(self.data_manager.get_current_price())

        # Sectorial and geographical exposure
        sectorial_exposure = defaultdict(float)
        geographical_exposure = defaultdict(float)
        for curr_ticker in self.portfolio.get_tickers():
            sector, region = self.data_manager.get_assets_info(curr_ticker)
            sectorial_exposure[sector] += weights[curr_ticker]
            geographical_exposure[region] += weights[curr_ticker]
        
        
        positions = self.portfolio.get_positions()
        return positions, weights, pd.DataFrame({"Sector": sectorial_exposure.keys(), "Value": sectorial_exposure.values()}), pd.DataFrame({"Country": geographical_exposure.keys(), "Value": geographical_exposure.values()})


    ##################################################################
    #                         Value at Risk                          #
    ##################################################################

    def get_historical_VaR_cVar(self, frequency = 'D', window_size=1, confidence_level=0.95):
        
        # We compute the portfolio returns and its current value
        portfolio_returns = self.portfolio.get_portfolio_returns(self.data_manager.get_market_data(window_size))
        portfolio_value = self.portfolio.get_portfolio_value(self.data_manager.get_current_price())

        # Depending on the frequency, we combine returns
        portfolio_returns = portfolio_returns.rolling(window=frequency_horizon_converter[frequency]).sum()

        # Computing the VaR and cVaR
        var = portfolio_returns.quantile(1-confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return - var * portfolio_value, - cvar * portfolio_value
    
    def get_variance_covariance_VaR_cVar(self, frequency = 'D', window_size=1, confidence_level=0.95):
        #TODO: factor var/covar matrices computations 
        # We get the usefull values
        portfolio_value = self.portfolio.get_portfolio_value(self.data_manager.get_current_price())
        ptf_weights = self.portfolio.get_weights(self.data_manager.get_current_price())
        mean_returns = self.data_manager.get_mean_returns(window_size, "ret")
        cov_matrix = self.data_manager.get_covariance_matrix(window_size, "ret")

        # Parametric approach utils computations
        alpha = 1-confidence_level
        Z_score = norm.ppf(alpha)
        mu_p = ptf_weights.dot(mean_returns) * frequency_horizon_converter[frequency]
        sigma_p = np.sqrt(ptf_weights @ cov_matrix @ ptf_weights) * np.sqrt(frequency_horizon_converter[frequency])

        # Computing the VaR and cVaR
        var = - (mu_p + Z_score * sigma_p) * portfolio_value
        cvar = (- mu_p + sigma_p * (norm.pdf(Z_score) / alpha)) * portfolio_value
        return var, cvar
    
    def get_Monte_Carlo_VaR_cVar(self, frequency = 'D', window_size=1, confidence_level=0.95):
        #TODO: factor var/covar matrices computations
        # We get the usefull values
        portfolio_value = self.portfolio.get_portfolio_value(self.data_manager.get_current_price())
        ptf_weights = self.portfolio.get_weights(self.data_manager.get_current_price())
        mean_returns = self.data_manager.get_mean_returns(window_size, "logret")
        cov_matrix = self.data_manager.get_covariance_matrix(window_size, "logret")

        # We perform the Monte Carlo to get asset returns
        returns = monte_carlo_returns_simulation(mean_returns, cov_matrix, self.num_simu_monte_carlo, frequency_horizon_converter[frequency])
        returns = pd.DataFrame(returns, index=ptf_weights.index).T

        # We compute the usefull portfolio values
        ptf_returns = (ptf_weights * returns).sum(axis=1)
        var = np.quantile(ptf_returns, 1-confidence_level)
        cvar = (ptf_returns[ptf_returns <= var]).mean()
        return - var * portfolio_value, - cvar * portfolio_value
    
    ##################################################################
    #                              Metrics                           #
    ##################################################################
    
    def get_portfolio_metrics(self, frequency = 'D', window_size=1):
        ptf_value = self.portfolio.get_portfolio_value(self.data_manager.get_current_price())
        portfolio_returns = self.portfolio.get_portfolio_returns(self.data_manager.get_market_data(window_size))
        total_return = (((1 + portfolio_returns).cumprod() - 1) * 100).iloc[-1]
        annual_volatility = portfolio_returns.std(ddof=1) * np.sqrt(252/frequency_horizon_converter[frequency]) * 100
        sharpe_ratio =  (portfolio_returns.mean() - self.risk_free_rate) / portfolio_returns.std(ddof=1)
        sortino_ratio =  (portfolio_returns.mean() - self.risk_free_rate) / portfolio_returns[portfolio_returns < 0].std(ddof=1)
        metrics_dict = {
            "Ptf. value ($)": ptf_value, 
            "1Y return (%)": total_return, 
            "Annual Volatility (%)": annual_volatility, 
            "Sharpe Ratio": sharpe_ratio, 
            "Sortino Ratio": sortino_ratio
        }
        return pd.DataFrame({"Metrics": metrics_dict.keys(), "Value": metrics_dict.values()})
    
    def get_risk_metrics(self, frequency = 'D', window_size=1):
        portfolio_returns = self.portfolio.get_portfolio_returns(self.data_manager.get_market_data(window_size))
        benchmark_returns = self.data_manager.get_benchmark_returns(window_size=window_size)
        daily_volatility = portfolio_returns.std(ddof=1)
        annual_volatility = daily_volatility * np.sqrt(252/frequency_horizon_converter[frequency]) * 100
        max_drawdown = ((1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()) - 1).min() * 100
        risk_metrics_dict = {
            "Daily Vol.": daily_volatility,
            "Annual Vol.": annual_volatility,
            "Max drawdown": max_drawdown,
            "Mkt Beta": portfolio_returns.cov(benchmark_returns)/benchmark_returns.var()
        }
        return pd.DataFrame({"Metrics": risk_metrics_dict.keys(), "Value": risk_metrics_dict.values()})     
    
    
    ##################################################################
    #                              Graphs                            #
    ##################################################################

    def get_portfolio_values(self, frequency = 'D', window_size=1):
        return self.portfolio.get_portfolio_values(self.data_manager.get_market_data(window_size))
    
    def get_drawdowns(self, frequency = 'D', window_size=1):
        portfolio_returns = self.portfolio.get_portfolio_returns(self.data_manager.get_market_data(window_size))
        return ((1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()) - 1) * 100
    
    ##################################################################
    #                        Risk contributions                      #
    ##################################################################
    
    def get_portfolio_volatility_risk_contribution(self, frequency = 'D', window_size=1, relative=False):
        cov_matrix = self.data_manager.get_covariance_matrix(window_size, "ret")
        weights = self.portfolio.get_weights(self.data_manager.get_current_price())
        
        ptf_th_vol = np.sqrt(weights @ cov_matrix @ weights)
        vol_risk_contrib = weights * (cov_matrix @ weights)/ptf_th_vol

        assert(abs(vol_risk_contrib.sum() - ptf_th_vol) < 1e-10)
        vol_risk_contrib.name = "Volatility"
        return 100*vol_risk_contrib/ptf_th_vol if relative else vol_risk_contrib
    
    def get_portfolio_var_risk_contribution(self, frequency = 'D', window_size=1, relative=False, confidence_level=0.95):
        portfolio_value = self.portfolio.get_portfolio_value(self.data_manager.get_current_price())
        cov_matrix = self.data_manager.get_covariance_matrix(window_size, "ret")
        mean_returns = self.data_manager.get_mean_returns(window_size, "ret")
        weights = self.portfolio.get_weights(self.data_manager.get_current_price())

        # Parametric approach utils computations
        alpha = 1-confidence_level
        Z_score = norm.ppf(alpha)

        # Value at Risk risk contribution        
        ptf_th_vol = np.sqrt(weights @ cov_matrix @ weights)
        var_risk_contrib = - weights * (mean_returns + Z_score * (cov_matrix @ weights)/ptf_th_vol) * portfolio_value

        # Asserting the results sums to the theoretical Value at Risk
        th_VaR, _ = self.get_variance_covariance_VaR_cVar(frequency = frequency, window_size=window_size, confidence_level=confidence_level)
        assert(abs(var_risk_contrib.sum() - th_VaR) < 1e-10)
        var_risk_contrib.name = "VaR"
        return 100*var_risk_contrib/th_VaR if relative else var_risk_contrib
