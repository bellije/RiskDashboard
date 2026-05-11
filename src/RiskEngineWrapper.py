"""Wrapper for RiskEngine providing application-friendly init/update workflow and prepared tables for UI display."""

import pandas as pd
import numpy as np
from .RiskEngine import RiskEngine

class RiskEngineWrapper:
    """A high-level wrapper around RiskEngine for Streamlit dashboard data orchestration."""

    def __init__(self):
        """Initialize wrapper state before risk engine is configured."""
        self.initial_computations_done = False

    def init_risk_engine(self, file_path, benchmark_ticker):
        """Initialize the risk engine from a CSV position file and benchmark.

        Parameters:
            file_path (str): Path to CSV containing columns Ticker, Quantity.
            benchmark_ticker (str): Benchmark ticker symbol.

        Returns:
            None

        Raises:
            FileNotFoundError: If the CSV path does not exist.
            Exception: On unknown read/parse error.
        """

        # We parse the csv file containing the positions
        try:
            curr_positions = pd.read_csv(file_path, index_col="Ticker")["Quantity"]
        except FileNotFoundError:
            print("The file path is not valid.")
            raise
        except Exception as e:
            raise

        # We initialize the risk engine
        self.risk_engine = RiskEngine(curr_positions, benchmark_ticker)

        # We make the needed computations
        self.perform_computations()
        self.initial_computations_done = True

    def update_positions(self, new_positions):
        """Update positions in the risk engine, refresh data and recompute metrics.

        Parameters:
            new_positions (dict): Mutated rows, added rows, and deleted row indexes from UI data editor.

        Returns:
            None
        """

        # We assert that the risk_engine has been initialized and first computations have been done
        if not self.initial_computations_done:
            raise Exception("Computations have not been done before")
        
        # We update the positions in the risk engine
        self.risk_engine.update_positions(new_positions)

        # Initializing parameters
        

        # We perform the computations
        self.perform_computations()

    def perform_computations(self, frequency='D', window_size=1):
        """Perform all metrics and chart data calculations for the dashboard.

        Parameters:
            frequency (str): Frequency key, e.g. 'D', 'W', 'M'.
            window_size (float): Lookback window in years.

        Returns:
            None
        """
        # TODO: manage the frequencies

        # Setting window_size
        self.window_size = window_size

        # Exposition
        self.positions, self.weights, self.sectorial_exposure, self.geographical_exposure = self.risk_engine.get_exposition()
        self.positions = self.positions.reset_index()
        self.weights.name = "Weights"
        self.weights = self.weights.reset_index()

        # Prices graph
        self.portfolio_values = self.risk_engine.get_portfolio_values(frequency, self.window_size)
        self.portfolio_values.name = "Portfolio value ($)"
        self.portfolio_values = self.portfolio_values.reset_index()

        # Benchmark graph
        cumulative_benchmmark_returns = np.exp(self.risk_engine.data_manager.get_benchmark_returns(self.window_size, type="logret").cumsum())
        benchmark_values = cumulative_benchmmark_returns * self.portfolio_values.iloc[0, 1]
        benchmark_values[self.portfolio_values.iloc[0, 0]] = self.portfolio_values.iloc[0, 1]
        benchmark_values.sort_index(inplace=True)
        self.portfolio_values["Benchmark value ($)"] = benchmark_values.values

        # Performance metrics
        self.ptf_metrics = self.risk_engine.get_portfolio_metrics(frequency, self.window_size)
        self.ptf_metrics["Value"] = self.ptf_metrics["Value"].map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')

        # VaR and cVaRs
        historical_var, historical_cvar = self.risk_engine.get_historical_VaR_cVar(frequency=frequency, window_size=self.window_size)
        parametric_var, parametric_cvar = self.risk_engine.get_variance_covariance_VaR_cVar(frequency=frequency, window_size=self.window_size)
        monte_carlo_var, monte_carlo_cvar = self.risk_engine.get_Monte_Carlo_VaR_cVar(frequency=frequency, window_size=self.window_size)
        self.risk_metrics_overview = pd.DataFrame([
            {"Method": "Historical", "Metric": "VaR 95%", "Value ($)": historical_var},
            {"Method": "Historical", "Metric": "cVaR 95%", "Value ($)": historical_cvar},
            {"Method": "Parametric", "Metric": "VaR 95%", "Value ($)": parametric_var},
            {"Method": "Parametric", "Metric": "cVaR 95%", "Value ($)": parametric_cvar},
            {"Method": "Monte Carlo", "Metric": "VaR 95%", "Value ($)": monte_carlo_var},
            {"Method": "Monte Carlo", "Metric": "cVaR 95%", "Value ($)": monte_carlo_cvar}
        ]).set_index(["Method", "Metric"])
        self.risk_metrics_overview["Value ($)"] = self.risk_metrics_overview["Value ($)"].map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')

        # Computing the drawdown
        self.drawdowns = self.risk_engine.get_drawdowns(frequency, self.window_size)
        self.drawdowns.name = "Drawdown (%)"
        self.drawdowns = self.drawdowns.reset_index()

        # Computing the risk metrics
        self.risk_metrics = self.risk_engine.get_risk_metrics(frequency, self.window_size)
        self.risk_metrics["Value"] = self.risk_metrics["Value"].map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')
        
        # Filling the Customizable VaR/cVaR table
        self.update_var_cvar_horizon()

        # Getting the risk contributions
        # Volatility risk contribution
        self.risk_contribution = pd.DataFrame(self.risk_engine.get_portfolio_volatility_risk_contribution(frequency, self.window_size, relative=True))
        # Weights to compare
        self.risk_contribution.loc[:, "Weights"] = self.weights.set_index("Ticker")*100
        # VaR to compare
        self.risk_contribution.loc[:, "Value at Risk"] = self.risk_engine.get_portfolio_var_risk_contribution(frequency, self.window_size, relative=True)
        # Rounding the results
        self.risk_contribution = self.risk_contribution.map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')
        # Switching to long format for graphs creation
        self.risk_contribution = pd.melt(self.risk_contribution.reset_index(), id_vars="Ticker", value_vars=["Value at Risk", "Volatility", "Weights"], var_name="Type", value_name="Value")


    def update_var_cvar_horizon(self, new_horizon='D', confidence=0.95):
        """Recompute the customizable VaR/cVaR table for a given horizon and confidence level.

        Parameters:
            new_horizon (str): Horizon frequency code (e.g., 'D', 'W', 'M').
            confidence (float): Confidence level for VaR/cVaR.

        Returns:
            None
        """

        up_historical_var, up_historical_cvar = self.risk_engine.get_historical_VaR_cVar(frequency=new_horizon, window_size=self.window_size, confidence_level=confidence)
        up_parametric_var, up_parametric_cvar = self.risk_engine.get_variance_covariance_VaR_cVar(frequency=new_horizon, window_size=self.window_size, confidence_level=confidence)
        up_monte_carlo_var, up_monte_carlo_cvar = self.risk_engine.get_Monte_Carlo_VaR_cVar(frequency=new_horizon, window_size=self.window_size, confidence_level=confidence)

        self.updatable_var_cvar = pd.DataFrame([
            {"Method": "Historical", "Metric": "VaR", "Value ($)": up_historical_var},
            {"Method": "Historical", "Metric": "cVaR", "Value ($)": up_historical_cvar},
            {"Method": "Parametric", "Metric": "VaR", "Value ($)": up_parametric_var},
            {"Method": "Parametric", "Metric": "cVaR", "Value ($)": up_parametric_cvar},
            {"Method": "Monte Carlo", "Metric": "VaR", "Value ($)": up_monte_carlo_var},
            {"Method": "Monte Carlo", "Metric": "cVaR", "Value ($)": up_monte_carlo_cvar}
        ]).set_index(["Method", "Metric"])
        self.updatable_var_cvar["Value ($)"] = self.updatable_var_cvar["Value ($)"].map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')

    def update_benchmark(self, new_benchmark_ticker, frequency="D"):
        """Update benchmark, recompute risk metrics and portfolio benchmark series.

        Parameters:
            new_benchmark_ticker (str): Benchmark ticker symbol.
            frequency (str): Horizon code for metrics (e.g., 'D').

        Returns:
            None
        """

        self.risk_engine.data_manager.update_benchmark(new_benchmark_ticker)

        # Updating the benchmark related computations
        self.risk_metrics = self.risk_engine.get_risk_metrics(frequency, self.window_size)
        self.risk_metrics["Value"] = self.risk_metrics["Value"].map(lambda x: "{:.2f}".format(x) if pd.notna(x) else 'NaN')

        # Benchmark graph
        cumulative_benchmmark_returns = np.exp(self.risk_engine.data_manager.get_benchmark_returns(self.window_size, type="logret").cumsum())
        benchmark_values = cumulative_benchmmark_returns * self.portfolio_values.iloc[0, 1]
        benchmark_values[self.portfolio_values.iloc[0, 0]] = self.portfolio_values.iloc[0, 1]
        benchmark_values.sort_index(inplace=True)
        self.portfolio_values["Benchmark value ($)"] = benchmark_values.values

