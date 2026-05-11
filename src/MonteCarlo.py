"""MonteCarlo module for simulating asset returns with multivariate normal assumptions."""

import numpy as np

from .Utils import frequency_horizon_converter

def monte_carlo_returns_simulation(drifts, covariance_matrix, num_simu, period_size=1):
    """Generate Monte Carlo asset return scenarios from estimated drift and covariance.

    Parameters:
        drifts (pd.Series): Expected asset returns (log returns) per period.
        covariance_matrix (pd.DataFrame): Asset covariance matrix.
        num_simu (int): Number of scenarios to simulate.
        period_size (float): Time horizon multiplier in periods (e.g., 1 day =1, 5 days=5).

    Returns:
        np.ndarray: Simulated log returns shape (num_assets, num_simu).
    """

    # Converting to numpy
    drifts = drifts.values
    covariance_matrix = covariance_matrix.values

    # Time step
    deltaT = period_size

    # Managing the correlation between the assets
    L = np.linalg.cholesky(covariance_matrix)
    Z = np.random.normal(size=(drifts.shape[0], num_simu))

    # Simulating the returns
    log_returns = (drifts - 0.5 * np.diag(covariance_matrix))[:, np.newaxis] * deltaT + np.sqrt(deltaT) * (L @ Z)

    return log_returns