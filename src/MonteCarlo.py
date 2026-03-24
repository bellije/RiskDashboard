import numpy as np

from .Utils import frequency_horizon_converter

def monte_carlo_returns_simulation(drifts, covariance_matrix, num_simu, period_size=1):

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