# RiskDashboard

A lightweight interactive risk engine for equity portfolios built with Python and Streamlit.

## Project purpose

Analyze portfolio performance and risk using historical price data, with support for:
- Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
- Historical, parametric (variance-covariance) and Monte Carlo methodologies
- Drawdown, volatility, Sharpe, Sortino, market beta and more
- Portfolio exposure by weight, sector and geography
- Dynamic portfolio editing and multiple benchmarks

## Features

- Import portfolio from CSV: `Ticker,Quantity`
- Auto-fetch 3 years of historical quotes from Yahoo Finance (`yfinance`)
- Realtime portfolio and benchmark evolution charts (Plotly)
- Interactive dashboard (Streamlit) with tabs:
  - Overview
  - Risk
- Custom horizon VaR/CVaR (1 day / 1 week / 1 month)
- Risk contribution by asset (volatility + VaR)
- Simple portfolio drawdown analysis

## Repository structure

- `RiskDashboard.py`: Streamlit app entrypoint
- `src/`
  - `DataManager.py`: market and benchmark data retrieval + derived returns/covariances
  - `Portfolio.py`: positions, values, returns, weights
  - `RiskEngine.py`: risk calculations and metrics
  - `RiskEngineWrapper.py`: UI-friendly computed results
  - `MonteCarlo.py`: Monte Carlo return simulation
  - `Utils.py`: horizon and lookup converters
- `file.csv`: example portfolio input
- `requirements.txt`: Python dependencies

## Requirements

- Python >= 3.10

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run RiskDashboard.py
```

Browse: `http://localhost:8501`

## Portfolio input format

CSV must include 2 columns (header is required):

```csv
Ticker,Quantity
AAPL,48
MSFT,60
AMD,92
```

- Ticker symbols must be valid Yahoo ticker names
- Quantity is numerical number of shares

## Notes

- The project uses 252 trading days for annualization
- If a symbol fails in yfinance or is invalid, it may raise in data load
- Risk free rate is currently fixed at 1% in `RiskEngine`

## Roadmap (future improvements)

- Better error handling/validation for tickers and CSV upload
- Support for non-US market calendars and missing data imputation
- Stress testing and scenario analysis
- GARCH volatility modeling
- Factor risk decomposition (Fama-French)
- Real-time updates and websocket data
- Multiple portfolio comparison
