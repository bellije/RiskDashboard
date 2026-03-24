# Risk engine for equity portfolios

Interactive web application for equity portfolio risk analysis, computing VaR, Expected Shortfall, performance metrics, and drawdown visualization.

## Overview
This project delivers a risk management dashboard for analyzing portfolio risk and performance using historical market data.
It implements multiple risk measures (VaR, CVaR/ES, drawdowns) across methodologies, like historical, parametric, and Monte Carlo, and presents results via interactive charts.

## Key Pages:

**Navbar**: Lookback size; Benchmark choice (MSCI World, S&P500, NASDAQ); Portfolio visualization with possibility to add, remove or modify assets in the portfolio. 
**Overview**: Portfolio evolution compared to normalized benchmark; portfolio metrics (Value, 1Y return, Annual Vol., Sharpe ratio, Sortino ratio); Visual sectorial and geographical exposures; Risk metrics (VaR and cVaR 95%) for a 1 day period.
**Risk**: Portfolio evolution and drawdown; Volatility and VaR contributions of assets; General risk metrics (Daily vol., Annual vol., Max drawdown, Market beta); Custom VaR/cVaR (1 day, 1 week, 1 month for 95%, 97.5%, 99% confidence). 

## Tech Stack

**Backend**: Python 3.10+ (NumPy, pandas for calculations; yfinance for data)
**Frontend**: Streamlit (Single page application with tabs) + Plotly/ (interactive charts)
**Data**: yFinance (retrieved live)

## Quick Start

### Prerequisites

```console
Python >= 3.10
pip install -r requirements.txt
streamlit run RiskDashboard.py
```
Open http://localhost:8501.

### Your portfolio

To import a portfolio in the risk engine, use a csv file structured as follow (an example is given in file.csv):
Ticker, Quantity
AAPL, 48
MSFT, 60
AMD, 92
...
Those tickers have to exist as yFinance tickers.

### Usage Workflow

**Data Input**: Load portfolio via CSV and choose benchmark (auto-fetch 3Y history of prices for assets and benchmark).

#### Navigate Pages:
**Overview**: Visualize portfolio evolution, exposure and performance metrics.
**Risk**: Visualize drawdowns, risk contribution and risk metrics.

## Tech Stack

**Backend**: Python 3.10+ (NumPy, pandas, SciPy for risk calcs; yfinance for data)
**Frontend**: Streamlit (multipage app) + Plotly/Altair (interactive charts)
**Data**: CSV upload or live tickers (Yahoo Finance, local files)

## Roadmap

- Errors management
- Documentation detailing
- Stress testing
- GARCH volatility models
- Factor risk decomposition (Fama-French)
- Real-time data via WebSocket
- Multi-portfolio comparison