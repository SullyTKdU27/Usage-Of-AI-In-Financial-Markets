# Enhanced Stock Trader: Machine Learning Stock Price Prediction
## Overview
This is a sophisticated stock trading analysis tool that uses machine learning (Random Forest Regression) to predict stock prices and simulate trading strategies. The code combines technical analysis indicators with predictive modeling to generate investment insights.

## Original Code from Github

https://github.com/austin-starks/Deep-RL-Stocks

## Features
- Advanced technical indicator calculation (SMA, EMA, MACD, RSI, Bollinger Bands)
- Machine learning price prediction using Random Forest
- Portfolio simulation and backtesting
- Visualization of stock price predictions and portfolio performance

## Required Imports
pip install yfinance pandas numpy matplotlib scikit-learn

## Specific Python libraries needed:
- `yfinance`: For downloading stock data
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `scikit-learn`: Machine learning (RandomForestRegressor, StandardScaler, train_test_split)

## Data Source
- Stock: Apple Inc. (AAPL)
- Source: Yahoo Finance, https://finance.yahoo.com/quote/AAPL/history/?guccounter=1&period1=1685577600&period2=1717200000
- Date Range: 
  - Full data range: June 1, 2019 to June 1, 2024 (5 years)
  - Actual model training: Approximately 1 year due to Random Forest complexity limitations, June 1, 2023 - June 1, 2024 (Input into Code)

## Key Methodology
- Uses a 30-day sliding window for feature extraction
- Generates multiple technical indicators
- Applies Random Forest Regression for price prediction
- Simulates trading decisions based on predicted price movements
- Provides portfolio value tracking and performance analysis

## Note
This is a research and learning project. Not financial advice. Always consult professional financial advisors before making investment decisions.
