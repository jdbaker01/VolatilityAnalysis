# Make src directory a Python package
from .calculations import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_volatility,
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    calculate_rolling_var,
    calculate_covariance_matrix
)
from .correlation import calculate_correlation_matrix, format_correlation_matrix
from .data_handler import get_multiple_stocks_data
from .logger import logger

__all__ = [
    'calculate_daily_returns',
    'calculate_cumulative_returns',
    'calculate_volatility',
    'calculate_portfolio_returns',
    'calculate_portfolio_volatility',
    'calculate_rolling_var',
    'calculate_covariance_matrix',
    'calculate_correlation_matrix',
    'format_correlation_matrix',
    'get_multiple_stocks_data',
    'logger'
]
