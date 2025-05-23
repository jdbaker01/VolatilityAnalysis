import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.calculations import (
    calculate_daily_returns,
    calculate_volatility,
    calculate_cumulative_returns,
    calculate_portfolio_returns,
    calculate_portfolio_volatility
)

# Test Data Setup
@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = [100, 102, 99, 101, 103, 102, 105, 107, 106, 108]
    return pd.DataFrame({
        'Close': prices
    }, index=dates)

@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    returns = [0.02, -0.03, 0.02, 0.02, -0.01, 0.03, 0.019, -0.009, 0.019]
    return pd.Series(returns, index=dates[1:])

@pytest.fixture
def multi_stock_returns():
    """Create sample returns data for multiple stocks."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    stock1 = [0.02, -0.03, 0.02, 0.02, -0.01, 0.03, 0.019, -0.009, 0.019]
    stock2 = [0.01, -0.02, 0.03, 0.01, -0.02, 0.02, 0.015, -0.005, 0.025]
    return {
        'STOCK1': pd.Series(stock1, index=dates[1:]),
        'STOCK2': pd.Series(stock2, index=dates[1:])
    }

# Daily Returns Tests
def test_calculate_daily_returns_valid(sample_stock_data):
    """Test daily returns calculation with valid data."""
    returns = calculate_daily_returns(sample_stock_data)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_stock_data) - 1  # One less due to pct_change
    assert returns.iloc[0] == pytest.approx(0.02)  # (102-100)/100

def test_calculate_daily_returns_empty():
    """Test daily returns calculation with empty data."""
    empty_df = pd.DataFrame({'Close': []})
    with pytest.raises(ValueError, match="No valid data for calculating returns"):
        calculate_daily_returns(empty_df)

def test_calculate_daily_returns_missing_column():
    """Test daily returns calculation with missing Close column."""
    df = pd.DataFrame({'Open': [100, 101, 102]})
    with pytest.raises(ValueError, match="DataFrame must contain a 'Close' price column"):
        calculate_daily_returns(df)

# Volatility Tests
def test_calculate_volatility_valid(sample_returns):
    """Test volatility calculation with valid data."""
    window = 5
    vol = calculate_volatility(sample_returns, window)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_returns) - window + 1
    assert all(v >= 0 for v in vol)  # Volatility should be non-negative

def test_calculate_volatility_invalid_window(sample_returns):
    """Test volatility calculation with invalid window size."""
    with pytest.raises(ValueError, match="Window size must be at least 2 days"):
        calculate_volatility(sample_returns, window=1)

def test_calculate_volatility_empty():
    """Test volatility calculation with empty data."""
    empty_series = pd.Series([])
    with pytest.raises(ValueError, match="No data provided for volatility calculation"):
        calculate_volatility(empty_series)

# Cumulative Returns Tests
def test_calculate_cumulative_returns_valid(sample_returns):
    """Test cumulative returns calculation with valid data."""
    cum_returns = calculate_cumulative_returns(sample_returns)
    assert isinstance(cum_returns, pd.Series)
    assert len(cum_returns) == len(sample_returns)
    # Verify cumulative calculation: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    expected_final = np.prod([1 + r for r in sample_returns]) - 1
    assert cum_returns.iloc[-1] == pytest.approx(expected_final)

def test_calculate_cumulative_returns_empty():
    """Test cumulative returns calculation with empty data."""
    empty_series = pd.Series([])
    with pytest.raises(ValueError, match="No data provided for cumulative returns calculation"):
        calculate_cumulative_returns(empty_series)

# Portfolio Tests
def test_calculate_portfolio_returns_valid(multi_stock_returns):
    """Test portfolio returns calculation with valid data."""
    portfolio_returns = calculate_portfolio_returns(multi_stock_returns)
    assert isinstance(portfolio_returns, pd.Series)
    assert len(portfolio_returns) == len(next(iter(multi_stock_returns.values())))
    # Test equal weighting
    for date in portfolio_returns.index:
        expected = sum(returns[date] for returns in multi_stock_returns.values()) / len(multi_stock_returns)
        assert portfolio_returns[date] == pytest.approx(expected)

def test_calculate_portfolio_returns_empty():
    """Test portfolio returns calculation with empty data."""
    with pytest.raises(ValueError, match="No stock returns provided"):
        calculate_portfolio_returns({})

def test_calculate_portfolio_volatility_valid(multi_stock_returns):
    """Test portfolio volatility calculation with valid data."""
    window = 5
    portfolio_vol = calculate_portfolio_volatility(multi_stock_returns, window)
    assert isinstance(portfolio_vol, pd.Series)
    assert len(portfolio_vol) > 0
    assert all(v >= 0 for v in portfolio_vol)  # Volatility should be non-negative

def test_calculate_portfolio_volatility_invalid_window(multi_stock_returns):
    """Test portfolio volatility calculation with invalid window size."""
    with pytest.raises(ValueError):
        calculate_portfolio_volatility(multi_stock_returns, window=1)
