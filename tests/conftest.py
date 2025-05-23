"""
Test configuration and shared fixtures for all test modules.
"""
import pytest
import pandas as pd
from datetime import datetime

@pytest.fixture
def sample_dates():
    """Provide a standard set of dates for testing."""
    return pd.date_range(start='2023-01-01', end='2023-01-10')

@pytest.fixture
def sample_prices():
    """Provide a standard set of prices for testing."""
    return [100, 102, 99, 101, 103, 102, 105, 107, 106, 108]

@pytest.fixture
def sample_returns():
    """Provide a standard set of returns for testing."""
    return [0.02, -0.03, 0.02, 0.02, -0.01, 0.03, 0.019, -0.009, 0.019]

@pytest.fixture
def test_start_date():
    """Provide a standard start date for testing."""
    return datetime(2023, 1, 1)

@pytest.fixture
def test_end_date():
    """Provide a standard end date for testing."""
    return datetime(2023, 1, 10)
