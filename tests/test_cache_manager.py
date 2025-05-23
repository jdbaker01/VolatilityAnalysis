import pytest
import pandas as pd
import os
import shutil
from datetime import datetime
import pytz
from src.cache_manager import CacheManager

# Test Data Setup
@pytest.fixture
def test_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    return cache_dir

@pytest.fixture
def cache_manager(test_cache_dir):
    """Create a CacheManager instance for testing."""
    return CacheManager(cache_dir=str(test_cache_dir))

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D', tz='UTC')
    data = {
        'Close': [100, 102, 99, 101, 103, 102, 105, 107, 106, 108],
        'Volume': [1000, 1200, 900, 1100, 1300, 1100, 1400, 1500, 1200, 1300]
    }
    df = pd.DataFrame(data, index=dates)
    return df

# Directory Structure Tests
def test_cache_directory_creation(test_cache_dir):
    """Test cache directory is created properly."""
    CacheManager(cache_dir=str(test_cache_dir))
    assert os.path.exists(test_cache_dir)

def test_symbol_directory_creation(cache_manager):
    """Test symbol-specific directory is created."""
    symbol = "AAPL"
    symbol_dir = cache_manager._get_symbol_dir(symbol)
    assert os.path.exists(symbol_dir)
    assert symbol_dir.endswith(symbol)

# Data Storage Tests
def test_save_and_load_data(cache_manager, sample_stock_data):
    """Test saving and loading data for a symbol."""
    symbol = "AAPL"
    cache_manager.save_to_cache(symbol, sample_stock_data)
    
    start_date = pytz.UTC.localize(datetime(2023, 1, 1))
    end_date = pytz.UTC.localize(datetime(2023, 1, 10))
    
    loaded_data = cache_manager.get_cached_data(symbol, start_date, end_date)
    pd.testing.assert_frame_equal(loaded_data, sample_stock_data, check_freq=False)

def test_save_overlapping_data(cache_manager):
    """Test saving overlapping date ranges."""
    symbol = "AAPL"
    
    # First dataset: 2023-01-01 to 2023-01-05
    dates1 = pd.date_range(start='2023-01-01', end='2023-01-05', tz='UTC')
    data1 = pd.DataFrame({'Close': [100, 102, 99, 101, 103]}, index=dates1)
    
    # Second dataset: 2023-01-03 to 2023-01-07
    dates2 = pd.date_range(start='2023-01-03', end='2023-01-07', tz='UTC')
    data2 = pd.DataFrame({'Close': [98, 100, 102, 101, 104]}, index=dates2)
    
    # Save both datasets
    cache_manager.save_to_cache(symbol, data1)
    cache_manager.save_to_cache(symbol, data2)
    
    # Load the entire range
    start_date = pytz.UTC.localize(datetime(2023, 1, 1))
    end_date = pytz.UTC.localize(datetime(2023, 1, 7))
    loaded_data = cache_manager.get_cached_data(symbol, start_date, end_date)
    
    # Check the data is merged properly
    expected_range = pd.date_range(start='2023-01-01', end='2023-01-07', tz='UTC')
    assert len(loaded_data) == len(expected_range)
    assert loaded_data.index[0] == expected_range[0]
    assert loaded_data.index[-1] == expected_range[-1]
    
    # Check that the data values are correct (should keep the latest values for overlapping dates)
    assert loaded_data.loc[expected_range[0], 'Close'] == 100  # From data1
    assert loaded_data.loc[expected_range[2], 'Close'] == 98   # From data2 (overlapping date)
    assert loaded_data.loc[expected_range[-1], 'Close'] == 104 # From data2

# Cache Validation Tests
def test_has_cached_data(cache_manager, sample_stock_data):
    """Test cache validation for different date ranges."""
    symbol = "AAPL"
    cache_manager.save_to_cache(symbol, sample_stock_data)
    
    # Test exact date range
    assert cache_manager.has_cached_data(
        symbol,
        pytz.UTC.localize(datetime(2023, 1, 1)),
        pytz.UTC.localize(datetime(2023, 1, 10))
    )
    
    # Test subset of date range
    assert cache_manager.has_cached_data(
        symbol,
        pytz.UTC.localize(datetime(2023, 1, 2)),
        pytz.UTC.localize(datetime(2023, 1, 5))
    )
    
    # Test date range not in cache
    assert not cache_manager.has_cached_data(
        symbol,
        pytz.UTC.localize(datetime(2022, 12, 1)),
        pytz.UTC.localize(datetime(2022, 12, 31))
    )

def test_date_range_merging(cache_manager):
    """Test merging of overlapping date ranges in metadata."""
    ranges = [
        ["2023-01-01", "2023-01-05"],
        ["2023-01-03", "2023-01-07"],
        ["2023-01-10", "2023-01-15"]
    ]
    
    merged = cache_manager._merge_date_ranges(ranges)
    assert len(merged) == 2  # Should merge first two ranges
    assert merged[0] == ["2023-01-01", "2023-01-07"]
    assert merged[1] == ["2023-01-10", "2023-01-15"]

# Cache Clearing Tests
def test_clear_specific_symbol(cache_manager, sample_stock_data):
    """Test clearing cache for a specific symbol."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Save data for multiple symbols
    for symbol in symbols:
        cache_manager.save_to_cache(symbol, sample_stock_data)
    
    # Clear one symbol
    cache_manager.clear_cache("AAPL")
    
    # Check AAPL is cleared but others remain
    aapl_dir = os.path.join(cache_manager.cache_dir, "AAPL")
    msft_dir = os.path.join(cache_manager.cache_dir, "MSFT")
    googl_dir = os.path.join(cache_manager.cache_dir, "GOOGL")
    
    assert not os.path.exists(aapl_dir)
    assert os.path.exists(msft_dir)
    assert os.path.exists(googl_dir)

def test_clear_all_cache(cache_manager, sample_stock_data):
    """Test clearing entire cache."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Save data for multiple symbols
    for symbol in symbols:
        cache_manager.save_to_cache(symbol, sample_stock_data)
    
    # Clear all cache
    cache_manager.clear_cache()
    
    # Check all symbol directories are removed
    for symbol in symbols:
        symbol_dir = os.path.join(cache_manager.cache_dir, symbol)
        assert not os.path.exists(symbol_dir)

# Error Handling Tests
def test_invalid_date_range(cache_manager, sample_stock_data):
    """Test handling of invalid date ranges."""
    symbol = "AAPL"
    cache_manager.save_to_cache(symbol, sample_stock_data)
    
    # End date before start date
    result = cache_manager.get_cached_data(
        symbol,
        pytz.UTC.localize(datetime(2023, 1, 10)),
        pytz.UTC.localize(datetime(2023, 1, 1))
    )
    assert result is None

def test_missing_symbol(cache_manager):
    """Test handling of non-existent symbol."""
    result = cache_manager.get_cached_data(
        "INVALID",
        pytz.UTC.localize(datetime(2023, 1, 1)),
        pytz.UTC.localize(datetime(2023, 1, 10))
    )
    assert result is None
