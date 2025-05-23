import pytest
import pandas as pd
import numpy as np
from src.correlation import calculate_correlation_matrix, format_correlation_matrix

# Test Data Setup
@pytest.fixture
def sample_returns():
    """Create sample returns data for multiple stocks."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', tz='UTC')
    # Create returns with known correlation:
    # stock1 and stock2: perfectly positively correlated (1.0)
    # stock1 and stock3: perfectly negatively correlated (-1.0)
    # stock2 and stock3: perfectly negatively correlated (-1.0)
    stock1 = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.02]
    stock2 = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.02]  # Same as stock1
    stock3 = [-0.01, -0.02, 0.01, -0.03, 0.02, -0.01, -0.02, 0.01, -0.02]  # Inverse of stock1
    
    return {
        'STOCK1': pd.Series(stock1, index=dates[1:]),
        'STOCK2': pd.Series(stock2, index=dates[1:]),
        'STOCK3': pd.Series(stock3, index=dates[1:])
    }

@pytest.fixture
def sample_correlation_matrix():
    """Create a sample correlation matrix."""
    return pd.DataFrame(
        [[1.0, 0.5, -0.5],
         [0.5, 1.0, 0.25],
         [-0.5, 0.25, 1.0]],
        index=['STOCK1', 'STOCK2', 'STOCK3'],
        columns=['STOCK1', 'STOCK2', 'STOCK3']
    )

# Correlation Matrix Calculation Tests
def test_calculate_correlation_matrix_valid(sample_returns):
    """Test correlation matrix calculation with valid data."""
    matrix = calculate_correlation_matrix(sample_returns)
    
    # Check matrix properties
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (3, 3)  # 3x3 matrix for 3 stocks
    assert list(matrix.index) == list(matrix.columns)  # Symmetric matrix
    
    # Check correlation values
    assert matrix.loc['STOCK1', 'STOCK2'] == pytest.approx(1.0)  # Perfect positive correlation
    assert matrix.loc['STOCK1', 'STOCK3'] == pytest.approx(-1.0)  # Perfect negative correlation
    assert matrix.loc['STOCK2', 'STOCK3'] == pytest.approx(-1.0)  # Perfect negative correlation
    
    # Check diagonal values (self-correlation)
    assert all(matrix.loc[stock, stock] == 1.0 for stock in matrix.index)

def test_calculate_correlation_matrix_single_stock():
    """Test correlation matrix calculation with single stock (should fail)."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', tz='UTC')
    single_stock = {'STOCK1': pd.Series([0.01, 0.02, -0.01], index=dates[:3])}
    
    with pytest.raises(ValueError, match="At least 2 stocks are required for correlation analysis"):
        calculate_correlation_matrix(single_stock)

def test_calculate_correlation_matrix_empty():
    """Test correlation matrix calculation with empty data."""
    with pytest.raises(ValueError):
        calculate_correlation_matrix({})

def test_calculate_correlation_matrix_invalid_data():
    """Test correlation matrix calculation with invalid data."""
    invalid_data = {
        'STOCK1': pd.Series([]),
        'STOCK2': pd.Series([])
    }
    with pytest.raises(ValueError, match="No valid data for correlation calculation"):
        calculate_correlation_matrix(invalid_data)

# Correlation Matrix Formatting Tests
def test_format_correlation_matrix_valid(sample_correlation_matrix):
    """Test correlation matrix formatting with valid data."""
    formatted = format_correlation_matrix(sample_correlation_matrix)
    
    # Check formatting
    assert isinstance(formatted, pd.DataFrame)
    assert formatted.shape == sample_correlation_matrix.shape
    
    # Check percentage conversion and formatting
    assert formatted.loc['STOCK1', 'STOCK2'] == "50.00%"  # 0.5 -> 50.00%
    assert formatted.loc['STOCK1', 'STOCK3'] == "-50.00%"  # -0.5 -> -50.00%
    assert formatted.loc['STOCK2', 'STOCK3'] == "25.00%"  # 0.25 -> 25.00%
    
    # Check diagonal values
    assert all(formatted.loc[stock, stock] == "100.00%" for stock in formatted.index)

def test_format_correlation_matrix_range(sample_returns):
    """Test that formatted correlation values are within valid range (-100% to 100%)."""
    matrix = calculate_correlation_matrix(sample_returns)
    formatted = format_correlation_matrix(matrix)
    
    # Extract numeric values from percentage strings
    numeric_values = formatted.map(lambda x: float(x.strip('%')) / 100)
    
    # Check range
    assert numeric_values.min().min() >= -1.0
    assert numeric_values.max().max() <= 1.0

def test_format_correlation_matrix_symmetry(sample_correlation_matrix):
    """Test that formatted correlation matrix maintains symmetry."""
    formatted = format_correlation_matrix(sample_correlation_matrix)
    
    # Check symmetry by comparing string values
    for i in formatted.index:
        for j in formatted.columns:
            if i != j:
                # Remove % sign and convert to float for comparison
                val1 = float(formatted.loc[i, j].strip('%'))
                val2 = float(formatted.loc[j, i].strip('%'))
                assert val1 == val2
