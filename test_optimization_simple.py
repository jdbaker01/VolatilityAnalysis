"""
Simple test script for the optimization module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.optimization import MinVolatilityStrategy, MaxSharpeStrategy, PortfolioOptimizer

def create_sample_returns():
    """Create sample returns data for testing."""
    # Create a date range
    dates = [datetime.now(pytz.UTC) - timedelta(days=i) for i in range(100)]
    
    # Create sample returns for 3 assets
    data = {
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'MSFT': np.random.normal(0.0012, 0.018, 100),
        'GOOGL': np.random.normal(0.0008, 0.022, 100)
    }
    
    return pd.DataFrame(data, index=dates)

def test_max_sharpe_strategy():
    """Test the MaxSharpeStrategy."""
    returns = create_sample_returns()
    strategy = MaxSharpeStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(returns, {})
    
    print("MaxSharpeStrategy weights:", weights)
    print("Sum of weights:", sum(weights.values()))
    
    # Check that weights are within bounds [0, 1]
    for asset, weight in weights.items():
        print(f"{asset}: {weight:.4f} (0 <= weight <= 1: {0 <= weight <= 1})")

def test_min_volatility_strategy():
    """Test the MinVolatilityStrategy."""
    returns = create_sample_returns()
    strategy = MinVolatilityStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(returns, {})
    
    print("MinVolatilityStrategy weights:", weights)
    print("Sum of weights:", sum(weights.values()))
    
    # Check that weights are within bounds [0, 1]
    for asset, weight in weights.items():
        print(f"{asset}: {weight:.4f} (0 <= weight <= 1: {0 <= weight <= 1})")

if __name__ == "__main__":
    print("Testing MaxSharpeStrategy...")
    test_max_sharpe_strategy()
    
    print("\nTesting MinVolatilityStrategy...")
    test_min_volatility_strategy()