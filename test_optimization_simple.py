"""
Simple test script for the optimization module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.optimization import MinVolatilityStrategy, MaxSharpeStrategy, EqualRiskStrategy, MaxReturnStrategy, PortfolioOptimizer, calculate_risk_contribution

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

def test_equal_risk_strategy():
    """Test the EqualRiskStrategy."""
    returns = create_sample_returns()
    strategy = EqualRiskStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(returns, {})
    
    print("EqualRiskStrategy weights:", weights)
    print("Sum of weights:", sum(weights.values()))
    
    # Check that weights are within bounds [0, 1]
    for asset, weight in weights.items():
        print(f"{asset}: {weight:.4f} (0 <= weight <= 1: {0 <= weight <= 1})")
    
    # Calculate risk contribution for each asset
    risk_contrib = calculate_risk_contribution(weights, returns)
    print("\nRisk contributions:")
    for asset, risk in risk_contrib.items():
        print(f"{asset}: {risk:.4f}")

def test_max_return_strategy():
    """Test the MaxReturnStrategy."""
    returns = create_sample_returns()
    strategy = MaxReturnStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(returns, {})
    
    print("MaxReturnStrategy weights:", weights)
    print("Sum of weights:", sum(weights.values()))
    
    # Check that weights are within bounds [0, 1]
    for asset, weight in weights.items():
        print(f"{asset}: {weight:.4f} (0 <= weight <= 1: {0 <= weight <= 1})")
    
    # Print the mean returns for each asset
    mean_returns = returns.mean() * 252  # Annualized returns
    print("\nAnnualized mean returns:")
    for asset, mean_return in mean_returns.items():
        print(f"{asset}: {mean_return:.4f}")

if __name__ == "__main__":
    print("Testing MaxSharpeStrategy...")
    test_max_sharpe_strategy()
    
    print("\nTesting MinVolatilityStrategy...")
    test_min_volatility_strategy()
    
    print("\nTesting EqualRiskStrategy...")
    test_equal_risk_strategy()
    
    print("\nTesting MaxReturnStrategy...")
    test_max_return_strategy()