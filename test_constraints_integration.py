"""
Test script for the integration of constraints with optimization strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.optimization import MinVolatilityStrategy, MaxSharpeStrategy, EqualRiskStrategy, MaxReturnStrategy
from src.constraints import ConstraintSet

def create_sample_returns():
    """Create sample returns data for testing."""
    # Create a date range
    dates = [datetime.now(pytz.UTC) - timedelta(days=i) for i in range(100)]
    
    # Create sample returns for 3 assets with fixed seed for reproducibility
    np.random.seed(42)
    data = {
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'MSFT': np.random.normal(0.0012, 0.018, 100),
        'GOOGL': np.random.normal(0.0008, 0.022, 100)
    }
    
    return pd.DataFrame(data, index=dates)

def test_with_constraint_set():
    """Test optimization strategies with ConstraintSet."""
    returns = create_sample_returns()
    
    # Create a constraint set
    constraints = ConstraintSet()
    constraints.set_weight_bounds('AAPL', 0.2, 0.4)
    constraints.set_weight_bounds('MSFT', 0.1, 0.3)
    constraints.set_weight_bounds('GOOGL', 0.3, 0.5)
    
    # Test MinVolatilityStrategy
    min_vol_strategy = MinVolatilityStrategy()
    min_vol_weights = min_vol_strategy.optimize(returns, constraints)
    
    print("MinVolatilityStrategy with ConstraintSet:")
    print(f"  weights: {min_vol_weights}")
    print(f"  sum of weights: {sum(min_vol_weights.values())}")
    print(f"  AAPL weight within bounds: {0.2 <= min_vol_weights['AAPL'] <= 0.4}")
    print(f"  MSFT weight within bounds: {0.1 <= min_vol_weights['MSFT'] <= 0.3}")
    print(f"  GOOGL weight within bounds: {0.3 <= min_vol_weights['GOOGL'] <= 0.5}")
    
    # Test MaxSharpeStrategy
    max_sharpe_strategy = MaxSharpeStrategy()
    max_sharpe_weights = max_sharpe_strategy.optimize(returns, constraints)
    
    print("\nMaxSharpeStrategy with ConstraintSet:")
    print(f"  weights: {max_sharpe_weights}")
    print(f"  sum of weights: {sum(max_sharpe_weights.values())}")
    print(f"  AAPL weight within bounds: {0.2 <= max_sharpe_weights['AAPL'] <= 0.4}")
    print(f"  MSFT weight within bounds: {0.1 <= max_sharpe_weights['MSFT'] <= 0.3}")
    print(f"  GOOGL weight within bounds: {0.3 <= max_sharpe_weights['GOOGL'] <= 0.5}")
    
    # Test EqualRiskStrategy
    equal_risk_strategy = EqualRiskStrategy()
    equal_risk_weights = equal_risk_strategy.optimize(returns, constraints)
    
    print("\nEqualRiskStrategy with ConstraintSet:")
    print(f"  weights: {equal_risk_weights}")
    print(f"  sum of weights: {sum(equal_risk_weights.values())}")
    print(f"  AAPL weight within bounds: {0.2 <= equal_risk_weights['AAPL'] <= 0.4}")
    print(f"  MSFT weight within bounds: {0.1 <= equal_risk_weights['MSFT'] <= 0.3}")
    print(f"  GOOGL weight within bounds: {0.3 <= equal_risk_weights['GOOGL'] <= 0.5}")
    
    # Test MaxReturnStrategy
    max_return_strategy = MaxReturnStrategy()
    max_return_weights = max_return_strategy.optimize(returns, constraints)
    
    print("\nMaxReturnStrategy with ConstraintSet:")
    print(f"  weights: {max_return_weights}")
    print(f"  sum of weights: {sum(max_return_weights.values())}")
    print(f"  AAPL weight within bounds: {0.2 <= max_return_weights['AAPL'] <= 0.4}")
    print(f"  MSFT weight within bounds: {0.1 <= max_return_weights['MSFT'] <= 0.3}")
    print(f"  GOOGL weight within bounds: {0.3 <= max_return_weights['GOOGL'] <= 0.5}")

if __name__ == "__main__":
    test_with_constraint_set()