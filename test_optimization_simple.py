"""
Simple test script for the optimization module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.optimization import MinVolatilityStrategy, MaxSharpeStrategy, EqualRiskStrategy, MaxReturnStrategy, PortfolioOptimizer, calculate_portfolio_metrics, calculate_risk_contribution
from src.constraints import ConstraintSet

class MockStrategy:
    """Mock strategy for testing the optimizer."""
    
    def optimize(self, returns, constraints):
        """Return equal weights for testing."""
        assets = returns.columns
        return {asset: 1.0 / len(assets) for asset in assets}
    
    def get_name(self):
        return "Mock Strategy"

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

def test_portfolio_optimizer():
    """Test the PortfolioOptimizer class."""
    returns = create_sample_returns()
    strategy = MockStrategy()
    
    # Test initialization
    optimizer = PortfolioOptimizer(returns, strategy)
    print("PortfolioOptimizer initialization:")
    print(f"  returns shape: {optimizer.returns.shape}")
    print(f"  strategy name: {optimizer.strategy.get_name()}")
    print(f"  constraints type: {type(optimizer.constraints)}")
    
    # Test setting constraints
    constraints = {'allow_short': False, 'max_position_size': 0.5}
    optimizer.set_constraints(constraints)
    print("\nAfter setting constraints:")
    print(f"  constraints type: {type(optimizer.constraints)}")
    print(f"  allow_short: {optimizer.constraints.allow_short}")
    print(f"  max_position_size: {optimizer.constraints.max_position_size}")
    
    # Test optimization
    weights = optimizer.optimize()
    print("\nOptimization results:")
    print(f"  weights: {weights}")
    print(f"  sum of weights: {sum(weights.values())}")
    
    # Test with ConstraintSet
    constraint_set = ConstraintSet()
    constraint_set.set_weight_bounds('AAPL', 0.2, 0.4)
    constraint_set.set_weight_bounds('MSFT', 0.1, 0.3)
    constraint_set.set_weight_bounds('GOOGL', 0.3, 0.5)
    
    optimizer.set_constraints(constraint_set)
    print("\nAfter setting ConstraintSet:")
    print(f"  constraints type: {type(optimizer.constraints)}")
    print(f"  AAPL bounds: {optimizer.constraints.weight_bounds['AAPL']}")
    print(f"  MSFT bounds: {optimizer.constraints.weight_bounds['MSFT']}")
    print(f"  GOOGL bounds: {optimizer.constraints.weight_bounds['GOOGL']}")

def test_max_sharpe_strategy():
    """Test the MaxSharpeStrategy."""
    returns = create_sample_returns()
    strategy = MaxSharpeStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(returns, {})
    
    print("MaxSharpeStrategy weights:")
    print(f"  weights: {weights}")
    print(f"  sum of weights: {sum(weights.values())}")
    
    # Test with ConstraintSet
    constraint_set = ConstraintSet()
    constraint_set.set_weight_bounds('AAPL', 0.2, 0.4)
    constraint_set.set_weight_bounds('MSFT', 0.1, 0.3)
    constraint_set.set_weight_bounds('GOOGL', 0.3, 0.5)
    
    weights = strategy.optimize(returns, constraint_set)
    print("\nMaxSharpeStrategy with ConstraintSet:")
    print(f"  weights: {weights}")
    print(f"  sum of weights: {sum(weights.values())}")
    print(f"  AAPL weight within bounds: {0.2 <= weights['AAPL'] <= 0.4}")
    print(f"  MSFT weight within bounds: {0.1 <= weights['MSFT'] <= 0.3}")
    print(f"  GOOGL weight within bounds: {0.3 <= weights['GOOGL'] <= 0.5}")

if __name__ == "__main__":
    print("Testing PortfolioOptimizer...")
    test_portfolio_optimizer()
    
    print("\nTesting MaxSharpeStrategy...")
    test_max_sharpe_strategy()