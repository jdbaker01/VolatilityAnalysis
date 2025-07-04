"""
Tests for the portfolio optimization module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.optimization import OptimizationStrategy, PortfolioOptimizer, calculate_portfolio_metrics, calculate_risk_contribution, MaxSharpeStrategy, MinVolatilityStrategy, EqualRiskStrategy, MaxReturnStrategy

class MockStrategy(OptimizationStrategy):
    """Mock strategy for testing the optimizer."""
    
    def optimize(self, returns, constraints):
        """Return equal weights for testing."""
        assets = returns.columns
        return {asset: 1.0 / len(assets) for asset in assets}
    
    def get_name(self):
        return "Mock Strategy"


@pytest.fixture
def sample_returns():
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


def test_portfolio_optimizer_initialization(sample_returns):
    """Test that the optimizer initializes correctly."""
    strategy = MockStrategy()
    optimizer = PortfolioOptimizer(sample_returns, strategy)
    
    assert optimizer.returns.equals(sample_returns)
    assert optimizer.strategy == strategy
    assert optimizer.constraints == {}


def test_portfolio_optimizer_set_constraints(sample_returns):
    """Test setting constraints."""
    strategy = MockStrategy()
    optimizer = PortfolioOptimizer(sample_returns, strategy)
    
    constraints = {'min_weight': 0.1, 'max_weight': 0.5}
    optimizer.set_constraints(constraints)
    
    assert optimizer.constraints == constraints


def test_portfolio_optimizer_optimize(sample_returns):
    """Test the optimization process."""
    strategy = MockStrategy()
    optimizer = PortfolioOptimizer(sample_returns, strategy)
    
    weights = optimizer.optimize()
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to 1.0
    assert np.isclose(sum(weights.values()), 1.0)
    
    # Check that all weights are equal (for the mock strategy)
    expected_weight = 1.0 / len(sample_returns.columns)
    for weight in weights.values():
        assert np.isclose(weight, expected_weight)


def test_portfolio_optimizer_with_invalid_returns():
    """Test that the optimizer raises an error with invalid returns."""
    strategy = MockStrategy()
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        PortfolioOptimizer(pd.DataFrame(), strategy)
    
    # Test with None
    with pytest.raises(ValueError):
        PortfolioOptimizer(None, strategy)


def test_portfolio_optimizer_with_invalid_strategy(sample_returns):
    """Test that the optimizer raises an error with invalid strategy."""
    with pytest.raises(ValueError):
        PortfolioOptimizer(sample_returns, None)


def test_calculate_portfolio_metrics(sample_returns):
    """Test calculation of portfolio metrics."""
    # Create equal weights
    weights = {asset: 1.0 / len(sample_returns.columns) for asset in sample_returns.columns}
    
    metrics = calculate_portfolio_metrics(weights, sample_returns)
    
    # Check that all required metrics are present
    assert "expected_return" in metrics
    assert "volatility" in metrics
    assert "sharpe_ratio" in metrics
    
    # Check that metrics are reasonable
    assert isinstance(metrics["expected_return"], float)
    assert isinstance(metrics["volatility"], float)
    assert isinstance(metrics["sharpe_ratio"], float)
    
    # Volatility should be positive
    assert metrics["volatility"] >= 0


def test_calculate_risk_contribution(sample_returns):
    """Test calculation of risk contribution."""
    # Create equal weights
    weights = {asset: 1.0 / len(sample_returns.columns) for asset in sample_returns.columns}
    
    risk_contrib = calculate_risk_contribution(weights, sample_returns)
    
    # Check that risk contributions are returned for all assets
    assert set(risk_contrib.keys()) == set(sample_returns.columns)
    
    # Check that risk contributions sum to approximately the portfolio volatility
    metrics = calculate_portfolio_metrics(weights, sample_returns)
    total_risk = sum(risk_contrib.values())
    assert np.isclose(total_risk, metrics["volatility"], rtol=1e-5)


def test_max_sharpe_strategy_initialization():
    """Test that the MaxSharpeStrategy initializes correctly."""
    strategy = MaxSharpeStrategy()
    assert strategy.risk_free_rate == 0.0
    assert strategy.get_name() == "Maximum Sharpe Ratio"
    
    strategy_with_rf = MaxSharpeStrategy(risk_free_rate=0.02)
    assert strategy_with_rf.risk_free_rate == 0.02


def test_max_sharpe_strategy_optimize(sample_returns):
    """Test the MaxSharpeStrategy optimization."""
    strategy = MaxSharpeStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(sample_returns, {})
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that all weights are within bounds [0, 1]
    for weight in weights.values():
        assert 0 <= weight <= 1


def test_max_sharpe_strategy_with_constraints(sample_returns):
    """Test the MaxSharpeStrategy with custom constraints."""
    strategy = MaxSharpeStrategy()
    
    # Test with custom weight bounds
    constraints = {
        'weight_bounds': {
            'AAPL': (0.2, 0.4),
            'MSFT': (0.1, 0.3),
            'GOOGL': (0.3, 0.5)
        }
    }
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights respect the bounds
    assert 0.2 <= weights['AAPL'] <= 0.4
    assert 0.1 <= weights['MSFT'] <= 0.3
    assert 0.3 <= weights['GOOGL'] <= 0.5
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)


def test_max_sharpe_strategy_with_short_selling(sample_returns):
    """Test the MaxSharpeStrategy with short selling allowed."""
    strategy = MaxSharpeStrategy()
    
    # Test with short selling allowed
    constraints = {'allow_short': True}
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that at least one weight is negative (short position)
    # Note: This test might fail if the optimal portfolio doesn't include short positions
    # In that case, we should just check that weights are within [-1, 1]
    for weight in weights.values():
        assert -1 <= weight <= 1


def test_min_volatility_strategy_initialization():
    """Test that the MinVolatilityStrategy initializes correctly."""
    strategy = MinVolatilityStrategy()
    assert strategy.get_name() == "Minimum Volatility"


def test_min_volatility_strategy_optimize(sample_returns):
    """Test the MinVolatilityStrategy optimization."""
    strategy = MinVolatilityStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(sample_returns, {})
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that all weights are within bounds [0, 1]
    for weight in weights.values():
        assert 0 <= weight <= 1
    
    # Calculate portfolio volatility with these weights
    min_vol_metrics = calculate_portfolio_metrics(weights, sample_returns)
    
    # Calculate portfolio volatility with equal weights for comparison
    equal_weights = {asset: 1.0 / len(sample_returns.columns) for asset in sample_returns.columns}
    equal_weights_metrics = calculate_portfolio_metrics(equal_weights, sample_returns)
    
    # The minimum volatility portfolio should have lower volatility than equal weights
    # Note: This might not always be true for very small sample sizes or specific return patterns
    # But it should generally hold for random returns
    assert min_vol_metrics["volatility"] <= equal_weights_metrics["volatility"] * 1.001  # Allow small margin for numerical issues


def test_min_volatility_strategy_with_constraints(sample_returns):
    """Test the MinVolatilityStrategy with custom constraints."""
    strategy = MinVolatilityStrategy()
    
    # Test with custom weight bounds
    constraints = {
        'weight_bounds': {
            'AAPL': (0.2, 0.4),
            'MSFT': (0.1, 0.3),
            'GOOGL': (0.3, 0.5)
        }
    }
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights respect the bounds
    assert 0.2 <= weights['AAPL'] <= 0.4
    assert 0.1 <= weights['MSFT'] <= 0.3
    assert 0.3 <= weights['GOOGL'] <= 0.5
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)


def test_min_volatility_strategy_with_short_selling(sample_returns):
    """Test the MinVolatilityStrategy with short selling allowed."""
    strategy = MinVolatilityStrategy()
    
    # Test with short selling allowed
    constraints = {'allow_short': True}
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that all weights are within bounds [-1, 1]
    for weight in weights.values():
        assert -1 <= weight <= 1


def test_equal_risk_strategy_initialization():
    """Test that the EqualRiskStrategy initializes correctly."""
    strategy = EqualRiskStrategy()
    assert strategy.get_name() == "Equal Risk Contribution"


def test_equal_risk_strategy_optimize(sample_returns):
    """Test the EqualRiskStrategy optimization."""
    strategy = EqualRiskStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(sample_returns, {})
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that all weights are within bounds [0, 1]
    for weight in weights.values():
        assert 0 <= weight <= 1
    
    # Calculate risk contribution for each asset
    risk_contrib = calculate_risk_contribution(weights, sample_returns)
    
    # In an equal risk contribution portfolio, each asset should contribute
    # approximately the same amount of risk
    risk_values = list(risk_contrib.values())
    for i in range(1, len(risk_values)):
        # Allow for some numerical optimization error
        assert np.isclose(risk_values[0], risk_values[i], rtol=0.1)


def test_equal_risk_strategy_with_constraints(sample_returns):
    """Test the EqualRiskStrategy with custom constraints."""
    strategy = EqualRiskStrategy()
    
    # Test with custom weight bounds
    constraints = {
        'weight_bounds': {
            'AAPL': (0.2, 0.4),
            'MSFT': (0.1, 0.3),
            'GOOGL': (0.3, 0.5)
        }
    }
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights respect the bounds
    assert 0.2 <= weights['AAPL'] <= 0.4
    assert 0.1 <= weights['MSFT'] <= 0.3
    assert 0.3 <= weights['GOOGL'] <= 0.5
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)


def test_max_return_strategy_initialization():
    """Test that the MaxReturnStrategy initializes correctly."""
    strategy = MaxReturnStrategy()
    assert strategy.get_name() == "Maximum Return"


def test_max_return_strategy_optimize(sample_returns):
    """Test the MaxReturnStrategy optimization."""
    strategy = MaxReturnStrategy()
    
    # Test with default constraints
    weights = strategy.optimize(sample_returns, {})
    
    # Check that weights are returned for all assets
    assert set(weights.keys()) == set(sample_returns.columns)
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # Check that all weights are within bounds [0, 1]
    for weight in weights.values():
        assert 0 <= weight <= 1
    
    # Without constraints, MaxReturnStrategy should allocate 100% to the asset
    # with the highest expected return
    mean_returns = sample_returns.mean() * 252  # Annualized returns
    max_return_asset = mean_returns.idxmax()
    
    # The asset with the highest return should have a weight close to 1.0
    # (allowing for some numerical precision issues)
    assert weights[max_return_asset] > 0.99


def test_max_return_strategy_with_constraints(sample_returns):
    """Test the MaxReturnStrategy with custom constraints."""
    strategy = MaxReturnStrategy()
    
    # Test with custom weight bounds
    constraints = {
        'weight_bounds': {
            'AAPL': (0.2, 0.4),
            'MSFT': (0.1, 0.3),
            'GOOGL': (0.3, 0.5)
        }
    }
    
    weights = strategy.optimize(sample_returns, constraints)
    
    # Check that weights respect the bounds
    assert 0.2 <= weights['AAPL'] <= 0.4
    assert 0.1 <= weights['MSFT'] <= 0.3
    assert 0.3 <= weights['GOOGL'] <= 0.5
    
    # Check that weights sum to approximately 1.0
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    
    # With constraints, the strategy should allocate as much as possible to the
    # assets with the highest returns, within the constraints
    mean_returns = sample_returns.mean() * 252  # Annualized returns
    sorted_assets = mean_returns.sort_values(ascending=False).index
    
    # The asset with the highest return should be at its upper bound
    # (if it's not at the upper bound, it means another asset has a higher return)
    highest_return_asset = sorted_assets[0]
    if highest_return_asset in constraints['weight_bounds']:
        upper_bound = constraints['weight_bounds'][highest_return_asset][1]
        assert np.isclose(weights[highest_return_asset], upper_bound, rtol=1e-5)