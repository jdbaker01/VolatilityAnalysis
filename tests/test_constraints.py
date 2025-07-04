"""
Tests for the constraint handling module.
"""

import pytest
from src.constraints import ConstraintSet, create_constraint_set_from_dict


def test_constraint_set_initialization():
    """Test that the ConstraintSet initializes with default values."""
    constraints = ConstraintSet()
    
    assert constraints.weight_bounds == {}
    assert constraints.default_bounds == (0.0, 1.0)
    assert constraints.allow_short is False
    assert constraints.max_position_size == 1.0
    assert constraints.sector_constraints == {}
    assert constraints.asset_sectors == {}


def test_set_weight_bounds():
    """Test setting weight bounds for specific assets."""
    constraints = ConstraintSet()
    
    # Set valid bounds
    constraints.set_weight_bounds('AAPL', 0.1, 0.5)
    assert constraints.weight_bounds['AAPL'] == (0.1, 0.5)
    
    # Set bounds with min = max
    constraints.set_weight_bounds('MSFT', 0.2, 0.2)
    assert constraints.weight_bounds['MSFT'] == (0.2, 0.2)
    
    # Test invalid bounds (min > max)
    with pytest.raises(ValueError):
        constraints.set_weight_bounds('GOOGL', 0.6, 0.4)
    
    # Test invalid bounds (min < 0 with short selling disabled)
    with pytest.raises(ValueError):
        constraints.set_weight_bounds('TSLA', -0.1, 0.3)
    
    # Test invalid bounds (min < -1.0)
    constraints.set_allow_short(True)
    with pytest.raises(ValueError):
        constraints.set_weight_bounds('TSLA', -1.1, 0.3)
    
    # Test invalid bounds (max > 1.0)
    with pytest.raises(ValueError):
        constraints.set_weight_bounds('TSLA', 0.0, 1.1)


def test_set_default_bounds():
    """Test setting default bounds for assets without specific bounds."""
    constraints = ConstraintSet()
    
    # Set valid default bounds
    constraints.set_default_bounds(0.05, 0.3)
    assert constraints.default_bounds == (0.05, 0.3)
    
    # Test invalid bounds (min > max)
    with pytest.raises(ValueError):
        constraints.set_default_bounds(0.4, 0.2)
    
    # Test invalid bounds (min < 0 with short selling disabled)
    with pytest.raises(ValueError):
        constraints.set_default_bounds(-0.1, 0.3)
    
    # Test invalid bounds (min < -1.0)
    constraints.set_allow_short(True)
    with pytest.raises(ValueError):
        constraints.set_default_bounds(-1.1, 0.3)
    
    # Test invalid bounds (max > 1.0)
    with pytest.raises(ValueError):
        constraints.set_default_bounds(0.0, 1.1)


def test_set_allow_short():
    """Test setting the allow_short flag and its effect on weight bounds."""
    constraints = ConstraintSet()
    
    # Enable short selling
    constraints.set_allow_short(True)
    assert constraints.allow_short is True
    
    # Set negative weight bounds
    constraints.set_weight_bounds('AAPL', -0.2, 0.5)
    constraints.set_default_bounds(-0.1, 0.3)
    
    # Disable short selling and check that bounds are adjusted
    constraints.set_allow_short(False)
    assert constraints.allow_short is False
    assert constraints.weight_bounds['AAPL'] == (0.0, 0.5)
    assert constraints.default_bounds == (0.0, 0.3)


def test_set_max_position_size():
    """Test setting the maximum position size and its effect on weight bounds."""
    constraints = ConstraintSet()
    
    # Set weight bounds
    constraints.set_weight_bounds('AAPL', 0.1, 0.8)
    constraints.set_weight_bounds('MSFT', 0.2, 0.6)
    constraints.set_default_bounds(0.05, 0.7)
    
    # Set maximum position size
    constraints.set_max_position_size(0.5)
    assert constraints.max_position_size == 0.5
    
    # Check that weight bounds are adjusted
    assert constraints.weight_bounds['AAPL'] == (0.1, 0.5)
    assert constraints.weight_bounds['MSFT'] == (0.2, 0.5)
    assert constraints.default_bounds == (0.05, 0.5)
    
    # Test invalid max position size
    with pytest.raises(ValueError):
        constraints.set_max_position_size(0.0)
    
    with pytest.raises(ValueError):
        constraints.set_max_position_size(1.1)


def test_set_sector_constraint():
    """Test setting sector constraints."""
    constraints = ConstraintSet()
    
    # Set valid sector constraints
    constraints.set_sector_constraint('Technology', 0.4)
    constraints.set_sector_constraint('Finance', 0.3)
    
    assert constraints.sector_constraints['Technology'] == 0.4
    assert constraints.sector_constraints['Finance'] == 0.3
    
    # Test invalid sector constraints
    with pytest.raises(ValueError):
        constraints.set_sector_constraint('Energy', 0.0)
    
    with pytest.raises(ValueError):
        constraints.set_sector_constraint('Materials', 1.1)


def test_set_asset_sector():
    """Test setting asset sectors."""
    constraints = ConstraintSet()
    
    # Set asset sectors
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    constraints.set_asset_sector('JPM', 'Finance')
    
    assert constraints.asset_sectors['AAPL'] == 'Technology'
    assert constraints.asset_sectors['MSFT'] == 'Technology'
    assert constraints.asset_sectors['JPM'] == 'Finance'


def test_validate():
    """Test constraint validation."""
    constraints = ConstraintSet()
    
    # Valid constraints
    is_valid, _ = constraints.validate()
    assert is_valid
    
    # Set sector constraints
    constraints.set_sector_constraint('Technology', 0.4)
    constraints.set_sector_constraint('Finance', 0.7)
    
    # Valid sector constraints (sum > 1.0)
    is_valid, _ = constraints.validate()
    assert is_valid
    
    # Set asset sectors
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    constraints.set_asset_sector('JPM', 'Finance')
    
    # Set weight bounds
    constraints.set_weight_bounds('AAPL', 0.1, 0.3)
    
    # Valid constraints with sectors and bounds
    is_valid, _ = constraints.validate()
    assert is_valid
    
    # Invalid: asset with weight bounds but no sector
    constraints.set_weight_bounds('GOOGL', 0.1, 0.3)
    is_valid, error_msg = constraints.validate()
    assert not is_valid
    assert "has no sector assigned" in error_msg


def test_get_bounds():
    """Test getting bounds for a list of assets."""
    constraints = ConstraintSet()
    
    # Set weight bounds
    constraints.set_weight_bounds('AAPL', 0.1, 0.5)
    constraints.set_weight_bounds('MSFT', 0.2, 0.6)
    constraints.set_default_bounds(0.05, 0.3)
    
    # Get bounds for assets
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL'])
    assert bounds == [(0.1, 0.5), (0.2, 0.6), (0.05, 0.3)]
    
    # Set maximum position size
    constraints.set_max_position_size(0.4)
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL'])
    assert bounds == [(0.1, 0.4), (0.2, 0.4), (0.05, 0.3)]
    
    # Enable short selling and set negative bounds
    constraints.set_allow_short(True)
    constraints.set_weight_bounds('TSLA', -0.2, 0.3)
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
    assert bounds == [(0.1, 0.4), (0.2, 0.4), (0.05, 0.3), (-0.2, 0.3)]
    
    # Disable short selling and check that negative bounds are adjusted
    constraints.set_allow_short(False)
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
    assert bounds == [(0.1, 0.4), (0.2, 0.4), (0.05, 0.3), (0.0, 0.3)]


def test_to_dict():
    """Test converting constraints to a dictionary."""
    constraints = ConstraintSet()
    
    # Set constraints
    constraints.set_allow_short(True)
    constraints.set_max_position_size(0.4)
    constraints.set_weight_bounds('AAPL', -0.1, 0.3)
    constraints.set_weight_bounds('MSFT', 0.2, 0.4)
    constraints.set_sector_constraint('Technology', 0.5)
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    
    # Convert to dictionary
    constraints_dict = constraints.to_dict()
    
    assert constraints_dict['allow_short'] is True
    assert constraints_dict['max_position_size'] == 0.4
    assert constraints_dict['weight_bounds']['AAPL'] == (-0.1, 0.3)
    assert constraints_dict['weight_bounds']['MSFT'] == (0.2, 0.4)
    assert constraints_dict['sector_constraints']['Technology'] == 0.5
    assert constraints_dict['asset_sectors']['AAPL'] == 'Technology'
    assert constraints_dict['asset_sectors']['MSFT'] == 'Technology'


def test_to_scipy_constraints():
    """Test converting constraints to scipy format."""
    constraints = ConstraintSet()
    
    # Set constraints
    constraints.set_sector_constraint('Technology', 0.5)
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    
    # Convert to scipy constraints
    scipy_constraints = constraints.to_scipy_constraints(['AAPL', 'MSFT', 'GOOGL'])
    
    # Should have 2 constraints: sum to 1 and sector constraint
    assert len(scipy_constraints) == 2
    assert scipy_constraints[0]['type'] == 'eq'  # Sum to 1
    assert scipy_constraints[1]['type'] == 'ineq'  # Sector constraint


def test_create_constraint_set_from_dict():
    """Test creating a ConstraintSet from a dictionary."""
    constraints_dict = {
        'allow_short': True,
        'max_position_size': 0.4,
        'weight_bounds': {
            'AAPL': (-0.1, 0.3),
            'MSFT': (0.2, 0.4)
        },
        'sector_constraints': {
            'Technology': 0.5,
            'Finance': 0.3
        },
        'asset_sectors': {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'JPM': 'Finance'
        }
    }
    
    constraints = create_constraint_set_from_dict(constraints_dict)
    
    assert constraints.allow_short is True
    assert constraints.max_position_size == 0.4
    assert constraints.weight_bounds['AAPL'] == (-0.1, 0.3)
    assert constraints.weight_bounds['MSFT'] == (0.2, 0.4)
    assert constraints.sector_constraints['Technology'] == 0.5
    assert constraints.sector_constraints['Finance'] == 0.3
    assert constraints.asset_sectors['AAPL'] == 'Technology'
    assert constraints.asset_sectors['MSFT'] == 'Technology'
    assert constraints.asset_sectors['JPM'] == 'Finance'