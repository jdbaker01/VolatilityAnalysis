"""
Simple test script for position size and sector constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.constraints import ConstraintSet

def create_sample_returns():
    """Create sample returns data for testing."""
    # Create a date range
    dates = [datetime.now(pytz.UTC) - timedelta(days=i) for i in range(100)]
    
    # Create sample returns for 5 assets with fixed seed for reproducibility
    np.random.seed(42)
    data = {
        'AAPL': np.random.normal(0.001, 0.02, 100),  # Technology
        'MSFT': np.random.normal(0.0012, 0.018, 100),  # Technology
        'GOOGL': np.random.normal(0.0008, 0.022, 100),  # Technology
        'JPM': np.random.normal(0.0005, 0.015, 100),  # Finance
        'BAC': np.random.normal(0.0006, 0.017, 100)  # Finance
    }
    
    return pd.DataFrame(data, index=dates)

def test_position_size_constraints():
    """Test position size constraints."""
    # Create a constraint set with maximum position size
    constraints = ConstraintSet()
    constraints.set_max_position_size(0.3)
    
    # Set some weight bounds
    constraints.set_weight_bounds('AAPL', 0.1, 0.5)
    constraints.set_weight_bounds('MSFT', 0.2, 0.6)
    
    # Get bounds for assets
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC'])
    
    print("Position size constraints (max_position_size = 0.3):")
    print(f"  AAPL bounds: {bounds[0]}")
    print(f"  MSFT bounds: {bounds[1]}")
    print(f"  GOOGL bounds: {bounds[2]}")
    print(f"  JPM bounds: {bounds[3]}")
    print(f"  BAC bounds: {bounds[4]}")
    
    # Check that all bounds respect the maximum position size
    all_respect_max = all(bound[1] <= 0.3 for bound in bounds)
    print(f"  All bounds respect max position size: {all_respect_max}")

def test_sector_constraints():
    """Test sector constraints."""
    # Create a constraint set with sector constraints
    constraints = ConstraintSet()
    
    # Set sector constraints
    constraints.set_sector_constraint('Technology', 0.6)
    constraints.set_sector_constraint('Finance', 0.4)
    
    # Set asset sectors
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    constraints.set_asset_sector('GOOGL', 'Technology')
    constraints.set_asset_sector('JPM', 'Finance')
    constraints.set_asset_sector('BAC', 'Finance')
    
    # Convert to scipy constraints
    scipy_constraints = constraints.to_scipy_constraints(['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC'])
    
    print("\nSector constraints (Technology <= 0.6, Finance <= 0.4):")
    print(f"  Number of scipy constraints: {len(scipy_constraints)}")
    print(f"  Types of scipy constraints: {[c['type'] for c in scipy_constraints]}")
    
    # Test the sector constraints with some weights
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights
    
    # Check the sum to 1 constraint
    sum_constraint = scipy_constraints[0]['fun'](weights)
    print(f"  Sum constraint (should be 0): {sum_constraint}")
    
    # Check the Technology sector constraint
    tech_constraint = scipy_constraints[1]['fun'](weights)
    print(f"  Technology constraint (should be >= 0): {tech_constraint}")
    
    # Check the Finance sector constraint
    finance_constraint = scipy_constraints[2]['fun'](weights)
    print(f"  Finance constraint (should be >= 0): {finance_constraint}")

def test_combined_constraints():
    """Test combined position size and sector constraints."""
    # Create a constraint set with both position size and sector constraints
    constraints = ConstraintSet()
    
    # Set maximum position size
    constraints.set_max_position_size(0.25)
    
    # Set sector constraints
    constraints.set_sector_constraint('Technology', 0.6)
    constraints.set_sector_constraint('Finance', 0.4)
    
    # Set asset sectors
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    constraints.set_asset_sector('GOOGL', 'Technology')
    constraints.set_asset_sector('JPM', 'Finance')
    constraints.set_asset_sector('BAC', 'Finance')
    
    # Get bounds for assets
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC'])
    
    # Convert to scipy constraints
    scipy_constraints = constraints.to_scipy_constraints(['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC'])
    
    print("\nCombined constraints (max_position_size = 0.25, Technology <= 0.6, Finance <= 0.4):")
    print(f"  AAPL bounds: {bounds[0]}")
    print(f"  MSFT bounds: {bounds[1]}")
    print(f"  GOOGL bounds: {bounds[2]}")
    print(f"  JPM bounds: {bounds[3]}")
    print(f"  BAC bounds: {bounds[4]}")
    print(f"  Number of scipy constraints: {len(scipy_constraints)}")
    print(f"  Types of scipy constraints: {[c['type'] for c in scipy_constraints]}")
    
    # Test the constraints with some weights
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights
    
    # Check the sum to 1 constraint
    sum_constraint = scipy_constraints[0]['fun'](weights)
    print(f"  Sum constraint (should be 0): {sum_constraint}")
    
    # Check the Technology sector constraint
    tech_constraint = scipy_constraints[1]['fun'](weights)
    print(f"  Technology constraint (should be >= 0): {tech_constraint}")
    
    # Check the Finance sector constraint
    finance_constraint = scipy_constraints[2]['fun'](weights)
    print(f"  Finance constraint (should be >= 0): {finance_constraint}")

if __name__ == "__main__":
    test_position_size_constraints()
    test_sector_constraints()
    test_combined_constraints()