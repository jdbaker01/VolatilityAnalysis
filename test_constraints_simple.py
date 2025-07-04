"""
Simple test script for the constraints module.
"""

from src.constraints import ConstraintSet, create_constraint_set_from_dict

def test_constraint_set():
    """Test basic functionality of the ConstraintSet class."""
    print("Testing ConstraintSet initialization...")
    constraints = ConstraintSet()
    
    print("Default values:")
    print(f"  weight_bounds: {constraints.weight_bounds}")
    print(f"  default_bounds: {constraints.default_bounds}")
    print(f"  allow_short: {constraints.allow_short}")
    print(f"  max_position_size: {constraints.max_position_size}")
    print(f"  sector_constraints: {constraints.sector_constraints}")
    print(f"  asset_sectors: {constraints.asset_sectors}")
    
    print("\nSetting weight bounds...")
    constraints.set_weight_bounds('AAPL', 0.1, 0.5)
    constraints.set_weight_bounds('MSFT', 0.2, 0.6)
    print(f"  AAPL bounds: {constraints.weight_bounds['AAPL']}")
    print(f"  MSFT bounds: {constraints.weight_bounds['MSFT']}")
    
    print("\nSetting default bounds...")
    constraints.set_default_bounds(0.05, 0.3)
    print(f"  default_bounds: {constraints.default_bounds}")
    
    print("\nSetting maximum position size...")
    constraints.set_max_position_size(0.4)
    print(f"  max_position_size: {constraints.max_position_size}")
    print(f"  AAPL bounds after max position size: {constraints.weight_bounds['AAPL']}")
    print(f"  MSFT bounds after max position size: {constraints.weight_bounds['MSFT']}")
    print(f"  default_bounds after max position size: {constraints.default_bounds}")
    
    print("\nSetting sector constraints...")
    constraints.set_sector_constraint('Technology', 0.5)
    constraints.set_sector_constraint('Finance', 0.3)
    print(f"  sector_constraints: {constraints.sector_constraints}")
    
    print("\nSetting asset sectors...")
    constraints.set_asset_sector('AAPL', 'Technology')
    constraints.set_asset_sector('MSFT', 'Technology')
    constraints.set_asset_sector('JPM', 'Finance')
    print(f"  asset_sectors: {constraints.asset_sectors}")
    
    print("\nValidating constraints...")
    is_valid, error_msg = constraints.validate()
    print(f"  is_valid: {is_valid}")
    if not is_valid:
        print(f"  error_msg: {error_msg}")
    
    print("\nGetting bounds for assets...")
    bounds = constraints.get_bounds(['AAPL', 'MSFT', 'GOOGL', 'JPM'])
    print(f"  bounds: {bounds}")
    
    print("\nConverting to dictionary...")
    constraints_dict = constraints.to_dict()
    print(f"  constraints_dict: {constraints_dict}")
    
    print("\nConverting to scipy constraints...")
    scipy_constraints = constraints.to_scipy_constraints(['AAPL', 'MSFT', 'GOOGL', 'JPM'])
    print(f"  Number of scipy constraints: {len(scipy_constraints)}")
    print(f"  Types of scipy constraints: {[c['type'] for c in scipy_constraints]}")
    
    print("\nCreating constraint set from dictionary...")
    new_constraints = create_constraint_set_from_dict(constraints_dict)
    print(f"  allow_short: {new_constraints.allow_short}")
    print(f"  max_position_size: {new_constraints.max_position_size}")
    print(f"  weight_bounds: {new_constraints.weight_bounds}")
    print(f"  sector_constraints: {new_constraints.sector_constraints}")
    print(f"  asset_sectors: {new_constraints.asset_sectors}")

if __name__ == "__main__":
    test_constraint_set()