"""
Test imports to identify issues.
"""

try:
    from src.optimization import OptimizationStrategy, PortfolioOptimizer
    print("Successfully imported OptimizationStrategy and PortfolioOptimizer")
except Exception as e:
    print(f"Error importing from optimization.py: {str(e)}")

try:
    from src.constraints import ConstraintSet, create_constraint_set_from_dict
    print("Successfully imported ConstraintSet and create_constraint_set_from_dict")
except Exception as e:
    print(f"Error importing from constraints.py: {str(e)}")

try:
    import numpy as np
    import pandas as pd
    from scipy import optimize
    print("Successfully imported numpy, pandas, and scipy.optimize")
except Exception as e:
    print(f"Error importing dependencies: {str(e)}")

print("Import test complete")