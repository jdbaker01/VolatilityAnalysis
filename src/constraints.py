"""
Constraint handling for portfolio optimization.

This module provides classes and functions for defining and validating
constraints for portfolio optimization.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from src.logger import logger


class ConstraintSet:
    """
    Container for all portfolio optimization constraints.
    
    This class provides a structured way to define and validate constraints
    for portfolio optimization.
    """
    
    def __init__(self):
        """Initialize an empty constraint set."""
        # Weight bounds for each asset (min, max)
        self.weight_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Default bounds for assets without specific bounds
        self.default_bounds: Tuple[float, float] = (0.0, 1.0)
        
        # Whether to allow short selling
        self.allow_short: bool = False
        
        # Maximum position size as a fraction of the portfolio
        self.max_position_size: float = 1.0
        
        # Sector constraints (sector name -> maximum allocation)
        self.sector_constraints: Dict[str, float] = {}
        
        # Asset to sector mapping
        self.asset_sectors: Dict[str, str] = {}
    
    def set_weight_bounds(self, asset: str, min_weight: float, max_weight: float) -> None:
        """
        Set minimum and maximum weight bounds for a specific asset.
        
        Args:
            asset (str): Asset symbol
            min_weight (float): Minimum weight (0.0 to 1.0, or negative if short selling allowed)
            max_weight (float): Maximum weight (0.0 to 1.0)
            
        Raises:
            ValueError: If bounds are invalid
        """
        if max_weight < min_weight:
            raise ValueError(f"Maximum weight ({max_weight}) must be greater than or equal to minimum weight ({min_weight})")
        
        if not self.allow_short and min_weight < 0:
            raise ValueError("Negative minimum weight not allowed when short selling is disabled")
        
        if min_weight < -1.0:
            raise ValueError("Minimum weight cannot be less than -1.0")
        
        if max_weight > 1.0:
            raise ValueError("Maximum weight cannot exceed 1.0")
        
        self.weight_bounds[asset] = (min_weight, max_weight)
    
    def set_default_bounds(self, min_weight: float, max_weight: float) -> None:
        """
        Set default minimum and maximum weight bounds for assets without specific bounds.
        
        Args:
            min_weight (float): Minimum weight (0.0 to 1.0, or negative if short selling allowed)
            max_weight (float): Maximum weight (0.0 to 1.0)
            
        Raises:
            ValueError: If bounds are invalid
        """
        if max_weight < min_weight:
            raise ValueError(f"Maximum weight ({max_weight}) must be greater than or equal to minimum weight ({min_weight})")
        
        if not self.allow_short and min_weight < 0:
            raise ValueError("Negative minimum weight not allowed when short selling is disabled")
        
        if min_weight < -1.0:
            raise ValueError("Minimum weight cannot be less than -1.0")
        
        if max_weight > 1.0:
            raise ValueError("Maximum weight cannot exceed 1.0")
        
        self.default_bounds = (min_weight, max_weight)
    
    def set_allow_short(self, allow_short: bool) -> None:
        """
        Set whether to allow short selling.
        
        Args:
            allow_short (bool): Whether to allow short selling
        """
        self.allow_short = allow_short
        
        # If short selling is disabled, ensure all minimum weights are non-negative
        if not allow_short:
            for asset, (min_weight, max_weight) in self.weight_bounds.items():
                if min_weight < 0:
                    logger.warning(f"Adjusting minimum weight for {asset} from {min_weight} to 0.0 because short selling is disabled")
                    self.weight_bounds[asset] = (0.0, max_weight)
            
            if self.default_bounds[0] < 0:
                logger.warning(f"Adjusting default minimum weight from {self.default_bounds[0]} to 0.0 because short selling is disabled")
                self.default_bounds = (0.0, self.default_bounds[1])
    
    def set_max_position_size(self, max_size: float) -> None:
        """
        Set maximum position size as a fraction of the portfolio.
        
        Args:
            max_size (float): Maximum position size (0.0 to 1.0)
            
        Raises:
            ValueError: If max_size is invalid
        """
        if max_size <= 0.0 or max_size > 1.0:
            raise ValueError("Maximum position size must be between 0.0 and 1.0")
        
        self.max_position_size = max_size
        
        # Adjust any weight bounds that exceed the maximum position size
        for asset, (min_weight, max_weight) in self.weight_bounds.items():
            if max_weight > max_size:
                logger.warning(f"Adjusting maximum weight for {asset} from {max_weight} to {max_size} due to maximum position size constraint")
                self.weight_bounds[asset] = (min_weight, max_size)
        
        if self.default_bounds[1] > max_size:
            logger.warning(f"Adjusting default maximum weight from {self.default_bounds[1]} to {max_size} due to maximum position size constraint")
            self.default_bounds = (self.default_bounds[0], max_size)
    
    def set_sector_constraint(self, sector: str, max_allocation: float) -> None:
        """
        Set maximum allocation for a sector.
        
        Args:
            sector (str): Sector name
            max_allocation (float): Maximum allocation (0.0 to 1.0)
            
        Raises:
            ValueError: If max_allocation is invalid
        """
        if max_allocation <= 0.0 or max_allocation > 1.0:
            raise ValueError("Maximum sector allocation must be between 0.0 and 1.0")
        
        self.sector_constraints[sector] = max_allocation
    
    def set_asset_sector(self, asset: str, sector: str) -> None:
        """
        Set the sector for an asset.
        
        Args:
            asset (str): Asset symbol
            sector (str): Sector name
        """
        self.asset_sectors[asset] = sector
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate that all constraints are consistent.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check that sector allocations sum to at least 1.0
        if self.sector_constraints:
            sector_sum = sum(self.sector_constraints.values())
            if sector_sum < 1.0:
                return False, f"Sum of sector allocations ({sector_sum}) is less than 1.0"
        
        # Check that all assets with sector constraints have a sector assigned
        for asset in self.weight_bounds.keys():
            if self.sector_constraints and asset not in self.asset_sectors:
                return False, f"Asset {asset} has no sector assigned but sector constraints are defined"
        
        return True, ""
    
    def get_bounds(self, assets: List[str]) -> List[Tuple[float, float]]:
        """
        Get bounds for all assets, applying default bounds where needed.
        
        Args:
            assets (List[str]): List of asset symbols
            
        Returns:
            List[Tuple[float, float]]: List of (min_weight, max_weight) tuples
        """
        bounds = []
        for asset in assets:
            if asset in self.weight_bounds:
                bounds.append(self.weight_bounds[asset])
            else:
                bounds.append(self.default_bounds)
        
        # Apply maximum position size constraint
        bounds = [(min(b[0], self.max_position_size), min(b[1], self.max_position_size)) for b in bounds]
        
        # Apply short selling constraint
        if not self.allow_short:
            bounds = [(max(0.0, b[0]), b[1]) for b in bounds]
        
        return bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraints to a dictionary for use with optimization algorithms.
        
        Returns:
            Dict[str, Any]: Dictionary of constraints
        """
        return {
            'weight_bounds': self.weight_bounds,
            'allow_short': self.allow_short,
            'max_position_size': self.max_position_size,
            'sector_constraints': self.sector_constraints,
            'asset_sectors': self.asset_sectors
        }
    
    def to_scipy_constraints(self, assets: List[str]) -> List[Dict[str, Any]]:
        """
        Convert constraints to a format usable by scipy.optimize.minimize.
        
        Args:
            assets (List[str]): List of asset symbols
            
        Returns:
            List[Dict[str, Any]]: List of constraint dictionaries for scipy
        """
        constraints = []
        
        # Constraint: weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Sector constraints
        if self.sector_constraints and self.asset_sectors:
            for sector, max_allocation in self.sector_constraints.items():
                # Get indices of assets in this sector
                sector_indices = [i for i, asset in enumerate(assets) if self.asset_sectors.get(asset) == sector]
                
                if sector_indices:
                    # Create a constraint function for this sector
                    def sector_constraint(weights, indices=sector_indices, max_alloc=max_allocation):
                        return max_alloc - sum(weights[i] for i in indices)
                    
                    constraints.append({
                        'type': 'ineq',
                        'fun': sector_constraint
                    })
        
        return constraints


def create_constraint_set_from_dict(constraints_dict: Dict[str, Any]) -> ConstraintSet:
    """
    Create a ConstraintSet from a dictionary.
    
    Args:
        constraints_dict (Dict[str, Any]): Dictionary of constraints
        
    Returns:
        ConstraintSet: Constraint set object
    """
    constraint_set = ConstraintSet()
    
    # Set allow_short first to ensure weight bounds are validated correctly
    if 'allow_short' in constraints_dict:
        constraint_set.set_allow_short(constraints_dict['allow_short'])
    
    # Set weight bounds
    if 'weight_bounds' in constraints_dict:
        for asset, (min_weight, max_weight) in constraints_dict['weight_bounds'].items():
            constraint_set.set_weight_bounds(asset, min_weight, max_weight)
    
    # Set maximum position size
    if 'max_position_size' in constraints_dict:
        constraint_set.set_max_position_size(constraints_dict['max_position_size'])
    
    # Set sector constraints
    if 'sector_constraints' in constraints_dict:
        for sector, max_allocation in constraints_dict['sector_constraints'].items():
            constraint_set.set_sector_constraint(sector, max_allocation)
    
    # Set asset sectors
    if 'asset_sectors' in constraints_dict:
        for asset, sector in constraints_dict['asset_sectors'].items():
            constraint_set.set_asset_sector(asset, sector)
    
    return constraint_set