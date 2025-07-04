"""
Portfolio optimization module for the Stock Volatility Analysis application.

This module provides the core functionality for optimizing portfolio weights
based on different strategies such as Maximum Sharpe Ratio, Minimum Volatility,
Equal Risk Contribution, and Maximum Return.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from scipy import optimize
from src.logger import logger


from src.constraints import ConstraintSet, create_constraint_set_from_dict

class OptimizationStrategy(ABC):
    """
    Abstract base class for all portfolio optimization strategies.
    
    All concrete optimization strategies must implement the optimize method.
    """
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio weights based on the specific strategy.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            constraints (Dict): Dictionary of constraints to apply
            
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        pass
    
    def _validate_returns(self, returns: pd.DataFrame) -> None:
        """
        Validate the returns data.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            
        Raises:
            ValueError: If returns data is invalid
        """
        if returns is None or returns.empty:
            raise ValueError("Returns data is empty or None")
        
        if returns.isnull().any().any():
            logger.warning("Returns data contains NaN values. These will be filled with 0.")
            returns.fillna(0, inplace=True)
    
    def _get_constraint_set(self, constraints: Dict) -> ConstraintSet:
        """
        Convert a constraints dictionary to a ConstraintSet object.
        
        Args:
            constraints (Dict): Dictionary of constraints
            
        Returns:
            ConstraintSet: Constraint set object
        """
        if isinstance(constraints, ConstraintSet):
            return constraints
        return create_constraint_set_from_dict(constraints)


class MinVolatilityStrategy(OptimizationStrategy):
    """
    Implementation of Minimum Volatility strategy.
    
    This strategy aims to find the portfolio weights that minimize the overall
    portfolio volatility, regardless of expected returns.
    """
    
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "Minimum Volatility"
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio weights to minimize volatility.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            constraints (Dict): Dictionary of constraints to apply
            
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        # Validate inputs
        self._validate_returns(returns)
        
        # Get asset names and count
        assets = returns.columns
        n_assets = len(assets)
        
        # Calculate annualized covariance matrix
        cov_matrix = returns.cov() * 252  # Assuming 252 trading days in a year
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Convert constraints to ConstraintSet
        constraint_set = self._get_constraint_set(constraints)
        
        # Get bounds for all assets
        bounds = constraint_set.get_bounds(assets)
        
        # Get scipy constraints
        constraints_list = constraint_set.to_scipy_constraints(assets)
        
        # Define the portfolio volatility function to minimize
        def portfolio_volatility(weights):
            weights_array = np.array(weights)
            portfolio_variance = weights_array.T @ cov_matrix @ weights_array
            return np.sqrt(portfolio_variance)
        
        # Optimize using scipy's minimize function
        optimization_result = optimize.minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Check if optimization was successful
        if not optimization_result['success']:
            logger.warning(f"Optimization failed: {optimization_result['message']}")
        
        # Get the optimized weights
        optimized_weights = optimization_result['x']
        
        # Convert to dictionary
        weights_dict = {asset: weight for asset, weight in zip(assets, optimized_weights)}
        
        return weights_dict


class MaxReturnStrategy(OptimizationStrategy):
    """
    Implementation of Maximum Return strategy.
    
    This strategy aims to find the portfolio weights that maximize the expected
    return, regardless of risk. Without constraints, this will typically allocate
    100% to the asset with the highest expected return.
    """
    
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "Maximum Return"
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio weights to maximize expected return.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            constraints (Dict): Dictionary of constraints to apply
            
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        # Validate inputs
        self._validate_returns(returns)
        
        # Get asset names and count
        assets = returns.columns
        n_assets = len(assets)
        
        # Calculate annualized mean returns
        mean_returns = returns.mean() * 252  # Assuming 252 trading days in a year
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Convert constraints to ConstraintSet
        constraint_set = self._get_constraint_set(constraints)
        
        # Get bounds for all assets
        bounds = constraint_set.get_bounds(assets)
        
        # Get scipy constraints
        constraints_list = constraint_set.to_scipy_constraints(assets)
        
        # Define the negative expected return function to minimize
        def negative_expected_return(weights):
            weights_array = np.array(weights)
            portfolio_return = np.sum(mean_returns * weights_array)
            return -portfolio_return  # Negative because we want to maximize return
        
        # Optimize using scipy's minimize function
        optimization_result = optimize.minimize(
            negative_expected_return,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Check if optimization was successful
        if not optimization_result['success']:
            logger.warning(f"Optimization failed: {optimization_result['message']}")
        
        # Get the optimized weights
        optimized_weights = optimization_result['x']
        
        # Convert to dictionary
        weights_dict = {asset: weight for asset, weight in zip(assets, optimized_weights)}
        
        return weights_dict


class EqualRiskStrategy(OptimizationStrategy):
    """
    Implementation of Equal Risk Contribution strategy.
    
    This strategy aims to find portfolio weights where each asset contributes
    equally to the total portfolio risk (risk parity).
    """
    
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "Equal Risk Contribution"
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio weights for equal risk contribution.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            constraints (Dict): Dictionary of constraints to apply
            
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        # Validate inputs
        self._validate_returns(returns)
        
        # Get asset names and count
        assets = returns.columns
        n_assets = len(assets)
        
        # Calculate annualized covariance matrix
        cov_matrix = returns.cov() * 252  # Assuming 252 trading days in a year
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define the risk contribution objective function
        def risk_contribution_objective(weights):
            weights_array = np.array(weights)
            portfolio_volatility = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
            
            # Calculate marginal contribution to risk for each asset
            mcr = cov_matrix @ weights_array
            
            # Calculate risk contribution for each asset
            risk_contribution = weights_array * mcr / portfolio_volatility
            
            # Target risk contribution (equal for all assets)
            target_risk = portfolio_volatility / n_assets
            
            # Calculate the sum of squared deviations from target risk contribution
            deviation = sum((risk_contribution - target_risk) ** 2)
            
            return deviation
        
        # Convert constraints to ConstraintSet
        constraint_set = self._get_constraint_set(constraints)
        
        # Get bounds for all assets
        bounds = constraint_set.get_bounds(assets)
        
        # Get scipy constraints
        constraints_list = constraint_set.to_scipy_constraints(assets)
        
        # Optimize using scipy's minimize function
        optimization_result = optimize.minimize(
            risk_contribution_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        # Check if optimization was successful
        if not optimization_result['success']:
            logger.warning(f"Optimization failed: {optimization_result['message']}")
        
        # Get the optimized weights
        optimized_weights = optimization_result['x']
        
        # Convert to dictionary
        weights_dict = {asset: weight for asset, weight in zip(assets, optimized_weights)}
        
        return weights_dict


class MaxSharpeStrategy(OptimizationStrategy):
    """
    Implementation of Maximum Sharpe Ratio strategy.
    
    This strategy aims to find the portfolio weights that maximize the Sharpe ratio,
    which is a measure of risk-adjusted return.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the Maximum Sharpe Ratio strategy.
        
        Args:
            risk_free_rate (float): Risk-free rate used in Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "Maximum Sharpe Ratio"
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio weights to maximize the Sharpe ratio.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            constraints (Dict): Dictionary of constraints to apply
            
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        # Validate inputs
        self._validate_returns(returns)
        
        # Get asset names and count
        assets = returns.columns
        n_assets = len(assets)
        
        # Calculate annualized mean returns and covariance matrix
        mean_returns = returns.mean() * 252  # Assuming 252 trading days in a year
        cov_matrix = returns.cov() * 252  # Annualized covariance
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Convert constraints to ConstraintSet
        constraint_set = self._get_constraint_set(constraints)
        
        # Get bounds for all assets
        bounds = constraint_set.get_bounds(assets)
        
        # Get scipy constraints
        constraints_list = constraint_set.to_scipy_constraints(assets)
        
        # Define the negative Sharpe ratio function to minimize
        def negative_sharpe_ratio(weights):
            weights_array = np.array(weights)
            portfolio_return = np.sum(mean_returns * weights_array)
            portfolio_volatility = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
            
            # Handle case where volatility is zero or very small
            if portfolio_volatility < 1e-8:
                return -999  # Large negative number to avoid division by zero
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe  # Negative because we want to maximize Sharpe ratio
        
        # Optimize using scipy's minimize function
        optimization_result = optimize.minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Check if optimization was successful
        if not optimization_result['success']:
            logger.warning(f"Optimization failed: {optimization_result['message']}")
        
        # Get the optimized weights
        optimized_weights = optimization_result['x']
        
        # Convert to dictionary
        weights_dict = {asset: weight for asset, weight in zip(assets, optimized_weights)}
        
        return weights_dict


class PortfolioOptimizer:
    """
    Main controller for portfolio optimization.
    
    This class coordinates the optimization process, applying the selected
    strategy with the specified constraints.
    """
    
    def __init__(self, returns: pd.DataFrame, strategy: OptimizationStrategy):
        """
        Initialize the optimizer with returns data and a strategy.
        
        Args:
            returns (pd.DataFrame): Historical returns for all assets
            strategy (OptimizationStrategy): The optimization strategy to use
        """
        self.returns = returns
        self.strategy = strategy
        self.constraints = {}
        
        # Validate inputs
        if returns is None or returns.empty:
            raise ValueError("Returns data is empty or None")
        
        if strategy is None:
            raise ValueError("Optimization strategy cannot be None")
    
    def set_constraints(self, constraints: Dict) -> None:
        """
        Set optimization constraints.
        
        Args:
            constraints (Dict): Dictionary of constraints to apply
        """
        if isinstance(constraints, ConstraintSet):
            self.constraints = constraints
        else:
            self.constraints = create_constraint_set_from_dict(constraints)
    
    def optimize(self) -> Dict[str, float]:
        """
        Run optimization with selected strategy and constraints.
        
        Returns:
            Dict[str, float]: Optimized weights for each asset
        """
        try:
            logger.info(f"Starting optimization with strategy: {self.strategy.get_name()}")
            
            # Run the optimization
            optimized_weights = self.strategy.optimize(self.returns, self.constraints)
            
            # Validate the results
            self._validate_weights(optimized_weights)
            
            logger.info(f"Optimization complete. Optimized weights: {optimized_weights}")
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise ValueError(f"Optimization failed: {str(e)}")
    
    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """
        Validate the optimized weights.
        
        Args:
            weights (Dict[str, float]): Optimized weights for each asset
            
        Raises:
            ValueError: If weights are invalid
        """
        if not weights:
            raise ValueError("Optimization produced empty weights")
        
        # Check that all assets have weights
        if set(weights.keys()) != set(self.returns.columns):
            raise ValueError("Optimized weights do not match the assets in returns data")
        
        # Check that weights sum to approximately 1.0
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            logger.warning(f"Weights sum to {weight_sum}, not 1.0. Normalizing weights.")
            # Normalize weights
            for asset in weights:
                weights[asset] /= weight_sum


def calculate_portfolio_metrics(weights: Dict[str, float], returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate key metrics for a portfolio.
    
    Args:
        weights (Dict[str, float]): Portfolio weights
        returns (pd.DataFrame): Historical returns for all assets
        
    Returns:
        Dict[str, float]: Dictionary with expected return, volatility, and Sharpe ratio
    """
    # Convert weights to numpy array in the same order as returns columns
    weight_array = np.array([weights[asset] for asset in returns.columns])
    
    # Calculate expected returns (annualized)
    mean_returns = returns.mean() * 252  # Assuming 252 trading days in a year
    expected_return = np.sum(mean_returns * weight_array)
    
    # Calculate portfolio volatility (annualized)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    portfolio_variance = weight_array.T @ cov_matrix @ weight_array
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate Sharpe ratio (assuming 0 risk-free rate for simplicity)
    # In a real implementation, this should use a configurable risk-free rate
    risk_free_rate = 0.0
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        "expected_return": expected_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio
    }


def calculate_risk_contribution(weights: Dict[str, float], returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the risk contribution of each asset in the portfolio.
    
    Args:
        weights (Dict[str, float]): Portfolio weights
        returns (pd.DataFrame): Historical returns for all assets
        
    Returns:
        Dict[str, float]: Risk contribution of each asset
    """
    # Convert weights to numpy array in the same order as returns columns
    assets = list(returns.columns)
    weight_array = np.array([weights[asset] for asset in assets])
    
    # Calculate portfolio volatility and marginal contributions
    cov_matrix = returns.cov() * 252  # Annualized covariance
    portfolio_variance = weight_array.T @ cov_matrix @ weight_array
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal contribution to risk
    mcr = cov_matrix @ weight_array
    
    # Component contribution to risk
    ccr = weight_array * mcr
    
    # Risk contribution
    rc = ccr / portfolio_volatility if portfolio_volatility > 0 else np.zeros_like(weight_array)
    
    # Convert to dictionary
    risk_contribution = {asset: rc[i] for i, asset in enumerate(assets)}
    
    return risk_contribution