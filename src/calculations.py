import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_portfolio_returns(stock_returns: Dict[str, pd.Series]) -> pd.Series:
    """
    Calculate equal-weighted portfolio returns from individual stock returns.
    
    Args:
        stock_returns (Dict[str, pd.Series]): Dictionary of stock symbols to their daily returns
        
    Returns:
        pd.Series: Daily portfolio returns as decimals
    """
    try:
        if not stock_returns:
            raise ValueError("No stock returns provided")
            
        # Combine all returns into a DataFrame
        returns_df = pd.DataFrame(stock_returns)
        
        # Calculate equal-weighted portfolio returns
        portfolio_returns = returns_df.mean(axis=1)
        
        if portfolio_returns.empty:
            raise ValueError("No valid data for portfolio returns calculation")
            
        return portfolio_returns
        
    except Exception as e:
        raise ValueError(f"Error calculating portfolio returns: {str(e)}")

def calculate_portfolio_volatility(stock_returns: Dict[str, pd.Series], window: int = 21) -> pd.Series:
    """
    Calculate equal-weighted portfolio volatility.
    
    Args:
        stock_returns (Dict[str, pd.Series]): Dictionary of stock symbols to their daily returns
        window (int): Rolling window size in trading days (default: 21 for monthly)
        
    Returns:
        pd.Series: Rolling annualized portfolio volatility as decimals
    """
    try:
        # Calculate portfolio returns first
        portfolio_returns = calculate_portfolio_returns(stock_returns)
        
        # Calculate portfolio volatility using the standard volatility function
        portfolio_volatility = calculate_volatility(portfolio_returns, window)
        
        return portfolio_volatility
        
    except Exception as e:
        raise ValueError(f"Error calculating portfolio volatility: {str(e)}")

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from daily returns.
    
    Args:
        returns (pd.Series): Daily percentage returns as decimals
        
    Returns:
        pd.Series: Cumulative returns as decimals (e.g., 0.15 for 15% total return)
    """
    try:
        if returns.empty:
            raise ValueError("No data provided for cumulative returns calculation")
            
        # Calculate cumulative returns using compound effect
        # Formula: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        cumulative_returns = (1 + returns).cumprod() - 1
        
        return cumulative_returns
        
    except Exception as e:
        raise ValueError(f"Error calculating cumulative returns: {str(e)}")

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily percentage returns from stock data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with 'Close' prices
        
    Returns:
        pd.Series: Daily percentage returns as decimals (e.g., 0.05 for 5%)
    """
    try:
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Calculate daily returns using percentage change of closing prices
        daily_returns = df['Close'].pct_change()
        
        # Drop the first row which will be NaN
        daily_returns = daily_returns.dropna()
        
        if daily_returns.empty:
            raise ValueError("No valid data for calculating returns")
            
        return daily_returns
        
    except KeyError:
        raise ValueError("DataFrame must contain a 'Close' price column")
    except Exception as e:
        raise ValueError(f"Error calculating daily returns: {str(e)}")

def calculate_var(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR) using either historical or parametric method.
    
    Args:
        returns (pd.Series): Daily percentage returns as decimals
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        method (str): Method to use ('historical' or 'parametric')
        
    Returns:
        float: Value at Risk as a decimal (e.g., 0.02 means 2% potential loss)
    """
    try:
        if returns.empty:
            raise ValueError("No data provided for VaR calculation")
            
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        if method == 'historical':
            # Historical VaR is simply the percentile of historical returns
            var = -returns.quantile(1 - confidence_level)
            
        elif method == 'parametric':
            # Parametric VaR assumes normal distribution
            # Calculate z-score for the confidence level
            z_score = -np.percentile(np.random.standard_normal(10000), (1 - confidence_level) * 100)
            
            # Calculate mean and standard deviation of returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Calculate VaR
            var = -(mean_return + z_score * std_return)
            
        else:
            raise ValueError("Method must be either 'historical' or 'parametric'")
            
        return var
        
    except Exception as e:
        raise ValueError(f"Error calculating VaR: {str(e)}")

def calculate_rolling_var(returns: pd.Series, window: int = 21, confidence_level: float = 0.95, method: str = 'historical') -> pd.Series:
    """
    Calculate rolling Value at Risk (VaR) over a specified window.
    
    Args:
        returns (pd.Series): Daily percentage returns as decimals
        window (int): Rolling window size in trading days
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        method (str): Method to use ('historical' or 'parametric')
        
    Returns:
        pd.Series: Rolling VaR values as decimals
    """
    try:
        if window < 2:
            raise ValueError("Window size must be at least 2 days")
            
        # Calculate rolling VaR
        rolling_var = returns.rolling(window=window, min_periods=window).apply(
            lambda x: calculate_var(x, confidence_level, method)
        )
        
        return rolling_var
        
    except Exception as e:
        raise ValueError(f"Error calculating rolling VaR: {str(e)}")

def calculate_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate rolling annualized volatility from daily returns.
    
    Args:
        returns (pd.Series): Daily percentage returns as decimals
        window (int): Rolling window size in trading days (default: 21 for monthly)
        
    Returns:
        pd.Series: Rolling annualized volatility as decimals (e.g., 0.15 for 15%)
    """
    try:
        if window < 2:
            raise ValueError("Window size must be at least 2 days")
            
        if returns.empty:
            raise ValueError("No data provided for volatility calculation")
            
        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=window, min_periods=window).std()
        
        # Annualize the volatility (multiply by sqrt of trading days in a year)
        # Assuming 252 trading days in a year
        annualized_vol = rolling_std * np.sqrt(252)
        
        # Drop NaN values from the start of the window
        annualized_vol = annualized_vol.dropna()
        
        return annualized_vol
        
    except Exception as e:
        raise ValueError(f"Error calculating volatility: {str(e)}")
