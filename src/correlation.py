import pandas as pd
import numpy as np
from typing import Dict

def calculate_correlation_matrix(stock_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple stock returns.
    
    Args:
        stock_returns (Dict[str, pd.Series]): Dictionary mapping symbols to their daily returns
        
    Returns:
        pd.DataFrame: Correlation matrix with stock symbols as index and columns
        
    Raises:
        ValueError: If less than 2 stocks are provided or if data is invalid
    """
    try:
        if len(stock_returns) < 2:
            raise ValueError("At least 2 stocks are required for correlation analysis")
            
        # Combine all returns into a DataFrame
        returns_df = pd.DataFrame(stock_returns)
        
        if returns_df.empty:
            raise ValueError("No valid data for correlation calculation")
            
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Round to 4 decimal places
        correlation_matrix = correlation_matrix.round(4)
        
        return correlation_matrix
        
    except Exception as e:
        raise ValueError(f"Error calculating correlation matrix: {str(e)}")

def format_correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Format correlation matrix for display, converting decimals to percentages.
    
    Args:
        matrix (pd.DataFrame): Raw correlation matrix
        
    Returns:
        pd.DataFrame: Formatted correlation matrix with percentage values
    """
    # Convert to percentage
    formatted_matrix = matrix * 100
    
    # Format as strings with % symbol
    formatted_matrix = formatted_matrix.map(lambda x: f"{x:.2f}%")
    
    return formatted_matrix
