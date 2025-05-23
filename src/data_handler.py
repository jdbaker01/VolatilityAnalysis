import yfinance as yf
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from cache_manager import CacheManager

# Initialize cache manager
_cache_manager = CacheManager()

def get_stock_data(symbol: str, start_date: datetime, end_date: datetime, use_cache: bool = True) -> pd.DataFrame:
    """
    Retrieve stock data for a single symbol, using cache if available.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        use_cache (bool): Whether to use cache (default: True)
        
    Returns:
        pd.DataFrame: DataFrame containing stock data with columns:
                     Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
                     
    Raises:
        ValueError: If symbol is invalid, dates are invalid, or no data is available
    """
    # Validate inputs
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Invalid stock symbol")
        
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("Start and end dates must be datetime objects")
        
    if end_date <= start_date:
        raise ValueError("End date must be after start date")
    
    try:
        # Check cache first if enabled
        if use_cache:
            cached_data = _cache_manager.get_cached_data(symbol, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Download stock data if not in cache
        stock = yf.Ticker(symbol.upper())  # Convert to uppercase
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol} in the specified date range")
            
        # Verify required columns exist
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Sort by date to ensure proper order for calculations
        df = df.sort_index()
        
        # Save to cache if enabled
        if use_cache:
            _cache_manager.save_to_cache(symbol, df)
            
        return df
        
    except Exception as e:
        if "Invalid ticker" in str(e):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        raise ValueError(f"Error retrieving data for {symbol}: {str(e)}")
def get_multiple_stocks_data(symbols: List[str], start_date: datetime, end_date: datetime, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Retrieve data for multiple stock symbols in parallel, using cache when available.
    
    Args:
        symbols (List[str]): List of stock symbols
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        use_cache (bool): Whether to use cache (default: True)
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbols to their respective DataFrames
        
    Raises:
        ValueError: If symbols are invalid, dates are invalid, or no data is available
    """
    # Validate inputs
    if not symbols:
        raise ValueError("No stock symbols provided")
        
    if not all(isinstance(s, str) and s.strip() for s in symbols):
        raise ValueError("Invalid stock symbols")
        
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("Start and end dates must be datetime objects")
        
    if end_date <= start_date:
        raise ValueError("End date must be after start date")
    
    # Remove duplicates and clean symbols
    symbols = list(set(s.strip().upper() for s in symbols))
    
    def fetch_single_stock(symbol: str) -> tuple[str, pd.DataFrame]:
        """Helper function to fetch data for a single stock"""
        try:
            df = get_stock_data(symbol, start_date, end_date, use_cache=use_cache)
            return symbol, df
        except Exception as e:
            # Return empty DataFrame for failed fetches
            return symbol, pd.DataFrame()
    
    # Fetch data in parallel
    stock_data = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_single_stock, symbols)
        
        for symbol, df in results:
            if not df.empty:
                stock_data[symbol] = df
    
    if not stock_data:
        raise ValueError("No valid data retrieved for any of the provided symbols")
    
    return stock_data
