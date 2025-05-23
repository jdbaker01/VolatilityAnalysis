import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from src.cache_manager import CacheManager
from src.logger import logger

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
    # Convert dates to UTC timezone if they're naive
    if start_date.tzinfo is None:
        start_date = pytz.UTC.localize(start_date)
    if end_date.tzinfo is None:
        end_date = pytz.UTC.localize(end_date)
        
    # Validate inputs
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid stock symbol: {symbol}")
            raise ValueError("Invalid stock symbol")
            
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            logger.error(f"Invalid date types: start_date={type(start_date)}, end_date={type(end_date)}")
            raise ValueError("Start and end dates must be datetime objects")
            
        if end_date <= start_date:
            logger.error(f"Invalid date range: start={start_date}, end={end_date}")
            raise ValueError("End date must be after start date")
    
    try:
        # Check cache first if enabled
        if use_cache:
            cached_data = _cache_manager.get_cached_data(symbol, start_date, end_date)
            if cached_data is not None:
                logger.info(f"Cache hit for {symbol} from {start_date} to {end_date}")
                return cached_data
            logger.info(f"Cache miss for {symbol} from {start_date} to {end_date}")
        
        # Download stock data if not in cache
        stock = yf.Ticker(symbol.upper())  # Convert to uppercase
        # Convert dates to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        df = stock.history(start=start_str, end=end_str)
        
        # Ensure the index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        if df.empty:
            logger.warning(f"No data available for {symbol} from {start_date} to {end_date}")
            raise ValueError(f"No data available for {symbol} in the specified date range")
            
        # Verify required columns exist
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns for {symbol}: {', '.join(missing_columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Sort by date to ensure proper order for calculations
        df = df.sort_index()
        
        # Save to cache if enabled
        if use_cache:
            _cache_manager.save_to_cache(symbol, df)
            logger.info(f"Saved data to cache for {symbol} from {start_date} to {end_date}")
            
        return df
        
    except Exception as e:
        if "Invalid ticker" in str(e):
            logger.error(f"Invalid stock symbol: {symbol}")
            raise ValueError(f"Invalid stock symbol: {symbol}")
        logger.error(f"Error retrieving data for {symbol}: {str(e)}")
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
    
    def fetch_single_stock(symbol: str) -> tuple[str, pd.DataFrame, str]:
        """Helper function to fetch data for a single stock"""
        try:
            df = get_stock_data(symbol, start_date, end_date, use_cache=use_cache)
            return symbol, df, ""
        except Exception as e:
            return symbol, pd.DataFrame(), str(e)
    
    # Fetch data in parallel
    stock_data = {}
    errors = []
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_single_stock, symbols)
        
        for symbol, df, error in results:
            if not df.empty:
                stock_data[symbol] = df
            else:
                errors.append(f"{symbol}: {error}")
    
    if not stock_data:
        error_details = "\n".join(errors)
        raise ValueError(f"No valid data retrieved for any of the provided symbols:\n{error_details}")
    
    return stock_data
