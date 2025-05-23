import os
import json
import shutil
from datetime import datetime
import pytz
from typing import Optional, Dict, Any
import pandas as pd
from src.logger import logger

class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Base directory for cache storage
        """
        self.cache_dir = cache_dir
        self._ensure_cache_directory()
    
    def _ensure_cache_directory(self) -> None:
        """Create the cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            logger.info(f"Creating cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir)
    
    def _get_symbol_dir(self, symbol: str) -> str:
        """Get the directory path for a specific symbol."""
        symbol_dir = os.path.join(self.cache_dir, symbol.upper())
        if not os.path.exists(symbol_dir):
            logger.info(f"Creating symbol directory: {symbol_dir}")
            os.makedirs(symbol_dir)
        return symbol_dir
    
    def _get_metadata_path(self, symbol: str) -> str:
        """Get the metadata file path for a symbol."""
        return os.path.join(self._get_symbol_dir(symbol), "metadata.json")
    
    def _get_data_path(self, symbol: str) -> str:
        """Get the data file path for a symbol."""
        return os.path.join(self._get_symbol_dir(symbol), "data.csv")
    
    def _load_metadata(self, symbol: str) -> Dict[str, Any]:
        """Load metadata for a symbol."""
        metadata_path = self._get_metadata_path(symbol)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"date_ranges": []}
    
    def _save_metadata(self, symbol: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for a symbol."""
        metadata_path = self._get_metadata_path(symbol)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _merge_date_ranges(self, ranges: list) -> list:
        """Merge overlapping date ranges."""
        if not ranges:
            return []
            
        # Sort ranges by start date
        ranges = sorted(ranges, key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"))
        
        merged = [ranges[0]]
        for current in ranges[1:]:
            previous = merged[-1]
            
            # Convert dates to datetime for comparison
            prev_end = datetime.strptime(previous[1], "%Y-%m-%d")
            curr_start = datetime.strptime(current[0], "%Y-%m-%d")
            
            # If current range overlaps with previous
            if curr_start <= prev_end:
                # Update end date if current range extends further
                curr_end = datetime.strptime(current[1], "%Y-%m-%d")
                prev_end = datetime.strptime(previous[1], "%Y-%m-%d")
                if curr_end > prev_end:
                    merged[-1][1] = current[1]
            else:
                merged.append(current)
                
        return merged
    
    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Convert naive datetime to timezone-aware UTC datetime."""
        if dt.tzinfo is None:
            return pytz.UTC.localize(dt)
        return dt.astimezone(pytz.UTC)

    def has_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Check if data for the given symbol and date range is in cache.
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            bool: True if data is available in cache
        """
        # Ensure dates are timezone-aware
        start_date = self._ensure_timezone_aware(start_date)
        end_date = self._ensure_timezone_aware(end_date)
        
        if end_date < start_date:
            return False
            
        metadata = self._load_metadata(symbol)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        for date_range in metadata["date_ranges"]:
            range_start = pytz.UTC.localize(datetime.strptime(date_range[0], "%Y-%m-%d"))
            range_end = pytz.UTC.localize(datetime.strptime(date_range[1], "%Y-%m-%d"))
            
            if range_start <= start_date and range_end >= end_date:
                return True
                
        return False
    
    def get_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        logger.debug(f"Attempting to retrieve cached data for {symbol} from {start_date} to {end_date}")
        """
        Retrieve data from cache for the given symbol and date range.
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Optional[pd.DataFrame]: Cached data if available, None otherwise
        """
        # Ensure dates are timezone-aware
        start_date = self._ensure_timezone_aware(start_date)
        end_date = self._ensure_timezone_aware(end_date)
        
        if end_date < start_date:
            return None
            
        if not self.has_cached_data(symbol, start_date, end_date):
            return None
            
        data_path = self._get_data_path(symbol)
        if not os.path.exists(data_path):
            return None
            
        try:
            df = pd.read_csv(data_path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)  # Ensure index is timezone-aware datetime
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]
            if filtered_df.empty:
                logger.debug(f"No data found in cache for {symbol} in specified date range")
                return None
            logger.info(f"Successfully retrieved cached data for {symbol}")
            return filtered_df
        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {str(e)}")
            return None
    
    def save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        logger.debug(f"Attempting to save data to cache for {symbol}")
        """
        Save data to cache for the given symbol.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to cache
        """
        if data.empty:
            return
            
        # Ensure data is sorted by date and index is datetime
        data = data.copy()  # Create a copy to avoid modifying original
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data = data.sort_index()
        
        # Get existing data and metadata
        data_path = self._get_data_path(symbol)
        metadata = self._load_metadata(symbol)
        
        # If we have existing data, merge it
        if os.path.exists(data_path):
            existing_data = pd.read_csv(data_path, index_col=0)
            existing_data.index = pd.to_datetime(existing_data.index, utc=True)
            # Ensure both DataFrames have timezone-aware indices
            if existing_data.index.tz is None:
                existing_data.index = existing_data.index.tz_localize('UTC')
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
                
            # Merge data, keeping the latest values for duplicates
            data = pd.concat([existing_data, data])
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
        
        # Update metadata with new date range
        start_str = data.index.min().strftime("%Y-%m-%d")
        end_str = data.index.max().strftime("%Y-%m-%d")
        
        # Add new date range and merge overlapping ranges
        metadata["date_ranges"].append([start_str, end_str])
        metadata["date_ranges"] = self._merge_date_ranges(metadata["date_ranges"])
        
        # Save data and metadata
        data.to_csv(data_path, date_format='%Y-%m-%d')
        self._save_metadata(symbol, metadata)
        logger.info(f"Successfully saved data to cache for {symbol}")
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache for a specific symbol or all symbols.
        
        Args:
            symbol (Optional[str]): Symbol to clear cache for, or None to clear all
        """
        try:
            if symbol:
                symbol_dir = os.path.join(self.cache_dir, symbol.upper())
                if os.path.exists(symbol_dir):
                    logger.info(f"Clearing cache for symbol: {symbol}")
                    shutil.rmtree(symbol_dir)
                    # Don't recreate symbol directory
            else:
                if os.path.exists(self.cache_dir):
                    logger.info("Clearing entire cache")
                    shutil.rmtree(self.cache_dir)
                    # Don't recreate cache directory
        except Exception as e:
            raise ValueError(f"Error clearing cache: {str(e)}")
