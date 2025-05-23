import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Set up a logger that works well with Streamlit.
    
    Args:
        name (str): Name for the logger (default: module name)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stream_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'  # Simpler format for Streamlit output
    )
    
    # File handler - daily rotating log file
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(f'logs/volatility_analysis_{today}.log')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Stream handler for Streamlit output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.WARNING)  # Only warnings and errors to Streamlit
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger('volatility_analysis')
