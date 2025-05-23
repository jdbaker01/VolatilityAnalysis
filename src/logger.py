import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Set up a logger that outputs to stdout.
    
    Args:
        name (str): Name for the logger (default: module name)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter for stdout
    formatter = logging.Formatter(
        '%(levelname)s - %(message)s'  # Simple format for stdout
    )
    
    # Stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)  # Log everything INFO and above
    
    # Add handler
    logger.addHandler(stream_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger('volatility_analysis')
