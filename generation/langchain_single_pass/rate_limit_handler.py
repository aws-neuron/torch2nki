# rate_limit_handler.py
import random
import time
import logging
from functools import wraps
from typing import Callable, Any, Dict, List, Optional, Union

# Set up logging with NullHandler to suppress console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.NullHandler()]  # Use NullHandler instead of default StreamHandler
)
logger = logging.getLogger('rate_limit_handler')

def retry_with_backoff(max_retries=60, base_delay=10, max_delay=150):
    """
    Decorator for implementing exponential backoff with jitter.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay in seconds
        max_delay (float): Maximum delay in seconds
    
    Returns:
        Function wrapped with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a rate limit related error
                    error_msg = str(e).lower()
                    rate_limited = any(phrase in error_msg for phrase in 
                                       ["rate limit", "too many requests", "throttle", 
                                        "quota exceeded", "capacity", "429"])
                    
                    # If it's not a rate limit error or we've used all retries, re-raise
                    if (not rate_limited) or (retries >= max_retries):
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(max_delay, base_delay * (2 ** retries))
                    # Add jitter (random value between 0 and delay)
                    jitter = random.uniform(0, delay)
                    wait_time = delay + jitter
                    
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds "
                        f"(attempt {retries+1}/{max_retries}). Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                    retries += 1
                    
        return wrapper
    return decorator

# Create a function specifically for chain invocation
def invoke_chain_with_retry(chain, params, max_retries=60, base_delay=10, max_delay=150, log_to_file_func=None):
    """
    Safely invoke a LangChain chain with retry logic for rate limiting.
    
    Args:
        chain: The LangChain chain to invoke
        params (dict): Parameters to pass to the chain
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay in seconds
        max_delay (float): Maximum delay in seconds
        log_to_file_func (callable, optional): Function to log messages to a file
        
    Returns:
        The result from the chain invocation
    """
    retries = 0
    stats = {"retries": 0}
    
    while True:
        try:
            return chain.invoke(params)
        except Exception as e:
            # Check if this is a rate limit related error
            error_msg = str(e).lower()
            rate_limited = any(phrase in error_msg for phrase in 
                              ["rate limit", "too many requests", "throttle", 
                               "quota exceeded", "capacity", "429"])
            
            # If it's not a rate limit error or we've used all retries, re-raise
            if (not rate_limited) or (retries >= max_retries):
                if log_to_file_func:
                    log_to_file_func(f"Error after {retries} retries: {str(e)}")
                raise
            
            # Calculate delay with exponential backoff and jitter
            delay = min(max_delay, base_delay * (2 ** retries))
            jitter = random.uniform(0, delay)
            wait_time = delay + jitter
            
            # Log the retry
            retry_msg = f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds (attempt {retries+1}/{max_retries}). Error: {str(e)}"
            logger.warning(retry_msg)
            if log_to_file_func:
                log_to_file_func(retry_msg)
            
            stats["retries"] += 1
            time.sleep(wait_time)
            retries += 1