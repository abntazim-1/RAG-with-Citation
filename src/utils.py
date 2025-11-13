import logging
import os
import functools
import time
import traceback

def setup_logger(log_file="logs/app.log"):
    """
    Setup a logger that logs messages to both console and file.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# Initialize default logger
logger = setup_logger()

def log_exceptions(func=None, *, reraise=False):
    """
    Decorator to catch and log exceptions in any function.
    
    Args:
        func: function to wrap
        reraise: if True, re-raises exception after logging
    """
    if func is None:
        return lambda f: log_exceptions(f, reraise=reraise)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            if reraise:
                raise
            return None  # or default value if needed
    return wrapper

def log_time(func):
    """
    Decorator to log execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result
    return wrapper
