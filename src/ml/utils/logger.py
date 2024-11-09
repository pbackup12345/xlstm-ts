# src/ml/utils/logger.py

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.DEBUG  # Default log level


def is_notebook():
    """Check if the environment is a Jupyter Notebook."""
    try:
        from IPython import get_ipython
        if 'ipykernel' in sys.modules:
            return True
    except ImportError:
        return False
    return False


def get_logger(name, log_to_file=False, file_path="xlstm-ts.log"):
    """Set up and return a logger with handlers based on the environment.

    Args:
        name (str): Logger name.
        log_to_file (bool): Whether to log to a file (Desktop app).
        file_path (str): Path to the log file (if log_to_file is True).
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        # Stream handler for console (default)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(stream_handler)

        # Notebook-specific adjustments
        if is_notebook():
            # Special handler for cleaner notebook output
            from IPython.display import display
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            # Add file handler for desktop app
            if log_to_file:
                file_handler = logging.FileHandler(file_path)
                file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
                logger.addHandler(file_handler)

        logger.setLevel(LOG_LEVEL)
    return logger
