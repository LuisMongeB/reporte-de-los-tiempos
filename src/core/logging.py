"""
Centralized logging configuration for the application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from src.core.config import get_config


# Color codes for terminal output
class ColorCodes:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GREY = '\033[90m'


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds color to log levels in terminal output"""
    
    # Mapping of log levels to colors
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.GREY,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.PURPLE,
    }
    
    def format(self, record):
        # Add color to the log level
        if record.levelno in self.LEVEL_COLORS and sys.stdout.isatty():
            levelname = record.levelname
            record.levelname = (
                f"{self.LEVEL_COLORS[record.levelno]}{levelname}{ColorCodes.RESET}"
            )
            
        # Format the message
        formatted = super().format(record)
        
        # Restore the original levelname (for handlers that might use it later)
        record.levelname = levelname if 'levelname' in locals() else record.levelname
        
        return formatted


class ContextFilter(logging.Filter):
    """Filter that adds contextual information to log records"""
    
    def __init__(self, app_name: str):
        super().__init__()
        self.app_name = app_name
    
    def filter(self, record):
        # Add app name to all records
        record.app_name = self.app_name
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    app_name: Optional[str] = None,
    enable_colors: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        app_name: Application name for context
        enable_colors: Whether to enable colored output in terminal
        log_format: Custom log format string
        date_format: Custom date format string
    
    Returns:
        Logger: Configured root logger
    """
    # Get configuration if not provided
    config = get_config()
    log_level = log_level or config.log_level
    app_name = app_name or config.app_name
    
    # Default formats
    if not log_format:
        log_format = (
            '%(asctime)s | %(app_name)s | %(levelname)-8s | '
            '%(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
    
    if not date_format:
        date_format = '%Y-%m-%d %H:%M:%S'
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Set log level
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create context filter
    context_filter = ContextFilter(app_name)
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColorFormatter(log_format, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(context_filter)
    root_logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # File handler uses plain formatting (no colors)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(context_filter)
        
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    configure_third_party_loggers(log_level)
    
    # Log initial setup message
    root_logger.info(
        f"Logging initialized | Level: {log_level} | "
        f"File: {log_file or 'None'} | Colors: {enable_colors}"
    )
    
    return root_logger


def configure_third_party_loggers(log_level: str):
    """
    Configure logging levels for third-party libraries.
    
    Args:
        log_level: Application log level
    """
    # Map of third-party loggers and their minimum levels
    third_party_config = {
        # Suppress verbose logs from these libraries unless in DEBUG mode
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'uvicorn.access': logging.INFO,
        'uvicorn.error': logging.INFO,
        'sqlalchemy.engine': logging.WARNING,
        'sqlalchemy.pool': logging.WARNING,
        'openai': logging.INFO,
        'aiosqlite': logging.WARNING,
    }
    
    # If app is in DEBUG mode, allow INFO level for third-party libs
    if log_level == 'DEBUG':
        for logger_name in third_party_config:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    else:
        # Apply configured levels
        for logger_name, level in third_party_config.items():
            logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exception: Exception, message: str = None):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        message: Optional message to prepend
    """
    if message:
        logger.error(f"{message}: {type(exception).__name__}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"{type(exception).__name__}: {str(exception)}", exc_info=True)


def create_audit_logger(log_file: Path) -> logging.Logger:
    """
    Create a dedicated audit logger for security-sensitive operations.
    
    Args:
        log_file: Path to audit log file
    
    Returns:
        Logger: Configured audit logger
    """
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't propagate to root logger
    
    # Create audit log directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler with rotation
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=10,
        encoding='utf-8'
    )
    
    # Use detailed format for audit logs
    formatter = logging.Formatter(
        '%(asctime)s | AUDIT | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)
    
    return audit_logger
