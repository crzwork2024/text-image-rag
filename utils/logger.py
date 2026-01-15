"""
Logger Management Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Configures and manages system logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored Log Formatter"""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Purple
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record"""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_format: str = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s',
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    Setup Logger

    Args:
        name: Logger name
        log_level: Log level
        log_format: Log format
        log_dir: Log file directory
        enable_file_logging: Whether to enable file logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console Handler (Colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    if enable_file_logging and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file by date
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file_path = log_dir / log_filename

        file_handler = logging.FileHandler(
            log_file_path,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
