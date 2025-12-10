"""
Centralized logging configuration for the Codebase RAG MCP Server.

This module provides a unified logging setup that supports:
- Console output (stdout/stderr)
- Optional file logging with rotation
- Environment variable configuration

Usage:
    from utils.logging_config import setup_logging
    setup_logging()  # Call once at application startup
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    stream: object = None,
    verbose: bool = False,
    suppress_third_party: bool = True,
) -> logging.Logger:
    """
    Configure centralized logging for the application.

    This function sets up logging based on environment variables and parameters.
    It should be called once at application startup.

    Args:
        stream: Output stream for console handler (default: sys.stderr)
                Use sys.stdout for CLI tools, sys.stderr for MCP server
        verbose: If True, set log level to DEBUG regardless of LOG_LEVEL env var
        suppress_third_party: If True, reduce noise from third-party libraries

    Returns:
        The root logger instance

    Environment Variables:
        LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
        LOG_FILE_ENABLED: Enable file logging (true/false). Default: false
        LOG_FILE_PATH: Path to log file. Default: logs/codebase-rag.log
        LOG_FILE_MAX_SIZE: Max file size in MB before rotation. Default: 10
        LOG_FILE_BACKUP_COUNT: Number of backup files to keep. Default: 5
    """
    # Determine log level
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_stream = stream if stream is not None else sys.stderr
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    file_enabled = os.getenv("LOG_FILE_ENABLED", "false").lower() == "true"
    if file_enabled:
        file_handler = _create_file_handler(formatter, log_level)
        if file_handler:
            root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    if suppress_third_party and not verbose:
        _suppress_third_party_loggers()

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={logging.getLevelName(log_level)}, file_enabled={file_enabled}")

    return root_logger


def _create_file_handler(
    formatter: logging.Formatter,
    log_level: int,
) -> RotatingFileHandler | None:
    """
    Create a rotating file handler based on environment configuration.

    Args:
        formatter: Log formatter to use
        log_level: Logging level

    Returns:
        RotatingFileHandler instance, or None if creation fails
    """
    # Get file configuration from environment
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/codebase-rag.log")
    max_size_mb = int(os.getenv("LOG_FILE_MAX_SIZE", "10"))
    backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    try:
        # Ensure log directory exists
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        logger = logging.getLogger(__name__)
        logger.info(f"File logging enabled: {log_file_path} (max: {max_size_mb}MB, backups: {backup_count})")

        return file_handler

    except (OSError, PermissionError) as e:
        # Log to stderr since file logging failed
        sys.stderr.write(f"Warning: Failed to create log file at {log_file_path}: {e}\n")
        return None


def _suppress_third_party_loggers() -> None:
    """Suppress verbose logging from third-party libraries."""
    noisy_loggers = [
        "qdrant_client",
        "httpx",
        "httpcore",
        "urllib3",
        "transformers",
        "torch",
        "huggingface_hub",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    This is a convenience function that ensures consistent logger naming.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
