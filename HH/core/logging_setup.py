"""
HarnessHarvester - Logging Setup

Rotating file logs + console output with configurable levels.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from core.constants import LOGS_DIR, LOG_LEVEL, LOG_MAX_SIZE_MB, LOG_BACKUP_COUNT

_configured = False


def setup_logging(
    level: str = None,
    log_dir: str = None,
    console: bool = True,
    name: str = "harnessharvester",
) -> logging.Logger:
    """
    Configure and return the application logger.
    Called once at startup; subsequent calls return existing logger.
    """
    global _configured

    logger = logging.getLogger(name)

    if _configured:
        return logger

    effective_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(effective_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── File handler with rotation ──
    effective_dir = log_dir or LOGS_DIR
    os.makedirs(effective_dir, exist_ok=True)
    log_path = os.path.join(effective_dir, f"{name}.log")

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(effective_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── Console handler ──
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _configured = True
    return logger


def get_logger(name: str = "harnessharvester") -> logging.Logger:
    """Get (or create) a child logger."""
    parent = logging.getLogger("harnessharvester")
    if not parent.handlers:
        setup_logging()
    if name == "harnessharvester":
        return parent
    return parent.getChild(name)
