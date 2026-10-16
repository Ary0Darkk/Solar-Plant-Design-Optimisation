"""
Synchronized, multi-process logging facility for MPI and multiprocessing ranks.
"""

import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from ..config.defaults import PROJECT_ROOT

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for readable terminal inspection."""

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: MAGENTA + BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"


def setup_rank_logger(
    rank: int = 0,
    timestamp: Optional[str] = None,
    log_folder: Union[str, Path] = "logs",
    logger_name: str = "NYS_Optimisation",
) -> logging.Logger:
    """
    Set up file and console logger for the specified MPI rank process.
    """
    folder = Path(log_folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    time_str = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = folder / f"opt_{time_str}_rank_{rank}.log"

    # File Handler
    file_formatter = logging.Formatter(
        "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (Rank 0 only to avoid cluttered console)
    if rank == 0:
        console_formatter = ColoredFormatter(
            "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
