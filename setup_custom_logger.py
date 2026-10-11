import logging
import sys
import os
import multiprocessing
from datetime import datetime

# Terminal color codes
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGNETA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
LIGHT_GRAY = "\033[2m"


def setup_custom_logger(
    name="NYS_Optimisation",
    log_folder="logs",
    existing_file=None,
    rank=0,
):
    """
    Log to the console and a timestamped, MPI-rank-specific file.

    Parameters
    ----------
    name : str
        Base name of the logger.
    log_folder : str
        Directory where log files will be stored.
    existing_file : str or None
        Optional explicit log file path.
    rank : int
        MPI rank, obtained from MPI.COMM_WORLD.Get_rank().
    """

    # Race-safe directory creation
    os.makedirs(log_folder, exist_ok=True)

    # Create a distinct logger for each MPI rank
    logger_name = f"{name}.rank{rank}"
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if this logger is initialized again
    if logger.handlers:
        return logger

    # Generate a timestamped, rank-specific filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if existing_file:
        log_file = existing_file
    else:
        filename = f"{name}_{timestamp}_rank{rank}.log"
        log_file = os.path.join(log_folder, filename)

    # File log format
    file_formatter = logging.Formatter(
        "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | "
        "%(filename)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    # Console log format with colors
    console_formatter = logging.Formatter(
        f"{BLUE}%(asctime)s{RESET} | "
        f"{MAGNETA}PID:%(process)-5d{RESET} | "
        f"{GREEN}%(levelname)-8s{RESET} | "
        "%(filename)s:%(funcName)s:%(lineno)d | "
        f"{LIGHT_GRAY}%(message)s{RESET}"
    )

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initialization only from the main process
    if multiprocessing.current_process().name == "MainProcess":
        logger.info(
            "Logger initialized | MPI rank: %s | File: %s",
            rank,
            log_file,
        )

    return logger
