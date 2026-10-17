#!/usr/bin/env python3
"""
Custom logger configuration for the Concentrated Solar Power (CSP) optimization project.
Provides unbuffered immediate-flushing file and console handlers with ANSI formatting.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from optimized_pipeline.tracker import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent


def setup_custom_logger(
    name: str = "NYS_Optimisation",
    log_folder: str = "logs",
    existing_file: Optional[str] = None,
    rank: int = 0,
    timestamp: Optional[str] = None,
) -> logging.Logger:
    """
    Log to console (sys.stdout) and timestamped, MPI-rank-specific file with zero buffering delay.

    Parameters
    ----------
    name : str
        Base name of the logger (default: "NYS_Optimisation").
    log_folder : str
        Directory where log files will be stored.
    existing_file : str or None
        Optional explicit log file path.
    rank : int
        MPI rank (default: 0).
    timestamp : str or None
        Optional synchronized timestamp string.
    """
    folder = Path(log_folder) if os.path.isabs(log_folder) else (PROJECT_ROOT / log_folder)
    log_file_path = Path(existing_file) if existing_file else None
    return setup_logger(
        rank=rank,
        timestamp=timestamp,
        log_dir=folder,
        log_file=log_file_path,
        name=name,
        propagate=True,
        use_stdout=True,
    )


__all__ = ["setup_custom_logger"]
