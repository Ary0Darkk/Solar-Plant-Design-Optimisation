#!/usr/bin/env python3
"""
Convenience entry point for running the memory-optimized CSP pipeline.

Usage:
    python optimized_pipeline/run.py --season winter
    python optimized_pipeline/run.py --season summer --pop-size 112 --generations 100
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimized_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
