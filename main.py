#!/usr/bin/env python3
"""
Primary entry point for Concentrated Solar Power (CSP) GA Optimization.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from solar_opt.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
