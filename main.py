"""
Main entry point for Concentrated Solar Power (CSP) Genetic Algorithm Optimization.
"""

import sys

# Ensure UTF-8 output streams on Windows to prevent MLflow emoji charmap encode errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from solar_opt.cli import main

if __name__ == "__main__":
    main()
